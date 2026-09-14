"""Run a query through a tagger container and read back the vector it emits.

The protocol
------------
https://docs.eluv.io/docs/ai-ml/ai-ml-tagger/model-protocol/ — an OCI image that
stays alive, reads newline-separated *file paths* on stdin, and appends
newline-delimited JSON messages to the `--output-path` it was started with:

    {"type": "tag",      "data": {"start_time":…, "end_time":…, "source_media":…,
                                  "vector": […], "tag": …, "additional_info": {…}}}
    {"type": "progress", "data": {"source_media": "…"}}   # that file is done
    {"type": "error",    "data": {"message": "…", "source_media": "…"}}

Every model this service can query therefore runs the same way, and this module
is the whole of what it takes to query one. There is no per-model Python here and
no model weights in this process: the container owns the tower, its dependencies
and its device, which is what lets three mutually incompatible stacks (SigLIP 2
on torch 2.x, Qwen on transformers ≥4.57, InsightFace on numpy <1.20) be queried
from one interpreter that has none of them installed.

One addition to the protocol
----------------------------
A **`.txt` file of newline-separated queries** is a valid input, and the
container answers it with one vector tag per line. That is how a *text* query
reaches a container whose stdin is already spoken for by the file-path stream:
the query is written to a file like any other medium.

Why the container is long-lived rather than one run per query
-------------------------------------------------------------
Because the protocol says it may be — "the container must stay alive and accept
newline separated input files via stdin". Loading a multi-GB checkpoint costs
tens of seconds, and a run-per-query would pay it on every search. So a session
is started on the first query against a model and then kept, which is the same
bargain the in-process towers used to make by keeping their weights resident.

The consequence is that results arrive by *tailing* the output file rather than
by waiting for the process to exit: a query is finished when a `progress` (or
`error`) message naming its file appears. `_Reader` therefore holds its position
in the file across queries, so a slow or abandoned query's messages are still
read in order by the next one rather than being skipped or re-read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import config

logger = logging.getLogger(__name__)


class TaggerError(RuntimeError):
    """A tagger container could not be run, or answered with an error."""


def device_args(device: Any, runtime: str = "") -> List[str]:
    """Runtime flags placing the container on `device`.

    Accepts a CUDA device index (`0`), several (`[0, 1]` or `"0,1"`), or `all`.
    Every container runs on a GPU, so there is no value meaning "no GPU" and no
    default: a model whose placement is not configured raises, rather than
    running somewhere nobody chose. Being wrong about this does not fail, it
    just takes minutes per query, which reads as the service hanging.

    Inside the container the GPUs named here are renumbered from zero, so a
    model that asks for a bare `cuda` lands on the card configured for it
    without knowing anything about this. That is the whole reason to place a
    model by configuration rather than by having it choose: with one container
    per model, each one picking "the emptiest GPU" independently is how two
    large models end up on the same card.

    docker and podman spell it differently -- `--gpus` against the nvidia
    runtime, CDI devices against podman's -- so the runtime picks the spelling.
    """
    if device is None or (isinstance(device, str) and not device.strip()):
        raise TaggerError(
            "no CUDA device is configured for this container. Set `device:` on "
            "the model, or `container: device:` for all of them -- every "
            "container runs on a GPU."
        )
    if isinstance(device, bool):
        # `device: true` is not a GPU index and has no sensible reading.
        raise TaggerError("`device` must be a CUDA device index, a list of them, or 'all'")

    podman = "podman" in (runtime or RUNTIME)
    if isinstance(device, str) and device.strip().lower() == "all":
        return ["--device", "nvidia.com/gpu=all"] if podman else ["--gpus", "all"]

    raw = device if isinstance(device, (list, tuple)) else str(device).split(",")
    indices = []
    for item in raw:
        text = str(item).strip()
        if not text.isdigit():
            raise TaggerError(
                f"{device!r} is not a CUDA device index. Use an index (0), a list "
                "of them ([0, 1]), or 'all'."
            )
        indices.append(int(text))
    if not indices:
        raise TaggerError("`device` names no CUDA device")
    if podman:
        return [arg for i in indices for arg in ("--device", f"nvidia.com/gpu={i}")]
    return ["--gpus", f"device={','.join(str(i) for i in indices)}"]


# Where this service's staging directory is mounted inside the container. The
# paths written to stdin and the --output-path are both under it, so the
# container never sees a host path and nothing outside the staging directory is
# exposed to it.
MOUNT = "/ev"

# Everything below comes from `config.yml`'s `container:` section, with the
# environment winning over it; see `config.setting`.

# `docker` or `podman` -- the protocol names both, and they take the same flags
# for what is used here (-i, --rm, -v).
RUNTIME = str(config.setting("container", "runtime", "EV_CONTAINER_RUNTIME", "docker"))

# Extra arguments for every container, e.g. `--gpus all` or a bind mount of a
# weight cache. Per-model arguments live under that model instead. A list in the
# file; a single shell-quoted string in the environment, which is all an env var
# can carry.
EXTRA_ARGS = config.as_args(config.setting("container", "args", "EV_CONTAINER_ARGS", []))

# The GPU a model that names none of its own is run on. Per-model `device:` wins
# over it; see `device_args` for what it accepts. There is no fallback beyond
# this one: a container with no device configured is a configuration error.
DEFAULT_DEVICE = config.setting("container", "device", "EV_CONTAINER_DEVICE", None)

# How long to wait for a query's `progress` message. Generous because it covers
# the first query's model load, which is the slow one: on a cold container the
# checkpoint is read and moved onto the device before any tag is written.
QUERY_TIMEOUT = float(config.setting("container", "timeout", "EV_CONTAINER_TIMEOUT", 900))

# How often the output file is checked for new lines while waiting.
POLL_SECONDS = 0.05

# How often the keepalive sweep checks that every started container is still
# running. A container that dies between queries -- OOM-killed, or the daemon
# restarted -- is otherwise only noticed by the query that needs it, which then
# pays the whole cold start while someone waits for a search.
KEEPALIVE_SECONDS = float(
    config.setting("container", "keepalive", "EV_CONTAINER_KEEPALIVE", 30)
)

# Ceiling on the sweep's backoff for a container that keeps dying, so a
# crash-looping image is retried occasionally rather than every sweep -- and so
# one that was waiting on something external still comes back on its own.
MAX_RESTART_DELAY = 600.0

# Where each container's own output is kept. Beside the service log rather than
# in the staging directory it used to share with the query files: the staging
# directory is removed when a container stops, which deleted the log exactly
# when something had gone wrong enough to stop it. These survive a restart, so
# the log of the container that died is still there after the one that replaced
# it has started.
CONTAINER_LOG_DIR = Path(
    config.setting(
        "container", "log_dir", "EV_CONTAINER_LOG_DIR",
        Path(__file__).resolve().parents[1] / "logs" / "containers",
    )
)

# A container log is rotated once when it passes this, so a model that prints a
# progress bar per query cannot fill a disk. One generation is enough: these are
# for reading after something went wrong, not an audit trail.
CONTAINER_LOG_MAX_BYTES = 32 * 1024 * 1024

# Tail of the container's stderr quoted when it dies or times out. Enough to
# carry a traceback's last frames, not so much that it buries the message.
STDERR_TAIL = 4000


class TaggerSession:
    """One running container, queried one file at a time.

    Serialized by a lock: the protocol's output is a single stream keyed by
    `source_media`, and taggers are not required to interleave, so two
    concurrent queries could not be told apart reliably. Searches are
    interactive and one at a time per user, so nothing is lost by it.

    `params` is fixed for the session because `--params` is a *launch* argument.
    Two indexes built with different recipes therefore get different sessions --
    see `session_for`, which keys on exactly that.
    """

    def __init__(
        self,
        image: str,
        params: Optional[Dict[str, Any]] = None,
        args: Sequence[str] = (),
        device: Any = None,
    ) -> None:
        self.image = image
        self.params = dict(params or {})
        self.args = list(args)
        # Validated here rather than at launch, so a bad index in the config is
        # reported when the session is asked for and not minutes later.
        self.device = device
        self.device_args = device_args(device)
        self.log_path = _log_path(
            json.dumps([image, self.params, self.args, device], sort_keys=True), image
        )
        # Where the runtime writes the container's id, once it is started.
        self._cidfile: Optional[Path] = None
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._stage: Optional[Path] = None
        self._reader: Optional[_Reader] = None
        self._stderr_path: Optional[Path] = None
        # Backoff for the keepalive sweep; see `_restart_delay`.
        self._failures = 0
        self._retry_after = 0.0

    # -------------------------------------------------------------- querying

    def tag_file(self, data: bytes, suffix: str) -> List[Dict[str, Any]]:
        """Feed one file to the container; return the `tag` messages it emits.

        `suffix` is kept on the staged file because a tagger picks its decoder by
        extension -- a clip written as `.bin` is not read as the container it is.
        """
        with self._lock:
            self._ensure_running()
            assert self._stage is not None and self._proc is not None

            name = f"{uuid.uuid4().hex}{suffix}"
            host_path = self._stage / "in" / name
            host_path.write_bytes(data)
            try:
                tags = self._run_one(f"{MOUNT}/in/{name}")
                # It answered, so it is not crash-looping: let the sweep put it
                # back promptly again if it dies later.
                self._failures, self._retry_after = 0, 0.0
                return tags
            finally:
                try:
                    host_path.unlink()
                except OSError:
                    pass

    def tag_text(self, queries: Sequence[str]) -> List[Dict[str, Any]]:
        """Embed newline-separated queries via the `.txt` input the protocol takes.

        A query may not contain a newline -- the file's one-query-per-line shape
        is the whole of its structure, so an embedded newline would silently
        become two queries and return a vector for half of what was asked.
        """
        cleaned = [q.strip() for q in queries]
        if not any(cleaned):
            raise TaggerError("text query is empty")
        for query in cleaned:
            if "\n" in query or "\r" in query:
                raise TaggerError(
                    "a text query cannot contain a newline: the query file is "
                    "one query per line"
                )
        return self.tag_file(("\n".join(cleaned) + "\n").encode("utf-8"), ".txt")

    def _run_one(self, container_path: str) -> List[Dict[str, Any]]:
        """Write one path to stdin and collect messages until that file is done."""
        proc, reader = self._proc, self._reader
        assert proc is not None and proc.stdin is not None and reader is not None

        try:
            proc.stdin.write(container_path + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            # Read before shutting down: _shutdown removes the staging directory
            # the log lives in, and the log is the whole diagnosis.
            printed = self._stderr()
            self._shutdown()
            raise TaggerError(
                f"{self.image} exited before it could be given a query: {printed}"
            ) from exc

        tags: List[Dict[str, Any]] = []
        deadline = time.monotonic() + QUERY_TIMEOUT
        while True:
            for message in reader.read():
                kind = message.get("type")
                data = message.get("data") or {}
                if not _same_media(data.get("source_media"), container_path):
                    # Another file's message -- a leftover from a query that
                    # timed out, since queries are serialized. Logged rather
                    # than dropped silently, because a container that answers
                    # the wrong path for every query would otherwise just hang.
                    logger.warning(
                        f"{self.image}: {kind} message for "
                        f"{data.get('source_media')!r}, expected {container_path!r}"
                    )
                    continue
                if kind == "error":
                    raise TaggerError(
                        f"{self.image}: {data.get('message') or 'unspecified error'}"
                    )
                if kind == "tag":
                    tags.append(data)
                elif kind == "progress":
                    return tags

            if proc.poll() is not None:
                printed = self._stderr()   # before _shutdown removes the log
                self._shutdown()
                raise TaggerError(
                    f"{self.image} exited with code {proc.returncode} while embedding "
                    f"the query: {printed}"
                )
            if time.monotonic() > deadline:
                raise TaggerError(
                    f"{self.image} did not finish the query within {QUERY_TIMEOUT:.0f}s. "
                    "The first query also loads the model, so raise EV_CONTAINER_TIMEOUT "
                    f"if that is simply slow here. Last output: {self._stderr()}"
                )
            time.sleep(POLL_SECONDS)

    # ------------------------------------------------------------- lifecycle

    @property
    def container_id(self) -> Optional[str]:
        """The running container's id, for `podman logs` / `stats` / `exec`.

        Read on demand rather than remembered: the runtime writes it a moment
        after the process starts, so reading it once at launch is a race, and
        nothing this service does depends on having it.
        """
        return _read_cidfile(self._cidfile) if self._cidfile else None

    @property
    def alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    @property
    def _restart_delay(self) -> float:
        """How long the sweep waits before trying a container that keeps dying.

        A container that starts and immediately exits -- a wrong image, a
        missing mount, a model that cannot find its weights -- would otherwise
        be restarted every sweep for as long as the service runs, filling the
        log and pulling an image on a loop. So each consecutive failed restart
        doubles the wait, up to a ceiling, and a query that actually succeeds
        clears it.

        This delays nothing a user waits on: a *query* always tries to start the
        container it needs, backoff or not, and reports what went wrong.
        """
        return min(KEEPALIVE_SECONDS * 2 ** self._failures, MAX_RESTART_DELAY)

    def due_for_restart(self) -> bool:
        """Whether the keepalive sweep should try this session now."""
        return not self.alive and time.monotonic() >= self._retry_after

    def ensure_running(self, blocking: bool = True) -> bool:
        """Start the container if it is not up. Returns whether it is now.

        `blocking=False` is for the keepalive sweep: a query holds the lock for
        as long as the container takes to answer, and a sweep has no reason to
        queue behind one. A session that is mid-query is running by definition,
        and one whose query is what killed it is restarted by the next query
        anyway.
        """
        if self.alive:
            return True
        if not self._lock.acquire(blocking=blocking):
            return self.alive
        try:
            self._ensure_running()
        finally:
            self._lock.release()
        return self.alive

    def _ensure_running(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._proc is not None:
            logger.warning(f"{self.image} is no longer running; restarting it")
            self._shutdown()
        self._start()

    def _start(self) -> None:
        _evict_conflicting(self)
        stage = Path(tempfile.mkdtemp(prefix="ev-tagger-"))
        (stage / "in").mkdir()
        out = stage / "out.jsonl"
        out.touch()
        # World-writable: the container usually runs as root, but it may not run
        # as *this* uid, and it has to be able to write the output file and read
        # the inputs staged beside it.
        for path in (stage, stage / "in"):
            path.chmod(0o777)
        out.chmod(0o666)

        cidfile = stage / "cid"
        cmd = [
            RUNTIME, "run", "-i", "--rm",
            "-v", f"{stage}:{MOUNT}",
            # A host path, not a container one: the runtime writes the id here
            # so a running container can be reached by it -- `podman stats`,
            # `exec`, or `logs` for the runtime's own copy of this output.
            "--cidfile", str(cidfile),
            *self.device_args,
            *EXTRA_ARGS,
            *self.args,
            self.image,
            "--output-path", f"{MOUNT}/out.jsonl",
        ]
        if self.params:
            # Only when there is something to say: a container is free to reject
            # an unexpected `--params`, and an empty object says nothing anyway.
            cmd += ["--params", json.dumps(self.params, sort_keys=True)]

        log_path = self._prepare_log(cmd)
        logger.info(
            f"starting tagger container: {' '.join(shlex.quote(c) for c in cmd)} "
            f"(its output: {log_path})"
        )
        try:
            # The container's own logging. A file rather than a pipe because
            # nothing reads it while a query runs, and a full pipe buffer would
            # deadlock the container mid-query. Appended to, so the log of a
            # container that died outlives it. Closed here because Popen dups
            # the descriptor.
            with log_path.open("ab") as log:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
        except FileNotFoundError as exc:
            # Nothing was started, so the staging directory would otherwise be
            # left behind on every retry.
            shutil.rmtree(stage, ignore_errors=True)
            raise TaggerError(
                f"{RUNTIME} is not installed or not on PATH, so no query can be "
                "embedded. Set EV_CONTAINER_RUNTIME if the runtime is podman or "
                "is installed elsewhere."
            ) from exc
        except OSError as exc:
            shutil.rmtree(stage, ignore_errors=True)
            raise TaggerError(f"could not start {self.image}: {exc}") from exc

        self._proc = proc
        self._stage = stage
        self._stderr_path = log_path
        self._reader = _Reader(out)
        self._cidfile = cidfile
        # The only read that waits: the runtime writes the id a moment after
        # the process starts, and this one is for a log line, not a query.
        container_id = _read_cidfile(cidfile, wait=2.0)
        if container_id:
            logger.info(f"{self.image} is container {container_id[:12]}")

    def _prepare_log(self, cmd: Sequence[str]) -> Path:
        """The container's log file, rotated if large, with this start recorded.

        The header matters as much as the output: these files are appended to
        across restarts, so without it a crash loop reads as one incoherent
        stream rather than as the same failure N times.
        """
        path = self.log_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file() and path.stat().st_size > CONTAINER_LOG_MAX_BYTES:
                path.replace(path.with_suffix(".log.1"))
            with path.open("a", encoding="utf-8") as log:
                log.write(
                    f"\n=== {datetime.now().isoformat(timespec='seconds')} "
                    f"{' '.join(shlex.quote(c) for c in cmd)}\n"
                )
        except OSError as exc:
            # Not fatal: a container that cannot be logged can still answer
            # queries, and saying so beats refusing to start it.
            logger.warning(f"could not prepare {path}: {exc}")
        return path

    def _shutdown(self) -> None:
        """Stop the container and forget its staging directory."""
        proc = self._proc
        if proc is not None:
            try:
                if proc.stdin and not proc.stdin.closed:
                    # Closing stdin is how the protocol says "no more files";
                    # a container that honours it exits on its own.
                    proc.stdin.close()
            except OSError:
                pass
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        if self._reader is not None:
            self._reader.close()
        if self._stage is not None:
            shutil.rmtree(self._stage, ignore_errors=True)
        # _stderr_path is deliberately kept: it is outside the staging directory
        # now, and the log of a container that just died is the thing most worth
        # reading. container_id is cleared -- that one really is gone.
        self._proc = self._stage = self._reader = self._cidfile = None

    def close(self) -> None:
        with self._lock:
            self._shutdown()

    def _stderr(self) -> str:
        """The tail of what the container printed, for an error message."""
        if self._stderr_path is None or not self._stderr_path.is_file():
            return "(it printed nothing)"
        try:
            text = self._stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(its output could not be read)"
        text = text.strip()
        if not text:
            return "(it printed nothing)"
        return text[-STDERR_TAIL:]


# Log filenames already claimed, so two sessions on the same image (different
# recipes) do not write to one file. The common case is one session per image
# and therefore a filename that is just the image, which is the point: a log is
# only useful if it can be found without looking anything up.
_log_names: Dict[str, str] = {}
_log_names_lock = threading.Lock()


def _log_path(key: str, image: str) -> Path:
    """The file this session's container output is appended to."""
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", image.rsplit("/", 1)[-1].split(":")[0]) or "container"
    with _log_names_lock:
        if key not in _log_names:
            name = slug
            if name in _log_names.values():
                name = f"{slug}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"
            _log_names[key] = name
    return CONTAINER_LOG_DIR / f"{_log_names[key]}.log"


class _Reader:
    """Incremental reader for the container's newline-delimited output file.

    Holds its offset across calls, and buffers a trailing partial line: the
    container appends to this file while it is being read, so a read can land
    mid-line, and treating that as a whole line would lose the message.
    """

    def __init__(self, path: Path) -> None:
        self._handle = path.open("r", encoding="utf-8", errors="replace")
        self._partial = ""

    def read(self) -> List[Dict[str, Any]]:
        """Every complete message appended since the last call."""
        messages: List[Dict[str, Any]] = []
        chunk = self._handle.read()
        if not chunk:
            return messages
        self._partial += chunk
        *lines, self._partial = self._partial.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"tagger wrote a line that is not JSON: {line[:200]!r}")
                continue
            if isinstance(message, dict):
                messages.append(message)
        return messages

    def close(self) -> None:
        try:
            self._handle.close()
        except OSError:
            pass


def _read_cidfile(path: Path, wait: float = 0.0) -> Optional[str]:
    """The container id the runtime wrote, or None if it never appeared."""
    deadline = time.monotonic() + wait
    while True:
        try:
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
        except OSError:
            pass
        if time.monotonic() >= deadline:
            return None
        time.sleep(POLL_SECONDS)


def _same_media(reported: Any, expected: str) -> bool:
    """Whether a message's `source_media` names the file that was submitted.

    The protocol says it echoes the input path, and it usually does. The
    basename is accepted too, because a container that resolves the path (to an
    absolute one, or relative to its own workdir) is still unambiguous here:
    staged names are UUIDs, and only one query is in flight at a time.
    """
    if not isinstance(reported, str) or not reported:
        return False
    return reported == expected or Path(reported).name == Path(expected).name


# One session per (image, args, device) -- deliberately NOT per `--params`.
#
# `--params` is a launch argument, so keying on it meant an index whose rows
# stamped a different recipe started a second container: two copies of the same
# multi-GB checkpoint on one card. Parameters are configuration now, fixed per
# model in `config.yml`, so there is exactly one container per model and a
# second one cannot appear.
# Sessions live for the process, like the
# loaded indexes they serve.
_sessions: Dict[str, TaggerSession] = {}
_sessions_lock = threading.Lock()


def session_for(
    image: str,
    params: Optional[Dict[str, Any]] = None,
    args: Sequence[str] = (),
    device: Any = None,
) -> TaggerSession:
    """The container for this image, started once and shared.

    `params` configures the session the first time it is asked for and is
    ignored afterwards -- it is a launch argument, and the whole point of not
    keying on it is that a second caller wanting different ones gets this
    container rather than a second copy of the model. A disagreement is logged,
    because it means `config.yml` and something else expect different recipes.
    """
    key = json.dumps([image, list(args), device], sort_keys=True)
    with _sessions_lock:
        session = _sessions.get(key)
        if session is None:
            session = _sessions[key] = TaggerSession(image, params, args, device)
        elif params and params != session.params:
            logger.warning(
                f"{image} is already running under params {session.params or 'none'}; "
                f"ignoring a request for {params}. Parameters are fixed per model in "
                "config.yml so that one model is one container."
            )
        return session


def _evict_conflicting(starting: TaggerSession) -> None:
    """Stop any other container already holding this one's image and GPU.

    Two sessions exist for one image when two indexes stamp different recipes:
    `--params` is fixed at launch, so a query cannot borrow a container started
    under different ones. What they *cannot* share is the card -- two copies of
    an 8B checkpoint do not fit on a 24 GB GPU, and the second one fails to load
    rather than the first one yielding.

    So the card holds one container per image at a time, and the cost of an
    index with an unusual recipe is a cold start rather than an out-of-memory.
    A session that is mid-query is left alone: it is answering someone.
    """
    for other in all_sessions():
        if other is starting or not other.alive:
            continue
        if (other.image, other.device) != (starting.image, starting.device):
            continue
        # Non-blocking, and this runs while `starting` holds its own lock: a
        # blocking acquire here is how two sessions evicting each other would
        # deadlock. Skipping is safe -- the worst case is the pair coexisting
        # for one more query.
        if not other._lock.acquire(blocking=False):
            logger.warning(
                f"{other.image} is busy on CUDA device {other.device}, so two of it "
                "will run until that query finishes"
            )
            continue
        try:
            logger.info(
                f"stopping {other.image} on CUDA device {other.device} "
                f"(params {other.params or 'none'}) to make room for the same image "
                f"under params {starting.params or 'none'}"
            )
            other._shutdown()
        finally:
            other._lock.release()


def status() -> List[Dict[str, Any]]:
    """What every started container is, where it runs, and where its log is.

    Reported rather than logged so it can be asked for while the service is up:
    the question "why did that query fail" is usually "which of these is not
    running, and what did it print", and both answers are here.
    """
    return [
        {
            "image": session.image,
            "device": session.device,
            "running": session.alive,
            "container_id": (session.container_id or "")[:12] or None,
            "log": str(session.log_path),
            "params": session.params or None,
            "failed_restarts": session._failures,
        }
        for session in all_sessions()
    ]


def all_sessions() -> List[TaggerSession]:
    with _sessions_lock:
        return list(_sessions.values())


def start_keepalive(interval: float = KEEPALIVE_SECONDS) -> threading.Thread:
    """Sweep every started session, restarting any container that has died.

    Only sessions that have been started are swept: a session exists because
    something asked for it, so this keeps up what is meant to be up and never
    starts a container nobody has asked for.
    """

    def loop() -> None:
        while True:
            time.sleep(interval)
            for session in all_sessions():
                if not session.due_for_restart():
                    continue
                # Counted before the attempt, and cleared by the next query that
                # the container answers: a restart that "works" but exits again
                # a second later is a crash loop, not a recovery, and only a
                # query can tell the two apart.
                session._failures += 1
                session._retry_after = time.monotonic() + session._restart_delay
                try:
                    # Non-blocking: a session busy with a query is not dead, and
                    # a sweep must not queue behind a slow search.
                    if session.ensure_running(blocking=False):
                        logger.info(f"restarted tagger container {session.image}")
                except TaggerError as exc:
                    # Logged, not raised: the sweep outlives any one container,
                    # and the next query reports the failure to whoever asked.
                    logger.warning(
                        f"could not restart {session.image} (retrying in "
                        f"{session._restart_delay:.0f}s): {exc}"
                    )

    thread = threading.Thread(target=loop, name="tagger-keepalive", daemon=True)
    thread.start()
    return thread
