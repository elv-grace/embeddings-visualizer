"""Which container embeds a query for an index, and what it will accept.

How the model is determined
---------------------------
From the **batch** a vector was written in, not from anything the vector itself
carries. Each row names its `batch_id`; `GET /indexes/{qid}/batches/{batch_id}`
returns that batch's `model`, and the mapping from model to container is 1-1
because a batch is one tagger run and a tagger has exactly one embedder.

Every model is queried the same way
-----------------------------------
By running its tagger container over the query, through the protocol in
`tagger.py`. There are no per-model towers in this service any more: no torch,
no transformers, no checkpoint resolution, no second interpreter for a model
whose pins conflict with the rest. A model is now one row of configuration --
an image and the modalities it accepts -- and adding one is adding that row.

That is also the only way a query is *guaranteed* to land in the index's space.
The old towers imported the taggers' code to avoid drifting from it, which
worked only as long as the import resolved to the same version the index was
built with; here the query runs the tagger itself.

Modalities
----------
Which kinds of query a space accepts is a property of the model, so it is
configured beside the image rather than declared per index. A face index takes
images only because InsightFace identity vectors have no text tower to put a
description into the same space -- not because of anything about a given index.

The default container
---------------------
An index whose batches report no model at all still queries, through
`DEFAULT_MODEL`'s container. It is text-only: text is the one modality every
embedding model accepts, so it is the only assumption that is safe to make about
a space nothing has identified.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import config
from tagger import DEFAULT_DEVICE, TaggerError, session_for, start_keepalive

logger = logging.getLogger(__name__)

# Width the visualizer pads query vectors to when an index is wider than the
# model; see pad_vector.
DEFAULT_TARGET_SIZE = 1024


class EmbeddingError(RuntimeError):
    """A query could not be embedded."""


# The entry used when an index's batches report no model. Text-only on purpose;
# see the module docstring.
DEFAULT_MODEL = "default"

# An index's `model` -> the tagger container that embeds a query into that
# space, and the query modalities that container accepts.
#
#   image       the OCI image to run. `None` until the built images are named --
#               fill it in here or, better, in the config file below, which is
#               how a deployment points at its own registry without a patch.
#   modalities  which of text/image/video this model's space can be queried with.
#   device      the CUDA device index this model's container runs on. Unset
#               takes the `container:` default; with neither set, a query
#               against that model raises rather than running unplaced.
#   params      the container's `--params`: the recipe its vectors were embedded
#               under. Configuration, not something read off an index -- see
#               `configured_params`.
#   args        extra arguments for the container runtime, e.g. a bind mount of
#               a weight cache. Global ones go in EV_CONTAINER_ARGS instead.
MODEL_CONTAINERS: Dict[str, Dict[str, Any]] = {
    DEFAULT_MODEL: {"image": None, "modalities": ["text"], "args": [], "device": None, "params": {}},
    "frame_vectors": {
        "image": None, "modalities": ["text", "image"], "args": [], "device": None,
        "params": {},
    },
    "video_vectors": {
        "image": None, "modalities": ["text", "image", "video"], "args": [],
        "device": None, "params": {},
    },
    # Face embeddings from model-celeb-vector. `image` only, and deliberately:
    # these are InsightFace identity vectors, so there is no text tower that
    # could put a description into the same space.
    "face_vectors": {"image": None, "modalities": ["image"], "args": [], "device": None, "params": {}},
}

# `config.yml`'s `models:` section is merged over the table above, key by key,
# at import. That is where image names belong: they differ per registry and per
# retag, and nothing else here has to change when one moves. An entry naming
# only an image keeps that model's built-in modalities.
#
#     models:
#       frame_vectors:
#         image: localhost/model-frame-vector:latest

ALL_MODALITIES = ("text", "image", "video")

# What a query of each modality is staged as for the container. Text is the
# protocol's newline-separated query file; the other two keep the upload's own
# suffix, since a tagger picks its decoder by extension.
DEFAULT_SUFFIX = {"image": ".png", "video": ".mp4"}


def _load_config() -> None:
    """Merge `config.yml`'s `models:` section over MODEL_CONTAINERS, in place.

    Key by key rather than entry by entry, so a file that names only images
    keeps the modalities configured here -- which is the shape most deployments
    want, since an image moves far more often than a model gains a modality.

    A model the file names and this table does not is kept: it is how a tagger
    is added without a patch, and `modalities` is the only key it has to carry
    beyond the image.
    """
    models = config.section("models")
    for model, entry in models.items():
        if not isinstance(entry, dict):
            logger.warning(f"{config.CONFIG_PATH}: entry for {model!r} is not a mapping")
            continue
        current = MODEL_CONTAINERS.setdefault(
            model, {"image": None, "modalities": [], "args": [], "device": None, "params": {}}
        )
        # `image:` with nothing after it parses as None, which is the file
        # saying "not set yet" rather than "unset what is configured".
        current.update({k: v for k, v in entry.items() if v is not None})
    if models:
        logger.info(f"model containers configured for: {sorted(models)}")


_load_config()


# Keys a tagger may stamp in `additional_info` that change the vector a query
# must be embedded into. A whitelist, so the provenance a tagger also stamps
# (box, score, upscale, crop_padding, detector, text) never reaches a container.
#
# They are read to be *checked*, not applied: the container runs on the recipe
# configured beside it (`configured_params`), and `tuning_mismatch` reports where
# an index's rows disagree with it. Applying them per index is what started a
# second container per recipe -- `--params` is a launch argument -- so the recipe
# is stated once in config.yml, and an index wanting a different one is a config
# change rather than another copy of the model.
TUNING_KEYS = (
    "normalize",        # both: whether cosine reduces to a dot product
    "max_num_patches",  # SigLIP 2: the NaFlex patch budget
    "prompt",           # Qwen: the instruction the vectors were embedded under
    "fps",              # Qwen: video sampling rate
    "max_frames",       # Qwen: video frame budget
    "max_length",       # Qwen: token budget
    "dim",              # Qwen: the MRL width vectors were truncated to
    "det_confidence",   # celeb: the MTCNN gate deciding which faces were kept
    "min_box_size",     # celeb: the smallest face the tagger embedded
)


def read_tuning(metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tuning parameters stamped on the rows, or {} when nothing stamped any.

    A query embedded under different parameters than the index lands in a
    different space and quietly returns the wrong neighbours, so these are worth
    reading when present. They are *optional*: every one has a default in the
    container that owns it, and an index that stamps nothing still queries
    correctly.

    The first row carrying any of them wins. Parameters are invariant across a
    tagger's run, so scanning further would only cost time; where an index mixes
    taggers its batches differ too, which `models` already surfaces.
    """
    for row in metadata:
        info = row.get("additional_info")
        if not isinstance(info, dict):
            continue
        found = {k: info[k] for k in TUNING_KEYS if info.get(k) is not None}
        if found:
            return found
    return {}


def model_spec(model: Optional[str]) -> Dict[str, Any]:
    """The registry entry for an index's `model`, or the default container's.

    Unregistered rather than absent is worth distinguishing in the log: a model
    name nothing knows about probably wants a config entry, whereas no model at
    all is the ordinary case for an index whose batches could not be read.
    """
    name = (model or "").strip()
    if name and name not in MODEL_CONTAINERS:
        logger.warning(
            f"no container is configured for index model {name!r}; using the "
            f"default text-only container. Known models: {sorted(MODEL_CONTAINERS)}"
        )
    return MODEL_CONTAINERS.get(name) or MODEL_CONTAINERS[DEFAULT_MODEL]


def modalities_for(model: Optional[str]) -> List[str]:
    """Which query kinds this model's space accepts."""
    spec = model_spec(model)
    return [m for m in (spec.get("modalities") or []) if m in ALL_MODALITIES]


def build_embedder_for_model(model: Optional[str], target_size: int) -> "QueryEmbedder":
    """The query embedder for an index whose batches report `model`.

    Everything about the container -- image, device, `--params` -- comes from
    configuration, so every index on a given model shares one container. Nothing
    an index carries can start a second.
    """
    spec = model_spec(model)
    image = (spec.get("image") or "").strip()
    if not image:
        raise EmbeddingError(
            f"no container image is configured for index model {model or DEFAULT_MODEL!r}. "
            f"Set one under `models:` in {config.CONFIG_PATH} (or point EV_CONFIG "
            "elsewhere) -- a query runs the tagger itself, so there is nothing to "
            "fall back to."
        )
    return QueryEmbedder(
        image=image,
        modalities=modalities_for(model),
        target_size=target_size,
        params=configured_params(spec),
        args=container_args(spec),
        device=container_device(spec),
    )


class QueryEmbedder:
    """Embeds queries for one index by running its tagger container.

    Holds no model and no weights: the container does, and it is shared with
    every other index built under the same model and recipe (`tagger.session_for`),
    started on the first query against it and kept for the process.
    """

    def __init__(
        self,
        image: str,
        modalities: Sequence[str],
        target_size: int = DEFAULT_TARGET_SIZE,
        params: Optional[Dict[str, Any]] = None,
        args: Sequence[str] = (),
        device: Any = None,
    ) -> None:
        self.image = image
        self.modalities = list(modalities)
        self.target_size = target_size
        self.params = dict(params or {})
        self.args = list(args)
        self.device = device

    @property
    def session(self):
        return session_for(self.image, self.params, self.args, self.device)

    def embed_text(self, query: str) -> List[float]:
        if not (query or "").strip():
            raise EmbeddingError("text query is empty")
        try:
            tags = self.session.tag_text([query])
        except TaggerError as exc:
            raise EmbeddingError(str(exc)) from exc
        return self._vector(tags, "text")

    def embed_file(self, mode: str, data: bytes, filename: Optional[str] = None) -> List[float]:
        """Embed an uploaded image or clip.

        The upload's suffix is kept because a tagger picks its decoder by
        extension -- a clip staged as `.bin` is not read as the container it is.
        """
        if not data:
            raise EmbeddingError(f"the uploaded {mode} is empty")
        suffix = Path(filename or "").suffix or DEFAULT_SUFFIX.get(mode, "")
        try:
            tags = self.session.tag_file(data, suffix)
        except TaggerError as exc:
            raise EmbeddingError(str(exc)) from exc
        return self._vector(tags, mode)

    def _vector(self, tags: List[Dict[str, Any]], mode: str) -> List[float]:
        """The one vector to search with, out of what the container emitted."""
        vectors = [t for t in tags if isinstance(t.get("vector"), list) and t["vector"]]
        if not vectors:
            if tags:
                raise EmbeddingError(
                    f"{self.image} tagged this {mode} query but emitted no vector, so "
                    "there is nothing to search with. It is a tagging model, not an "
                    "embedding one -- check the image configured for this index's model."
                )
            raise EmbeddingError(_nothing_found(mode))
        # More than one is the detector case: a face model emits a tag per
        # detected face, and the subject of a query photo is the face the
        # uploader framed, so the largest box wins. Ties cannot matter --
        # equal-area detections are equally good candidates. With no boxes at
        # all (a plain whole-input embedder, or a `.txt` holding one query) the
        # first is the only one.
        best = max(vectors, key=lambda t: _box_area(t.get("frame_info")))
        return pad_vector([float(x) for x in best["vector"]], self.target_size)


def _nothing_found(mode: str) -> str:
    if mode == "image":
        return (
            "the model found nothing to embed in this image. If it is a face index, "
            "no face was detected -- try a closer or sharper crop."
        )
    return f"the model returned no tags for this {mode} query"


def _box_area(frame_info: Any) -> float:
    """Area of a tag's bounding box, or 0 when it has none.

    The protocol's `frame_info` carries the box beside `frame_idx`; the key
    names follow the vectors' own metadata (`x1`/`y1`/`x2`/`y2`, normalized).
    """
    if not isinstance(frame_info, dict):
        return 0.0
    box = frame_info.get("box") if isinstance(frame_info.get("box"), dict) else frame_info
    try:
        return max(0.0, float(box["x2"]) - float(box["x1"])) * max(
            0.0, float(box["y2"]) - float(box["y1"])
        )
    except (KeyError, TypeError, ValueError):
        return 0.0


def container_args(spec: Dict[str, Any]) -> List[str]:
    """One model's extra runtime arguments, however the file wrote them."""
    return config.as_args(spec.get("args"))


def configured_params(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The `--params` this model's container is launched with.

    Configuration, deliberately, and not the tuning stamped on an index's rows.
    `--params` is a launch argument: deriving it per index meant a second index
    with a different recipe started a second container holding a second copy of
    the same multi-GB checkpoint. One model is one container, so the recipe it
    runs under is stated once, here.

    The stamped tuning is still read -- `read_tuning` -- and still reported, and
    `tuning_mismatch` says when the two disagree. That disagreement is worth
    seeing: it means queries are being embedded under a different recipe than
    the index was built with, which does not fail, it just returns worse
    neighbours.
    """
    params = spec.get("params")
    return dict(params) if isinstance(params, dict) else {}


def tuning_mismatch(
    model: Optional[str], tuning: Dict[str, Any]
) -> Dict[str, Any]:
    """Stamped parameters that disagree with the configured ones.

    Only keys the config actually sets are compared: a container falls back to
    its own defaults for anything unset, and those are usually the right ones --
    it is a *stated* parameter differing from a *stamped* one that means the
    query and the index are in different spaces.
    """
    configured = configured_params(model_spec(model))
    return {
        key: {"index": value, "configured": configured[key]}
        for key, value in (tuning or {}).items()
        if key in configured and configured[key] != value
    }


def container_device(spec: Dict[str, Any]) -> Any:
    """The CUDA device this model's container runs on.

    Falls back to the `container:` default, so a box whose models all share one
    card configures it once. Beyond that there is no fallback: `tagger` rejects
    an unset device rather than placing the container nowhere in particular.
    """
    device = spec.get("device")
    return DEFAULT_DEVICE if device is None else device


def warm_containers() -> List[str]:
    """Start every configured model's container, and keep them started.

    Called at boot so the first search of the day does not also pay for pulling
    an image and loading a multi-GB checkpoint. Containers are long-lived by
    design (see `tagger`), so this only brings forward a cost that would
    otherwise land on a user mid-query, and `start_keepalive` puts back any that
    later dies.

    These are the containers, not a warm copy of them: `--params` comes from
    config.yml, so the container started here is the one every query against
    that model uses. Nothing an index carries starts another.

    Returns the images that were started, so the caller can log what is up.
    """
    started: List[str] = []
    for model, spec in MODEL_CONTAINERS.items():
        image = (spec.get("image") or "").strip()
        if not image:
            logger.info(f"no container image configured for {model!r}; not starting one")
            continue
        try:
            # Inside the try: building the session validates `device`, and one
            # model with a bad index must not stop the others from starting.
            session = session_for(
                image, configured_params(spec), container_args(spec),
                container_device(spec),
            )
            if session.alive:
                continue   # another model is served by the same image and recipe
            # Sequential and blocking on purpose: these compete for the same GPU
            # and the same image store, and starting them all at once is how a
            # boot turns into a stampede.
            if session.ensure_running():
                started.append(image)
                logger.info(
                    f"started tagger container for {model!r} on CUDA device "
                    f"{session.device}: {image}"
                )
        except TaggerError as exc:
            # Not fatal. An index still loads, plots and queries every *other*
            # model, and the query that needs this one reports why it cannot.
            # This also catches a `device:` the config got wrong or left unset,
            # which is rejected when the session is built rather than at launch.
            logger.warning(f"could not start the container for {model!r} ({image}): {exc}")
    start_keepalive()
    return started


def pad_vector(vec: List[float], target_size: int) -> List[float]:
    """Right-pad with zeros to the index's width.

    SigLIP 2 base emits 768 dims into a 1024-wide index. Cosine ranking is
    unaffected for unit vectors. Raises when the model is wider than the index,
    which catches the wrong container before any vectorstore call.
    """
    if len(vec) > target_size:
        raise EmbeddingError(
            f"vector of length {len(vec)} cannot be padded to smaller target {target_size}; "
            "the container that embedded this query does not match this index"
        )
    return vec + [0.0] * (target_size - len(vec))
