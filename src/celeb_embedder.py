"""Query side for a celeb (face) vector index, built by model-celeb-vector.

What the index holds
--------------------
One vector per detected *face*, not per frame: `CelebVectorizer.tag_frame` runs
MTCNN over a frame, crops each face above `det_confidence`, resizes to
InsightFace's 112x112, and emits the r100 embedding with the face's normalized
box. The vectors are L2-normalized, so cosine reduces to a dot product.

A query is therefore a **face**, and the only sensible query mode is `image`:
there is no text tower for face identity, so a text query has nothing to embed
into this space. `embedder.INDEX_MODELS` says so, and the UI disables the mode.

Not a reimplementation
----------------------
Like `qwen_embedder`, this imports the tagger's own model rather than copying
it. That matters more here than anywhere else: a query has to go through the
same detect -> crop -> resize -> transpose -> `get_batch_features` path, and any
divergence (a different detector, a different crop convention, a missing
re-normalize) moves the vector without failing.

Why it runs in a second interpreter
-----------------------------------
`model-celeb-vector/setup.py` pins `numpy<1.20.0`, `torch==1.9.0` and mxnet.
Those cannot coexist with what the rest of this service needs -- SigLIP 2 and
Qwen run on torch 2.13 and numpy 2.x -- so installing them here would break
every other tower plus UMAP and scikit-learn.

So the tagger's model runs in its own interpreter (`.celebenv`, built by
`tools/setup_celeb_env.sh`) and this module talks to it over a pipe, via
`tools/celeb_worker.py`. That is cheaper than it sounds: the tagger sets
`gpu: -1` and keeps InsightFace on CPU deliberately, so CPU wheels suffice --
no CUDA 10.1 and none of the ~10 GB its conda container would cost.

Verified not to change the vector: max abs diff 0.0, cosine 1.0 against calling
`tag_frame` directly on the same image.

Two paths, in order. The worker is preferred when `.celebenv` (or `CELEB_PYTHON`)
exists, which is the normal case. A plain in-process import is the fallback, and
is what works *inside* the tagger's own container where `/elv` is WORKDIR and the
stack is already present. With neither, a query fails naming the missing module
rather than being silently absent.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import IO, Any, Dict, List, Optional

import numpy as np
from PIL import Image

from embedder import EmbeddingError, decode_image, pad_vector

logger = logging.getLogger(__name__)

# An interpreter with the tagger's stack (numpy<1.20, torch 1.9, mxnet 1.9).
# Built by tools/setup_celeb_env.sh; CELEB_PYTHON overrides the location.
CELEB_VENV_PYTHON = Path(__file__).resolve().parents[1] / ".celebenv" / "bin" / "python"
CELEB_WORKER = Path(__file__).resolve().parents[1] / "tools" / "celeb_worker.py"

# Where tools/setup_celeb_env.sh clones the tagger's source when there is no
# checkout beside this repo -- see that script for why it fetches rather than
# vendors the files.
_CLONED = Path(__file__).resolve().parents[1] / ".celebenv" / "src"

COMMON_ML_CANDIDATES = (
    "/elv",
    str(Path(__file__).resolve().parents[2] / "common-ml"),
    str(_CLONED / "common-ml"),
)


def _worker_python() -> Optional[Path]:
    """The isolated interpreter, or None when there is none to use."""
    override = os.environ.get("CELEB_PYTHON")
    for path in ([Path(override)] if override else []) + [CELEB_VENV_PYTHON]:
        if path.is_file():
            return path
    return None

# The tagger stamps this as `additional_info.model`; it also names the tower in
# `embedder.INDEX_MODELS`.
MODEL_ID = "insightface-r100-ii"

# InsightFace r100 emits 512 dims; a wider index is zero-padded by pad_vector.
NATIVE_DIM = 512

# Searched in order, after a plain import has already been tried.
#   /elv            the tagger container's WORKDIR
#   ../..           a sibling checkout, for a plain local layout
#   .celebenv/src   a clone the setup script made, when there was no checkout
CELEB_PATH_CANDIDATES = (
    "/elv",
    str(Path(__file__).resolve().parents[2] / "model-celeb" / "model-celeb-vector"),
    str(_CLONED / "model-celeb" / "model-celeb-vector"),
)

# Where the InsightFace r100 weights live. config.yml names /ml/models/celeb_detection
# locally and `models` inside the container; CELEB_MODEL_PATH overrides both.
# config.yml names /ml/models/celeb_detection, but the r100 checkpoint actually
# lives at /ml/models/celeb (models/model-r100-ii/model-{symbol.json,0000.params}),
# which is what `model_input_path` is joined against. Container path last.
CELEB_WEIGHT_CANDIDATES = ("/ml/models/celeb", "/ml/models/celeb_detection", "models")


def _celeb_search_paths() -> List[str]:
    override = os.environ.get("CELEB_EMBEDDING_PATH")
    return ([override] if override else []) + list(CELEB_PATH_CANDIDATES)


def _load_vectorizer_class():
    """Import the tagger's `CelebVectorizer`, or explain exactly what is missing.

    Only `model-celeb-vector` needs to go on `sys.path`: the tagger's own module
    adds `model-celeb` (three levels up from it) so that `celeb.face_model`
    resolves.
    """
    try:
        from celeb_vector.model import CelebVectorizer  # noqa: PLC0415

        return CelebVectorizer
    except ImportError:
        pass

    for path in _celeb_search_paths():
        if not Path(path, "celeb_vector", "model.py").is_file():
            continue
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            from celeb_vector.model import CelebVectorizer  # noqa: PLC0415

            return CelebVectorizer
        except ImportError as exc:
            raise EmbeddingError(
                f"found the tagger at {path} but could not import its model: {exc}. "
                "It needs mxnet-cu101, facenet-pytorch, easydict and scikit-image, "
                "which pin numpy<1.20 and torch 1.9 and so cannot be installed "
                "alongside the SigLIP 2 / Qwen towers. Run this where that stack "
                "exists (the tagger's own container) or query that index there."
            ) from exc

    raise EmbeddingError(
        "celeb query embedding needs model-celeb-vector importable. Install it, run "
        "where it is on sys.path, or point CELEB_EMBEDDING_PATH at it. Looked in: "
        f"{', '.join(_celeb_search_paths())}"
    )


def _tagger_path() -> Optional[str]:
    """The model-celeb-vector checkout, or None when none is on the candidates."""
    for path in _celeb_search_paths():
        if Path(path, "celeb_vector", "model.py").is_file():
            return path
    return None


def _common_ml_path() -> str:
    """common-ml, which the tagger imports for FrameModel/FrameTag."""
    override = os.environ.get("COMMON_ML_PATH")
    for path in ([override] if override else []) + list(COMMON_ML_CANDIDATES):
        if path and Path(path, "common_ml").is_dir():
            return path
    return override or COMMON_ML_CANDIDATES[-1]


def _pick_face(faces):
    if not faces:
        raise EmbeddingError(
            "no face was detected in this image, so there is nothing to "
            "match against a face index. Try a closer or sharper crop."
        )
    return max(faces, key=lambda f: _box_area(getattr(f, "box", None)))


def _weights_path() -> str:
    """The r100 weight directory, or the first candidate for the error to name."""
    override = os.environ.get("CELEB_MODEL_PATH")
    for path in ([override] if override else []) + list(CELEB_WEIGHT_CANDIDATES):
        if path and Path(path).is_dir():
            return path
    return override or CELEB_WEIGHT_CANDIDATES[0]


class CelebQueryEmbedder:
    """Embeds a face from an uploaded image into a celeb index's space.

    Loaded lazily, like the other towers: an index plots without the weights,
    and only a query pays for them.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        target_size: int = 1024,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_id = model_id
        self.target_size = target_size
        self.params = params or {}
        self._vectorizer = None

    @property
    def vectorizer(self):
        if self._vectorizer is None:
            cls = _load_vectorizer_class()
            from celeb_vector.config import RuntimeConfig  # noqa: PLC0415

            # Only what the tagger actually recorded; its own defaults are the
            # right fallback for anything absent.
            cfg_kwargs = {
                k: self.params[k]
                for k in ("det_confidence", "min_box_size")
                if self.params.get(k) is not None
            }
            try:
                self._vectorizer = cls(
                    model_input_path=_weights_path(), cfg=RuntimeConfig(**cfg_kwargs)
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"could not load {self.model_id} from {_weights_path()!r}: {exc}. "
                    "Set CELEB_MODEL_PATH to the InsightFace r100 weight directory."
                ) from exc
        return self._vectorizer

    def embed_text(self, query: str) -> List[float]:
        raise EmbeddingError(
            "a celeb index holds face embeddings, which have no text tower -- "
            "query it with an image of a face instead."
        )

    def embed_image(self, image: IO[bytes]) -> List[float]:
        """The largest face in the upload, embedded as the tagger would embed it.

        Largest rather than first or highest-confidence: a query photo often has
        bystanders, and the subject is the face the uploader framed. Ties cannot
        matter -- equal-area faces are equally good candidates.

        Decoding happens here rather than in the worker so that EXIF rotation is
        applied (a phone photo records it in metadata, and `cv2.imread` ignores
        it) and so the worker receives one predictable format.
        """
        vector = self._embed_array(decode_image(image))
        return pad_vector([float(x) for x in vector], self.target_size)

    def _embed_array(self, img: np.ndarray) -> List[float]:
        """One (H, W, 3) RGB frame -> the largest face's embedding.

        Prefers the isolated interpreter when there is one, because that is the
        normal case outside the tagger's container; in-process is the path that
        works *inside* it, where the stack is already present.
        """
        python = _worker_python()
        if python is not None:
            return self._embed_remote(img, python)

        faces = self.vectorizer.tag_frame(img)
        best = _pick_face(faces)
        vector = list(getattr(best, "vector", None) or [])
        if not vector:
            raise EmbeddingError("the tagger returned a face with no embedding")
        return vector

    def _embed_remote(self, img: np.ndarray, python: Path) -> List[float]:
        """Hand one frame to the worker over its pipe.

        A PNG on disk rather than the array over the pipe: the two interpreters
        have incompatible numpy ABIs, so nothing numpy-shaped can cross between
        them, and PNG is lossless so the pixels the tagger sees are the pixels
        decoded here.
        """
        proc = self._worker(python)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(img).save(tmp.name)
            path = tmp.name
        try:
            proc.stdin.write(json.dumps({"image_path": path}) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                self._proc = None   # died; the next query starts a fresh one
                raise EmbeddingError(
                    "the celeb worker exited without answering; see the service log"
                )
            reply = json.loads(line)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        if reply.get("error"):
            if "no face" in str(reply["error"]).lower():
                raise EmbeddingError(
                    "no face was detected in this image, so there is nothing to "
                    "match against a face index. Try a closer or sharper crop."
                )
            raise EmbeddingError(f"celeb worker: {reply['error']}")
        return list(reply.get("vector") or [])

    def _worker(self, python: Path):
        """The running worker, started on first use and then kept.

        Long-lived because loading r100 costs a couple of seconds; this mirrors
        the in-process towers, which keep their weights resident too.
        """
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            return proc

        # Checkout paths are optional. tools/setup_celeb_env.sh pip-installs
        # celeb_vector, celeb and common_ml into the environment, so the worker
        # usually imports them with no help; these only matter when running
        # against a checkout instead (local development, or the tagger's own
        # container). The worker puts on sys.path whichever of them is set.
        env = {
            **os.environ,
            "CELEB_MODEL_PATH": _weights_path(),
            "CELEB_PARAMS": json.dumps(self.params or {}),
        }
        tagger = _tagger_path()
        if tagger:
            env["CELEB_EMBEDDING_PATH"] = tagger
        common_ml = _common_ml_path()
        if common_ml and Path(common_ml, "common_ml").is_dir():
            env["COMMON_ML_PATH"] = common_ml
        logger.info(f"starting celeb worker: {python} (weights {_weights_path()})")
        proc = subprocess.Popen(
            [str(python), str(CELEB_WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,          # the tagger's logging goes to the service's stderr
            env=env,              # carries the paths the worker puts on sys.path
            text=True,
            bufsize=1,
        )
        ready = proc.stdout.readline()
        if not ready or not json.loads(ready or "{}").get("ready"):
            proc.kill()
            raise EmbeddingError(
                "the celeb worker failed to start; its stack or weights are "
                "missing. See the service log for what it reported."
            )
        self._proc = proc
        return proc


def _box_area(box: Optional[Dict[str, Any]]) -> float:
    if not isinstance(box, dict):
        return 0.0
    try:
        return max(0.0, float(box["x2"]) - float(box["x1"])) * max(
            0.0, float(box["y2"]) - float(box["y1"])
        )
    except (KeyError, TypeError, ValueError):
        return 0.0


def build(target_size: int, params: Optional[Dict[str, Any]] = None) -> CelebQueryEmbedder:
    """The celeb query embedder, configured for one index.

    `params` only tunes an already-chosen tower -- `det_confidence` and
    `min_box_size`, the gates deciding which faces the tagger kept. The tagger
    stamps neither today, so this is normally empty and its defaults apply.
    """
    return CelebQueryEmbedder(
        model_id=MODEL_ID, target_size=target_size, params=dict(params or {})
    )
