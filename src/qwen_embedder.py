"""Query side for a Qwen3-VL video index (to support video embeddings visualization).

Not a copy of the tagger's embedding module
-------------------------------------------
`model-qwenvl-video-vector/embedding/qwen3_vl_embedding.py` defines a custom
embedding head (`Qwen3VLForEmbedding`) and an input pipeline whose `process()`
already accepts text, image and video uniformly — a query is just another item
of `{text|image|video, instruction, fps, max_frames}`. So the query side is an
adapter over that class, exactly as `embedder.py` is an adapter over SigLIP 2's
towers, and the model definition is imported rather than duplicated. Copying 600
lines that track a checkpoint would be a fourth copy of a contract that is
already hard enough to keep in sync.

Where it is imported from
-------------------------
A plain import first: inside the tagger's container the code sits at `/elv`,
which is WORKDIR and therefore already on `sys.path`, and a `pip install -e` of
the `qwenvl-embedding` package has the same effect anywhere else. Failing that,
`QWEN_EMBEDDING_PATH`, then `/elv`, then a sibling checkout beside this repo.
No path is hardcoded to a home directory.

Model weights are not this module's problem: transformers resolves them through
the usual cache, which the container pins with `HF_HOME=/root/.cache`.

The tagger's top-level package is named `embedding`, generic enough to collide
with something else on the path, so the import is deferred until a Qwen query is
actually made rather than run at startup.

Recipe parameters
-----------------
Every recipe key that changes a vector is threaded through: `prompt` becomes the
instruction (an embedding model conditions on it, so a query embedded under a
different one lands elsewhere), `fps`/`max_frames`/`max_length` are the sampling
budget a video query has to match, and `dim` is the MRL width the tagger
truncated and re-normalized to.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import IO, Any, Dict, List, Optional

from embedder import EmbeddingError, pad_vector

# Searched in order, after a plain import has already been tried.
#   /elv                  the tagger container's WORKDIR
#   ../../model-vector/…  a sibling checkout, for a plain local layout
QWEN_PATH_CANDIDATES = (
    "/elv",
    str(Path(__file__).resolve().parents[2] / "model-vector" / "model-qwenvl-video-vector"),
)

# Videos are handed to the embedder as a path: its reader opens files, not
# streams. Suffix matters — qwen-vl-utils picks its decoder by extension.
DEFAULT_VIDEO_SUFFIX = ".mp4"


def _qwen_search_paths() -> List[str]:
    override = os.environ.get("QWEN_EMBEDDING_PATH")
    return ([override] if override else []) + list(QWEN_PATH_CANDIDATES)


def _load_embedder_class():
    """Import the tagger's embedder, or explain exactly what is missing."""
    try:
        from embedding.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: PLC0415

        return Qwen3VLEmbedder
    except ImportError:
        pass

    for path in _qwen_search_paths():
        if not Path(path, "embedding", "qwen3_vl_embedding.py").is_file():
            continue
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            from embedding.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: PLC0415

            return Qwen3VLEmbedder
        except ImportError as exc:
            raise EmbeddingError(
                f"found the tagger at {path} but could not import its embedder: {exc}. "
                "Its dependencies (qwen-vl-utils, decord, transformers>=4.57.3) have to "
                "be installed in this environment."
            ) from exc

    raise EmbeddingError(
        "Qwen query embedding needs model-qwenvl-video-vector importable. Install it "
        "(`pip install -e <checkout>`), run where it is on sys.path, or point "
        f"QWEN_EMBEDDING_PATH at it. Looked in: {', '.join(_qwen_search_paths())}"
    )


class QwenQueryEmbedder:
    """Embeds text, image and video queries into a Qwen3-VL index's space.

    Loaded lazily: the weights are several GB, and an index plots without them.
    """

    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        normalize: bool = True,
        target_size: int = 1024,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.normalize = normalize
        self.target_size = target_size
        self.params = params or {}
        # The instruction the indexed vectors were embedded under.
        self.instruction = self.params.get("prompt")
        self.fps = self.params.get("fps")
        self.max_frames = self.params.get("max_frames")
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            cls = _load_embedder_class()
            kwargs: Dict[str, Any] = {"model_name_or_path": self.model_id, "revision": self.revision}
            # Only pass what the recipe actually recorded; the tagger's own
            # defaults are the right fallback for anything it omitted.
            for key, value in (
                ("fps", self.fps),
                ("max_frames", self.max_frames),
                ("max_length", self.params.get("max_length")),
                ("embedding_dim", self.params.get("dim")),
            ):
                if value is not None:
                    kwargs[key] = value
            if self.instruction:
                kwargs["default_instruction"] = self.instruction
            try:
                self._embedder = cls(**kwargs)
            except Exception as exc:
                raise EmbeddingError(f"could not load {self.model_id}: {exc}") from exc
        return self._embedder

    def embed_text(self, query: str) -> List[float]:
        if not (query or "").strip():
            raise EmbeddingError("text query is empty")
        return self._embed({"text": query})

    def embed_image(self, image: IO[bytes]) -> List[float]:
        from PIL import Image, ImageOps  # noqa: PLC0415

        try:
            decoded = Image.open(image)
            # A phone photo records rotation in EXIF rather than pixel data.
            decoded = ImageOps.exif_transpose(decoded) or decoded
            decoded = decoded.convert("RGB")
        except Exception as exc:
            raise EmbeddingError(f"could not decode image: {exc}") from exc
        return self._embed({"image": decoded})

    def embed_video(self, video: IO[bytes], filename: Optional[str] = None) -> List[float]:
        """Embed an uploaded clip.

        Written to a temp file because the video reader opens paths, not
        streams, and keeps the upload's suffix because the decoder is chosen by
        extension.
        """
        suffix = Path(filename or "").suffix or DEFAULT_VIDEO_SUFFIX
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(video.read())
            tmp.flush()
            tmp.close()
            item: Dict[str, Any] = {"video": tmp.name}
            # The sampling budget the indexed vectors were built under; a query
            # sampled differently is not comparable to them.
            if self.fps is not None:
                item["fps"] = self.fps
            if self.max_frames is not None:
                item["max_frames"] = self.max_frames
            return self._embed(item)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _embed(self, item: Dict[str, Any]) -> List[float]:
        if self.instruction:
            item.setdefault("instruction", self.instruction)
        try:
            # process() returns (batch, dim); one item in, one row out.
            embeddings = self.embedder.process([item], normalize=self.normalize)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"could not embed query: {exc}") from exc
        vector = embeddings[0].float().cpu().numpy().tolist()
        return pad_vector([float(x) for x in vector], self.target_size)
