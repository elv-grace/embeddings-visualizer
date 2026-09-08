"""SigLIP 2 image and text towers, for embedding a search query into an index.

Self-contained copy
-------------------
This mirrors content-search's ``content_search/frame_search/embedder.py``, which
itself mirrors the tagger's ``FeatureExtractor`` in
model-vector/model-siglip2-frame-vector. Keeping the visualizer independent of a
running content-search costs a third copy of the same contract, and all three
have to agree or query vectors land in a different space than the indexed ones.

TODO: collapse these into one shared package. The values below are load-bearing
and silent when wrong -- a mismatch does not raise, it just returns bad
neighbours. In particular the text tower's padding="max_length"/max_length=64 is
the fixed-length padding SigLIP was trained on; tokenized any other way, the
query vector moves far enough that results collapse onto whatever content
dominates the index.

The index does not record which model produced it. GET /indexes/{qid} returns
only {qid, vector_size}, and the per-vector metadata carries no model fields, so
these constants cannot be read back from the index and are supplied by the
caller instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import IO, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

# Must match what the index was tagged with; see the module docstring.
DEFAULT_MODEL_ID = "google/siglip2-base-patch16-naflex"
DEFAULT_REVISION = "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"
DEFAULT_NORMALIZE = True
DEFAULT_MAX_NUM_PATCHES = 256
DEFAULT_TARGET_SIZE = 1024


class EmbeddingError(RuntimeError):
    """A query could not be embedded."""


def _resolve_device_dtype(dtype: Optional[torch.dtype]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is not None:
        return device, dtype
    if device.type == "cuda":
        # bf16 needs Ampere+; fall back to fp16 otherwise.
        return device, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return device, torch.float32  # half precision is unstable/slow on CPU


SIGLIP_PATH_CANDIDATES = (
    "/elv",
    str(Path(__file__).resolve().parents[2] / "model-vector" / "model-siglip2-frame-vector"),
)


def _load_tagger_extractor():
    """The tagger's own `FeatureExtractor`, or None if it is not importable.

    Same resolution order as `qwen_embedder`: plain import (inside the tagger's
    container `/elv` is WORKDIR and already on `sys.path`), then
    `SIGLIP_EMBEDDING_PATH`, then a sibling checkout.
    """
    paths = [os.environ["SIGLIP_EMBEDDING_PATH"]] if os.environ.get("SIGLIP_EMBEDDING_PATH") else []
    paths += list(SIGLIP_PATH_CANDIDATES)
    for path in [None] + paths:
        if path is not None:
            if not Path(path, "siglip_frame", "model.py").is_file():
                continue
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            from siglip_frame.config import RuntimeConfig
            from siglip_frame.model import FeatureExtractor

            return FeatureExtractor, RuntimeConfig
        except ImportError:
            continue
    return None


class Siglip2ImageEmbedder:
    """Embeds an image query with the tagger's own vision tower.

    The image half is where a mismatch is both silent and fatal — preprocessing
    budget, pooling and normalize all move the vector — so it runs the tagger's
    `FeatureExtractor` rather than a parallel implementation of it. The local
    tower below is the fallback for when the tagger is not importable, and is
    kept byte-for-byte equivalent.

    The text half has no counterpart to import: this tagger only ever loads the
    vision tower, so `Siglip2TextEmbedder` is necessarily query-side code.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        normalize: bool = DEFAULT_NORMALIZE,
        max_num_patches: int = DEFAULT_MAX_NUM_PATCHES,
        revision: Optional[str] = DEFAULT_REVISION,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if max_num_patches < 1:
            raise EmbeddingError(f"max_num_patches must be >= 1, got {max_num_patches!r}")
        self.normalize = normalize
        self.max_num_patches = max_num_patches
        self.device, self.dtype = _resolve_device_dtype(dtype)

        tagger = _load_tagger_extractor()
        if tagger is not None:
            FeatureExtractor, RuntimeConfig = tagger
            cfg = RuntimeConfig(normalize=normalize, max_num_patches=max_num_patches)
            self._extractor = FeatureExtractor(
                cfg, model_id=model_id, revision=revision, dtype=dtype
            )
            self.processor = self._extractor.processor
            self.model = self._extractor.model
            return

        from transformers import Siglip2ImageProcessor, Siglip2VisionModel

        self._extractor = None
        # Vision tower only; transformers logs the checkpoint's text-tower keys as
        # UNEXPECTED, which is the discarded half and is expected.
        self.processor = Siglip2ImageProcessor.from_pretrained(model_id, revision=revision)
        self.model = Siglip2VisionModel.from_pretrained(
            model_id, revision=revision, dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    def embed_image(self, img: np.ndarray) -> np.ndarray:
        if self._extractor is not None:
            return self._extractor._embed_frame(img)

        inputs = self._preprocess(img)
        with torch.no_grad():
            # .float() before normalizing: dividing in bf16 lands ~0.1% off unit
            # length, which a cosine index reads as a real score difference.
            vector = self.model(**inputs).pooler_output.float()
            if self.normalize:
                vector = F.normalize(vector, p=2, dim=-1)
        return vector.squeeze(0).cpu().numpy()

    def _preprocess(self, img: np.ndarray) -> Dict[str, torch.Tensor]:
        """Turn an (H, W, 3) uint8 RGB image into the NaFlex vision-tower inputs."""
        inputs = self.processor(
            images=Image.fromarray(img),
            return_tensors="pt",
            max_num_patches=self.max_num_patches,
        )
        out = {k: v.to(self.device) for k, v in inputs.items()}
        # Only pixel_values is float; the mask and spatial shapes stay integer.
        out["pixel_values"] = out["pixel_values"].to(self.dtype)
        return out


class Siglip2TextEmbedder:
    """Embeds a text query into the same space as the frame vectors."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        normalize: bool = DEFAULT_NORMALIZE,
        revision: Optional[str] = DEFAULT_REVISION,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        from transformers import Siglip2Processor, Siglip2TextModel

        self.normalize = normalize
        self.device, self.dtype = _resolve_device_dtype(dtype)

        # Siglip2Processor rather than a bare tokenizer, for its text defaults:
        # padding="max_length", truncation=True, max_length=64.
        self.processor = Siglip2Processor.from_pretrained(model_id, revision=revision)
        self.model = Siglip2TextModel.from_pretrained(
            model_id, revision=revision, dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    def embed_text(self, query: str) -> np.ndarray:
        inputs = self.processor(text=[query], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            vector = self.model(**inputs).pooler_output.float()
            if self.normalize:
                vector = F.normalize(vector, p=2, dim=-1)
        return vector.squeeze(0).cpu().numpy()


class QueryEmbedder:
    """Both towers behind one interface, loaded lazily.

    Lazy because the weights are several GB and the projection half of the
    service is useful without them: an index loads and plots with no model
    resident, and only a search pays the load.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: Optional[str] = DEFAULT_REVISION,
        normalize: bool = DEFAULT_NORMALIZE,
        max_num_patches: int = DEFAULT_MAX_NUM_PATCHES,
        target_size: int = DEFAULT_TARGET_SIZE,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.normalize = normalize
        self.max_num_patches = max_num_patches
        self.target_size = target_size
        self._image: Optional[Siglip2ImageEmbedder] = None
        self._text: Optional[Siglip2TextEmbedder] = None

    def embed_text(self, query: str) -> List[float]:
        if not (query or "").strip():
            raise EmbeddingError("text query is empty")
        if self._text is None:
            self._text = Siglip2TextEmbedder(self.model_id, self.normalize, self.revision)
        return self._to_index_width(self._text.embed_text(query))

    def embed_image(self, image: IO[bytes]) -> List[float]:
        if self._image is None:
            self._image = Siglip2ImageEmbedder(
                self.model_id, self.normalize, self.max_num_patches, self.revision
            )
        return self._to_index_width(self._image.embed_image(decode_image(image)))

    def _to_index_width(self, vector: Sequence[float]) -> List[float]:
        return pad_vector([float(x) for x in vector], self.target_size)


def build_embedder(recipe, target_size: int) -> QueryEmbedder:
    """Pick and configure a query embedder from a tagger's stamped recipe.

    Dispatch is on the checkpoint id rather than a `kind`, because what a query
    has to be embedded *with* is the model, not what the indexed vectors are.
    Every recipe parameter that changes a vector is threaded through, so a query
    is embedded under the same recipe the index was built with.

    An unrecognised embedder raises rather than falling back to SigLIP 2: a
    wrong-model query does not fail, it silently returns meaningless neighbours.
    """
    embedder = (recipe.embedder or "").lower()

    if "siglip" in embedder:
        return QueryEmbedder(
            model_id=recipe.embedder,
            revision=recipe.revision,
            normalize=recipe.normalize,
            max_num_patches=int(recipe.params.get("max_num_patches", DEFAULT_MAX_NUM_PATCHES)),
            target_size=target_size,
        )

    if "qwen" in embedder:
        # Imported here: qwen_embedder imports back for pad_vector/EmbeddingError.
        from qwen_embedder import QwenQueryEmbedder

        return QwenQueryEmbedder(
            model_id=recipe.embedder,
            revision=recipe.revision,
            normalize=recipe.normalize,
            target_size=target_size,
            # dim is the MRL width; the rest of the sampling budget rides in params.
            params={**recipe.params, "dim": recipe.dim},
        )

    raise EmbeddingError(
        f"no query embedder is registered for {recipe.embedder!r}; "
        "add one rather than querying with a different model"
    )


def pad_vector(vec: List[float], target_size: int) -> List[float]:
    """Right-pad with zeros to the index's width.

    SigLIP 2 base emits 768 dims into a 1024-wide index. Cosine ranking is
    unaffected for unit vectors. Raises when the model is wider than the index,
    which catches the wrong checkpoint before any vectorstore call.
    """
    if len(vec) > target_size:
        raise EmbeddingError(
            f"vector of length {len(vec)} cannot be padded to smaller target {target_size}; "
            "the loaded model does not match this index"
        )
    return vec + [0.0] * (target_size - len(vec))


def decode_image(image: IO[bytes]) -> np.ndarray:
    """Decode an upload into the (H, W, 3) uint8 RGB array the model expects."""
    try:
        decoded = Image.open(image)
        # A phone photo records rotation in EXIF rather than pixel data, and
        # Image.open does not apply it, so without this an upload embeds sideways.
        decoded = ImageOps.exif_transpose(decoded) or decoded
        # CMYK, grayscale and RGBA uploads all have to reach the model as RGB.
        decoded = decoded.convert("RGB")
    except Exception as exc:
        raise EmbeddingError(f"could not decode image: {exc}") from exc
    return np.asarray(decoded, dtype=np.uint8)
