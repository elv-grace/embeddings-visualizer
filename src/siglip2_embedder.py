"""SigLIP 2 image and text towers, for embedding a search query into an index.

Where the image tower comes from
--------------------------------
The tagger's own, imported rather than reimplemented: `Siglip2CropEmbedder` from
model-frame-vector (the repo formerly called model-detection; its package is
still `general_detection`). It replaced model-vector's `FeatureExtractor` on
2026-09-10 and was checked to be bit-identical on a non-square image -- max abs
diff 0.0, cosine 1.0 -- so queries embed into the same space as before. The
local tower further down is the fallback for when neither is importable, and is
kept equivalent to both.

The *text* tower has nothing to import: these taggers only ever load the vision
half, so it stays query-side code duplicated with content-search's copy. That
copy is load-bearing and silent when wrong -- a mismatch does not raise, it just
returns bad neighbours. In particular padding="max_length"/max_length=64 is the
fixed-length padding SigLIP was trained on; tokenized any other way the query
vector moves far enough that results collapse onto whatever content dominates
the index.

TODO: collapse the text tower into one shared package.

`embedder.py` owns the registry that decides *which* tower a given index needs;
this module only knows how to be the SigLIP 2 one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, IO, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from embedder import EmbeddingError, decode_image, pad_vector

# Must match what the index was tagged with.
DEFAULT_MODEL_ID = "google/siglip2-base-patch16-naflex"
DEFAULT_REVISION = "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"
DEFAULT_NORMALIZE = True
DEFAULT_MAX_NUM_PATCHES = 256
DEFAULT_TARGET_SIZE = 1024


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
    # The model-detection repo was renamed model-frame-vector (2026-09-10); the
    # package inside it is still `general_detection`. The old name stays as a
    # fallback so a checkout that has not been renamed still resolves.
    str(Path(__file__).resolve().parents[2] / "model-frame-vector"),
    str(Path(__file__).resolve().parents[2] / "model-detection"),
)


def _load_tagger_extractor():
    """model-frame-vector's `Siglip2CropEmbedder`, or None if it is not importable.

    Same resolution order as `qwen_embedder`: plain import (inside the tagger's
    container `/elv` is WORKDIR and already on `sys.path`), then
    `SIGLIP_EMBEDDING_PATH`, then a sibling checkout.

    Substituting it for model-vector's `FeatureExtractor` is safe because the two
    compute the same thing for a whole image: `Siglip2ImageProcessor` at a patch
    budget, `Siglip2VisionModel(...).pooler_output.float()`, then an optional
    `F.normalize`. Its extra lever is `max_upscale`, which shrinks the budget for
    small *crops*; left at its default of None every input gets the full
    `max_num_patches`, which is the fixed budget the frame tagger always used.
    Its defaults match this module's constants exactly (256 patches, normalize
    on), so a query embeds into the same space as before.
    """
    paths = [os.environ["SIGLIP_EMBEDDING_PATH"]] if os.environ.get("SIGLIP_EMBEDDING_PATH") else []
    paths += list(SIGLIP_PATH_CANDIDATES)
    for path in [None] + paths:
        if path is not None:
            if not Path(path, "general_detection", "embedder.py").is_file():
                continue
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            from general_detection.config import RuntimeConfig
            from general_detection.embedder import Siglip2CropEmbedder

            return Siglip2CropEmbedder, RuntimeConfig
        except ImportError:
            continue
    return None


class Siglip2ImageEmbedder:
    """Embeds an image query with the tagger's own vision tower.

    The image half is where a mismatch is both silent and fatal — preprocessing
    budget, pooling and normalize all move the vector — so it runs the tagger's
    `Siglip2CropEmbedder` rather than a parallel implementation of it. The local
    tower below is the fallback for when the tagger is not importable, and is
    kept numerically equivalent.

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
            Siglip2CropEmbedder, RuntimeConfig = tagger
            # max_upscale stays None: a query is a whole image, so it gets the
            # full patch budget, which is what the frame tagger always used.
            self._cfg = RuntimeConfig(normalize=normalize, max_num_patches=max_num_patches)
            self._extractor = Siglip2CropEmbedder(
                model_id=model_id, revision=revision, dtype=dtype
            )
            self.processor = self._extractor.processor
            self.model = self._extractor.model
            return

        from transformers import Siglip2ImageProcessor, Siglip2VisionModel

        self._extractor = None
        self._cfg = None
        # Vision tower only; transformers logs the checkpoint's text-tower keys as
        # UNEXPECTED, which is the discarded half and is expected.
        self.processor = Siglip2ImageProcessor.from_pretrained(model_id, revision=revision)
        self.model = Siglip2VisionModel.from_pretrained(
            model_id, revision=revision, dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    def embed_image(self, img: np.ndarray) -> np.ndarray:
        if self._extractor is not None:
            # embed() is batched and also returns per-crop upscale factors, which
            # are a tagging-side diagnostic; one image in, one vector out.
            # return self._extractor._embed_frame(img)
            vectors, _upscales = self._extractor.embed([img], self._cfg)
            return vectors[0]

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



MODEL_ID = DEFAULT_MODEL_ID


def build(target_size: int, params: Optional[Dict[str, Any]] = None) -> "QueryEmbedder":
    """The SigLIP 2 query embedder, configured for one index.

    `params` only tunes an already-chosen tower; it never selects one. An index
    that stamped nothing still queries correctly, on the constants above.
    """
    params = params or {}
    return QueryEmbedder(
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_REVISION,
        normalize=bool(params.get("normalize", DEFAULT_NORMALIZE)),
        max_num_patches=int(params.get("max_num_patches", DEFAULT_MAX_NUM_PATCHES)),
        target_size=target_size,
    )
