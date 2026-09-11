"""Which embedding model an index was built with, and the tower that matches it.

How the model is determined
---------------------------
From the **batch** a vector was written in, not from anything the vector itself
carries. Each row names its `batch_id`; `GET /indexes/{qid}/batches/{batch_id}`
returns that batch's `model`, and the mapping from model to embedder is 1-1
because a batch is one tagger run and a tagger has exactly one embedder.

This replaced reading the model out of each vector's `additional_info`. A batch
is the right place for it: the field cannot go missing the way a stamped one can
(taggers have already stopped stamping `kind`), it is authoritative for every
row in the run rather than per-row, and it costs one lookup per batch instead of
trusting whatever the first row happened to carry.

`query_modes` lives here for the same reason -- which kinds of query a space
accepts is a property of the model, not something a vector should have to say.

The towers themselves are in `siglip2_embedder` and `qwen_embedder`; this module
only decides between them. Both are imported lazily, inside `build_embedder_for_model`,
so the weights load only for a query and so the tower modules can import the
shared helpers below without a cycle.
"""

from __future__ import annotations

from typing import IO, Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps

# Width the visualizer pads query vectors to when an index is wider than the
# model; see pad_vector.
DEFAULT_TARGET_SIZE = 1024


class EmbeddingError(RuntimeError):
    """A query could not be embedded."""


# An index's `model` -> the tower that embeds into that space, and the query
# kinds it accepts. Adding a tagger means adding one entry here.
INDEX_MODELS: Dict[str, Dict[str, Any]] = {
    "frame_vectors": {"tower": "siglip2_embedder", "query_modes": ["text", "image"]},
    "video_vectors": {"tower": "qwen_embedder", "query_modes": ["text", "image", "video"]},
    # Face embeddings from model-celeb-vector. `image` only, and deliberately:
    # these are InsightFace identity vectors, so there is no text tower that
    # could put a description into the same space.
    "face_vectors": {"tower": "celeb_embedder", "query_modes": ["image"]},
}


# Keys a tagger may stamp in `additional_info` that change the vector a query
# must be embedded into. A whitelist, so the provenance a tagger also stamps
# (box, score, upscale, crop_padding, detector, text) never reaches a tower.
#
# These TUNE a tower; they never choose one. The model comes from the batch --
# see this module's docstring -- and nothing here can override it.
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
    tower that owns it, and an index that stamps nothing still queries correctly.

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


def model_spec(model: Optional[str]) -> Optional[Dict[str, Any]]:
    """The registry entry for an index's `model`, or None if it is unregistered."""
    return INDEX_MODELS.get((model or "").strip())


def query_modes_for(model: Optional[str]) -> List[str]:
    """Which query kinds this model's space accepts; empty when unregistered."""
    spec = model_spec(model)
    return list(spec["query_modes"]) if spec else []


def build_embedder_for_model(
    model: Optional[str], target_size: int, params: Optional[Dict[str, Any]] = None
):
    """The query embedder for an index whose batches report `model`.

    `params` only tunes an already-chosen tower (patch budget, sampling budget,
    MRL width) and never selects one, so an index that stamped no parameters
    still queries correctly on the tagger's own defaults.

    An unregistered model raises rather than defaulting to one of the towers: a
    wrong-model query does not fail, it silently returns meaningless neighbours.
    """
    spec = model_spec(model)
    if spec is None:
        raise EmbeddingError(
            f"no query embedder is registered for index model {model!r}; "
            f"known models are {sorted(INDEX_MODELS)}. Add one to INDEX_MODELS "
            "rather than querying with a different model."
        )
    # Imported here, not at module scope: the tower modules import the helpers
    # below, and the weights should load only when a query needs them.
    import importlib

    tower = importlib.import_module(spec["tower"])
    return tower.build(target_size, params)


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
