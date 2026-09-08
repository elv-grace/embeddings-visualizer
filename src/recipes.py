"""The embedding recipe a tagger stamps onto every vector it emits.

Where it comes from
-------------------
The vectorstore stores `additional_info` as JSONB and returns it verbatim on
every search hit, so the recipe arrives on the same row as the vector and no
second lookup is needed. `read_recipes` therefore parses what `get_vectors`
already fetched.

What the recipe is for
----------------------
An index does not record which model built it (`GET /indexes/{qid}` returns only
`{qid, vector_size}`), so without this the caller has to declare the model and
its query modes by hand. The stamped recipe answers both, and carries the
per-model parameters a query has to be embedded under to be comparable —
a query that differs on any of them lands in a different space, silently.

`kind` is what the vectors are, and it replaces guessing modality from which
positional fields happen to be set. That guess is wrong for whole-video tags:
they carry start == end == 0 and no frame_idx, which reads as "unknown".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# What the vectors are -> how the visualizer labels them.
KIND_TO_MODALITY = {"frame": "image", "image": "image", "video": "video", "text": "text"}


@dataclass
class Recipe:
    """The embedding recipe stamped on a track's tags."""

    embedder: str
    revision: Optional[str] = None
    dim: Optional[int] = None
    normalize: bool = True
    kind: Optional[str] = None
    query_modes: List[str] = field(default_factory=list)
    # Everything model-specific: max_num_patches for SigLIP 2; prompt, fps,
    # max_frames, max_length for Qwen. Passed through to the embedder.
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def modality(self) -> Optional[str]:
        return KIND_TO_MODALITY.get(self.kind or "")

    @classmethod
    def from_additional_info(cls, info: Dict[str, Any]) -> Optional["Recipe"]:
        """Parse one tag's additional_info, or None if it carries no recipe."""
        if not isinstance(info, dict) or not info.get("embedder"):
            return None
        known = {"embedder", "revision", "dim", "normalize", "kind", "query_modes"}
        return cls(
            embedder=str(info["embedder"]),
            revision=info.get("revision"),
            dim=info.get("dim"),
            normalize=bool(info.get("normalize", True)),
            kind=info.get("kind"),
            query_modes=list(info.get("query_modes") or []),
            params={k: v for k, v in info.items() if k not in known},
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "embedder": self.embedder,
            "revision": self.revision,
            "dim": self.dim,
            "normalize": self.normalize,
            "kind": self.kind,
            "modality": self.modality,
            "query_modes": self.query_modes,
            "params": self.params,
        }


def read_recipes(metadata: List[Dict[str, Any]]) -> Dict[str, Recipe]:
    """Recipes per track, read off rows the search already returned.

    Per track rather than per index because one index can hold tracks from
    different taggers, and each stamps its own recipe. The first stamped row of
    a track wins — the recipe is invariant across a tagger's output, so scanning
    further would only cost time.

    Tracks with no stamped recipe are simply absent: anything tagged before
    taggers started stamping has none, and the caller falls back to a declared
    model rather than failing.
    """
    found: Dict[str, Recipe] = {}
    for row in metadata:
        track = row.get("track") or ""
        if track in found:
            continue
        recipe = Recipe.from_additional_info(row.get("additional_info") or {})
        if recipe:
            found[track] = recipe
    return found
