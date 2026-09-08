"""HTTP service for the embeddings visualizer.

Serves the frontend and the projection/search API from one origin, so the page
runs inside an elv-core-js iframe without cross-origin requests.

The auth token is supplied per request by the frontend, which obtains it from
core via the FrameClient. It is forwarded to the vectorstore and never stored.

Vectors stay server-side. The browser receives 2D coordinates plus metadata
(~50 bytes/point rather than ~12 KB), which is what makes a large index
displayable at all.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from embedder import (
    DEFAULT_MAX_NUM_PATCHES,
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    EmbeddingError,
    QueryEmbedder,
    build_embedder,
)
from recipes import Recipe, read_recipes
from projection import METHODS, Projector, ProjectionError, anchor_to_neighbours, top_k_similar
from vectors_api import DEFAULT_SAMPLE_SIZE, VectorStoreError, get_track_counts, get_vectors, modality

PORT = 8099

# Modes a caller may declare. The index does not record which model built it, so
# the caller supplies the model and its modes; see embedder.py's docstring.
ALL_MODES = ("text", "image", "video")
DEFAULT_MODES = ("text", "image")

TOP_K = 10

# Loaded indexes are held for the process's life, and each pins its full vector
# matrix plus a fitted projector. Without a bound, a session of reloads walks the
# process out of memory.
MAX_LOADED = 4

_lock = threading.Lock()


@dataclass
class LoadedIndex:
    """One projected index, held for the lifetime of the process."""

    index_qid: str
    vectors: np.ndarray
    metadata: List[Dict[str, Any]]
    coords: np.ndarray
    projector: Projector
    modes: List[str]
    model_id: str
    tracks: Dict[str, int] = field(default_factory=dict)
    # Per-track recipes read off the search rows. Empty when the content was
    # tagged before taggers stamped `additional_info`, which is when the
    # caller-declared model and modes are used instead.
    recipes: Dict[str, Recipe] = field(default_factory=dict)
    embedder: Optional[QueryEmbedder] = None

    @property
    def recipe(self) -> Optional[Recipe]:
        """The recipe to embed a query with.

        One index can hold tracks from different taggers, but a query has to be
        embedded with one model, so this takes the first and the response
        reports every recipe found for the caller to notice a mismatch.
        """
        return next(iter(self.recipes.values()), None)


_indexes: Dict[str, LoadedIndex] = {}


def create_app(static_dir: str = "../web") -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "loaded": list(_indexes)})

    @app.post("/api/index")
    def load_index():
        """Fetch, project and cache an index. Returns the points to plot."""
        body = request.get_json(silent=True) or {}
        index_qid = (body.get("index_qid") or "").strip()
        if not index_qid:
            return _error("index_qid is required", 400)

        token = _token()
        if not token:
            return _error("missing Authorization token", 401)

        method = body.get("method") or "umap"
        if method not in METHODS:
            return _error(f"unknown method {method!r}, expected one of {list(METHODS)}", 400)

        modes = [m for m in (body.get("modes") or DEFAULT_MODES) if m in ALL_MODES]
        sources = body.get("sources") or None
        sample_size = int(body.get("sample_size") or DEFAULT_SAMPLE_SIZE)
        seed = int(body.get("seed") or 0)

        try:
            vectors, metadata = get_vectors(
                index_qid, token, sources=sources, sample_size=sample_size, seed=seed
            )
        except VectorStoreError as exc:
            return _error(str(exc), 400)
        except Exception as exc:
            return _error(f"could not read index: {exc}", 502)

        if not vectors:
            return _error(f"index {index_qid} returned no vectors", 404)

        try:
            tracks = get_track_counts(index_qid, token)
        except Exception:
            tracks = {}

        # The recipe the tagger stamped, read off the rows already fetched --
        # the vectorstore returns additional_info verbatim on every search hit.
        recipes = read_recipes(metadata)
        if recipes:
            detected = next(iter(recipes.values()))
            modes = detected.query_modes or modes

        matrix = np.asarray(vectors, dtype=np.float32)
        projector = Projector(method=method, seed=seed)
        try:
            coords = projector.fit(matrix)
        except ProjectionError as exc:
            return _error(str(exc), 400)

        key = uuid.uuid4().hex
        with _lock:
            # Oldest out first; dicts preserve insertion order.
            while len(_indexes) >= MAX_LOADED:
                _indexes.pop(next(iter(_indexes)))
            _indexes[key] = LoadedIndex(
                index_qid=index_qid,
                vectors=matrix,
                metadata=metadata,
                coords=coords,
                projector=projector,
                modes=modes,
                model_id=(
                    recipes[next(iter(recipes))].embedder
                    if recipes
                    else (body.get("model_id") or DEFAULT_MODEL_ID)
                ),
                tracks=tracks,
                recipes=recipes,
            )

        return jsonify(
            {
                "index_key": key,
                "index_qid": index_qid,
                "method": method,
                "modes": modes,
                "model_id": _indexes[key].model_id,
                "count": len(metadata),
                "vector_size": int(matrix.shape[1]),
                "tracks": tracks,
                # Empty when nothing was stamped; the UI then says the model and
                # modes were declared rather than detected.
                "recipes": {t: r.to_json() for t, r in recipes.items()},
                "bbox": projector.bbox,
                "explained_variance": projector.explained_variance,
                "points": _points(coords, metadata, recipes),
            }
        )

    @app.post("/api/search/<mode>")
    def search(mode: str):
        """Embed a query, place it in the fitted projection, rank in full dims."""
        if mode not in ALL_MODES:
            return _error(f"unknown mode {mode!r}", 404)

        key = request.form.get("index_key") or (request.get_json(silent=True) or {}).get("index_key")
        loaded = _indexes.get(key or "")
        if loaded is None:
            return _error("index not loaded; load an index first", 404)

        if mode not in loaded.modes:
            # Wording fixed by the spec in README.md.
            return _error(f"Index does not support {mode} embeddings.", 422)

        if loaded.embedder is None:
            width = int(loaded.vectors.shape[1])
            try:
                # A stamped recipe names the model and the parameters its vectors
                # were produced under; without one, fall back to the defaults the
                # caller declared against.
                loaded.embedder = (
                    build_embedder(loaded.recipe, width)
                    if loaded.recipe
                    else QueryEmbedder(
                        model_id=loaded.model_id,
                        revision=DEFAULT_REVISION,
                        max_num_patches=DEFAULT_MAX_NUM_PATCHES,
                        target_size=width,
                    )
                )
            except EmbeddingError as exc:
                return _error(str(exc), 422)

        try:
            if mode == "text":
                body = request.get_json(silent=True) or {}
                vector = loaded.embedder.embed_text(body.get("query") or "")
            else:
                upload = request.files.get("file")
                if upload is None:
                    return _error("a file upload is required for this mode", 400)
                if mode == "video":
                    embed_video = getattr(loaded.embedder, "embed_video", None)
                    if embed_video is None:
                        return _error(
                            f"{loaded.model_id} cannot embed video queries", 422
                        )
                    # The filename carries the container suffix the decoder picks by.
                    vector = embed_video(upload.stream, upload.filename)
                else:
                    vector = loaded.embedder.embed_image(upload.stream)
        except EmbeddingError as exc:
            return _error(str(exc), 400)
        except Exception as exc:
            return _error(f"could not embed query: {exc}", 500)

        query = np.asarray(vector, dtype=np.float32)
        try:
            neighbours = top_k_similar(loaded.vectors, query, k=TOP_K)
            # Anchored to its neighbours rather than projected: see
            # projection.anchor_to_neighbours for why the projected point of an
            # off-manifold query carries no information.
            x, y = anchor_to_neighbours(loaded.coords, neighbours)
            projected = loaded.projector.transform(query.reshape(1, -1))[0]
        except ProjectionError as exc:
            return _error(str(exc), 400)

        return jsonify(
            {
                "mode": mode,
                "query_point": {"x": x, "y": y},
                # Kept for comparison; not what the node is drawn at.
                "projected_point": {"x": float(projected[0]), "y": float(projected[1])},
                # Ranked in the original space: 2D proximity is not similarity.
                "neighbours": [
                    {"index": i, "similarity": s, "id": loaded.metadata[i].get("id")}
                    for i, s in neighbours
                ],
            }
        )

    @app.after_request
    def _no_store(response):
        """Never let a browser cache the frontend.

        This is edited live, and a stale copy fails in ways that look like the
        server is down rather than like a cache: a cached index.html that still
        loads app.js as a classic script hits a syntax error on its first
        `import` and the page renders nothing at all.
        """
        if not request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response

    @app.get("/")
    def root():
        return send_from_directory(app.root_path + "/" + static_dir, "index.html")

    @app.get("/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(app.root_path + "/" + static_dir, filename)

    return app


def _points(
    coords: np.ndarray,
    metadata: List[Dict[str, Any]],
    recipes: Optional[Dict[str, Recipe]] = None,
) -> List[Dict[str, Any]]:
    """The per-node payload: position, modality, and the metadata for the card.

    Modality comes from the track's stamped `kind` when there is one, and only
    falls back to inferring it from populated fields otherwise. The inference is
    wrong for whole-video tags — start == end == 0 and no frame_idx reads as
    "unknown" — which is exactly what `kind` exists to settle.
    """
    recipes = recipes or {}
    return [
        {
            "i": i,
            "x": float(coords[i][0]),
            "y": float(coords[i][1]),
            "modality": (
                (recipes.get(meta.get("track") or "") or Recipe(embedder="")).modality
                or modality(meta)
            ),
            "meta": meta,
        }
        for i, meta in enumerate(metadata)
    ]


def _token() -> Optional[str]:
    header = request.headers.get("Authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return header.strip() or None


def _error(message: str, status: int):
    return jsonify({"error": message}), status


def _warm_umap() -> None:
    """Fit UMAP on noise at boot.

    UMAP's numba kernels compile on first use, which costs ~30s. Paying it here
    means the first real index load is not the one that waits.
    """
    try:
        Projector(method="umap").fit(np.random.default_rng(0).random((64, 16), dtype=np.float32))
    except Exception:
        pass


if __name__ == "__main__":
    threading.Thread(target=_warm_umap, daemon=True).start()
    create_app().run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
