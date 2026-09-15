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

import logging
import logging.handlers
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from embedder import (
    EmbeddingError,
    build_embedder_for_model,
    modalities_for,
    read_tuning,
    tuning_mismatch,
    warm_containers,
)
from projection import METHODS, Projector, ProjectionError, anchor_to_neighbours, top_k_similar
from tagger import status as container_status
from vectors_api import (
    DEFAULT_SAMPLE_SIZE,
    SEARCH_LIMIT,
    VectorStoreError,
    batch_models,
    get_track_counts,
    get_vectors,
    modality,
    search_vectors,
)

logger = logging.getLogger(__name__)

PORT = int(os.environ.get("EV_PORT") or 8079)

# Beside the repo, not in /tmp: a scratch path is wiped between sessions and on
# reboot, which loses exactly the history worth having. Override with EV_LOG_FILE.
LOG_FILE = Path(
    os.environ.get("EV_LOG_FILE")
    or Path(__file__).resolve().parents[1] / "logs" / "visualizer.log"
)
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3


def configure_logging(path: Path = LOG_FILE) -> Path:
    """Send this service's output to a rotating file as well as the console.

    Everything interesting used to go through `print`, which is why redirecting
    the process to a file appeared to produce nothing: Python line-buffers
    stderr but *block*-buffers stdout when it is not a TTY, so werkzeug's
    request lines (stderr) showed up while every print sat in an 8 KB buffer
    that a long-running server never fills. Logging writes and flushes per
    record, so it does not depend on how the process was launched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Idempotent: a reload must not attach a second handler and double every line.
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        root.addHandler(handler)
    if not any(isinstance(h, logging.StreamHandler)
               and not isinstance(h, logging.handlers.RotatingFileHandler)
               for h in root.handlers):
        root.addHandler(logging.StreamHandler())
    # werkzeug logs the request lines; without this they stay on stderr only.
    logging.getLogger("werkzeug").setLevel(logging.INFO)
    return path

# Every modality a query can arrive as. Which of them a given index accepts is
# not declared by the caller: it follows the model its batches report, and an
# index whose batches report none falls through to the default container, which
# is text-only. See embedder.py's docstring.
ALL_MODES = ("text", "image", "video")

# An upload larger than this is refused before it is staged for a container. A
# query is a photo or a short clip; anything else is a mistake that would
# otherwise be copied to disk and handed to a model in full.
MAX_UPLOAD_BYTES = int(os.environ.get("EV_MAX_UPLOAD_MB") or 512) * 1024 * 1024

# Of the SEARCH_LIMIT hits a query pulls from the index, how many are linked
# and listed. The rest are plotted unlabelled: the point of fetching them is to
# show the neighbourhood the query landed in, not to rank 100 rows at a viewer.
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
    # The track this plot was loaded under, or None for the whole index. Held
    # because a query has to be filtered the same way -- see `search_vectors`.
    track: Optional[str] = None
    # The model each batch in this index reports, and the one queries embed
    # with. `model` is the first -- one index can hold batches from different
    # taggers, but a query has to be embedded with one model, so the response
    # returns them all for a mismatch to be visible.
    models: Dict[str, str] = field(default_factory=dict)
    # Parameters a tagger stamped in additional_info, when it stamped any. These
    # only tune the tower the batch's model already chose -- see embedder.TUNING_KEYS.
    tuning: Dict[str, Any] = field(default_factory=dict)
    embedder: Optional[Any] = None
    # Row id -> its position in `metadata`, built on the first merge. None until
    # then rather than empty, so an index whose rows carry no id does not
    # rebuild the map on every search.
    positions: Optional[Dict[Any, int]] = None

    @property
    def model(self) -> Optional[str]:
        return next(iter(self.models.values()), None)

    def merge(
        self, vectors: np.ndarray, metadata: List[Dict[str, Any]]
    ) -> Tuple[List[int], int]:
        """Fold search hits into the plotted set; return where each one landed.

        Returns (position per input row, the first position appended). A query
        searches the whole index, so most of its hits were never sampled and
        have to be projected and added before they can be linked to. They are
        placed with the same `transform` the out-of-sample rows of a large index
        use, so they land in the picture that is already on screen rather than
        moving it.

        A hit that *was* sampled keeps the position it already has. Appending it
        again would put a second node on top of the first and point the
        neighbour link at whichever of the two the viewer is not looking at.

        Caller holds the lock: this mutates the arrays a concurrent search reads.
        """
        if self.positions is None:
            self.positions = {
                m["id"]: i for i, m in enumerate(self.metadata) if m.get("id") is not None
            }

        added_from = len(self.metadata)
        fresh: List[int] = []
        at: List[int] = []
        for i, meta in enumerate(metadata):
            row_id = meta.get("id")
            known = self.positions.get(row_id) if row_id is not None else None
            if known is None:
                known = added_from + len(fresh)
                fresh.append(i)
                if row_id is not None:
                    self.positions[row_id] = known
            at.append(known)

        if fresh:
            rows = vectors[fresh]
            self.coords = np.vstack([self.coords, self.projector.transform(rows)])
            self.vectors = np.vstack([self.vectors, rows])
            self.metadata.extend(metadata[i] for i in fresh)

        return at, added_from


_indexes: Dict[str, LoadedIndex] = {}


def create_app(static_dir: str = "../web") -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.get("/api/health")
    def health():
        """What is loaded, and what every tagger container is doing.

        `containers` names each one's log file: a query that fails names the
        container and quotes the tail of what it printed, and this is where the
        rest of it is.
        """
        return jsonify(
            {"ok": True, "loaded": list(_indexes), "containers": container_status()}
        )

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

        sources = body.get("sources") or None
        sample_size = int(body.get("sample_size") or DEFAULT_SAMPLE_SIZE)
        seed = int(body.get("seed") or 0)
        # Empty string and absent mean the same thing: the whole index.
        track = (body.get("track") or "").strip() or None

        try:
            vectors, metadata = get_vectors(
                index_qid,
                token,
                sources=sources,
                sample_size=sample_size,
                seed=seed,
                track=track,
            )
        except VectorStoreError as exc:
            return _error(str(exc), 400)
        except Exception as exc:
            return _error(f"could not read index: {exc}", 502)

        if not vectors:
            # Say which of the two this is. The index answered (a missing or
            # unreadable one raises above and returns 400/502), so either it
            # genuinely holds nothing, or it holds rows the read returned
            # without their embeddings -- a row with no `vector` cannot be
            # projected and is dropped. The track counts separate those two
            # without another guess.
            try:
                counts = get_track_counts(index_qid, token)
            except Exception:
                counts = {}
            total = sum(counts.values())
            if track and track not in counts:
                # Much the likeliest cause once a filter is involved, and the
                # one the viewer can act on: name the tracks that do exist.
                logger.warning(f"index {index_qid} has no track {track!r}; tracks={counts}")
                return _error(
                    f"index {index_qid} has no track {track!r}; it holds "
                    + (f"{sorted(counts)}" if counts else "no tracks at all"),
                    404,
                )
            detail = (
                f"but its tracks report {total} rows ({counts}) -- the search returned "
                "no embeddings for them, so check the index holds vectors and not only "
                "documents"
                if total
                else "and its tracks report no rows either, so the index is empty"
            )
            logger.warning(f"index {index_qid} enumerated 0 rows; tracks={counts}")
            return _error(
                f"index {index_qid}"
                + (f" track {track!r}" if track else "")
                + f" returned no vectors, {detail}",
                404,
            )

        try:
            tracks = get_track_counts(index_qid, token)
        except Exception:
            tracks = {}

        # Which model built this index, from the batches its rows name. One
        # lookup per distinct batch, and authorized by the same index token.
        models = batch_models(index_qid, metadata, token)
        model = next(iter(models.values()), None)
        # Modes follow the model, not anything a vector carries and not anything
        # the caller asked for. With no model this is the default container's
        # text, which is the only modality safe to assume of an unidentified
        # space.
        modes = modalities_for(model)
        if model:
            logger.info(f"Index model: {model} (from {len(models)} batch(es))")

        # additional_info is read for one thing only now: the parameters that
        # tune the tower the batch already chose. It cannot select a model.
        tuning = read_tuning(metadata)
        if tuning:
            logger.info(f"Tuning parameters stamped on the rows: {sorted(tuning)}")
        # The container runs on the recipe in config.yml, not on this one -- one
        # model is one container, so the recipe is stated rather than derived.
        # Where the two disagree the queries are embedded under different
        # parameters than the index was built with, which does not fail, it just
        # quietly returns worse neighbours. So it is said out loud, and returned
        # so the UI can say it too.
        mismatch = tuning_mismatch(model, tuning)
        if mismatch:
            logger.warning(
                f"index {index_qid} was tagged with "
                + ", ".join(f"{k}={v['index']!r} (config says {v['configured']!r})"
                            for k, v in mismatch.items())
                + f" -- queries will use config.yml's values. Align `params:` for "
                f"{model!r} or re-tag."
            )

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
                model_id=model or "unknown",
                tracks=tracks,
                models=models,
                tuning=tuning,
                track=track,
            )

        return jsonify(
            {
                "index_key": key,
                "index_qid": index_qid,
                "method": method,
                "modes": modes,
                "model_id": _indexes[key].model_id,
                "models": models,
                "count": len(metadata),
                "vector_size": int(matrix.shape[1]),
                "tracks": tracks,
                # Echoed rather than assumed: the frontend keeps the header's
                # selector on whatever was actually loaded.
                "track": track,
                # Empty when nothing was stamped. When it is not, these are the
                # container's --params, so the query is embedded under the same
                # recipe the index was tagged with.
                "tuning": tuning,
                # Non-empty when the index's stamped recipe disagrees with the
                # one its container is configured to run.
                "tuning_mismatch": mismatch,
                "bbox": projector.bbox,
                "explained_variance": projector.explained_variance,
                "points": _points(coords, metadata),
            }
        )

    @app.post("/api/search/<mode>")
    def search(mode: str):
        """Embed a query, rank it against the whole index, and plot what it found.

        The ranking is the vectorstore's, over every row in the index, rather
        than a scan of the sample held here: the sample bounds what is *drawn*,
        and letting it bound what is *findable* would make every search a search
        of ten thousand arbitrary rows. The hits it returns are folded into the
        projection and sent back as new points, so a match is on screen whether
        or not it was sampled.
        """
        if mode not in ALL_MODES:
            return _error(f"unknown mode {mode!r}", 404)

        key = request.form.get("index_key") or (request.get_json(silent=True) or {}).get("index_key")
        loaded = _indexes.get(key or "")
        if loaded is None:
            return _error("index not loaded; load an index first", 404)

        # Loading the index did not keep its token -- it is forwarded and
        # dropped -- and the vectorstore authorizes per request, so the search
        # needs one of its own.
        token = _token()
        if not token:
            return _error("missing Authorization token", 401)

        if mode not in loaded.modes:
            # Wording fixed by the spec in README.md.
            return _error(f"Index does not support {mode} embeddings.", 422)

        if loaded.embedder is None:
            width = int(loaded.vectors.shape[1])
            # The batch's model chooses the container; its recipe is configured
            # beside it, so every index on this model shares that one container.
            try:
                loaded.embedder = build_embedder_for_model(loaded.model, width)
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
                data = upload.stream.read(MAX_UPLOAD_BYTES + 1)
                if len(data) > MAX_UPLOAD_BYTES:
                    return _error(
                        f"this {mode} is larger than the "
                        f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB query limit; "
                        "a query is meant to be a still or a short clip",
                        413,
                    )
                # The filename carries the suffix the container's decoder picks by.
                vector = loaded.embedder.embed_file(mode, data, upload.filename)
        except EmbeddingError as exc:
            return _error(str(exc), 400)
        except Exception as exc:
            return _error(f"could not embed query: {exc}", 500)

        query = np.asarray(vector, dtype=np.float32)

        try:
            hits, hit_meta = search_vectors(
                loaded.index_qid,
                token,
                query.tolist(),
                limit=SEARCH_LIMIT,
                track=loaded.track,
            )
        except VectorStoreError as exc:
            return _error(str(exc), 400)
        except Exception as exc:
            return _error(f"could not search index: {exc}", 502)
        if not hits:
            return _error(
                f"index {loaded.index_qid}"
                + (f" track {loaded.track!r}" if loaded.track else "")
                + " returned no hits for this query",
                404,
            )

        matrix = np.asarray(hits, dtype=np.float32)
        try:
            # Re-ranked here rather than read off the response's `distance`, so
            # one scale governs both these hits and anything else this service
            # scores. It also settles the width check before the hits are merged.
            ranked = top_k_similar(matrix, query, k=TOP_K)
            with _lock:
                at, added_from = loaded.merge(matrix, hit_meta)
                # Sliced under the same lock that appended it. A concurrent
                # search on this index appends too, and a slice taken after the
                # release would hand this response the other query's rows as
                # well -- which the frontend would then plot twice, once per
                # response. `coords` is rebound by a merge, never written in
                # place, so the reference taken here stays this snapshot.
                coords = loaded.coords
                added = _points(
                    coords[added_from:], loaded.metadata[added_from:], offset=added_from
                )
                count = len(loaded.metadata)
            neighbours = [(at[i], similarity) for i, similarity in ranked]
            # Anchored to its neighbours rather than projected: see
            # projection.anchor_to_neighbours for why the projected point of an
            # off-manifold query carries no information.
            x, y = anchor_to_neighbours(coords, neighbours)
            projected = loaded.projector.transform(query.reshape(1, -1))[0]
        except ProjectionError as exc:
            return _error(str(exc), 400)

        logger.info(
            f"query ({mode}) on {loaded.index_qid}: {len(hits)} hits from the index, "
            f"{len(added)} new to the plot, best {ranked[0][1]:.4f}"
        )

        return jsonify(
            {
                "mode": mode,
                "query_point": {"x": x, "y": y},
                # Kept for comparison; not what the node is drawn at.
                "projected_point": {"x": float(projected[0]), "y": float(projected[1])},
                # How many rows the index was asked for, and how many of them
                # were not already plotted. The frontend says so: a viewer has
                # to know the search saw the whole index, not just the sample.
                "searched": len(hits),
                "count": count,
                # Non-null when the plot is filtered, so the frontend can say
                # the search was narrowed the same way.
                "track": loaded.track,
                # The hits that were not in the sample, ready to append. `i` is
                # already their position in the loaded set, which is what
                # `neighbours[].index` refers to.
                "points": added,
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
    coords: np.ndarray, metadata: List[Dict[str, Any]], offset: int = 0
) -> List[Dict[str, Any]]:
    """The per-node payload: position, modality, and the metadata for the card.

    Modality is inferred from which fields a row populates -- see
    `vectors_api.modality`. `offset` is where this slice starts in the loaded
    index: `i` is the frontend's handle on a node and has to stay the position
    in the whole set, so a batch of search hits appended to the end numbers from
    there rather than from 0."""
    return [
        {
            "i": offset + i,
            "x": float(coords[i][0]),
            "y": float(coords[i][1]),
            "modality": modality(meta),
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


def _warm_containers() -> None:
    """Bring up the tagger containers, and keep them up.

    In a thread because it starts a container per configured model and an image
    that has to be pulled can take minutes; the service loads and plots indexes
    perfectly well while that happens, and only a *query* needs a container.
    """
    try:
        images = warm_containers()
    except Exception as exc:
        logger.warning(f"could not warm the tagger containers: {exc}")
        return
    logger.info(
        f"tagger containers running: {', '.join(images)}" if images
        else "no tagger containers were started; queries will start them on demand"
    )


if __name__ == "__main__":
    log_path = configure_logging()
    logging.getLogger(__name__).info(f"starting on port {PORT}, logging to {log_path}")
    # Also on stdout, so `python3 src/app.py` says where its log went even when
    # the console is where someone is looking.
    print(f"embeddings-visualizer: port {PORT}, log {log_path}", flush=True)
    threading.Thread(target=_warm_umap, daemon=True).start()
    threading.Thread(target=_warm_containers, daemon=True).start()
    create_app().run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
