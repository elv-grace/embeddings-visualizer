"""Retrieve a random sample of an index's vectors, and search the whole index.

Sampling
--------
`/search` takes a `shuffle_seed`: with one set the rows come back in a random
order rather than by distance, so the first `limit` of them are a uniform random
sample of whatever the filters selected. One call therefore draws the sample
directly -- no enumeration, no per-window quotas, and no reference vector whose
direction biases what is returned.

This replaces a timeline walk that bisected the index into windows small enough
to read exactly, counted each one without vectors, then refetched a
proportional quota from each. That existed because every read was a KNN and a
KNN is a cone, not a sample; `shuffle_seed` makes it unnecessary. Transfer is
O(sample) either way, but it is now one request instead of O(index/2000) of them.

Search
------
The sample is what gets *plotted*; it is not what a query searches. `search` is
a KNN over the whole index -- an HNSW lookup in the vectorstore's pgvector
partition, which is the one place that can see every row -- so a query finds its
true nearest vectors whether or not they were sampled. The hits come back with
their embeddings, which is what lets the caller project them into the picture
the sample fitted.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

VECTORSTORE_URL = "http://localhost:8108"

TIMEOUT_SECONDS = 60.0

DEFAULT_SAMPLE_SIZE = 10_000

# Hits one query pulls out of the index. Only the top few are linked and listed;
# the rest are plotted, so the neighbourhood the query landed in is visible
# rather than just its winner. Kept well under a sample's worth: these carry
# vectors and are fetched on every search.
SEARCH_LIMIT = 100


class VectorStoreError(RuntimeError):
    """A vectorstore call failed or returned something unusable."""


def _headers(auth_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {auth_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _check(response: requests.Response, what: str) -> requests.Response:
    """raise_for_status, but keeping the reason the vectorstore gave.

    The vectorstore does not evaluate access itself -- it forwards the token to
    the fabric node holding the index object and relays that verdict. So a 403
    here means the token was well-formed and signed (a malformed one comes back
    400 "unknown scheme") and the account was denied read on the index qid.
    That distinction lives only in the response body, which raise_for_status
    discards, leaving a bare "403 Client Error: Forbidden for url: ..." that
    cannot be told apart from a wrong URL or a stopped service.
    """
    if response.ok:
        return response
    reason = _reason(response)
    raise VectorStoreError(
        f"{what} failed: {response.status_code} {response.reason}"
        + (f" -- {reason}" if reason else "")
    )


def _reason(response: requests.Response) -> str:
    """The innermost `kind`/`reason` the fabric reported, or the raw body."""
    try:
        node = response.json()
    except ValueError:
        return (response.text or "").strip()[:300]

    # The fabric nests the real cause: {"error": {"cause": {"cause": {...}}}}.
    parts: List[str] = []
    seen = 0
    while isinstance(node, dict) and seen < 12:
        seen += 1
        for key in ("kind", "reason", "op"):
            value = node.get(key)
            if isinstance(value, str) and value not in parts:
                parts.append(value)
        node = node.get("cause") or node.get("error") or None
    return ", ".join(parts)[:300]


def _search(
    index_qid: str,
    auth_token: str,
    *,
    limit: int,
    vector: Optional[Sequence[float]] = None,
    shuffle_seed: Optional[int] = None,
    include_vector: bool = False,
    sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """POST /indexes/{qid}/search and return the raw result rows.

    `vector` only orders the results, so it is omitted for a shuffled read: a
    KNN that is about to be shuffled is work done to be thrown away.
    """
    body: Dict[str, Any] = {"limit": limit, "include_vector": include_vector}
    if vector is not None:
        body["vector"] = list(vector)
    if shuffle_seed is not None:
        body["shuffle_seed"] = shuffle_seed
    # Omitted rather than sent empty: an empty filter is not always a no-op.
    if sources:
        body["sources"] = sources

    response = requests.post(
        f"{VECTORSTORE_URL}/indexes/{index_qid}/search",
        json=body,
        headers=_headers(auth_token),
        timeout=TIMEOUT_SECONDS,
    )
    _check(response, f"search of {index_qid}")
    return (response.json() or {}).get("results", [])


def get_index(index_qid: str, auth_token: str) -> Dict[str, Any]:
    """GET /indexes/{qid} -- index metadata, including vector_size and config."""
    response = requests.get(
        f"{VECTORSTORE_URL}/indexes/{index_qid}",
        headers=_headers(auth_token),
        timeout=TIMEOUT_SECONDS,
    )
    # response.raise_for_status()
    _check(response, f"read of index {index_qid}")
    return response.json() or {}


def get_track_counts(index_qid: str, auth_token: str) -> Dict[str, int]:
    """GET /indexes/{qid}/tracks -- row count per track."""
    response = requests.get(
        f"{VECTORSTORE_URL}/indexes/{index_qid}/tracks",
        headers=_headers(auth_token),
        timeout=TIMEOUT_SECONDS,
    )
    # response.raise_for_status()
    _check(response, f"track counts for {index_qid}")
    tracks = (response.json() or {}).get("tracks", [])
    return {t.get("name", "?"): t.get("count", 0) for t in tracks}


def get_batch(index_qid: str, batch_id: str, auth_token: str) -> Dict[str, Any]:
    """GET /indexes/{qid}/batches/{batch_id} -- the batch a vector was written in.

    Its `model` field is what names the embedding model. `batch_id` must be a
    UUID: the handler validates the shape before authorizing, so a malformed one
    comes back 400 `Field validation for 'BatchID' failed on the 'uuid' tag`
    rather than "not found". Authorization is against the *index* qid, exactly
    like every other call here, so the same token serves.
    """
    response = requests.get(
        f"{VECTORSTORE_URL}/indexes/{index_qid}/batches/{batch_id}",
        headers=_headers(auth_token),
        timeout=TIMEOUT_SECONDS,
    )
    _check(response, f"read of batch {batch_id} in {index_qid}")
    return response.json() or {}


def batch_models(
    index_qid: str, metadata: List[Dict[str, Any]], auth_token: str
) -> Dict[str, str]:
    """Map each `batch_id` present in the rows to the batch's `model`.

    One lookup per distinct batch rather than per row: a batch is one tagger run,
    so every row in it shares a model. Batches that cannot be read are skipped
    rather than fatal -- an index whose model is unknown still plots, it just
    cannot embed a query.
    """
    models: Dict[str, str] = {}
    for batch_id in {(m.get("batch_id") or "") for m in metadata} - {""}:
        try:
            model = (get_batch(index_qid, batch_id, auth_token) or {}).get("model")
        except Exception as exc:
            logger.warning(f"could not read batch {batch_id}: {exc}")
            continue
        if model:
            models[batch_id] = str(model)
    return models



def _shuffle_seed(seed: int) -> int:
    """A non-zero `shuffle_seed` derived deterministically from `seed`.

    shuffle_seed is a plain int on the request struct, so a literal 0 cannot be
    told apart from an omitted field and the search would fall back to distance
    order. 0 is exactly the default `seed`, so that is the one case that must
    not quietly stop shuffling. Drawing through a seeded RNG keeps the mapping
    deterministic -- one seed always draws the same sample, so an index projects
    to the same picture on every run -- while staying non-zero and positive.
    """
    return random.Random(seed).randrange(1, 2 ** 31)


def _unpack(rows: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    """Split search results into (vectors, metadata), positionally aligned.

    A row whose embedding did not come back is dropped rather than padded: it
    cannot be projected or ranked, and a metadata entry with no vector behind it
    would silently misalign every index after it.
    """
    vectors: List[List[float]] = []
    metadata: List[Dict[str, Any]] = []
    for row in rows:
        entry = dict(row.get("vector") or {})
        vector = entry.pop("vector", None)
        if vector is None:
            continue
        vectors.append(vector)
        metadata.append(entry)
    return vectors, metadata


def get_vectors(
    index_qid: str,
    auth_token: str,
    sources: Optional[List[str]] = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 0,
) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    """Return (vectors, metadata) for a random sample of the index.

    Positionally aligned: metadata[i] describes vectors[i]. The sample is the
    whole index when it holds fewer than sample_size rows, and is drawn in one
    shuffled read -- see this module's docstring.
    """
    track_counts = get_track_counts(index_qid, auth_token)
    if track_counts:
        logger.info(f"Tracks: {track_counts} (total {sum(track_counts.values())})")

    logger.info(f"Reading up to {sample_size} vectors in shuffled order...")
    rows = _search(
        index_qid,
        auth_token,
        limit=sample_size,
        shuffle_seed=_shuffle_seed(seed),
        include_vector=True,
        sources=sources,
    )
    vectors, metadata = _unpack(rows)

    population = sum(track_counts.values())
    logger.info(
        f"Retrieved {len(vectors)}"
        + (f" of {population}" if population else "")
        + f" vectors from index `{index_qid}`"
    )
    return vectors, metadata


def search_vectors(
    index_qid: str,
    auth_token: str,
    vector: Sequence[float],
    limit: int = SEARCH_LIMIT,
    sources: Optional[List[str]] = None,
) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    """Return (vectors, metadata) for the `limit` nearest rows to `vector`.

    Unshuffled, so this is the vectorstore's own ranking over the *whole* index,
    not over the plotted sample. Embeddings are requested because the caller has
    to place the hits in a projection fitted elsewhere and rank them itself --
    the response's `distance` is not returned, so that one scale governs both
    the sampled points and these.
    """
    rows = _search(
        index_qid,
        auth_token,
        limit=limit,
        vector=vector,
        include_vector=True,
        sources=sources,
    )
    return _unpack(rows)


def modality(meta: Dict[str, Any]) -> str:
    """Classify one row as text / image / video from its populated fields.

    Order matters. Frame rows carry start_time == end_time (a frame is an instant,
    not an interval), so the video test has to be a strict inequality and has to
    run after the frame_idx test -- otherwise every frame reads as a video.

    A row whose end_time is missing or not greater than start_time is `unknown`,
    including a segment written before whole-media tags carried a real duration
    (those re-based the "whole of this media" sentinel into start == end). Such a
    row describes no extent, so rather than reconstructing one it is reported as
    what it is -- the detail panel names the defective field.
    """
    if (meta.get("text") or "").strip():
        return "text"
    if meta.get("frame_idx") is not None:
        return "image"
    start, end = meta.get("start_time"), meta.get("end_time")
    if start is not None and end is not None and end > start:
        return "video"
    return "unknown"
