"""Retrieve a representative sample of an index's vectors and their metadata.

Enumeration
-----------
The vectorstore exposes no scan endpoint: every read goes through /search, which
is a KNN around a reference vector. Membership in the result is nevertheless
independent of that vector whenever the filtered subset is smaller than `limit` --
the search filters (qids, track, start_time_gte/lte) are pre-filtered, so the
search runs an exact KNN over the filtered subset and returns all of it. Walking
the timeline in windows narrow enough to stay under `limit` therefore enumerates
the index exactly, with the reference vector only setting the order within a
window. Truncation is detected rather than assumed: a window that comes back full
is bisected and retried, so this stays correct even if the pre-filter semantics
differ from the ones documented in vectorstore-swagger.yaml (which describes
/spaces, not the deployed /indexes API).

Sampling
--------
A vector is ~12 KB of JSON, so enumerating an index with them inline does not
scale. The counting pass runs with include_vector=false (~150 B/row) to get each
window's population, each window is then given a quota proportional to that
population, and only the quota is refetched with vectors. Transfer is O(sample),
not O(index).

Within a window the quota is still the k nearest to the reference vector, so each
window draws its own random probe direction: the strata are covered
proportionally and the within-window bias varies independently across windows
instead of compounding into one global cone. A single fixed probe -- [0.5]*d or
one random draw alike -- returns a cone of the space around one direction, which
is not a sample of the index.
"""

from __future__ import annotations

import random
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

VECTORSTORE_URL = "http://localhost:8108"

TIMEOUT_SECONDS = 60.0

# Rows one /search call may return. Windows returning this many are treated as
# truncated and bisected.
WINDOW_LIMIT = 2000

# Upper bound of the timeline walk, in ms. start_time is an Int4 column, and a
# larger bound makes the search fail with a 500 rather than return nothing.
MAX_START_TIME_MS = 2 ** 31 - 1

DEFAULT_SAMPLE_SIZE = 10_000


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
    vector: Sequence[float],
    limit: int,
    include_vector: bool = False,
    sources: Optional[List[str]] = None,
    start_time_gte: Optional[int] = None,
    start_time_lte: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """POST /indexes/{qid}/search and return the raw result rows."""
    body: Dict[str, Any] = {
        "vector": list(vector),
        "limit": limit,
        "include_vector": include_vector,
    }
    # Omitted rather than sent empty: an empty filter is not always a no-op.
    if sources:
        body["sources"] = sources
    if start_time_gte is not None:
        body["start_time_gte"] = start_time_gte
    if start_time_lte is not None:
        body["start_time_lte"] = start_time_lte

    response = requests.post(
        f"{VECTORSTORE_URL}/indexes/{index_qid}/search",
        json=body,
        headers=_headers(auth_token),
        timeout=TIMEOUT_SECONDS,
    )
    # response.raise_for_status()
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


def _random_probe(vector_size: int, rng: random.Random) -> List[float]:
    """A uniformly random direction on the unit sphere."""
    raw = [rng.gauss(0.0, 1.0) for _ in range(vector_size)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


def count_windows(
    index_qid: str,
    auth_token: str,
    probe: Sequence[float],
    sources: Optional[List[str]] = None,
) -> Tuple[List[Tuple[int, int, int]], Dict[str, List[int]]]:
    """Partition the timeline into windows of at most WINDOW_LIMIT rows.

    Returns (start_time_gte, start_time_lte, row_count) per non-empty window,
    plus every row's start_time grouped by content qid. Runs without vectors, so
    the whole index costs ~150 B/row to walk.

    The start times are collected here, and not later, because this pass sees the
    whole index while the fetch that follows sees only the sample. Deriving
    shot bounds from the sample would stretch a shot across whatever its
    neighbours' rows the sampling happened to drop; deriving them here means a
    sampled row still carries the bound it has in the full index.
    """
    windows: List[Tuple[int, int, int]] = []
    starts_by_qid: Dict[str, List[int]] = {}
    # Explicit stack rather than recursion: the bisection can go ~40 deep.
    pending = [(0, MAX_START_TIME_MS)]

    while pending:
        lo, hi = pending.pop()
        rows = _search(
            index_qid,
            auth_token,
            vector=probe,
            limit=WINDOW_LIMIT,
            include_vector=False,
            sources=sources,
            start_time_gte=lo,
            start_time_lte=hi,
        )
        if not rows:
            continue
        if len(rows) >= WINDOW_LIMIT and hi > lo:
            mid = (lo + hi) // 2
            pending.append((lo, mid))
            pending.append((mid + 1, hi))
            continue
        if len(rows) >= WINDOW_LIMIT:
            # More than WINDOW_LIMIT rows share one timestamp; cannot split further.
            print(f"warning: window [{lo}, {hi}] is saturated, rows beyond {WINDOW_LIMIT} are invisible")
        windows.append((lo, hi, len(rows)))

        for row in rows:
            entry = row.get("vector") or {}
            # Frames and text are instants, not segments; including them would
            # invent an interval for a row that never had one.
            if entry.get("frame_idx") is not None or (entry.get("text") or "").strip():
                continue
            start = entry.get("start_time")
            if start is None:
                continue
            starts_by_qid.setdefault(entry.get("qid") or "", []).append(int(start))

    windows.sort()
    for starts in starts_by_qid.values():
        starts.sort()
    return windows, starts_by_qid


def derive_segment_ends(
    metadata: List[Dict[str, Any]], starts_by_qid: Dict[str, List[int]]
) -> int:
    """Fill in `derived_end_time` for segment rows whose own end is unusable.

    Why the end is missing
    ----------------------
    A tag-aligned tagger is handed one segment at a time as its whole input, so
    within that clip it sees a single window and stamps the "whole of this media"
    sentinel start == end == 0. The pipeline then re-bases the row into the parent
    timeline, shifting both fields by the segment's offset and leaving
    start == end == segment start -- a row that reads as "whole video" and plays
    to the end of the file instead of stopping at its own segment.

    Model-side windowing never lands here: a tagger that splits by
    `segment_length_s` sees several windows and stamps real bounds, so `end_time`
    is already right and nothing below fires.

    What is assumed
    ---------------
    Segments tile the timeline contiguously and do not overlap --
    true of shot alignment and of fixed-length clip alignment alike, which is why
    this keys off neighbouring starts rather than anything shot-specific. The next
    segment's start is this segment's end, so for tiling input this
    reconstructs the alignment track's bounds rather than approximating them.

    The last segment of each object has no successor and is left open, which
    plays it to the end of the video -- right for a final segment.

    Written to `derived_end_time`, never over `end_time`: the row keeps saying
    what the tagger actually recorded, so a reconstructed bound can be labelled
    as reconstructed.
    """
    filled = 0
    for meta in metadata:
        start, end = meta.get("start_time"), meta.get("end_time")
        if start is None or (end is not None and end > start):
            continue   # a real interval; nothing to derive
        if meta.get("frame_idx") is not None or (meta.get("text") or "").strip():
            continue
        starts = starts_by_qid.get(meta.get("qid") or "") or []
        nxt = bisect_right(starts, int(start))
        if nxt >= len(starts):
            continue   # last segment of this object: leave open
        meta["derived_end_time"] = starts[nxt]
        filled += 1
    return filled


def _quotas(counts: Sequence[int], sample_size: int) -> List[int]:
    """Apportion sample_size across windows proportionally (largest remainder)."""
    total = sum(counts)
    if total <= sample_size:
        return list(counts)

    exact = [c * sample_size / total for c in counts]
    quotas = [int(x) for x in exact]
    # Largest remainder, so the quotas sum to exactly sample_size.
    remainder = sample_size - sum(quotas)
    order = sorted(range(len(counts)), key=lambda i: exact[i] - quotas[i], reverse=True)
    for i in order[:remainder]:
        quotas[i] += 1
    return quotas


def get_vectors(
    index_qid: str,
    auth_token: str,
    sources: Optional[List[str]] = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 0,
) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    """Return (vectors, metadata) for a proportional sample of the index.

    Positionally aligned: metadata[i] describes vectors[i]. The sample is the
    whole index when it holds fewer than sample_size rows. `seed` fixes the probe
    directions so a given index projects to the same picture on every run.
    """
    rng = random.Random(seed)

    index = get_index(index_qid, auth_token)
    vector_size = index.get("vector_size")
    if not vector_size:
        raise VectorStoreError(f"index {index_qid} reported no vector_size: {index}")
    print(f"Index `{index_qid}`: vector_size={vector_size}")

    track_counts = get_track_counts(index_qid, auth_token)
    if track_counts:
        print(f"Tracks: {track_counts} (total {sum(track_counts.values())})")

    print("Counting rows per window (no vectors)...")
    # windows = count_windows(index_qid, auth_token, _random_probe(vector_size, rng), sources)
    # Also returns every row's start_time, which is what segment ends are derived
    # from -- collected here because this pass sees the whole index, not the sample.
    windows, starts_by_qid = count_windows(
        index_qid, auth_token, _random_probe(vector_size, rng), sources
    )
    population = sum(count for _, _, count in windows)
    if population == 0:
        return [], []
    print(f"Enumerated {population} rows across {len(windows)} windows")

    quotas = _quotas([count for _, _, count in windows], sample_size)

    vectors: List[List[float]] = []
    metadata: List[Dict[str, Any]] = []
    for (lo, hi, _count), quota in zip(windows, quotas):
        if quota <= 0:
            continue
        rows = _search(
            index_qid,
            auth_token,
            # A fresh direction per window, so no single direction biases the sample.
            vector=_random_probe(vector_size, rng),
            limit=quota,
            include_vector=True,
            sources=sources,
            start_time_gte=lo,
            start_time_lte=hi,
        )
        for row in rows:
            entry = dict(row.get("vector") or {})
            vector = entry.pop("vector", None)
            if vector is None:
                continue
            vectors.append(vector)
            metadata.append(entry)

    print(f"Retrieved {len(vectors)} of {population} vectors from index `{index_qid}`")

    filled = derive_segment_ends(metadata, starts_by_qid)
    if filled:
        print(f"Derived an end time for {filled} segment rows that carried none")
    return vectors, metadata


def modality(meta: Dict[str, Any]) -> str:
    """Classify one row as text / image / video from its populated fields.

    Order matters. Frame rows carry start_time == end_time (a frame is an instant,
    not an interval), so the video test has to be a strict inequality and has to
    run after the frame_idx test -- otherwise every frame reads as a video.
    """
    if (meta.get("text") or "").strip():
        return "text"
    if meta.get("frame_idx") is not None:
        return "image"
    start, end = meta.get("start_time"), meta.get("end_time")
    if start is not None and end is not None and end > start:
        return "video"
    return "unknown"


# Replaced by the windowed enumeration above; kept for reference.
#
# def get_vectors(index_qid: str, auth_token: str, sources: list[str]=[]):
#     vectorstore = "http://localhost:8108/indexes"
#
#     request_vector_size = ["curl", "-X", "GET",
#                            f"{vectorstore}/{index_qid}",
#                            "-H", f"Authorization: Bearer {auth_token}",
#                            "-H", "Accept: application/json"
#                            ]
#     print(f"Calling VectorStore API to retrieve vector size in index `{index_qid}`...")
#     result = subprocess.run(request_vector_size, capture_output=True, text=True, check=True)
#     data = json.loads(result.stdout)
#     vector_size = data.get("vector_size")
#     print(f"Vector size in index `{index_qid}`: {vector_size}")
#
#     request_vector_count = ["curl", "-X", "GET",
#                             f"{vectorstore}/{index_qid}/tracks",
#                             "-H", f"Authorization: Bearer {auth_token}",
#                             "-H", "Accept: application/json"
#                             ]
#     print(f"Calling VectorStore API to retrieve vector count in index `{index_qid}`...")
#     result = subprocess.run(request_vector_count, capture_output=True, text=True, check=True)
#     data = json.loads(result.stdout)
#     vector_count = 0
#     tracks = data.get("tracks", [])
#     for track in tracks:
#         vector_count += track.get("count", 0)
#     print(f"Vector count in index `{index_qid}`: {vector_count}")
#
#     ref_vector = [0.5] * vector_size
#     request_vectors = ["curl", "-X", "POST",
#                         f"{vectorstore}/{index_qid}/search",
#                         "-H", f"Authorization: Bearer {auth_token}",
#                         "-H", "Content-Type: application/json",
#                         "-H", "Accept: application/json",
#                         "-d", f"""{{"limit": {vector_count}, "sources": {sources}, "vector": {ref_vector}, "include_vector": true}}"""
#                         ]
#     print(f"Calling VectorStore API to retrieve vectors in index `{index_qid}`...")
#     result = subprocess.run(request_vectors, capture_output=True, text=True, check=True)
#     data = json.loads(result.stdout)
#     results = data.get("results", [])
#     vectors = []
#     metadata = []
#     for object in results:
#         vector = object.get("vector").get("vector")
#         vectors.append(vector)
#         metainfo = object.get("vector")
#         del metainfo["vector"]
#         metadata.append(metainfo)
#     assert len(vectors) == vector_count, f"Expected {vector_count} vectors, but got {len(vectors)}"
#     print(f"Retrieved {len(vectors)} vectors from index `{index_qid}`.")
#     return vectors, metadata
