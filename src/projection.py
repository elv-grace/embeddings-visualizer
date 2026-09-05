"""Project high-dimensional index vectors down to plottable coordinates.

Pipeline
--------
PCA to an intermediate width, then UMAP to 2. PCA first because UMAP's neighbour
search degrades in very high dimensions and because the index's vectors are
zero-padded (SigLIP 2 emits 768 dims, stored in a 1024-wide index), so a quarter
of the columns carry no variance at all. PCA discards them for free.

The `pca` method skips the UMAP stage and plots the first two components
directly. It preserves real distances and global structure but separates
clusters poorly; UMAP separates clusters well but its 2D distances are not
meaningful. Both are offered because they fail in opposite directions.

Out-of-sample projection
------------------------
A search query has to land in the same picture as the vectors already plotted,
so both stages keep their fitted state and expose `transform`. This rules out
t-SNE, which has no out-of-sample extension: placing a query would mean refitting
and reshuffling the whole map underneath the user.

Large indexes
-------------
UMAP fitting is superlinear, so above `fit_sample` rows the fit runs on a random
subset and every remaining row is placed with the same `transform` the query path
uses. The picture stays stable as the sample size grows.

2D distance is not similarity
-----------------------------
Any projection to 2D distorts neighbourhoods; a node that looks near the query
may not be among its nearest vectors. `top_k_similar` therefore ranks in the
original space, and the frontend draws those links explicitly rather than letting
viewers read distance off the plot.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

PCA_DIMS = 50

# Rows above which UMAP is fitted on a subset and the rest are transformed.
FIT_SAMPLE = 10_000

METHODS = ("umap", "pca")


class ProjectionError(RuntimeError):
    """A projection could not be fitted or applied."""


class Projector:
    """Fits a 2D projection of an index and places new vectors into it.

    Not thread-safe: fit once, then treat as read-only.
    """

    def __init__(self, method: str = "umap", seed: int = 0, fit_sample: int = FIT_SAMPLE):
        if method not in METHODS:
            raise ProjectionError(f"unknown projection method {method!r}, expected one of {METHODS}")
        self.method = method
        self.seed = seed
        self.fit_sample = fit_sample
        self.pca: Optional[PCA] = None
        self.umap = None
        self.bbox: Optional[Dict[str, float]] = None
        self.fitted_on: int = 0

    def fit(self, vectors: np.ndarray) -> np.ndarray:
        """Fit the projection and return the (n, 2) coordinates of `vectors`."""
        vectors = _as_matrix(vectors)
        n_samples, n_features = vectors.shape

        pca_dims = min(PCA_DIMS, n_samples, n_features)
        if self.method == "pca":
            pca_dims = min(2, n_samples, n_features)
        self.pca = PCA(n_components=pca_dims, random_state=self.seed)
        reduced = self.pca.fit_transform(vectors)

        if self.method == "pca":
            coords = _pad_to_2d(reduced)
        else:
            coords = self._fit_umap(reduced)

        self.fitted_on = n_samples
        self.bbox = _bounds(coords)
        return coords

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Place new vectors into the fitted projection, as (n, 2)."""
        if self.pca is None:
            raise ProjectionError("projector has not been fitted")
        vectors = _as_matrix(vectors)
        reduced = self.pca.transform(vectors)
        if self.method == "pca":
            return _pad_to_2d(reduced)
        return np.asarray(self.umap.transform(reduced), dtype=np.float32)

    @property
    def explained_variance(self) -> float:
        """Fraction of variance the plotted axes retain, for `pca` only.

        Meaningless for `umap`, whose axes are not linear combinations of the
        input, so it reports 0.0 there.
        """
        if self.pca is None or self.method != "pca":
            return 0.0
        return float(self.pca.explained_variance_ratio_[:2].sum())

    def _fit_umap(self, reduced: np.ndarray) -> np.ndarray:
        import umap  # deferred: importing umap costs several seconds of numba warmup

        n_samples = reduced.shape[0]
        # UMAP needs at least a few neighbours; tiny indexes get whatever they have.
        n_neighbors = int(max(2, min(15, n_samples - 1)))
        self.umap = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=self.seed,
        )

        if n_samples <= self.fit_sample:
            return np.asarray(self.umap.fit_transform(reduced), dtype=np.float32)

        rng = np.random.default_rng(self.seed)
        fit_idx = rng.choice(n_samples, size=self.fit_sample, replace=False)
        self.umap.fit(reduced[fit_idx])
        coords = np.empty((n_samples, 2), dtype=np.float32)
        coords[fit_idx] = np.asarray(self.umap.embedding_, dtype=np.float32)
        rest = np.setdiff1d(np.arange(n_samples), fit_idx, assume_unique=False)
        # Chunked so a million-row index does not build one enormous intermediate.
        for chunk in np.array_split(rest, max(1, len(rest) // 10_000)):
            coords[chunk] = np.asarray(self.umap.transform(reduced[chunk]), dtype=np.float32)
        return coords


def top_k_similar(vectors: np.ndarray, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
    """Rank `vectors` against `query` by cosine similarity in the original space.

    Returns (row index, similarity) pairs, most similar first. This is the honest
    ranking; the 2D coordinates are a layout and must not be measured against.
    """
    matrix = _as_matrix(vectors)
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    if matrix.shape[1] != q.shape[0]:
        raise ProjectionError(f"query has {q.shape[0]} dims, index has {matrix.shape[1]}")

    # Normalizing both sides makes the dot product a cosine regardless of whether
    # the index stored unit vectors.
    matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    q = q / max(float(np.linalg.norm(q)), 1e-12)

    sims = matrix @ q
    k = int(min(k, sims.shape[0]))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(i), float(sims[i])) for i in top]


def anchor_to_neighbours(
    coords: np.ndarray, neighbours: List[Tuple[int, float]], sharpness: float = 40.0
) -> Tuple[float, float]:
    """Place a query at the softmax-weighted centroid of its true neighbours' coords.

    A query vector is almost never near the fitted manifold, and UMAP's
    ``transform`` has nothing local to place it by: it drops every such point at
    the same averaged location. Measured on a SigLIP frame index, random unit
    vectors, a text query and an out-of-domain image query all land within ~0.2
    units of each other on a 30-unit-wide map, so the raw projected position
    carries no information about the query.

    The neighbours do. Weighting their plotted positions by similarity puts the
    node where its matches are, which is the question a viewer is actually
    asking. `sharpness` controls how strongly the best match dominates; high
    enough and the node sits beside the top hit rather than drifting into the
    empty space between scattered matches.
    """
    if not neighbours:
        raise ProjectionError("cannot anchor a query with no neighbours")

    idx = np.array([i for i, _ in neighbours], dtype=int)
    sims = np.array([s for _, s in neighbours], dtype=np.float64)

    # Softmax over similarities, shifted for numerical stability. Relative
    # spacing is what matters, which keeps this working for both the ~0.07
    # cross-modal range and the ~0.95 same-modality one.
    weights = np.exp(sharpness * (sims - sims.max()))
    weights /= weights.sum()

    point = (coords[idx] * weights[:, None]).sum(axis=0)
    return float(point[0]), float(point[1])


def _as_matrix(vectors) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ProjectionError(f"expected a non-empty 2D array of vectors, got shape {matrix.shape}")
    return matrix


def _pad_to_2d(reduced: np.ndarray) -> np.ndarray:
    """Widen a 1-component PCA result to 2 columns, for degenerate inputs."""
    if reduced.shape[1] >= 2:
        return np.asarray(reduced[:, :2], dtype=np.float32)
    return np.column_stack([reduced[:, 0], np.zeros(reduced.shape[0])]).astype(np.float32)


def _bounds(coords: np.ndarray) -> Dict[str, float]:
    """The fitted set's extent, so the frontend can frame its initial viewport."""
    return {
        "x_min": float(coords[:, 0].min()),
        "x_max": float(coords[:, 0].max()),
        "y_min": float(coords[:, 1].min()),
        "y_max": float(coords[:, 1].max()),
    }
