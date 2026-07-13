"""PCA / MNF dimensionality reduction and per-pixel categorisation.

All functions are pure (no Dash imports) and operate on the pipeline bundle
dict produced by :mod:`stroboscopicsnom.snom_pipeline`.
"""
from __future__ import annotations

import numpy as np


# ── Feature-matrix construction ───────────────────────────────────────────────

def build_feature_matrix(
    bundle: dict,
    harmonic: str,
    *,
    bgsub: bool = False,
    l2norm: bool = False,
    standardize: bool = False,
    det_spectra_cube: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Extract per-pixel detector spectra and apply optional preprocessing.

    Parameters
    ----------
    bundle:
        Pipeline bundle (must contain ``det_spectra``; ``fft_baseline_cube``
        is used when *bgsub* is ``True``).
    harmonic:
        Harmonic key, e.g. ``"1w"``.
    bgsub:
        Subtract the per-pixel spectral background (``fft_baseline_cube``)
        from the harmonic spectra before decomposition.
    l2norm:
        Divide each pixel spectrum by its L2 norm (preserves spectral shape,
        removes per-pixel intensity variation).
    standardize:
        Mean-centre across pixels then scale each detector column to unit
        variance (full StandardScaler behaviour).

    Returns
    -------
    X_valid : (n_valid, det) float64
        Preprocessed feature matrix for valid (finite) pixels.
    valid_idx : (n_valid,) int64
        Flat indices into the (ny*nx) ravel of valid pixels.
    X_full : (ny, nx, det) float64
        Preprocessed full spatial cube (needed by :func:`run_mnf` for spatial
        noise estimation).
    spatial_shape : (ny, nx)
    """
    if det_spectra_cube is not None:
        det_spectra = det_spectra_cube.astype(np.float64)
    else:
        det_spectra = bundle["det_spectra"][harmonic].astype(np.float64)  # (ny, nx, det)
    ny, nx, det = det_spectra.shape

    if bgsub and "fft_baseline_cube" in bundle:
        baseline = bundle["fft_baseline_cube"].astype(np.float64)  # (ny, nx, det)
        X_full = det_spectra - baseline
    else:
        X_full = det_spectra.copy()

    # Per-pixel L2 normalisation
    if l2norm:
        norms = np.linalg.norm(X_full, axis=2, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        X_full = X_full / norms

    # Global mean-centre + scale across pixels
    if standardize:
        flat_tmp = X_full.reshape(ny * nx, det)
        finite_mask = np.all(np.isfinite(flat_tmp), axis=1)
        if finite_mask.any():
            mean_spec = flat_tmp[finite_mask].mean(axis=0)
            std_spec = flat_tmp[finite_mask].std(axis=0)
            std_spec = np.where(std_spec == 0.0, 1.0, std_spec)
            X_full = (X_full - mean_spec[None, None, :]) / std_spec[None, None, :]

    # Valid-pixel mask (drop NaN/Inf rows)
    flat = X_full.reshape(ny * nx, det)
    valid_mask = np.all(np.isfinite(flat), axis=1)
    valid_idx = np.flatnonzero(valid_mask)
    X_valid = flat[valid_idx]

    return X_valid, valid_idx, X_full, (ny, nx)


# ── Dimensionality reduction ──────────────────────────────────────────────────

def run_pca(
    X_valid: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard PCA on the (n_valid, det) feature matrix.

    Returns
    -------
    scores : (n_valid, k)
    loadings : (k, det) – principal component vectors.
    explained_var_ratio : (k,) – fraction of total variance per component.
    """
    from sklearn.decomposition import PCA  # lazy import – avoid hard dep at import time

    k = max(1, min(n_components, X_valid.shape[0] - 1, X_valid.shape[1]))
    pca = PCA(n_components=k)
    scores = pca.fit_transform(X_valid)
    return scores, pca.components_, pca.explained_variance_ratio_


def run_mnf(
    X_full: np.ndarray,
    valid_idx: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum Noise Fraction (MNF) transform (Green et al. 1988).

    Noise covariance is estimated from spatial nearest-neighbour differences.
    Components are ordered from highest to lowest SNR.

    Parameters
    ----------
    X_full : (ny, nx, det) float64
        Preprocessed spatial cube.  NaN pixels are excluded from noise
        estimation but may be present.
    valid_idx : (n_valid,) int64 – flat indices of valid pixels.
    n_components : number of MNF components to retain.

    Returns
    -------
    scores : (n_valid, k)
    loadings : (k, det) – component vectors in original detector space.
    snr : (k,) – estimated SNR per component (≥ 0), descending.
    """
    from sklearn.decomposition import PCA  # lazy import

    ny, nx, det = X_full.shape

    # ── Noise covariance from spatial gradients (NaN-safe) ────────────────────
    horiz = X_full[:, 1:, :] - X_full[:, :-1, :]    # (ny, nx-1, det)
    vert = X_full[1:, :, :] - X_full[:-1, :, :]     # (ny-1, nx, det)
    noise_samples = np.concatenate(
        [horiz.reshape(-1, det), vert.reshape(-1, det)], axis=0
    )
    finite_noise = np.all(np.isfinite(noise_samples), axis=1)
    noise_samples = noise_samples[finite_noise]

    if len(noise_samples) >= 2:
        Sigma_n = (noise_samples.T @ noise_samples) / (2.0 * len(noise_samples))
        reg = 1e-9 * (np.trace(Sigma_n) / det if det > 0 else 1.0)
        Sigma_n += np.eye(det) * reg
    else:
        # Fallback: identity noise covariance → equivalent to standard PCA
        Sigma_n = np.eye(det)

    # ── Noise-whitening transform W  (det × det) ──────────────────────────────
    eigvals, eigvecs = np.linalg.eigh(Sigma_n)
    eigvals = np.maximum(eigvals, 1e-12)
    W = eigvecs / np.sqrt(eigvals)[None, :]  # (det, det)

    # ── Whiten valid pixels and apply PCA ─────────────────────────────────────
    flat = X_full.reshape(ny * nx, det)
    X_valid = flat[valid_idx]
    X_white = X_valid @ W  # (n_valid, det)

    k = max(1, min(n_components, X_white.shape[0] - 1, X_white.shape[1]))
    pca = PCA(n_components=k)
    scores = pca.fit_transform(X_white)

    # Loadings back in original detector space
    loadings = pca.components_ @ W.T  # (k, det)

    # SNR: in noise-whitened space noise variance per dim = 1.
    # Total variance per component = pca.explained_variance_.
    # Signal variance ≈ total_var - 1; clamp to 0.
    snr = np.maximum(pca.explained_variance_ - 1.0, 0.0)

    return scores, loadings, snr


def run_gnmf(
    X_valid: np.ndarray,
    X_full: np.ndarray,
    valid_idx: np.ndarray,
    n_components: int,
    *,
    graph: str = "spatial",
    n_neighbors: int = 5,
    reg_lambda: float = 100.0,
    max_iter: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Graph-regularized NMF (Cai et al. 2011).

    Nonnegative matrix factorisation ``X ≈ W H`` with a graph-Laplacian
    smoothness penalty on ``W`` that favours similar rows (pixels) getting
    similar scores. ``X_valid`` is shifted to be nonnegative (NMF's
    requirement), so this also tolerates bgsub/standardize preprocessing.

    Parameters
    ----------
    X_valid : (n_valid, det)
    X_full : (ny, nx, det) – spatial cube, used to build the spatial graph.
    valid_idx : (n_valid,) – flat indices of valid pixels into ny*nx.
    n_components : number of factors to retain.
    graph : ``"spatial"`` (4-neighbor grid adjacency) or ``"spectral"``
        (k-NN in feature space).
    n_neighbors : neighbors per node for the spectral graph.
    reg_lambda : graph regularisation strength.
    max_iter : multiplicative-update iterations.

    Returns
    -------
    scores : (n_valid, k) – W, nonnegative.
    loadings : (k, det) – H, nonnegative.
    energy : (k,) – ``||W[:,j]|| * ||H[j,:]||`` per component, descending.
    """
    import scipy.sparse as sp

    n_valid, det = X_valid.shape
    k = max(1, min(n_components, n_valid - 1, det))

    Xnn = X_valid - X_valid.min()
    np.clip(Xnn, 0.0, None, out=Xnn)

    # ── Graph adjacency (n_valid × n_valid, sparse) ───────────────────────────
    if graph == "spectral":
        from sklearn.neighbors import kneighbors_graph

        kk = max(1, min(n_neighbors, n_valid - 1))
        A = kneighbors_graph(X_valid, kk, mode="connectivity", include_self=False)
        A = A.maximum(A.T)
    else:  # spatial: 4-neighbor grid adjacency among valid pixels
        ny, nx, _ = X_full.shape
        row_of = np.full(ny * nx, -1, dtype=np.int64)
        row_of[valid_idx] = np.arange(n_valid)
        flat_rc = valid_idx
        rows, cols, vals = [], [], []
        r = flat_rc // nx
        c = flat_rc % nx
        n_flat = ny * nx
        right = np.where(
            (c + 1 < nx), row_of[np.minimum(flat_rc + 1, n_flat - 1)], -1
        )
        down = np.where(
            (r + 1 < ny), row_of[np.minimum(flat_rc + nx, n_flat - 1)], -1
        )
        for src_row, nbr_flat in ((np.arange(n_valid), right), (np.arange(n_valid), down)):
            mask = nbr_flat >= 0
            rows.extend(src_row[mask])
            cols.extend(nbr_flat[mask])
            vals.extend([1.0] * int(mask.sum()))
        A = sp.coo_matrix((vals, (rows, cols)), shape=(n_valid, n_valid))
        A = A.maximum(A.T)
        A = A.tocsr()

    D = sp.diags(np.asarray(A.sum(axis=1)).ravel())

    # ── Multiplicative GNMF updates ───────────────────────────────────────────
    rng = np.random.default_rng(seed)
    W = rng.random((n_valid, k)) + 1e-3
    H = rng.random((k, det)) + 1e-3
    eps = 1e-10

    for _ in range(max_iter):
        H *= (W.T @ Xnn) / (W.T @ W @ H + eps)
        AW = A @ W
        DW = D @ W
        W *= (Xnn @ H.T + reg_lambda * AW) / (W @ H @ H.T + reg_lambda * DW + eps)

    energy = np.linalg.norm(W, axis=0) * np.linalg.norm(H, axis=1)
    order = np.argsort(energy)[::-1]
    return W[:, order], H[order], energy[order]


# ── Clustering ────────────────────────────────────────────────────────────────

def categorize(
    scores: np.ndarray,
    method: str,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster pixels in reduced-dimension score space.

    Parameters
    ----------
    scores : (n_valid, k)
    method : ``"kmeans"`` or ``"gmm"``.
    n_clusters : number of categories (clamped to n_valid).

    Returns
    -------
    labels : (n_valid,) int32
    centroids : (n_clusters, k) – cluster centres in score space.
    """
    n_clusters = max(2, min(n_clusters, len(scores)))
    if method == "gmm":
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=3)
        gmm.fit(scores)
        labels = gmm.predict(scores).astype(np.int32)
        centroids = gmm.means_
    else:  # kmeans (default)
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(scores).astype(np.int32)
        centroids = km.cluster_centers_
    return labels, centroids


# ── Per-category spectral statistics ─────────────────────────────────────────

def category_mean_spectra(
    X_raw: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std detector spectrum for each category.

    Parameters
    ----------
    X_raw : (n_valid, det) – raw (un-preprocessed) per-pixel spectra.
    labels : (n_valid,) int32
    n_clusters : total number of categories.

    Returns
    -------
    means : (n_clusters, det) float64
    stds  : (n_clusters, det) float64
    """
    det = X_raw.shape[1]
    means = np.full((n_clusters, det), np.nan, dtype=np.float64)
    stds = np.full((n_clusters, det), np.nan, dtype=np.float64)
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            means[k] = X_raw[mask].mean(axis=0)
            stds[k] = X_raw[mask].std(axis=0)
    return means, stds


# ── Spatial scatter-back ──────────────────────────────────────────────────────

def scatter_to_map(
    values: np.ndarray,
    valid_idx: np.ndarray,
    ny: int,
    nx: int,
    fill: float = np.nan,
) -> np.ndarray:
    """Scatter per-valid-pixel values back to a (ny, nx) spatial map.

    Parameters
    ----------
    values : (n_valid,)
    valid_idx : (n_valid,) – flat indices into (ny*nx).
    ny, nx : spatial dimensions.
    fill : fill value for invalid/missing pixels.

    Returns
    -------
    (ny, nx) float64 map.
    """
    out = np.full(ny * nx, fill, dtype=np.float64)
    out[valid_idx] = values
    return out.reshape(ny, nx)
