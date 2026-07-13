"""Tests for stroboscopicsnom.decomposition."""
from __future__ import annotations

import numpy as np
import pytest

from decomposition import (
    build_feature_matrix,
    categorize,
    category_mean_spectra,
    run_gnmf,
    run_mnf,
    run_pca,
    scatter_to_map,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_bundle(ny: int = 4, nx: int = 4, det: int = 8, n_freq: int = 5) -> dict:
    """Synthetic two-cluster bundle.

    Pixels where (iy + ix) is even have amplitude ~1; odd pixels have amplitude ~0.
    This planted structure should be recoverable by PCA / MNF.
    """
    rng = np.random.default_rng(42)
    det_spectra = np.zeros((ny, nx, det), dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            base = 1.0 if (iy + ix) % 2 == 0 else 0.0
            det_spectra[iy, ix] = base + rng.random(det).astype(np.float32) * 0.05
    fft_cube = np.ones((ny, nx, n_freq, det), dtype=np.float32) * 0.1
    fft_baseline_cube = np.ones((ny, nx, det), dtype=np.float32) * 0.05
    return {
        "det_spectra": {"1w": det_spectra},
        "fft_cube": fft_cube,
        "fft_baseline_cube": fft_baseline_cube,
        "det_axis": np.arange(det, dtype=np.int32),
        "grid": {"ny": ny, "nx": nx},
    }


# ── build_feature_matrix ──────────────────────────────────────────────────────

def test_build_feature_matrix_output_shapes():
    bundle = _make_bundle(ny=3, nx=3, det=8)
    X, valid_idx, X_full, shape = build_feature_matrix(
        bundle, "1w", bgsub=False, l2norm=False, standardize=False
    )
    assert X.shape == (9, 8)
    assert len(valid_idx) == 9
    assert X_full.shape == (3, 3, 8)
    assert shape == (3, 3)


def test_build_feature_matrix_bgsub_reduces_amplitude():
    bundle = _make_bundle(ny=2, nx=2, det=4)
    X_raw, _, _, _ = build_feature_matrix(bundle, "1w", bgsub=False, l2norm=False, standardize=False)
    X_sub, _, _, _ = build_feature_matrix(bundle, "1w", bgsub=True,  l2norm=False, standardize=False)
    assert X_sub.mean() < X_raw.mean()


def test_build_feature_matrix_l2norm_produces_unit_norms():
    bundle = _make_bundle(ny=3, nx=3, det=6)
    X, _, _, _ = build_feature_matrix(bundle, "1w", bgsub=False, l2norm=True, standardize=False)
    norms = np.linalg.norm(X, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-6)


def test_build_feature_matrix_standardize_zero_column_mean():
    bundle = _make_bundle(ny=3, nx=4, det=8)
    X, _, _, _ = build_feature_matrix(bundle, "1w", bgsub=False, l2norm=False, standardize=True)
    np.testing.assert_allclose(X.mean(axis=0), np.zeros(8), atol=1e-6)


def test_build_feature_matrix_nan_pixels_excluded():
    bundle = _make_bundle(ny=2, nx=2, det=4)
    # Poison one pixel
    bundle["det_spectra"]["1w"][0, 0, :] = np.nan
    X, valid_idx, _, _ = build_feature_matrix(bundle, "1w", bgsub=False, l2norm=False, standardize=False)
    assert X.shape == (3, 4)
    assert 0 not in valid_idx  # pixel (0,0) flat-index 0 excluded


# ── run_pca ───────────────────────────────────────────────────────────────────

def test_run_pca_output_shapes():
    rng = np.random.default_rng(0)
    X = rng.random((20, 10)).astype(np.float64)
    scores, loadings, explained = run_pca(X, n_components=3)
    assert scores.shape == (20, 3)
    assert loadings.shape == (3, 10)
    assert explained.shape == (3,)


def test_run_pca_explained_variance_between_0_and_1():
    rng = np.random.default_rng(1)
    X = rng.random((30, 12)).astype(np.float64)
    _, _, explained = run_pca(X, n_components=4)
    assert np.all(explained >= 0)
    assert np.sum(explained) <= 1.0 + 1e-6


def test_run_pca_first_component_captures_most_variance_for_planted_signal():
    rng = np.random.default_rng(7)
    group_a = rng.random((15, 8)) + 5.0
    group_b = rng.random((15, 8))
    X = np.vstack([group_a, group_b])
    _, _, explained = run_pca(X, n_components=3)
    assert explained[0] > 0.5


def test_run_pca_clamps_n_components_to_valid_range():
    X = np.eye(5, dtype=np.float64)  # 5 samples, 5 features
    scores, loadings, explained = run_pca(X, n_components=100)
    assert scores.shape[1] <= 5


# ── run_mnf ───────────────────────────────────────────────────────────────────

def test_run_mnf_output_shapes():
    rng = np.random.default_rng(3)
    ny, nx, det = 4, 4, 10
    X_full = rng.random((ny, nx, det)).astype(np.float64)
    valid_idx = np.arange(ny * nx)
    scores, loadings, snr = run_mnf(X_full, valid_idx, n_components=3)
    assert scores.shape == (ny * nx, 3)
    assert loadings.shape == (3, det)
    assert snr.shape == (3,)


def test_run_mnf_snr_nonnegative():
    rng = np.random.default_rng(4)
    X_full = rng.random((5, 5, 8)).astype(np.float64)
    valid_idx = np.arange(25)
    _, _, snr = run_mnf(X_full, valid_idx, n_components=4)
    assert np.all(snr >= 0)


def test_run_mnf_snr_descending():
    rng = np.random.default_rng(5)
    X_full = rng.random((6, 6, 12)).astype(np.float64)
    valid_idx = np.arange(36)
    _, _, snr = run_mnf(X_full, valid_idx, n_components=5)
    assert np.all(np.diff(snr) <= 1e-9)  # non-increasing


def test_run_mnf_handles_partial_nan_cube():
    rng = np.random.default_rng(6)
    X_full = rng.random((4, 4, 6)).astype(np.float64)
    X_full[0, 0, :] = np.nan  # one invalid pixel
    valid_idx = np.flatnonzero(
        np.all(np.isfinite(X_full.reshape(16, 6)), axis=1)
    )
    scores, _, _ = run_mnf(X_full, valid_idx, n_components=2)
    assert scores.shape[0] == len(valid_idx)


# ── run_gnmf ──────────────────────────────────────────────────────────────────

def test_run_gnmf_output_shapes():
    rng = np.random.default_rng(7)
    ny, nx, det = 4, 4, 6
    X_full = rng.random((ny, nx, det)).astype(np.float64)
    valid_idx = np.arange(ny * nx)
    X_valid = X_full.reshape(ny * nx, det)
    scores, loadings, energy = run_gnmf(
        X_valid, X_full, valid_idx, n_components=3, max_iter=20
    )
    assert scores.shape == (ny * nx, 3)
    assert loadings.shape == (3, det)
    assert energy.shape == (3,)


def test_run_gnmf_nonnegative_factors():
    rng = np.random.default_rng(8)
    ny, nx, det = 4, 4, 6
    X_full = rng.standard_normal((ny, nx, det))  # has negatives on purpose
    valid_idx = np.arange(ny * nx)
    X_valid = X_full.reshape(ny * nx, det)
    scores, loadings, _ = run_gnmf(
        X_valid, X_full, valid_idx, n_components=2, max_iter=20
    )
    assert np.all(scores >= 0)
    assert np.all(loadings >= 0)


def test_run_gnmf_energy_descending():
    rng = np.random.default_rng(9)
    X_full = rng.random((5, 5, 8)).astype(np.float64)
    valid_idx = np.arange(25)
    X_valid = X_full.reshape(25, 8)
    _, _, energy = run_gnmf(X_valid, X_full, valid_idx, n_components=4, max_iter=20)
    assert np.all(np.diff(energy) <= 1e-9)  # non-increasing


@pytest.mark.parametrize("graph", ["spatial", "spectral"])
def test_run_gnmf_both_graph_modes_run(graph):
    rng = np.random.default_rng(11)
    ny, nx, det = 4, 4, 6
    X_full = rng.random((ny, nx, det)).astype(np.float64)
    valid_idx = np.arange(ny * nx)
    X_valid = X_full.reshape(ny * nx, det)
    scores, loadings, energy = run_gnmf(
        X_valid, X_full, valid_idx, n_components=2, graph=graph,
        n_neighbors=3, max_iter=20,
    )
    assert np.all(np.isfinite(scores))
    assert np.all(np.isfinite(loadings))
    assert np.all(np.isfinite(energy))


# ── categorize ────────────────────────────────────────────────────────────────

def test_categorize_kmeans_produces_correct_number_of_labels():
    rng = np.random.default_rng(10)
    scores = rng.random((30, 4))
    labels, centroids = categorize(scores, "kmeans", n_clusters=3)
    assert labels.shape == (30,)
    assert set(labels).issubset({0, 1, 2})
    assert centroids.shape == (3, 4)


def test_categorize_gmm_produces_correct_number_of_labels():
    rng = np.random.default_rng(11)
    scores = rng.random((30, 4))
    labels, centroids = categorize(scores, "gmm", n_clusters=3)
    assert labels.shape == (30,)
    assert set(labels).issubset({0, 1, 2})
    assert centroids.shape == (3, 4)


def test_categorize_clamps_clusters_to_n_valid():
    rng = np.random.default_rng(12)
    scores = rng.random((3, 2))
    labels, centroids = categorize(scores, "kmeans", n_clusters=10)
    assert labels.shape == (3,)
    assert centroids.shape[0] <= 3


def test_categorize_two_cluster_signal_recoverable():
    """Tight planted clusters should be separated cleanly by KMeans."""
    rng = np.random.default_rng(20)
    group_a = rng.random((20, 3)) * 0.01       # near origin
    group_b = rng.random((20, 3)) * 0.01 + 10.0  # far from origin
    scores = np.vstack([group_a, group_b])
    labels, _ = categorize(scores, "kmeans", n_clusters=2)
    # Labels should separate both groups (allow either 0/1 assignment)
    assert len(set(labels[:20])) == 1
    assert len(set(labels[20:])) == 1
    assert labels[0] != labels[20]


# ── category_mean_spectra ─────────────────────────────────────────────────────

def test_category_mean_spectra_output_shapes():
    rng = np.random.default_rng(30)
    X = rng.random((20, 8))
    labels = np.array([0, 1, 2] * 6 + [0, 1], dtype=np.int32)
    means, stds = category_mean_spectra(X, labels, n_clusters=3)
    assert means.shape == (3, 8)
    assert stds.shape == (3, 8)


def test_category_mean_spectra_correct_values():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    labels = np.array([0, 0, 1], dtype=np.int32)
    means, stds = category_mean_spectra(X, labels, n_clusters=2)
    np.testing.assert_allclose(means[0], [2.0, 3.0])
    np.testing.assert_allclose(means[1], [5.0, 6.0])
    np.testing.assert_allclose(stds[1], [0.0, 0.0])


def test_category_mean_spectra_empty_category_is_nan():
    X = np.ones((5, 4), dtype=np.float64)
    labels = np.zeros(5, dtype=np.int32)  # all in category 0; category 1 empty
    means, stds = category_mean_spectra(X, labels, n_clusters=2)
    assert np.all(np.isnan(means[1]))


# ── scatter_to_map ────────────────────────────────────────────────────────────

def test_scatter_to_map_shape():
    result = scatter_to_map(np.array([1.0, 2.0]), np.array([0, 4]), ny=2, nx=3)
    assert result.shape == (2, 3)


def test_scatter_to_map_values_and_fill():
    values = np.array([10.0, 20.0, 30.0])
    valid_idx = np.array([0, 2, 5])
    result = scatter_to_map(values, valid_idx, ny=2, nx=3)
    flat = result.ravel()
    np.testing.assert_allclose(flat[[0, 2, 5]], [10.0, 20.0, 30.0])
    assert np.isnan(flat[1])
    assert np.isnan(flat[3])
    assert np.isnan(flat[4])


def test_scatter_to_map_custom_fill():
    result = scatter_to_map(np.array([7.0]), np.array([3]), ny=2, nx=3, fill=-1.0)
    assert result.ravel()[3] == 7.0
    assert result.ravel()[0] == -1.0


# ── Full pipeline smoke test ──────────────────────────────────────────────────

def test_full_pipeline_pca_kmeans_smoke():
    """End-to-end: bundle → feature matrix → PCA → KMeans → map."""
    bundle = _make_bundle(ny=5, nx=5, det=10, n_freq=6)
    X, valid_idx, X_full, (ny, nx) = build_feature_matrix(
        bundle, "1w", bgsub=True, l2norm=True, standardize=True
    )
    scores, _, explained = run_pca(X, n_components=3)
    labels, centroids = categorize(scores, "kmeans", n_clusters=2)
    means, stds = category_mean_spectra(
        bundle["det_spectra"]["1w"].astype(np.float64).reshape(ny * nx, -1)[valid_idx],
        labels,
        n_clusters=2,
    )
    label_map = scatter_to_map(labels.astype(np.float64), valid_idx, ny, nx)

    assert label_map.shape == (ny, nx)
    assert means.shape == (2, 10)
    assert scores.shape[0] == len(valid_idx)
    assert np.all(np.isfinite(means))


def test_full_pipeline_mnf_gmm_smoke():
    """End-to-end: bundle → feature matrix → MNF → GMM → map."""
    bundle = _make_bundle(ny=4, nx=4, det=8, n_freq=5)
    X, valid_idx, X_full, (ny, nx) = build_feature_matrix(
        bundle, "1w", bgsub=False, l2norm=False, standardize=True
    )
    scores, loadings, snr = run_mnf(X_full, valid_idx, n_components=3)
    labels, centroids = categorize(scores, "gmm", n_clusters=2)
    label_map = scatter_to_map(labels.astype(np.float64), valid_idx, ny, nx)

    assert label_map.shape == (ny, nx)
    assert snr.shape == (3,)
    assert np.all(snr >= 0)
    assert set(labels).issubset({0, 1})
