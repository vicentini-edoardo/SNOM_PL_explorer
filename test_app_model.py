from __future__ import annotations

import json

import h5py
import numpy as np

from app_model import MapSettings, SnomAppModel


def _write_grid_scan(path, ny: int = 2, nx: int = 2, frames_count: int = 40, det: int = 4) -> None:
    metadata = {
        "grid": {"ny": ny, "nx": nx},
        "n_block": frames_count,
        "trigger_frequency_hz": 40.0,
        "f_expected_hz": 4.0,
        "f_search_halfwidth_hz": 0.2,
        "roi_pixel_start": 1,
        "roi_pixel_end": det - 2,
        "roi_method": "mean",
        "window": "hann",
    }
    frame_index = np.arange(frames_count, dtype=np.float64)
    with h5py.File(path, "w") as h5:
        h5.attrs["metadata"] = json.dumps(metadata)
        scan = h5.create_group("scan")
        scan.create_dataset("coords_xy_nm", data=np.zeros((ny * nx, 2), dtype=np.float32))
        scan.create_dataset("coords_xyz_nm", data=np.zeros((ny * nx, 3), dtype=np.float32))
        scan.create_dataset("point_index_grid", data=np.arange(ny * nx, dtype=np.int32).reshape(ny, nx))

        points = h5.create_group("points")
        for iy in range(ny):
            for ix in range(nx):
                point = points.create_group(f"point_{iy:03d}_{ix:03d}")
                for key, value in {
                    "ix": ix,
                    "iy": iy,
                    "x_nm": float(ix),
                    "y_nm": float(iy),
                    "actual_x_nm": float(ix),
                    "actual_y_nm": float(iy),
                    "actual_z_nm": 0.0,
                }.items():
                    point.attrs[key] = value

                amp = 1.0 + ix + iy
                wave = amp * np.sin(2.0 * np.pi * 8.0 * frame_index / 40.0)
                frames = np.column_stack(
                    [100.0 + wave + pixel for pixel in range(det)]
                ).astype(np.float32)
                point.create_dataset("frames", data=frames)
                point.create_dataset("snom_xyz_nm", data=np.array([[0.0, 0.0, amp]], dtype=np.float32))
                point.create_dataset("snom_m_amp", data=np.array([[0.0, amp, amp + 1.0]], dtype=np.float32))
                point.create_dataset("snom_m_phase", data=np.array([[0.0, amp / 10.0, amp / 5.0]], dtype=np.float32))


def test_model_discovers_h5_files_by_relative_folder(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.h5").write_bytes(b"x")
    (tmp_path / "a.h5").write_bytes(b"x")

    model = SnomAppModel(tmp_path)

    assert model.folder_options() == [".", "nested"]
    assert model.file_options(".") == ["a.h5"]
    assert model.file_options("nested") == ["b.h5"]


def test_model_loads_scan_and_initializes_defaults(tmp_path):
    scan_path = tmp_path / "mini.h5"
    _write_grid_scan(scan_path)
    model = SnomAppModel(tmp_path)

    summary = model.load_scan(".", "mini.h5", recompute=True)

    assert summary.path == scan_path
    assert model.selected_pixel == (1, 1)
    assert model.roi_range == (1, 2)
    assert model.detector_range == (0, 3)
    assert model.line_rows == (0, 1)
    assert model.target_frequency_hz == 4.0
    assert (tmp_path / "_cache" / "mini.proc.npz").exists()


def test_model_computes_dash_equivalent_maps_and_inspector_data(tmp_path):
    _write_grid_scan(tmp_path / "mini.h5")
    model = SnomAppModel(tmp_path)
    model.load_scan(".", "mini.h5", recompute=True)
    model.select_pixel(0, 1)

    settings = MapSettings(harmonic="2w", compare_harmonic="1w", roi_range=model.roi_range)
    maps = model.compute_maps(settings)
    specs = [("2w", False), ("2w", True), ("1w", False), ("1w", True)]
    inspector = model.compute_inspector(settings, specs)
    profile = model.compute_line_profile(settings, rows=(0, 1))

    assert set(maps) == {"primary", "primary_bgsub", "compare", "compare_bgsub", "m1a", "m1p", "mechanical"}
    assert maps["primary"].shape == (2, 2)
    assert inspector["roi_trace"].shape == (40,)
    assert len(inspector["spectra"]) == 4
    assert all(spectrum.shape == (4,) for spectrum in inspector["spectra"])
    assert inspector["fft"].shape[1] == 4
    assert profile["x"].tolist() == [0.0, 1.0]
    assert profile["primary"].shape == (2,)


def test_model_computes_period_max_min_maps_and_spectra(tmp_path):
    _write_grid_scan(tmp_path / "mini.h5")
    model = SnomAppModel(tmp_path)
    model.load_scan(".", "mini.h5", recompute=True)
    model.select_pixel(0, 1)

    settings = model.map_settings()
    maps = model.compute_period_maps(settings)
    spectra = model.compute_period_spectra(settings)

    assert set(maps) == {"max", "min", "diff"}
    for m in maps.values():
        assert m.shape == (2, 2)
    assert np.allclose(maps["diff"], maps["max"] - maps["min"], equal_nan=True)
    assert spectra["max"].shape == (4,)
    assert np.allclose(spectra["diff"], spectra["max"] - spectra["min"])


def test_model_computes_period_trace_with_max_min_highlight(tmp_path):
    _write_grid_scan(tmp_path / "mini.h5")
    model = SnomAppModel(tmp_path)
    model.load_scan(".", "mini.h5", recompute=True)
    model.select_pixel(0, 1)

    settings = model.map_settings()
    trace = model.compute_period_trace(settings)

    assert trace["x"].shape == (40,)
    assert trace["trace"].shape == (40,)
    assert trace["max_mask"].sum() > 0
    assert trace["min_mask"].sum() > 0
    assert not np.any(trace["max_mask"] & trace["min_mask"])


def test_model_period_selected_pixel_shifts_move_trace_and_spectra(tmp_path):
    _write_grid_scan(tmp_path / "mini.h5")
    model = SnomAppModel(tmp_path)
    model.load_scan(".", "mini.h5", recompute=True)
    model.select_pixel(0, 1)

    base = model.map_settings(period_window=0)
    shifted = model.map_settings(period_window=0, period_max_shift=1, period_min_shift=-1)

    base_trace = model.compute_period_trace(base)
    shifted_trace = model.compute_period_trace(shifted)
    base_spectra = model.compute_period_spectra(base)
    shifted_spectra = model.compute_period_spectra(shifted)

    assert np.array_equal(np.flatnonzero(shifted_trace["max_mask"]), np.flatnonzero(base_trace["max_mask"]) + 1)
    assert np.array_equal(np.flatnonzero(shifted_trace["min_mask"]), np.flatnonzero(base_trace["min_mask"]) - 1)
    assert not np.allclose(shifted_spectra["max"], base_spectra["max"])
    assert not np.allclose(shifted_spectra["min"], base_spectra["min"])


def test_model_period_max_min_skips_interrupted_pixel(tmp_path):
    scan_path = tmp_path / "mini.h5"
    _write_grid_scan(scan_path)
    with h5py.File(scan_path, "a") as h5:
        grp = h5["points/point_000_001"]
        short_frames = grp["frames"][:5]
        del grp["frames"]
        grp.create_dataset("frames", data=short_frames)

    model = SnomAppModel(tmp_path)
    model.load_scan(".", "mini.h5", recompute=True)

    settings = model.map_settings()
    maps = model.compute_period_maps(settings)

    assert np.isnan(maps["max"][0, 1])
    assert not np.isnan(maps["max"][0, 0])


def test_model_computes_decomposition_summary(tmp_path):
    _write_grid_scan(tmp_path / "mini.h5")
    model = SnomAppModel(tmp_path)
    model.load_scan(".", "mini.h5", recompute=True)

    result = model.compute_decomposition(
        harmonic="2w",
        method="PCA",
        n_components=2,
        categorizer="kmeans",
        n_clusters=2,
        preprocess=("standardize",),
        detector_range=(0, 3),
    )

    assert result.label_map.shape == (2, 2)
    assert result.scree_values.shape[0] >= 1
    assert result.category_means.shape == (2, 4)
    assert result.centroids.shape[0] == 2
