from __future__ import annotations

import json

import h5py
import numpy as np

from snom_pipeline import PROCESSING_VERSION, cache_stamp, process_scan


def _write_one_point_scan(path, frames: np.ndarray) -> None:
    metadata = {
        "grid": {"ny": 1, "nx": 1},
        "n_block": int(frames.shape[0]),
        "trigger_frequency_hz": 40.0,
        "f_expected_hz": 4.0,
        "f_search_halfwidth_hz": 0.1,
        "roi_pixel_start": 0,
        "roi_pixel_end": int(frames.shape[1] - 1),
        "roi_method": "mean",
        "window": "hann",
    }
    with h5py.File(path, "w") as h5:
        h5.attrs["metadata"] = json.dumps(metadata)
        scan = h5.create_group("scan")
        scan.create_dataset("coords_xy_nm", data=np.zeros((1, 2), dtype=np.float32))
        scan.create_dataset("coords_xyz_nm", data=np.zeros((1, 3), dtype=np.float32))
        scan.create_dataset("point_index_grid", data=np.zeros((1, 1), dtype=np.int32))

        points = h5.create_group("points")
        point = points.create_group("point_000000")
        for key, value in {
            "ix": 0,
            "iy": 0,
            "x_nm": 0.0,
            "y_nm": 0.0,
            "actual_x_nm": 0.0,
            "actual_y_nm": 0.0,
            "actual_z_nm": 0.0,
        }.items():
            point.attrs[key] = value
        point.create_dataset("frames", data=frames.astype(np.float32))
        point.create_dataset("snom_xyz_nm", data=np.zeros((1, 3), dtype=np.float32))
        point.create_dataset("snom_m_amp", data=np.zeros((1, 3), dtype=np.float32))
        point.create_dataset("snom_m_phase", data=np.zeros((1, 3), dtype=np.float32))


def test_process_scan_original_demod_uses_roi_time_trace(tmp_path):
    frame_count = 120
    frame_index = np.arange(frame_count, dtype=np.float64)
    wave = 2.0 * np.sin(2.0 * np.pi * 8.0 * frame_index / 40.0)
    frames = np.column_stack([100.0 + wave, 100.0 - wave])
    path = tmp_path / "opposite_phase.h5"
    _write_one_point_scan(path, frames)

    bundle = process_scan(path)

    assert abs(float(bundle["demod_maps"]["2w"][0, 0])) < 1e-5
    assert float(bundle["demod_maps_bgsub"]["2w"][0, 0]) > 1.0


def test_cache_stamp_includes_processing_version(tmp_path):
    path = tmp_path / "scan.h5"
    path.write_bytes(b"scan")

    assert cache_stamp(path).startswith(f"{PROCESSING_VERSION}_")
