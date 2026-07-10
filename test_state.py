from __future__ import annotations

import numpy as np

from state import (
    clamp_pixel,
    discover_h5_files,
    extract_selected_pixel,
    get_detector_baseline_live,
    get_detector_spectrum_live,
    get_detector_spectrum_bgsub_live,
    finite_range,
    get_demod_map_bgsub_live,
    get_demod_map_live,
    get_nb_slice,
    nanmean_or_nan,
)


def test_discover_h5_files_groups_by_relative_folder(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "scan1.h5").write_bytes(b"x")
    (tmp_path / "scan2.h5").write_bytes(b"x")

    folders = discover_h5_files(tmp_path)

    assert list(folders) == [".", "a"]
    assert [p.name for p in folders["."]] == ["scan2.h5"]
    assert [p.name for p in folders["a"]] == ["scan1.h5"]


def test_extract_selected_pixel_uses_customdata_from_dash_click():
    click = {"points": [{"customdata": [4, 7], "x": 7, "y": 4}]}

    assert extract_selected_pixel(click, nx=10, ny=12) == (4, 7)


def test_extract_selected_pixel_falls_back_to_xy_and_clamps():
    click = {"points": [{"x": 99, "y": -2}]}

    assert extract_selected_pixel(click, nx=5, ny=8) == (0, 7)


def test_clamp_pixel_bounds_indices():
    assert clamp_pixel(ix=-5, iy=99, nx=4, ny=6) == (0, 5)


def test_get_nb_slice_handles_edges_and_single_pixel():
    assert get_nb_slice(ix=0, iy=0, nx=4, ny=5, avg3x3=True) == (
        slice(0, 2),
        slice(0, 2),
    )
    assert get_nb_slice(ix=2, iy=3, nx=4, ny=5, avg3x3=False) == (
        slice(3, 4),
        slice(2, 3),
    )


def test_get_demod_map_live_averages_detector_roi():
    fft_cube = np.zeros((1, 1, 4, 2), dtype=np.float32)
    fft_cube[0, 0, 1, :] = [10.0, 0.0]
    fft_cube[0, 0, 2, :] = [0.0, 6.0]
    bundle = {
        "fft_cube": fft_cube,
        "f_axis": np.array([0.0, 4.0, 8.0, 12.0], dtype=np.float32),
        "metadata": {
            "f_expected_hz": 4.0,
            "f_search_halfwidth_hz": 0.1,
            "trigger_frequency_hz": 40.0,
            "roi_method": "mean",
        },
    }

    result = get_demod_map_live(bundle, "2w", roi_ps=0, roi_pe=1)

    np.testing.assert_allclose(result, np.array([[3.0]], dtype=np.float32))


def test_get_demod_map_live_scales_1w_bin_instead_of_searching_2w_peak():
    fft_cube = np.zeros((1, 1, 5, 1), dtype=np.float32)
    fft_cube[0, 0, 2, 0] = 5.0
    fft_cube[0, 0, 3, 0] = 99.0
    bundle = {
        "fft_cube": fft_cube,
        "f_axis": np.array([0.0, 3.9, 7.8, 8.1, 11.7], dtype=np.float32),
        "metadata": {
            "f_expected_hz": 4.0,
            "f_search_halfwidth_hz": 0.5,
            "trigger_frequency_hz": 40.0,
            "roi_method": "mean",
        },
    }

    result = get_demod_map_live(bundle, "2w", roi_ps=0, roi_pe=0, target_frequency_hz=3.9)

    np.testing.assert_allclose(result, np.array([[5.0]], dtype=np.float32))


def test_get_demod_map_live_integrates_neighbor_bins_by_sum():
    fft_cube = np.zeros((1, 1, 5, 1), dtype=np.float32)
    fft_cube[0, 0, 1:4, 0] = [2.0, 5.0, 7.0]
    bundle = {
        "fft_cube": fft_cube,
        "f_axis": np.array([0.0, 4.0, 8.0, 12.0, 16.0], dtype=np.float32),
        "metadata": {
            "f_expected_hz": 4.0,
            "f_search_halfwidth_hz": 0.1,
            "trigger_frequency_hz": 40.0,
            "roi_method": "mean",
        },
    }

    result = get_demod_map_live(bundle, "2w", roi_ps=0, roi_pe=0, target_frequency_hz=4.0, neighbor_bins=1)

    np.testing.assert_allclose(result, np.array([[14.0]], dtype=np.float32))


def test_background_subtracted_map_uses_same_scaled_bins_as_original():
    fft_cube = np.zeros((1, 1, 5, 1), dtype=np.float32)
    fft_cube[0, 0, 2, 0] = 10.0
    fft_cube[0, 0, 3, 0] = 100.0
    fft_cube[0, 0, 4, 0] = 2.0
    bundle = {
        "fft_cube": fft_cube,
        "fft_baseline_cube": np.zeros((1, 1, 1), dtype=np.float32),
        "f_axis": np.array([0.0, 3.9, 7.8, 8.1, 16.0], dtype=np.float32),
        "metadata": {
            "f_expected_hz": 4.0,
            "f_search_halfwidth_hz": 0.5,
            "trigger_frequency_hz": 40.0,
            "roi_method": "mean",
        },
    }

    original = get_demod_map_live(bundle, "2w", 0, 0, target_frequency_hz=3.9)
    bgsub = get_demod_map_bgsub_live(bundle, "2w", 0, 0, bg_low=16.0, bg_high=16.0, target_frequency_hz=3.9)

    np.testing.assert_allclose(original, np.array([[10.0]], dtype=np.float32))
    np.testing.assert_allclose(bgsub, np.array([[8.0]], dtype=np.float32))


def test_single_frame_0w_background_subtraction_matches_original():
    fft_cube = np.array([[[[10.0, 20.0], [np.nan, np.nan]]]], dtype=np.float32)
    bundle = {
        "fft_cube": fft_cube,
        "f_axis": np.array([0.0], dtype=np.float32),
        "metadata": {
            "f_expected_hz": 4.0,
            "f_search_halfwidth_hz": 0.1,
            "trigger_frequency_hz": 40.0,
            "roi_method": "mean",
        },
    }

    spectrum = get_detector_spectrum_bgsub_live(bundle, "0w", slice(0, 1), slice(0, 1))
    demod_map = get_demod_map_bgsub_live(bundle, "0w", 0, 1)

    np.testing.assert_allclose(spectrum, np.array([10.0, 20.0], dtype=np.float32))
    np.testing.assert_allclose(demod_map, np.array([[15.0]], dtype=np.float32))


def test_get_detector_spectrum_live_uses_scaled_bins_and_neighbor_sum():
    fft_cube = np.zeros((1, 1, 5, 2), dtype=np.float32)
    fft_cube[0, 0, 1:4, :] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    bundle = {
        "fft_cube": fft_cube,
        "f_axis": np.array([0.0, 4.0, 8.0, 12.0, 16.0], dtype=np.float32),
        "metadata": {
            "f_expected_hz": 4.0,
            "f_search_halfwidth_hz": 0.1,
            "trigger_frequency_hz": 40.0,
        },
    }

    result = get_detector_spectrum_live(bundle, "2w", slice(0, 1), slice(0, 1), target_frequency_hz=4.0, neighbor_bins=1)

    np.testing.assert_allclose(result, np.array([9.0, 12.0], dtype=np.float32))


def test_get_demod_map_live_uses_roi_time_trace_for_default_roi():
    sample_rate_hz = 40.0
    frame_count = 120
    frames = np.arange(frame_count, dtype=np.float64)
    roi_trace = 100.0 + 2.0 * np.sin(2.0 * np.pi * 8.0 * frames / sample_rate_hz)
    fft_cube = np.zeros((1, 1, frame_count // 2 + 1, 2), dtype=np.float32)
    fft_cube[0, 0, 24, :] = [20.0, 20.0]
    bundle = {
        "roi_traces": roi_trace.reshape(1, 1, frame_count).astype(np.float32),
        "fft_cube": fft_cube,
        "f_axis": np.fft.rfftfreq(frame_count, d=1.0 / sample_rate_hz).astype(np.float32),
        "metadata": {
            "f_expected_hz": 4.0,
            "f_search_halfwidth_hz": 0.1,
            "trigger_frequency_hz": sample_rate_hz,
            "roi_method": "mean",
            "roi_pixel_start": 0,
            "roi_pixel_end": 1,
            "n_block": frame_count,
            "window": "hann",
        },
    }

    result = get_demod_map_live(bundle, "2w", roi_ps=0, roi_pe=1)

    np.testing.assert_allclose(result, np.array([[2.0]], dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_get_detector_spectrum_bgsub_live_subtracts_baseline_before_peak():
    fft_cube = np.zeros((1, 1, 4, 3), dtype=np.float32)
    fft_cube[0, 0, 2, :] = [6.0, 8.0, 10.0]
    baseline = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(1, 1, 3)
    bundle = {
        "fft_cube": fft_cube,
        "fft_baseline_cube": baseline,
        "f_axis": np.array([0.0, 4.0, 8.0, 12.0], dtype=np.float32),
        "metadata": {
            "f_expected_hz": 4.0,
            "f_search_halfwidth_hz": 0.1,
            "trigger_frequency_hz": 40.0,
        },
    }

    result = get_detector_spectrum_bgsub_live(bundle, "2w", slice(0, 1), slice(0, 1))

    np.testing.assert_allclose(result, np.array([5.0, 6.0, 7.0], dtype=np.float32))


def test_background_subtraction_uses_selected_frequency_range():
    fft_cube = np.zeros((1, 1, 5, 2), dtype=np.float32)
    fft_cube[0, 0, 2, :] = [10.0, 20.0]
    fft_cube[0, 0, 3, :] = [2.0, 6.0]
    fft_cube[0, 0, 4, :] = [4.0, 8.0]
    bundle = {
        "fft_cube": fft_cube,
        "fft_baseline_cube": np.zeros((1, 1, 2), dtype=np.float32),
        "f_axis": np.array([0.0, 4.0, 8.0, 16.0, 18.0], dtype=np.float32),
        "metadata": {
            "f_expected_hz": 4.0,
            "f_search_halfwidth_hz": 0.1,
            "trigger_frequency_hz": 40.0,
            "roi_method": "mean",
        },
    }

    spectrum = get_detector_spectrum_bgsub_live(bundle, "2w", slice(0, 1), slice(0, 1), bg_low=16.0, bg_high=18.0)
    demod_map = get_demod_map_bgsub_live(bundle, "2w", 0, 1, bg_low=16.0, bg_high=18.0)

    np.testing.assert_allclose(spectrum, np.array([7.0, 13.0], dtype=np.float32))
    np.testing.assert_allclose(demod_map, np.array([[10.0]], dtype=np.float32))


def test_background_subtraction_smooths_baseline_detector_profile():
    fft_cube = np.zeros((1, 1, 5, 3), dtype=np.float32)
    fft_cube[0, 0, 2, :] = [10.0, 20.0, 30.0]
    fft_cube[0, 0, 3, :] = [0.0, 30.0, 0.0]
    bundle = {
        "fft_cube": fft_cube,
        "fft_baseline_cube": np.zeros((1, 1, 3), dtype=np.float32),
        "f_axis": np.array([0.0, 4.0, 8.0, 16.0, 18.0], dtype=np.float32),
        "metadata": {
            "f_expected_hz": 4.0,
            "f_search_halfwidth_hz": 0.1,
            "trigger_frequency_hz": 40.0,
            "roi_method": "mean",
        },
    }

    spectrum = get_detector_spectrum_bgsub_live(
        bundle,
        "2w",
        slice(0, 1),
        slice(0, 1),
        bg_low=16.0,
        bg_high=16.0,
        baseline_smooth_px=3,
    )
    demod_map = get_demod_map_bgsub_live(
        bundle,
        "2w",
        0,
        2,
        bg_low=16.0,
        bg_high=16.0,
        baseline_smooth_px=3,
    )

    np.testing.assert_allclose(spectrum, np.array([-5.0, 10.0, 15.0], dtype=np.float32))
    np.testing.assert_allclose(demod_map, np.array([[20.0 / 3.0]], dtype=np.float32))


def test_background_subtraction_averages_spatial_neighbors_for_baseline():
    fft_cube = np.zeros((3, 3, 5, 1), dtype=np.float32)
    fft_cube[1, 1, 2, 0] = 100.0
    fft_cube[1, 1, 3, 0] = 90.0
    bundle = {
        "fft_cube": fft_cube,
        "fft_baseline_cube": np.zeros((3, 3, 1), dtype=np.float32),
        "f_axis": np.array([0.0, 4.0, 8.0, 16.0, 18.0], dtype=np.float32),
        "metadata": {
            "f_expected_hz": 4.0,
            "f_search_halfwidth_hz": 0.1,
            "trigger_frequency_hz": 40.0,
            "roi_method": "mean",
        },
    }

    spectrum = get_detector_spectrum_bgsub_live(
        bundle,
        "2w",
        slice(1, 2),
        slice(1, 2),
        bg_low=16.0,
        bg_high=16.0,
        background_neighbor_px=3,
    )
    baseline = get_detector_baseline_live(
        bundle,
        slice(1, 2),
        slice(1, 2),
        bg_low=16.0,
        bg_high=16.0,
        background_neighbor_px=3,
    )

    np.testing.assert_allclose(spectrum, np.array([90.0], dtype=np.float32))
    np.testing.assert_allclose(baseline, np.array([10.0], dtype=np.float32))


def test_get_detector_baseline_live_returns_smoothed_selected_profile():
    fft_cube = np.zeros((1, 1, 5, 3), dtype=np.float32)
    fft_cube[0, 0, 3, :] = [0.0, 30.0, 0.0]
    bundle = {
        "fft_cube": fft_cube,
        "fft_baseline_cube": np.zeros((1, 1, 3), dtype=np.float32),
        "f_axis": np.array([0.0, 4.0, 8.0, 16.0, 18.0], dtype=np.float32),
        "metadata": {
            "f_expected_hz": 4.0,
            "f_search_halfwidth_hz": 0.1,
            "trigger_frequency_hz": 40.0,
        },
    }

    baseline = get_detector_baseline_live(
        bundle,
        slice(0, 1),
        slice(0, 1),
        bg_low=16.0,
        bg_high=16.0,
        baseline_smooth_px=3,
    )

    np.testing.assert_allclose(baseline, np.array([15.0, 10.0, 15.0], dtype=np.float32))


def test_finite_range_uses_fallback_for_all_nan():
    arr = np.array([np.nan, np.nan])

    assert finite_range(arr) == (0.0, 1.0)


def test_nanmean_or_nan_returns_nan_columns_without_warning():
    arr = np.array([[np.nan, 1.0], [np.nan, 3.0]])

    result = nanmean_or_nan(arr, axis=0)

    np.testing.assert_allclose(result, np.array([np.nan, 2.0]), equal_nan=True)
