from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from snom_pipeline import BG_HIGH_HZ, BG_LOW_HZ, HARMONICS, _window_fn


def discover_h5_files(root_dir: Path) -> dict[str, list[Path]]:
    root_dir = Path(root_dir)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root_dir.rglob("*.h5")):
        folder = "." if path.parent == root_dir else str(path.parent.relative_to(root_dir))
        grouped[folder].append(path)
    return dict(sorted((folder, sorted(paths)) for folder, paths in grouped.items()))


def clamp_pixel(ix: int, iy: int, nx: int, ny: int) -> tuple[int, int]:
    return (
        int(np.clip(ix, 0, nx - 1)),
        int(np.clip(iy, 0, ny - 1)),
    )


def _get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_selected_pixel(click_data: dict | None, nx: int, ny: int) -> tuple[int, int] | None:
    points = _get(click_data, "points", [])
    if not points:
        return None
    point = points[0]
    customdata = _get(point, "customdata")
    if customdata is not None and len(customdata) >= 2:
        return clamp_pixel(int(round(float(customdata[0]))), int(round(float(customdata[1]))), nx, ny)
    raw_x, raw_y = _get(point, "x"), _get(point, "y")
    if raw_x is None or raw_y is None:
        return None
    return clamp_pixel(int(round(float(raw_y))), int(round(float(raw_x))), nx, ny)


def get_nb_slice(ix: int, iy: int, nx: int, ny: int, avg3x3: bool) -> tuple[slice, slice]:
    if avg3x3:
        return (
            slice(max(0, iy - 1), min(ny, iy + 2)),
            slice(max(0, ix - 1), min(nx, ix + 2)),
        )
    return slice(iy, iy + 1), slice(ix, ix + 1)


def _harmonic_index(harmonic: str) -> int:
    if harmonic == "0w":
        return 0
    if harmonic in {"1w", "2w", "3w"}:
        return int(harmonic[0])
    raise ValueError(f"Unsupported harmonic: {harmonic}")


def _collapse_fft_roi(fft_cube: np.ndarray, roi_ps: int, roi_pe: int, method: str) -> np.ndarray:
    roi_fft = fft_cube[:, :, :, roi_ps : roi_pe + 1]
    if method == "sum":
        return np.nansum(roi_fft, axis=-1)
    return nanmean_or_nan(roi_fft, axis=-1)


def _demod_target_frequency(metadata: dict, target_frequency_hz: float | None) -> float:
    if target_frequency_hz is None:
        return float(metadata["f_expected_hz"])
    return float(target_frequency_hz)


def _demod_neighbor_bins(neighbor_bins: int | float | None) -> int:
    if neighbor_bins is None:
        return 0
    return max(0, int(round(float(neighbor_bins))))


def _demod_bin_indices(
    f_axis: np.ndarray,
    metadata: dict,
    harmonic: str,
    target_frequency_hz: float | None = None,
    neighbor_bins: int | float | None = None,
) -> np.ndarray:
    harmonic_idx = _harmonic_index(harmonic)
    if harmonic_idx == 0:
        return np.array([0], dtype=np.int64)
    f_axis = np.asarray(f_axis, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(f_axis))
    if finite.size == 0:
        return np.array([], dtype=np.int64)
    target = _demod_target_frequency(metadata, target_frequency_hz)
    fundamental_bin = int(finite[np.argmin(np.abs(f_axis[finite] - target))])
    center = harmonic_idx * fundamental_bin
    if center < 0 or center >= f_axis.size:
        return np.array([], dtype=np.int64)
    radius = _demod_neighbor_bins(neighbor_bins)
    low = max(0, center - radius)
    high = min(f_axis.size - 1, center + radius)
    return np.arange(low, high + 1, dtype=np.int64)


def _integrated_demod_map_from_fft(
    fft_cube: np.ndarray,
    f_axis: np.ndarray,
    metadata: dict,
    harmonic: str,
    roi_ps: int,
    roi_pe: int,
    target_frequency_hz: float | None = None,
    neighbor_bins: int | float | None = None,
) -> np.ndarray:
    collapsed = _collapse_fft_roi(fft_cube, roi_ps, roi_pe, metadata.get("roi_method", "mean"))
    band_indices = _demod_bin_indices(f_axis, metadata, harmonic, target_frequency_hz, neighbor_bins)
    if band_indices.size == 0:
        return np.full(collapsed.shape[:2], np.nan, dtype=np.float32)

    band = collapsed[:, :, band_indices]
    finite = np.isfinite(band)
    counts = np.sum(finite, axis=2)
    totals = np.nansum(band, axis=2)
    return np.where(counts == 0, np.nan, totals).astype(np.float32)


def _smooth_detector_axis(values: np.ndarray, window_px: int | float | None) -> np.ndarray:
    if window_px is None:
        return values
    window = int(round(float(window_px)))
    if window <= 1:
        return values
    window = min(window, values.shape[-1])
    kernel = np.ones(window, dtype=np.float64)
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)

    totals = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), -1, filled)
    counts = np.apply_along_axis(lambda row: np.convolve(row.astype(np.float64), kernel, mode="same"), -1, finite)
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = totals / counts
    return np.where(counts == 0, np.nan, smoothed).astype(np.float32)


def _smooth_spatial_neighbors(values: np.ndarray, window_px: int | float | None) -> np.ndarray:
    if window_px is None:
        return values
    window = int(round(float(window_px)))
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    radius = window // 2
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    totals = np.zeros_like(filled, dtype=np.float64)
    counts = np.zeros_like(filled, dtype=np.float64)

    for dy in range(-radius, radius + 1):
        src_y = slice(max(0, -dy), values.shape[0] - max(0, dy))
        dst_y = slice(max(0, dy), values.shape[0] - max(0, -dy))
        for dx in range(-radius, radius + 1):
            src_x = slice(max(0, -dx), values.shape[1] - max(0, dx))
            dst_x = slice(max(0, dx), values.shape[1] - max(0, -dx))
            totals[dst_y, dst_x] += filled[src_y, src_x]
            counts[dst_y, dst_x] += finite[src_y, src_x]

    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = totals / counts
    return np.where(counts == 0, np.nan, smoothed).astype(np.float32)


def _baseline_cube_from_frequency_range(
    bundle: dict,
    bg_low: float | None = None,
    bg_high: float | None = None,
    baseline_smooth_px: int | float | None = None,
    background_neighbor_px: int | float | None = None,
) -> np.ndarray:
    smooth_window = 1 if baseline_smooth_px is None else int(round(float(baseline_smooth_px)))
    neighbor_window = 1 if background_neighbor_px is None else int(round(float(background_neighbor_px)))
    if bg_low is None and bg_high is None and smooth_window <= 1 and neighbor_window <= 1 and "fft_baseline_cube" in bundle:
        return bundle["fft_baseline_cube"]
    low = BG_LOW_HZ if bg_low is None else float(bg_low)
    high = BG_HIGH_HZ if bg_high is None else float(bg_high)
    if low > high:
        low, high = high, low
    mask = (bundle["f_axis"] >= low) & (bundle["f_axis"] <= high)
    if not np.any(mask):
        return np.full(bundle["fft_cube"].shape[:2] + (bundle["fft_cube"].shape[-1],), np.nan, dtype=np.float32)
    baseline = nanmean_or_nan(bundle["fft_cube"][:, :, mask, :], axis=2).astype(np.float32)
    baseline = _smooth_spatial_neighbors(baseline, neighbor_window)
    return _smooth_detector_axis(baseline, smooth_window)


def _integrated_detector_spectra_from_fft(
    fft_values: np.ndarray,
    f_axis: np.ndarray,
    metadata: dict,
    harmonic: str,
    target_frequency_hz: float | None = None,
    neighbor_bins: int | float | None = None,
) -> np.ndarray:
    band_indices = _demod_bin_indices(f_axis, metadata, harmonic, target_frequency_hz, neighbor_bins)
    if band_indices.size == 0:
        return np.full(fft_values.shape[:-2] + (fft_values.shape[-1],), np.nan, dtype=np.float32)

    band = fft_values[..., band_indices, :]
    finite = np.isfinite(band)
    counts = np.sum(finite, axis=-2)
    totals = np.nansum(band, axis=-2)
    return np.where(counts == 0, np.nan, totals).astype(np.float32)


def get_detector_spectrum_live(
    bundle: dict,
    harmonic: str,
    sl_y: slice,
    sl_x: slice,
    target_frequency_hz: float | None = None,
    neighbor_bins: int | float | None = None,
) -> np.ndarray:
    spectra = _integrated_detector_spectra_from_fft(
        bundle["fft_cube"][sl_y, sl_x],
        bundle["f_axis"],
        bundle["metadata"],
        harmonic,
        target_frequency_hz=target_frequency_hz,
        neighbor_bins=neighbor_bins,
    )
    if spectra.ndim == 1:
        return spectra
    return nanmean_or_nan(spectra, axis=tuple(range(spectra.ndim - 1))).astype(np.float32)


def get_detector_spectrum_bgsub_live(
    bundle: dict,
    harmonic: str,
    sl_y: slice,
    sl_x: slice,
    bg_low: float | None = None,
    bg_high: float | None = None,
    baseline_smooth_px: int | float | None = None,
    background_neighbor_px: int | float | None = None,
    target_frequency_hz: float | None = None,
    neighbor_bins: int | float | None = None,
) -> np.ndarray:
    baseline = _baseline_cube_from_frequency_range(bundle, bg_low, bg_high, baseline_smooth_px, background_neighbor_px)
    fft_bgsub = bundle["fft_cube"][sl_y, sl_x] - baseline[sl_y, sl_x, None, :]
    spectra = _integrated_detector_spectra_from_fft(
        fft_bgsub,
        bundle["f_axis"],
        bundle["metadata"],
        harmonic,
        target_frequency_hz=target_frequency_hz,
        neighbor_bins=neighbor_bins,
    )
    if spectra.ndim == 1:
        return spectra
    return nanmean_or_nan(spectra, axis=tuple(range(spectra.ndim - 1))).astype(np.float32)


def get_detector_baseline_live(
    bundle: dict,
    sl_y: slice,
    sl_x: slice,
    bg_low: float | None = None,
    bg_high: float | None = None,
    baseline_smooth_px: int | float | None = None,
    background_neighbor_px: int | float | None = None,
) -> np.ndarray:
    baseline = _baseline_cube_from_frequency_range(bundle, bg_low, bg_high, baseline_smooth_px, background_neighbor_px)
    selected = baseline[sl_y, sl_x]
    if selected.ndim == 1:
        return selected.astype(np.float32)
    return nanmean_or_nan(selected, axis=tuple(range(selected.ndim - 1))).astype(np.float32)


def get_fft_image_live(
    bundle: dict,
    sl_y: slice,
    sl_x: slice,
    bg_low: float | None = None,
    bg_high: float | None = None,
    baseline_smooth_px: int | float | None = None,
    background_neighbor_px: int | float | None = None,
    subtract_background: bool = False,
) -> np.ndarray:
    fft_values = bundle["fft_cube"][sl_y, sl_x]
    if subtract_background:
        baseline = _baseline_cube_from_frequency_range(bundle, bg_low, bg_high, baseline_smooth_px, background_neighbor_px)
        fft_values = fft_values - baseline[sl_y, sl_x, None, :]
    return nanmean_or_nan(fft_values, axis=(0, 1)).astype(np.float32)


def _is_default_roi(metadata: dict, roi_ps: int, roi_pe: int) -> bool:
    return (
        int(roi_ps) == int(metadata.get("roi_pixel_start", roi_ps))
        and int(roi_pe) == int(metadata.get("roi_pixel_end", roi_pe))
    )


def _trace_demod_value(
    roi: np.ndarray,
    f_axis: np.ndarray,
    metadata: dict,
    harmonic: str,
    target_frequency_hz: float | None = None,
    neighbor_bins: int | float | None = None,
) -> float:
    harmonic_idx = _harmonic_index(harmonic)
    chunk = np.asarray(roi, dtype=np.float64)[: int(metadata["n_block"])]
    finite = np.isfinite(chunk)
    if not np.any(finite):
        return float("nan")
    if harmonic_idx == 0:
        return float(np.nanmean(chunk))
    if chunk.size < 4:
        return float("nan")
    mean = float(np.nanmean(chunk))
    centered = np.where(finite, chunk - mean, 0.0)
    win = _window_fn(chunk.size, metadata.get("window", "hann"))
    gain = float(np.mean(win)) or 1.0
    spectrum = np.abs(np.fft.rfft(centered * win)) * 2.0 / (chunk.size * gain)
    band_indices = _demod_bin_indices(f_axis, metadata, harmonic, target_frequency_hz, neighbor_bins)
    band_indices = band_indices[band_indices < spectrum.size]
    if band_indices.size == 0:
        return float("nan")
    vals = spectrum[band_indices]
    return float(np.nansum(vals)) if np.any(np.isfinite(vals)) else float("nan")


def _demod_map_from_roi_traces(
    bundle: dict,
    harmonic: str,
    target_frequency_hz: float | None = None,
    neighbor_bins: int | float | None = None,
) -> np.ndarray:
    if harmonic not in HARMONICS:
        raise ValueError(f"Unsupported harmonic: {harmonic}")
    roi_traces = np.asarray(bundle["roi_traces"])
    out = np.full(roi_traces.shape[:2], np.nan, dtype=np.float32)
    for iy in range(roi_traces.shape[0]):
        for ix in range(roi_traces.shape[1]):
            out[iy, ix] = _trace_demod_value(
                roi_traces[iy, ix],
                bundle["f_axis"],
                bundle["metadata"],
                harmonic,
                target_frequency_hz=target_frequency_hz,
                neighbor_bins=neighbor_bins,
            )
    return out


def get_demod_map_live(
    bundle: dict,
    harmonic: str,
    roi_ps: int,
    roi_pe: int,
    target_frequency_hz: float | None = None,
    neighbor_bins: int | float | None = None,
) -> np.ndarray:
    if "roi_traces" in bundle and _is_default_roi(bundle["metadata"], roi_ps, roi_pe):
        return _demod_map_from_roi_traces(
            bundle,
            harmonic,
            target_frequency_hz=target_frequency_hz,
            neighbor_bins=neighbor_bins,
        )
    return _integrated_demod_map_from_fft(
        bundle["fft_cube"],
        bundle["f_axis"],
        bundle["metadata"],
        harmonic,
        roi_ps,
        roi_pe,
        target_frequency_hz=target_frequency_hz,
        neighbor_bins=neighbor_bins,
    )


def get_demod_map_bgsub_live(
    bundle: dict,
    harmonic: str,
    roi_ps: int,
    roi_pe: int,
    bg_low: float | None = None,
    bg_high: float | None = None,
    baseline_smooth_px: int | float | None = None,
    background_neighbor_px: int | float | None = None,
    target_frequency_hz: float | None = None,
    neighbor_bins: int | float | None = None,
) -> np.ndarray:
    baseline = _baseline_cube_from_frequency_range(bundle, bg_low, bg_high, baseline_smooth_px, background_neighbor_px)
    fft_bgsub = bundle["fft_cube"] - baseline[:, :, None, :]
    return _integrated_demod_map_from_fft(
        fft_bgsub,
        bundle["f_axis"],
        bundle["metadata"],
        harmonic,
        roi_ps,
        roi_pe,
        target_frequency_hz=target_frequency_hz,
        neighbor_bins=neighbor_bins,
    )


def finite_range(arr: np.ndarray, fallback=(0.0, 1.0)) -> tuple[float, float]:
    fin = arr[np.isfinite(arr)]
    if not fin.size:
        return fallback
    lo, hi = float(np.nanmin(fin)), float(np.nanmax(fin))
    return (lo - 1.0, hi + 1.0) if lo == hi else (lo, hi)


def nanmean_or_nan(arr: np.ndarray, axis=None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    counts = np.sum(np.isfinite(arr), axis=axis)
    totals = np.nansum(arr, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = totals / counts
    return np.where(counts == 0, np.nan, result)
