from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import h5py
import numpy as np

from decomposition import (
    build_feature_matrix,
    categorize,
    category_mean_spectra,
    run_gnmf,
    run_mnf,
    run_pca,
    scatter_to_map,
)
from snom_pipeline import (
    BG_HIGH_HZ,
    BG_LOW_HZ,
    HARMONICS,
    cache_stamp,
    load_cache,
    process_scan,
    save_cache,
)
from state import (
    _baseline_cube_from_frequency_range,
    _integrated_detector_spectra_from_fft,
    correct_linear_drift,
    correct_row_leveling,
    discover_h5_files,
    get_demod_map_bgsub_live,
    get_demod_map_live,
    get_detector_baseline_live,
    get_detector_spectrum_bgsub_live,
    get_detector_spectrum_live,
    get_fft_image_live,
    get_nb_slice,
    nanmean_or_nan,
)


@dataclass(frozen=True)
class ScanSummary:
    path: Path
    status: str
    ny: int
    nx: int
    detector_pixels: int
    metadata: dict


@dataclass(frozen=True)
class MapSettings:
    harmonic: str = "0w"
    compare_harmonic: str = "1w"
    roi_range: tuple[int, int] = (0, 0)
    cmap: str = "viridis"
    range_mode: str = "auto"
    color_min: float | None = None
    color_max: float | None = None
    bg_low_hz: float = BG_LOW_HZ
    bg_high_hz: float = BG_HIGH_HZ
    baseline_smooth_px: int = 1
    background_neighbor_px: int = 1
    target_frequency_hz: float | None = None
    neighbor_bins: int = 0
    avg3x3: bool = True
    fft_bgsub: bool = False
    mechanical_channel: str = "M1P"
    period_window: int = 1
    period_max_shift: int = 0
    period_min_shift: int = 0
    z_drift_correct: bool = False
    z_drift_scan_mode: str = "raster"
    z_row_level_mode: str = "none"


@dataclass(frozen=True)
class DecompositionResult:
    label_map: np.ndarray
    scree_values: np.ndarray
    category_means: np.ndarray
    category_stds: np.ndarray
    centroids: np.ndarray
    detector_axis: np.ndarray
    method: str
    categorizer: str
    scores: np.ndarray
    labels: np.ndarray


def _parse_float(value, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return default


def _normal_range(lo: int, hi: int, max_value: int) -> tuple[int, int]:
    a = max(0, min(int(lo), max_value))
    b = max(0, min(int(hi), max_value))
    return (a, b) if a <= b else (b, a)


class SnomAppModel:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.cache_dir = self.root_dir / "_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.bundle: dict | None = None
        self.summary: ScanSummary | None = None
        self.selected_pixel: tuple[int, int] = (0, 0)
        self.roi_range: tuple[int, int] = (0, 0)
        self.detector_range: tuple[int, int] = (0, 0)
        self.line_rows: tuple[int, int] = (0, 0)
        self.target_frequency_hz: float = 4.0
        self._folders: dict[str, list[Path]] = {}
        self._period_cache: tuple | None = None
        self.refresh_files()

    def set_root_dir(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.cache_dir = self.root_dir / "_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.bundle = None
        self.summary = None
        self.refresh_files()

    def refresh_files(self) -> dict[str, list[Path]]:
        self._folders = discover_h5_files(self.root_dir)
        return self._folders

    def folder_options(self) -> list[str]:
        return list(self._folders)

    def file_options(self, folder: str) -> list[str]:
        return [path.name for path in self._folders.get(folder, [])]

    def path_from_selection(self, folder: str, filename: str) -> Path:
        return self.root_dir / filename if folder == "." else self.root_dir / folder / filename

    def load_scan(
        self,
        folder: str,
        filename: str,
        recompute: bool = False,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> ScanSummary:
        path = self.path_from_selection(folder, filename)
        cache_path = self.cache_dir / (path.stem + ".proc.npz")
        stamp = cache_stamp(path)
        status = "Computed and cached"
        bundle = None
        if cache_path.exists() and not recompute:
            cached = load_cache(cache_path)
            if cached.get("stamp") == stamp:
                bundle = cached
                status = "Loaded from cache"
        if bundle is None:
            bundle = process_scan(path, progress_cb=progress_cb)
            save_cache(bundle, cache_path, stamp)
        self.bundle = bundle
        self._period_cache = None

        grid = self.bundle["grid"]
        metadata = self.bundle["metadata"]
        ny, nx = int(grid["ny"]), int(grid["nx"])
        det_max = len(self.bundle["det_axis"]) - 1
        default_start = int(metadata.get("roi_pixel_start", 0))
        default_end = int(metadata.get("roi_pixel_end", det_max))
        self.roi_range = _normal_range(default_start, default_end, det_max)
        self.detector_range = (0, det_max)
        self.line_rows = (0, min(ny - 1, 4))
        self.selected_pixel = (nx // 2, ny // 2)
        self.target_frequency_hz = float(metadata.get("f_expected_hz", 4.0))
        self.summary = ScanSummary(path, status, ny, nx, det_max + 1, metadata)
        return self.summary

    def select_pixel(self, ix: int, iy: int) -> None:
        if self.bundle is None:
            self.selected_pixel = (0, 0)
            return
        nx, ny = int(self.bundle["grid"]["nx"]), int(self.bundle["grid"]["ny"])
        self.selected_pixel = (
            int(np.clip(ix, 0, nx - 1)),
            int(np.clip(iy, 0, ny - 1)),
        )

    def map_settings(self, **overrides) -> MapSettings:
        base = MapSettings(roi_range=self.roi_range, target_frequency_hz=self.target_frequency_hz)
        values = {**base.__dict__, **overrides}
        return MapSettings(**values)

    def _demod_kwargs(self, settings: MapSettings) -> tuple[dict, dict]:
        kwargs = dict(
            target_frequency_hz=settings.target_frequency_hz,
            neighbor_bins=settings.neighbor_bins,
        )
        bg_kwargs = dict(
            bg_low=settings.bg_low_hz,
            bg_high=settings.bg_high_hz,
            baseline_smooth_px=settings.baseline_smooth_px,
            background_neighbor_px=settings.background_neighbor_px,
            **kwargs,
        )
        return kwargs, bg_kwargs

    def demod_map(self, settings: MapSettings, harmonic: str, bgsub: bool) -> np.ndarray:
        bundle = self._require_bundle()
        roi_ps, roi_pe = settings.roi_range
        kwargs, bg_kwargs = self._demod_kwargs(settings)
        if bgsub:
            return get_demod_map_bgsub_live(bundle, harmonic, roi_ps, roi_pe, **bg_kwargs)
        return get_demod_map_live(bundle, harmonic, roi_ps, roi_pe, **kwargs)

    def snom_map(self, settings: MapSettings, channel: str) -> np.ndarray:
        bundle = self._require_bundle()
        ny, nx = int(bundle["grid"]["ny"]), int(bundle["grid"]["nx"])
        snom_maps = bundle.get("snom_maps", {})
        result = snom_maps.get(channel, np.full((ny, nx), np.nan, dtype=np.float32))
        if channel == "Z":
            if settings.z_drift_correct:
                result = correct_linear_drift(result, settings.z_drift_scan_mode)
            if settings.z_row_level_mode != "none":
                result = correct_row_leveling(result, settings.z_row_level_mode)
        return result

    def compute_maps(self, settings: MapSettings) -> dict[str, np.ndarray]:
        primary = self.demod_map(settings, settings.harmonic, False)
        primary_bgsub = self.demod_map(settings, settings.harmonic, True)
        compare = self.demod_map(settings, settings.compare_harmonic, False)
        compare_bgsub = self.demod_map(settings, settings.compare_harmonic, True)
        return {
            "primary": primary,
            "primary_bgsub": primary_bgsub,
            "compare": compare,
            "compare_bgsub": compare_bgsub,
            "m1a": self.snom_map(settings, "M1A"),
            "m1p": self.snom_map(settings, "M1P"),
            "mechanical": self.snom_map(settings, settings.mechanical_channel),
        }

    def compute_inspector(
        self, settings: MapSettings, specs: list[tuple[str, bool]] | None = None
    ) -> dict[str, np.ndarray]:
        bundle = self._require_bundle()
        ix, iy = self.selected_pixel
        nx, ny = int(bundle["grid"]["nx"]), int(bundle["grid"]["ny"])
        avg3x3 = settings.avg3x3 and int(bundle["metadata"].get("n_block", 0)) > 1
        sl_y, sl_x = get_nb_slice(ix, iy, nx, ny, avg3x3)
        common = dict(target_frequency_hz=settings.target_frequency_hz, neighbor_bins=settings.neighbor_bins)
        bg_common = dict(
            bg_low=settings.bg_low_hz,
            bg_high=settings.bg_high_hz,
            baseline_smooth_px=settings.baseline_smooth_px,
            background_neighbor_px=settings.background_neighbor_px,
        )
        if specs is None:
            specs = [(settings.harmonic, False)]
        spectra = [
            get_detector_spectrum_bgsub_live(bundle, harmonic, sl_y, sl_x, **bg_common, **common)
            if bgsub
            else get_detector_spectrum_live(bundle, harmonic, sl_y, sl_x, **common)
            for harmonic, bgsub in specs
        ]
        return {
            "det_axis": bundle["det_axis"],
            "f_axis": bundle["f_axis"],
            "roi_trace": bundle["roi_traces"][iy, ix],
            "spectra": spectra,
            "baseline": get_detector_baseline_live(bundle, sl_y, sl_x, **bg_common),
            "fft": get_fft_image_live(bundle, sl_y, sl_x, subtract_background=settings.fft_bgsub, **bg_common),
        }

    def compute_line_profile(self, settings: MapSettings, rows: tuple[int, int]) -> dict[str, np.ndarray]:
        bundle = self._require_bundle()
        ny = int(bundle["grid"]["ny"])
        row_lo, row_hi = _normal_range(rows[0], rows[1], ny - 1)
        maps = self.compute_maps(settings)
        return {
            "x": np.arange(maps["primary"].shape[1], dtype=np.float64),
            "primary": nanmean_or_nan(maps["primary"][row_lo : row_hi + 1, :], axis=0),
            "primary_bgsub": nanmean_or_nan(maps["primary_bgsub"][row_lo : row_hi + 1, :], axis=0),
            "compare": nanmean_or_nan(maps["compare"][row_lo : row_hi + 1, :], axis=0),
            "compare_bgsub": nanmean_or_nan(maps["compare_bgsub"][row_lo : row_hi + 1, :], axis=0),
            "mechanical": nanmean_or_nan(maps["mechanical"][row_lo : row_hi + 1, :], axis=0),
        }

    def _period_sample_indices(
        self,
        period_avg: np.ndarray,
        window: int,
        *,
        max_shift: int = 0,
        min_shift: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        period_len = period_avg.shape[0]
        ref = period_avg.mean(axis=1)
        p_max = (int(np.argmax(ref)) + int(max_shift)) % period_len
        p_min = (int(np.argmin(ref)) + int(min_shift)) % period_len
        offsets = np.arange(-window, window + 1)
        return (p_max + offsets) % period_len, (p_min + offsets) % period_len

    def _selected_pixel_frames(self) -> np.ndarray | None:
        bundle = self._require_bundle()
        summary = self.summary
        if summary is None:
            raise RuntimeError("No scan loaded")
        ix, iy = self.selected_pixel
        det = len(bundle["det_axis"])
        with h5py.File(summary.path, "r") as h5:
            for name in h5["points"]:
                grp = h5[f"points/{name}"]
                if int(grp.attrs["iy"]) == iy and int(grp.attrs["ix"]) == ix:
                    frames = np.asarray(grp["frames"], dtype=np.float64)
                    if frames.ndim == 2 and frames.shape[1] == det:
                        return frames
                    return None
        return None

    def _period_cubes(self, settings: MapSettings) -> dict[str, np.ndarray]:
        """Per-detector-pixel average max/min over the modulation period.

        Re-streams raw `frames` from disk (not cached in the bundle) and is
        memoized on (path, n_block, frequency, window) so repeated calls with
        unchanged settings are free.
        # ponytail: extra full frame pass, memoized per-key. If this map gets
        # used constantly / scans get huge, fold into process_scan + cache.
        """
        bundle = self._require_bundle()
        summary = self.summary
        if summary is None:
            raise RuntimeError("No scan loaded")
        meta = bundle["metadata"]
        n_block = int(meta["n_block"])
        fs = float(meta["trigger_frequency_hz"])
        f0 = settings.target_frequency_hz or float(meta.get("f_expected_hz", 4.0))
        window = max(0, int(settings.period_window))
        key = (str(summary.path), n_block, round(float(f0), 6), window)
        if self._period_cache is not None and self._period_cache[0] == key:
            return self._period_cache[1]

        grid = bundle["grid"]
        ny, nx = int(grid["ny"]), int(grid["nx"])
        det = len(bundle["det_axis"])
        max_cube = np.full((ny, nx, det), np.nan, dtype=np.float32)
        min_cube = np.full((ny, nx, det), np.nan, dtype=np.float32)

        period_len = max(2, round(fs / f0))
        n_periods = n_block // period_len
        if n_periods >= 1:
            with h5py.File(summary.path, "r") as h5:
                for name in h5["points"]:
                    grp = h5[f"points/{name}"]
                    iy = int(grp.attrs["iy"])
                    ix = int(grp.attrs["ix"])
                    frames = np.asarray(grp["frames"], dtype=np.float64)
                    if frames.ndim != 2 or frames.shape[1] != det:
                        continue
                    if frames.shape[0] < n_periods * period_len:
                        continue  # interrupted measurement, too few frames: leave NaN
                    trimmed = frames[: n_periods * period_len]
                    period_avg = trimmed.reshape(n_periods, period_len, det).mean(axis=0)
                    idx_max, idx_min = self._period_sample_indices(period_avg, window)
                    max_cube[iy, ix] = period_avg[idx_max].mean(axis=0)
                    min_cube[iy, ix] = period_avg[idx_min].mean(axis=0)

        cubes = {"max": max_cube, "min": min_cube, "det_axis": bundle["det_axis"]}
        self._period_cache = (key, cubes)
        return cubes

    def compute_period_maps(self, settings: MapSettings) -> dict[str, np.ndarray]:
        cubes = self._period_cubes(settings)
        ps, pe = settings.roi_range
        max_map = nanmean_or_nan(cubes["max"][:, :, ps : pe + 1], axis=2)
        min_map = nanmean_or_nan(cubes["min"][:, :, ps : pe + 1], axis=2)
        return {"max": max_map, "min": min_map, "diff": max_map - min_map}

    def compute_period_spectra(self, settings: MapSettings) -> dict[str, np.ndarray]:
        bundle = self._require_bundle()
        meta = bundle["metadata"]
        fs = float(meta["trigger_frequency_hz"])
        f0 = settings.target_frequency_hz or float(meta.get("f_expected_hz", 4.0))
        window = max(0, int(settings.period_window))
        period_len = max(2, round(fs / f0))
        det_axis = bundle["det_axis"]
        det = len(det_axis)
        max_spec = np.full(det, np.nan, dtype=np.float64)
        min_spec = np.full(det, np.nan, dtype=np.float64)
        frames = self._selected_pixel_frames()
        if frames is not None:
            n_periods = frames.shape[0] // period_len
            if n_periods >= 1:
                trimmed = frames[: n_periods * period_len]
                period_avg = trimmed.reshape(n_periods, period_len, det).mean(axis=0)
                idx_max, idx_min = self._period_sample_indices(
                    period_avg,
                    window,
                    max_shift=settings.period_max_shift,
                    min_shift=settings.period_min_shift,
                )
                max_spec = period_avg[idx_max].mean(axis=0)
                min_spec = period_avg[idx_min].mean(axis=0)
        return {
            "det_axis": det_axis,
            "max": max_spec,
            "min": min_spec,
            "diff": max_spec - min_spec,
        }

    def compute_period_trace(self, settings: MapSettings) -> dict[str, np.ndarray]:
        """ROI-averaged raw time trace at the selected pixel, with the frame
        indices used as max/min samples flagged for highlighting."""
        bundle = self._require_bundle()
        summary = self.summary
        if summary is None:
            raise RuntimeError("No scan loaded")
        meta = bundle["metadata"]
        fs = float(meta["trigger_frequency_hz"])
        f0 = settings.target_frequency_hz or float(meta.get("f_expected_hz", 4.0))
        window = max(0, int(settings.period_window))
        period_len = max(2, round(fs / f0))
        ps, pe = settings.roi_range
        det = len(bundle["det_axis"])
        frames = self._selected_pixel_frames()

        empty = np.zeros(0, dtype=np.float64)
        if frames is None or frames.ndim != 2 or frames.shape[1] != det:
            return {"x": empty, "trace": empty, "max_mask": empty.astype(bool), "min_mask": empty.astype(bool)}

        n = frames.shape[0]
        trace = frames[:, ps : pe + 1].mean(axis=1)
        max_mask = np.zeros(n, dtype=bool)
        min_mask = np.zeros(n, dtype=bool)
        n_periods = n // period_len
        if n_periods >= 1:
            trimmed = frames[: n_periods * period_len]
            period_avg = trimmed.reshape(n_periods, period_len, det).mean(axis=0)
            idx_max, idx_min = self._period_sample_indices(
                period_avg,
                window,
                max_shift=settings.period_max_shift,
                min_shift=settings.period_min_shift,
            )
            for p in range(n_periods):
                base = p * period_len
                max_mask[base + idx_max] = True
                min_mask[base + idx_min] = True

        return {"x": np.arange(n, dtype=np.float64), "trace": trace, "max_mask": max_mask, "min_mask": min_mask}

    def compute_decomposition(
        self,
        *,
        harmonic: str,
        method: str,
        n_components: int,
        categorizer: str,
        n_clusters: int,
        preprocess: Iterable[str],
        detector_range: tuple[int, int],
        settings: MapSettings | None = None,
        gnmf_graph: str = "spatial",
        gnmf_neighbors: int = 5,
        gnmf_lambda: float = 100.0,
    ) -> DecompositionResult:
        bundle = self._require_bundle()
        settings = settings or self.map_settings(harmonic=harmonic)
        options = set(preprocess)
        target_hz = settings.target_frequency_hz
        baseline_cube = None
        if "bgsub" in options and not (harmonic == "0w" and len(bundle["f_axis"]) < 2):
            baseline_cube = _baseline_cube_from_frequency_range(
                bundle,
                settings.bg_low_hz,
                settings.bg_high_hz,
                settings.baseline_smooth_px,
                settings.background_neighbor_px,
            )
            fft_values = bundle["fft_cube"] - baseline_cube[:, :, None, :]
        else:
            fft_values = bundle["fft_cube"]
        det_cube = _integrated_detector_spectra_from_fft(
            fft_values,
            bundle["f_axis"],
            bundle["metadata"],
            harmonic,
            target_frequency_hz=target_hz,
            neighbor_bins=settings.neighbor_bins,
        )
        det_max = det_cube.shape[2] - 1
        px_lo, px_hi = _normal_range(detector_range[0], detector_range[1], det_max)
        det_axis = bundle["det_axis"][px_lo : px_hi + 1]
        det_cube = det_cube[:, :, px_lo : px_hi + 1]
        X_valid, valid_idx, X_full, (ny, nx) = build_feature_matrix(
            bundle,
            harmonic,
            bgsub=False,
            l2norm="l2norm" in options,
            standardize="standardize" in options,
            det_spectra_cube=det_cube,
        )
        if len(X_valid) < 2:
            raise ValueError("Need at least two valid pixels for decomposition")
        n_components = max(1, int(n_components))
        n_clusters = max(2, min(int(n_clusters), len(X_valid)))
        if method == "MNF":
            scores, _, scree_values = run_mnf(X_full, valid_idx, n_components)
        elif method == "GNMF":
            scores, _, scree_values = run_gnmf(
                X_valid,
                X_full,
                valid_idx,
                n_components,
                graph=gnmf_graph,
                n_neighbors=gnmf_neighbors,
                reg_lambda=gnmf_lambda,
            )
        else:
            scores, _, scree_values = run_pca(X_valid, n_components)
        labels, centroids = categorize(scores, categorizer, n_clusters)
        raw_flat = det_cube.astype(np.float64).reshape(ny * nx, -1)
        means, stds = category_mean_spectra(raw_flat[valid_idx], labels, n_clusters)
        label_map = scatter_to_map(labels.astype(np.float64), valid_idx, ny, nx)
        return DecompositionResult(label_map, scree_values, means, stds, centroids, det_axis, method, categorizer, scores, labels)

    def _require_bundle(self) -> dict:
        if self.bundle is None:
            raise RuntimeError("No scan loaded")
        return self.bundle
