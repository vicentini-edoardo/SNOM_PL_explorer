from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from decomposition import (
    build_feature_matrix,
    categorize,
    category_mean_spectra,
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

    def load_scan(self, folder: str, filename: str, recompute: bool = False) -> ScanSummary:
        path = self.path_from_selection(folder, filename)
        cache_path = self.cache_dir / (path.stem + ".proc.npz")
        stamp = cache_stamp(path)
        status = "Computed and cached"
        if cache_path.exists() and not recompute:
            cached = load_cache(cache_path)
            if cached.get("stamp") == stamp:
                self.bundle = cached
                status = "Loaded from cache"
            else:
                self.bundle = None
        if self.bundle is None:
            self.bundle = process_scan(path)
            save_cache(self.bundle, cache_path, stamp)

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

    def compute_maps(self, settings: MapSettings) -> dict[str, np.ndarray]:
        bundle = self._require_bundle()
        roi_ps, roi_pe = settings.roi_range
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
        primary = get_demod_map_live(bundle, settings.harmonic, roi_ps, roi_pe, **kwargs)
        primary_bgsub = get_demod_map_bgsub_live(bundle, settings.harmonic, roi_ps, roi_pe, **bg_kwargs)
        compare = get_demod_map_live(bundle, settings.compare_harmonic, roi_ps, roi_pe, **kwargs)
        compare_bgsub = get_demod_map_bgsub_live(bundle, settings.compare_harmonic, roi_ps, roi_pe, **bg_kwargs)
        snom_maps = bundle.get("snom_maps", {})
        return {
            "primary": primary,
            "primary_bgsub": primary_bgsub,
            "compare": compare,
            "compare_bgsub": compare_bgsub,
            "m1a": snom_maps.get("M1A", np.full_like(primary, np.nan)),
            "m1p": snom_maps.get("M1P", np.full_like(primary, np.nan)),
        }

    def compute_inspector(self, settings: MapSettings) -> dict[str, np.ndarray]:
        bundle = self._require_bundle()
        ix, iy = self.selected_pixel
        nx, ny = int(bundle["grid"]["nx"]), int(bundle["grid"]["ny"])
        sl_y, sl_x = get_nb_slice(ix, iy, nx, ny, settings.avg3x3)
        common = dict(target_frequency_hz=settings.target_frequency_hz, neighbor_bins=settings.neighbor_bins)
        bg_common = dict(
            bg_low=settings.bg_low_hz,
            bg_high=settings.bg_high_hz,
            baseline_smooth_px=settings.baseline_smooth_px,
            background_neighbor_px=settings.background_neighbor_px,
        )
        return {
            "det_axis": bundle["det_axis"],
            "f_axis": bundle["f_axis"],
            "roi_trace": bundle["roi_traces"][iy, ix],
            "spectrum": get_detector_spectrum_live(bundle, settings.harmonic, sl_y, sl_x, **common),
            "spectrum_bgsub": get_detector_spectrum_bgsub_live(bundle, settings.harmonic, sl_y, sl_x, **bg_common, **common),
            "baseline": get_detector_baseline_live(bundle, sl_y, sl_x, **bg_common),
            "fft": get_fft_image_live(bundle, sl_y, sl_x, subtract_background=settings.fft_bgsub, **bg_common),
        }

    def compute_line_profile(self, settings: MapSettings, rows: tuple[int, int]) -> dict[str, np.ndarray]:
        bundle = self._require_bundle()
        ny = int(bundle["grid"]["ny"])
        row_lo, row_hi = _normal_range(rows[0], rows[1], ny - 1)
        maps = self.compute_maps(settings)
        phase = bundle.get("snom_maps", {}).get("M1P")
        if phase is None:
            phase = np.full_like(maps["primary"], np.nan)
        return {
            "x": np.arange(maps["primary"].shape[1], dtype=np.float64),
            "primary": nanmean_or_nan(maps["primary"][row_lo : row_hi + 1, :], axis=0),
            "primary_bgsub": nanmean_or_nan(maps["primary_bgsub"][row_lo : row_hi + 1, :], axis=0),
            "compare": nanmean_or_nan(maps["compare"][row_lo : row_hi + 1, :], axis=0),
            "compare_bgsub": nanmean_or_nan(maps["compare_bgsub"][row_lo : row_hi + 1, :], axis=0),
            "m1p": nanmean_or_nan(phase[row_lo : row_hi + 1, :], axis=0),
        }

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
    ) -> DecompositionResult:
        bundle = self._require_bundle()
        settings = settings or self.map_settings(harmonic=harmonic)
        options = set(preprocess)
        target_hz = settings.target_frequency_hz
        baseline_cube = None
        if "bgsub" in options:
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
        else:
            scores, _, scree_values = run_pca(X_valid, n_components)
        labels, centroids = categorize(scores, categorizer, n_clusters)
        raw_flat = det_cube.astype(np.float64).reshape(ny * nx, -1)
        means, stds = category_mean_spectra(raw_flat[valid_idx], labels, n_clusters)
        label_map = scatter_to_map(labels.astype(np.float64), valid_idx, ny, nx)
        return DecompositionResult(label_map, scree_values, means, stds, centroids, det_axis, method, categorizer)

    def _require_bundle(self) -> dict:
        if self.bundle is None:
            raise RuntimeError("No scan loaded")
        return self.bundle
