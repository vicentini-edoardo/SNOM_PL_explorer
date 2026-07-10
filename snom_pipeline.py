"""snom_pipeline.py – data loading and processing for stroboscopic-SNOM scans."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

logger = logging.getLogger(__name__)

HARMONICS = ("0w", "1w", "2w", "3w")
SNOM_CHANNELS = ("Z", "M1A", "M1P", "M2A", "M2P")
BG_LOW_HZ = 15.0
BG_HIGH_HZ = 20.0
PROCESSING_VERSION = "roi-trace-demod-v2"


# ── Low-level signal processing ───────────────────────────────────────────────

def _window_fn(n: int, name: str = "hann") -> np.ndarray:
    if name == "hann":
        return np.hanning(n)
    if name == "blackman":
        return np.blackman(n)
    return np.ones(n, dtype=np.float64)


def _parabolic_peak_amplitude(f_axis: np.ndarray, spectrum: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= spectrum.size - 1:
        return float(spectrum[idx])
    alpha = float(spectrum[idx - 1])
    beta = float(spectrum[idx])
    gamma = float(spectrum[idx + 1])
    denom = alpha - 2.0 * beta + gamma
    if abs(denom) < 1e-15:
        return beta
    offset = 0.5 * (alpha - gamma) / denom
    offset = float(np.clip(offset, -1.0, 1.0))
    amplitude = beta - 0.25 * (alpha - gamma) * offset
    return float(max(amplitude, 0.0))


def trace_demod_scalars(roi: np.ndarray, meta: dict) -> dict[str, float]:
    """
    Single scalar per harmonic from the detector ROI time trace.
    Matches the notebook's recompute_demod path for original demod maps.
    """
    chunk = np.asarray(roi, dtype=np.float64)[: int(meta["n_block"])]
    chunk = chunk[np.isfinite(chunk)]
    if chunk.size == 0:
        return {h: float("nan") for h in HARMONICS}

    out = {"0w": float(np.mean(chunk))}
    if chunk.size < 4:
        out.update({h: float("nan") for h in ("1w", "2w", "3w")})
        return out

    fs = float(meta["trigger_frequency_hz"])
    f0 = float(meta["f_expected_hz"])
    halfwidth = float(meta["f_search_halfwidth_hz"])
    centered = chunk - np.mean(chunk)
    win = _window_fn(chunk.size, meta.get("window", "hann"))
    gain = float(np.mean(win)) or 1.0
    spectrum = np.abs(np.fft.rfft(centered * win)) * 2.0 / (chunk.size * gain)
    f_axis = np.fft.rfftfreq(chunk.size, d=1.0 / fs)

    for i, key in enumerate(("1w", "2w", "3w"), 1):
        low = max(0.0, i * f0 - halfwidth)
        high = min(fs / 2.0, i * f0 + halfwidth)
        band_indices = np.flatnonzero((f_axis >= low) & (f_axis <= high))
        if band_indices.size == 0:
            out[key] = float("nan")
            continue
        peak_idx = int(band_indices[np.argmax(spectrum[band_indices])])
        out[key] = _parabolic_peak_amplitude(f_axis, spectrum, peak_idx)
    return out


def point_fft_matrix(frames: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Windowed rFFT along time axis for each detector column.
    Returns (f_axis [n_freq], fft [n_freq, det]) float32.
    Bin 0 (DC) holds per-column temporal mean.
    """
    arr = frames.astype(np.float64)
    n_block = int(meta["n_block"])
    chunk = arr[: min(len(arr), n_block)]
    n, det = chunk.shape
    fs = float(meta["trigger_frequency_hz"])
    f_axis = np.fft.rfftfreq(n_block, d=1.0 / fs)
    n_freq = len(f_axis)

    if n < 2:
        fft = np.full((n_freq, det), np.nan, dtype=np.float32)
        if n == 1:
            fft[0, :] = chunk[0].astype(np.float32)
        return f_axis, fft

    dc = np.mean(chunk, axis=0)  # (det,)
    win = _window_fn(n, meta.get("window", "hann"))
    gain = float(np.mean(win)) or 1.0
    fft = np.abs(np.fft.rfft((chunk - dc[None, :]) * win[:, None], n=n_block, axis=0))
    fft = (fft * 2.0 / (n * gain)).astype(np.float32)
    fft[0, :] = dc.astype(np.float32)  # restore DC
    return f_axis, fft


def _band_mask(f_axis: np.ndarray, meta: dict, h: int) -> np.ndarray:
    """Boolean mask for harmonic h (1..3) frequency band."""
    f0 = float(meta["f_expected_hz"])
    hw = float(meta["f_search_halfwidth_hz"])
    return (f_axis >= h * f0 - hw) & (f_axis <= h * f0 + hw)


def demod_spectra(fft: np.ndarray, f_axis: np.ndarray, meta: dict) -> dict[str, np.ndarray]:
    """
    Per-detector-pixel amplitude for each harmonic.
    0w = DC bin; 1w/2w/3w = peak amplitude within frequency band per column.
    Returns {h: (det,)} float32.
    """
    det = fft.shape[1]
    out: dict[str, np.ndarray] = {"0w": fft[0].copy()}
    for i, key in enumerate(("1w", "2w", "3w"), 1):
        mask = _band_mask(f_axis, meta, i)
        if not mask.any():
            out[key] = np.full(det, np.nan, dtype=np.float32)
        else:
            band = fft[mask]  # (n_band, det)
            peak_row = np.argmax(band, axis=0)  # (det,)
            out[key] = band[peak_row, np.arange(det)].astype(np.float32)
    return out


def roi_scalars(fft: np.ndarray, f_axis: np.ndarray, meta: dict) -> dict[str, float]:
    """
    Single scalar per harmonic: amplitude collapsed over ROI detector pixels.
    Used to fill the 2-D maps.
    """
    ps, pe = int(meta["roi_pixel_start"]), int(meta["roi_pixel_end"])
    roi = fft[:, ps : pe + 1]
    method = meta.get("roi_method", "mean")
    collapsed = np.nansum(roi, axis=1) if method == "sum" else np.nanmean(roi, axis=1)
    out = {"0w": float(collapsed[0])}
    for i, key in enumerate(("1w", "2w", "3w"), 1):
        mask = _band_mask(f_axis, meta, i)
        vals = collapsed[mask]
        out[key] = float(np.nanmax(vals)) if vals.size else float("nan")
    return out


def baseline_profile(fft: np.ndarray, f_axis: np.ndarray) -> np.ndarray:
    """Mean FFT amplitude in BG_LOW–BG_HIGH Hz band per detector column. (det,) float32."""
    mask = (f_axis >= BG_LOW_HZ) & (f_axis <= BG_HIGH_HZ)
    if not mask.any():
        return np.full(fft.shape[1], np.nan, dtype=np.float32)
    return np.nanmean(fft[mask], axis=0).astype(np.float32)


def extract_snom_scalar(grp: h5py.Group, key: str) -> float:
    if key == "Z":
        arr = np.asarray(grp["snom_xyz_nm"])
        return float(np.nanmean(arr[:, 2])) if arr.size else float("nan")
    _map = {
        "M1A": ("snom_m_amp", 1), "M1P": ("snom_m_phase", 1),
        "M2A": ("snom_m_amp", 2), "M2P": ("snom_m_phase", 2),
    }
    ds, col = _map[key]
    arr = np.asarray(grp[ds])
    return float(np.nanmean(arr[:, col])) if arr.size else float("nan")


def integrate_roi(frames: np.ndarray, ps: int, pe: int, method: str = "mean") -> np.ndarray:
    roi = frames[:, ps : pe + 1].astype(np.float64)
    return np.mean(roi, axis=1) if method == "mean" else np.sum(roi, axis=1)


# ── Scan I/O ──────────────────────────────────────────────────────────────────

REQUIRED_METADATA_KEYS = (
    "grid",
    "n_block",
    "trigger_frequency_hz",
    "f_expected_hz",
    "f_search_halfwidth_hz",
    "roi_pixel_start",
    "roi_pixel_end",
)


def validate_scan_file(h5: h5py.File, path: Path) -> dict:
    """Check *h5* looks like a supported scan and return its parsed metadata.

    Raises ValueError with a message naming the file and the exact problem,
    so the GUI can show something actionable instead of a KeyError.
    """
    if "metadata" not in h5.attrs:
        raise ValueError(f"{path.name}: not a supported scan file (missing 'metadata' attribute)")
    try:
        meta = json.loads(h5.attrs["metadata"])
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"{path.name}: 'metadata' attribute is not valid JSON ({exc})") from None
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in meta]
    if missing:
        raise ValueError(f"{path.name}: metadata is missing required keys: {', '.join(missing)}")
    grid = meta["grid"]
    if not isinstance(grid, dict) or "ny" not in grid or "nx" not in grid:
        raise ValueError(f"{path.name}: metadata 'grid' must contain 'ny' and 'nx'")
    if "points" not in h5 or len(h5["points"]) == 0:
        raise ValueError(f"{path.name}: no 'points' group with scan data")
    return meta


def cache_stamp(path: Path) -> str:
    """Unique string for path's mtime + size; used to detect stale cache."""
    s = path.stat()
    return f"{PROCESSING_VERSION}_{int(s.st_mtime_ns)}_{s.st_size}"


def process_scan(
    path: Path,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict:
    """
    Stream all points in *path* once, build all precomputed arrays.

    Returned bundle keys
    --------------------
    file_name, grid, metadata, f_axis, det_axis,
    demod_maps, demod_maps_bgsub  – {h: (ny,nx)} float32
    snom_maps                     – {k: (ny,nx)} float32
    roi_traces                    – (ny,nx,n_block) float32
    det_spectra                   – {h: (ny,nx,det)} float32
    fft_cube                      – (ny,nx,n_freq,det) float32
    fft_baseline_cube             – (ny,nx,det) float32
    fft_zlim, fft_bgsub_zlim      – (lo, hi) floats
    """
    path = Path(path)
    logger.info("Processing scan %s", path)
    try:
        h5_file = h5py.File(path, "r")
    except OSError as exc:
        raise ValueError(f"{path.name}: cannot open as HDF5 ({exc})") from None
    with h5_file as h5:
        meta = validate_scan_file(h5, path)
        grid = meta["grid"]
        ny, nx = int(grid["ny"]), int(grid["nx"])
        n_block = int(meta["n_block"])
        fs = float(meta["trigger_frequency_hz"])
        n_freq = n_block // 2 + 1
        f_axis = np.fft.rfftfreq(n_block, d=1.0 / fs)

        point_names = sorted(h5["points"].keys())
        n_total = len(point_names)

        # Detector width from first point
        det = int(h5[f"points/{point_names[0]}/frames"].shape[1])
        ps = int(meta["roi_pixel_start"])
        pe = int(meta["roi_pixel_end"])
        roi_method = meta.get("roi_method", "mean")

        # Pre-allocate
        demod_maps     = {h: np.full((ny, nx), np.nan, dtype=np.float32) for h in HARMONICS}
        demod_maps_bgsub = {h: np.full((ny, nx), np.nan, dtype=np.float32) for h in HARMONICS}
        snom_maps      = {k: np.full((ny, nx), np.nan, dtype=np.float32) for k in SNOM_CHANNELS}
        roi_traces     = np.full((ny, nx, n_block), np.nan, dtype=np.float32)
        det_spectra    = {h: np.full((ny, nx, det), np.nan, dtype=np.float32) for h in HARMONICS}
        fft_cube       = np.full((ny, nx, n_freq, det), np.nan, dtype=np.float32)
        fft_baseline_cube = np.full((ny, nx, det), np.nan, dtype=np.float32)

        fft_min, fft_max = np.inf, -np.inf
        bgsub_min, bgsub_max = np.inf, -np.inf

        for i, name in enumerate(point_names):
            if progress_cb is not None:
                progress_cb(i / n_total, name)

            grp = h5[f"points/{name}"]
            iy = int(grp.attrs["iy"])
            ix = int(grp.attrs["ix"])

            frames = np.asarray(grp["frames"], dtype=np.float64)
            if frames.ndim != 2 or frames.shape[1] != det:
                continue

            # ROI time trace
            roi = integrate_roi(frames, ps, pe, roi_method)
            n_roi = min(len(roi), n_block)
            roi_traces[iy, ix, :n_roi] = roi[:n_roi].astype(np.float32)

            # FFT matrix
            _, fft = point_fft_matrix(frames, meta)
            fft_cube[iy, ix] = fft

            # Baseline
            bl = baseline_profile(fft, f_axis)
            fft_baseline_cube[iy, ix] = bl

            # Per-detector-pixel demod spectra
            spec = demod_spectra(fft, f_axis, meta)
            for h in HARMONICS:
                det_spectra[h][iy, ix] = spec[h]

            # Map scalars – original, from ROI time trace as in the notebook.
            sc = trace_demod_scalars(roi, meta)
            for h in HARMONICS:
                demod_maps[h][iy, ix] = sc[h]

            # Map scalars – background-subtracted
            if n_block < 2:
                sc_bg = {"0w": sc["0w"], "1w": float("nan"), "2w": float("nan"), "3w": float("nan")}
                fft_bg = np.full_like(fft, np.nan, dtype=np.float32)
            else:
                fft_bg = (fft - bl[None, :]).astype(np.float32)
                sc_bg = roi_scalars(fft_bg, f_axis, meta)
            for h in HARMONICS:
                demod_maps_bgsub[h][iy, ix] = sc_bg[h]

            # SNOM scalars
            for k in SNOM_CHANNELS:
                snom_maps[k][iy, ix] = extract_snom_scalar(grp, k)

            # Track global color ranges (skip DC row for original FFT)
            fin = fft[1:][np.isfinite(fft[1:])]
            if fin.size:
                fft_min = min(fft_min, float(np.nanmin(fin)))
                fft_max = max(fft_max, float(np.nanmax(fin)))
            fin_bg = fft_bg[1:][np.isfinite(fft_bg[1:])]  # skip DC row
            if fin_bg.size:
                bgsub_min = min(bgsub_min, float(np.nanmin(fin_bg)))
                bgsub_max = max(bgsub_max, float(np.nanmax(fin_bg)))

    if progress_cb is not None:
        progress_cb(1.0, "done")

    return {
        "file_name": path.name,
        "grid": grid,
        "metadata": meta,
        "f_axis": f_axis.astype(np.float32),
        "det_axis": np.arange(det, dtype=np.int32),
        "demod_maps": demod_maps,
        "demod_maps_bgsub": demod_maps_bgsub,
        "snom_maps": snom_maps,
        "roi_traces": roi_traces,
        "det_spectra": det_spectra,
        "fft_cube": fft_cube,
        "fft_baseline_cube": fft_baseline_cube,
        "fft_zlim": (
            float(fft_min) if np.isfinite(fft_min) else 0.0,
            float(fft_max) if np.isfinite(fft_max) else 1.0,
        ),
        "fft_bgsub_zlim": (
            float(bgsub_min) if np.isfinite(bgsub_min) else 0.0,
            float(bgsub_max) if np.isfinite(bgsub_max) else 1.0,
        ),
    }


# ── Cache I/O ─────────────────────────────────────────────────────────────────

def save_cache(bundle: dict, cache_path: Path, stamp: str) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        _stamp=np.frombuffer(stamp.encode(), dtype=np.uint8),
        _meta_json=np.frombuffer(json.dumps(bundle["metadata"]).encode(), dtype=np.uint8),
        f_axis=bundle["f_axis"],
        det_axis=bundle["det_axis"],
        roi_traces=bundle["roi_traces"],
        fft_cube=bundle["fft_cube"],
        fft_baseline_cube=bundle["fft_baseline_cube"],
        fft_zlim=np.array(bundle["fft_zlim"], dtype=np.float32),
        fft_bgsub_zlim=np.array(bundle["fft_bgsub_zlim"], dtype=np.float32),
        **{f"demod_maps_{h}": bundle["demod_maps"][h] for h in HARMONICS},
        **{f"demod_maps_bgsub_{h}": bundle["demod_maps_bgsub"][h] for h in HARMONICS},
        **{f"snom_maps_{k}": bundle["snom_maps"][k] for k in SNOM_CHANNELS},
        **{f"det_spectra_{h}": bundle["det_spectra"][h] for h in HARMONICS},
    )


def load_cache(cache_path: Path) -> dict:
    npz = np.load(cache_path, allow_pickle=False)
    meta = json.loads(bytes(npz["_meta_json"]).decode())
    return {
        "stamp": bytes(npz["_stamp"]).decode(),
        "file_name": Path(cache_path).stem.removesuffix(".proc"),
        "grid": meta["grid"],
        "metadata": meta,
        "f_axis": npz["f_axis"],
        "det_axis": npz["det_axis"],
        "roi_traces": npz["roi_traces"],
        "fft_cube": npz["fft_cube"],
        "fft_baseline_cube": npz["fft_baseline_cube"],
        "fft_zlim": tuple(float(v) for v in npz["fft_zlim"]),
        "fft_bgsub_zlim": tuple(float(v) for v in npz["fft_bgsub_zlim"]),
        "demod_maps":      {h: npz[f"demod_maps_{h}"] for h in HARMONICS},
        "demod_maps_bgsub":{h: npz[f"demod_maps_bgsub_{h}"] for h in HARMONICS},
        "snom_maps":       {k: npz[f"snom_maps_{k}"] for k in SNOM_CHANNELS},
        "det_spectra":     {h: npz[f"det_spectra_{h}"] for h in HARMONICS},
    }
