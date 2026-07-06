# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Desktop PyQt6 viewer for stroboscopic SNOM (scanning near-field optical microscopy) photoluminescence scans stored in HDF5. Maps, background subtraction, line profiles, PCA/MNF decomposition, k-means/GMM clustering, PNG/SVG/CSV/NPZ export.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt   # or: pip install -e .[test]

python app.py                                # run the app

QT_QPA_PLATFORM=offscreen python -m pytest -q          # run full test suite (headless Qt)
QT_QPA_PLATFORM=offscreen python -m pytest -q test_state.py::test_name  # single test
```

Session persistence uses `QSettings`; set `SNOM_PL_NO_SETTINGS=1` to disable it (used by tests) or `SNOM_PL_SETTINGS_FILE=/path` to redirect to an ini file.

## Architecture

Layered, no framework beyond Qt:

- **`snom_pipeline.py`** — lowest layer. Loads raw `.h5` scans, does FFT/harmonic demodulation of detector ROI traces (`HARMONICS = 0w/1w/2w/3w`), disk caching of processed results (`cache_stamp`/`load_cache`/`save_cache`, versioned by `PROCESSING_VERSION`). No Qt imports — pure numpy/h5py.
- **`state.py`** — pixel/map query helpers operating on an already-processed scan: file discovery (`discover_h5_files`), pixel clamping/neighborhood slicing, live map/spectrum getters (`get_demod_map_live`, `get_fft_image_live`, etc.), background subtraction. No Qt imports.
- **`decomposition.py`** — PCA/MNF feature extraction and k-means/GMM categorization over map cubes, independent of Qt.
- **`app_model.py`** — `SnomAppModel`, the non-GUI application model. Wires `snom_pipeline` + `state` + `decomposition` together, holds `MapSettings`/`ScanSummary` dataclasses, exposes the operations the GUI calls. This is the seam to target for unit tests without touching Qt widgets.
- **`gui/`** — PyQt6 presentation layer, one-way dependency on `app_model`/`state`/`snom_pipeline` (never the reverse):
  - `main_window.py` — `MainWindow`, wires everything together, owns `QSettings` session persistence, dispatches long-running work to background threads.
  - `workers.py` — `Worker`/`WorkerSignals`, a `QRunnable` wrapper forwarding `finished`/`error`/`progress` signals so heavy processing (`app_model` calls) doesn't block the GUI thread. Functions passed with `wants_progress=True` receive a `progress_cb(fraction, message)` kwarg.
  - `tabs.py` — tab widgets (Maps, Inspector, Decomposition, LineProfile, Period) composed into the main window.
  - `plotting.py` — `ImagePlotWidget` and colormap helpers built on `pyqtgraph`.
  - `theme.py` — dark stylesheet + pyqtgraph styling helpers (`apply_theme`, `style_plot_item`).
- `app.py` — thin entry point re-exporting `gui.main_window` / `gui.plotting` names for `python app.py` / `from app import MainWindow`.

Derived/cached data is stored under `_cache/`, keyed by a stamp so `PROCESSING_VERSION` bumps invalidate stale caches.

## Testing notes

- Tests are flat files at repo root (`test_*.py`), one per source module, plus `test_qt_app.py` for GUI-level behavior via `pytest-qt`.
- `pytest.ini` sets `qt_api = pyqt6`.
- Always run with `QT_QPA_PLATFORM=offscreen` — there is no display in CI/most dev shells.
- Prefer testing through `app_model.py` / `state.py` / `snom_pipeline.py` (pure logic) over `gui/` widgets when a test doesn't need to verify actual Qt rendering/interaction.
