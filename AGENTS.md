# SNOM PL Explorer

Desktop PyQt6 viewer for stroboscopic SNOM photoluminescence HDF5 scans.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

Run tests with `QT_QPA_PLATFORM=offscreen`. Set `SNOM_PL_NO_SETTINGS=1` for test isolation.

## Architecture

- Keep `snom_pipeline.py`, `state.py`, `decomposition.py`, and `app_model.py` Qt-free.
- The GUI (`gui/`) depends on the model and pipeline layers; never reverse that dependency.
- Prefer unit tests through `app_model.py`, `state.py`, or `snom_pipeline.py` unless verifying real widget behavior.
- Preserve cache-version semantics: change `PROCESSING_VERSION` when a processing change invalidates cached data.

## Working rules

- Inspect the existing behavior and run the focused test before changing processing or state logic.
- Run the full headless test suite after code changes.
- Ask before destructive commands, deleting data, rewriting Git history, or modifying credentials.
