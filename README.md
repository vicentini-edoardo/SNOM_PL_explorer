# SNOM PL Explorer

Desktop viewer for stroboscopic SNOM photoluminescence scans stored in HDF5 files.

It provides:

- spatial maps with wavelength selection
- background subtraction and pixel inspection
- line profiles and point-to-point comparison
- PCA and MNF decomposition
- k-means and GMM clustering
- PNG and SVG export

## Requirements

- Python 3.10+
- a local environment with Qt support

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with:

```bash
.venv\Scripts\activate
```

## Run

```bash
python app.py
```

The app starts in the repository folder. Pick a directory containing a supported `.h5` scan file from the source controls. Derived cache files are stored under `_cache/`.

## Test

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

## License

No license is currently granted. All rights are reserved.
