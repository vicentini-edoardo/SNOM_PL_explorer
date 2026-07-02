"""Entry point for the SNOM PL explorer GUI.

The implementation lives in the ``gui`` package; this module re-exports the
public names so ``python app.py`` and ``from app import MainWindow`` keep
working.
"""
from __future__ import annotations

from gui.main_window import ROOT_DIR, MainWindow, main
from gui.plotting import ImagePlotWidget

__all__ = ["ImagePlotWidget", "MainWindow", "ROOT_DIR", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
