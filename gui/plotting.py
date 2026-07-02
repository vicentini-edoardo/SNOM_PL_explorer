"""Reusable pyqtgraph widgets and colormap helpers."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore

from gui.theme import style_plot_item

COLORMAPS = ["viridis", "plasma", "magma", "inferno", "cividis", "hot", "jet", "gray"]
PHASE_COLORMAP = "CET-C1"
CAT_PALETTE = np.array([
    [ 76, 114, 176, 255],
    [221,  95,  99, 255],
    [ 85, 168, 104, 255],
    [221, 160,  49, 255],
    [148, 103, 189, 255],
    [ 78, 195, 197, 255],
    [228, 143,  71, 255],
    [196, 142, 173, 255],
], dtype=np.uint8)


def finite_levels(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    return (lo - 1.0, hi + 1.0) if lo == hi else (lo, hi)


def pyqt_colormap(name: str):
    try:
        return pg.colormap.get(name)
    except Exception:
        return pg.colormap.get("viridis")


def colormap_lut(name: str):
    return pyqt_colormap(name).getLookupTable(0.0, 1.0, 256)


class ImagePlotWidget(pg.GraphicsLayoutWidget):
    pixel_selected = QtCore.pyqtSignal(int, int)
    pixel_hovered = QtCore.pyqtSignal(float, float)

    def __init__(self, title: str, *, default_cmap: str = "viridis", aspect_locked: bool = True):
        super().__init__()
        self.setMinimumSize(180, 150)
        self.image: np.ndarray | None = None
        self.default_cmap = default_cmap
        self._aspect_locked = aspect_locked
        self.colorbar_levels_locked = False
        self._suppress_colorbar_signal = False
        self.plot = self.addPlot(row=0, col=0, title=title)
        self.plot.setAspectLocked(aspect_locked)
        self.plot.getViewBox().invertY(True)
        style_plot_item(self.plot)
        self.item = pg.ImageItem()
        self.plot.addItem(self.item)
        self.line_item = self.plot.plot(
            [],
            [],
            pen=pg.mkPen(color="#4fc3f7", width=2),
            symbol="o",
            symbolSize=4,
            symbolBrush="#4fc3f7",
        )
        self.line_item.hide()
        self.row_region = pg.LinearRegionItem(
            values=(0, 1),
            orientation=pg.LinearRegionItem.Horizontal,
            movable=False,
            brush=pg.mkBrush(37, 99, 235, 48),
        )
        self.row_region.setZValue(5)
        for line in self.row_region.lines:
            line.setPen(pg.mkPen("#2563eb", width=2))
        self.row_region.hide()
        self.plot.addItem(self.row_region)
        self.marker = pg.ScatterPlotItem(size=14, symbol="x", pen=pg.mkPen("w", width=2), brush=None)
        self.plot.addItem(self.marker)
        crosshair_pen = pg.mkPen("#8b949e", width=1, style=QtCore.Qt.PenStyle.DashLine)
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
        for line in (self.crosshair_v, self.crosshair_h):
            line.setZValue(10)
            line.hide()
            self.plot.addItem(line, ignoreBounds=True)
        self.colorbar = pg.ColorBarItem(values=(0, 1), colorMap=pyqt_colormap(default_cmap), label=title, interactive=True, width=14)
        self.colorbar.setImageItem(self.item, insert_in=self.plot)
        self.colorbar.sigLevelsChanged.connect(self._on_colorbar_levels_changed)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def reset_colorbar_lock(self) -> None:
        self.colorbar_levels_locked = False

    def _on_colorbar_levels_changed(self, *args) -> None:
        if self._suppress_colorbar_signal:
            return
        self.colorbar_levels_locked = True

    def _set_colorbar_levels(self, levels: tuple[float, float]) -> None:
        self._suppress_colorbar_signal = True
        try:
            self.item.setLevels(levels)
            self.colorbar.setLevels(levels)
        finally:
            self._suppress_colorbar_signal = False

    def set_image(
        self,
        data: np.ndarray | None,
        title: str,
        *,
        cmap: str | None = None,
        levels: tuple[float, float] | None = None,
        selected: tuple[int, int] | None = None,
    ) -> None:
        self.plot.setTitle(title)
        self.colorbar.axis.setLabel(title)
        if data is None:
            self.image = None
            self.item.clear()
            self.item.hide()
            self.line_item.hide()
            self.row_region.hide()
            self.marker.setData([])
            self.colorbar.hide()
            return
        cmap = cmap or self.default_cmap
        array = np.asarray(data, dtype=np.float32)
        if 1 in array.shape:
            self.image = array
            values = array.reshape(-1)
            coords = np.arange(values.size)
            self.item.hide()
            self.marker.setData([])
            self.row_region.hide()
            self.colorbar.hide()
            self.line_item.show()
            self.line_item.setData(coords, values)
            self.plot.setAspectLocked(False)
            self.plot.enableAutoRange()
            return
        self.plot.setAspectLocked(self._aspect_locked)
        self.plot.getViewBox().invertY(True)
        self.line_item.hide()
        self.item.show()
        self.colorbar.show()
        self.image = array
        image = self.image.T
        image_levels = levels or finite_levels(image)
        self.item.setImage(image, autoLevels=False)
        self.item.setLookupTable(colormap_lut(cmap))
        self.colorbar.setColorMap(pyqt_colormap(cmap))
        if levels is not None or not self.colorbar_levels_locked:
            self._set_colorbar_levels(image_levels)
        self.plot.setRange(xRange=(0, image.shape[0]), yRange=(0, image.shape[1]), padding=0.02)
        if selected is None:
            self.marker.setData([])
        else:
            ix, iy = selected
            self.marker.setData([{"pos": (ix + 0.5, iy + 0.5)}])

    def set_row_region(self, rows: tuple[int, int] | None) -> None:
        if rows is None or self.image is None or 1 in self.image.shape:
            self.row_region.hide()
            return
        row_lo, row_hi = sorted((int(rows[0]), int(rows[1])))
        row_lo = int(np.clip(row_lo, 0, self.image.shape[0] - 1))
        row_hi = int(np.clip(row_hi, 0, self.image.shape[0] - 1))
        self.row_region.setRegion((row_lo, row_hi + 1))
        self.row_region.show()

    def set_crosshair(self, x: float | None, y: float | None = None) -> None:
        """Show the crosshair at view coordinates (x, y), or hide it when x is None."""
        if x is None or self.image is None or 1 in self.image.shape:
            self.crosshair_v.hide()
            self.crosshair_h.hide()
            return
        self.crosshair_v.setPos(x)
        self.crosshair_h.setPos(y)
        self.crosshair_v.show()
        self.crosshair_h.show()

    def _on_mouse_clicked(self, event) -> None:
        if self.image is None or not self.plot.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.plot.vb.mapSceneToView(event.scenePos())
        ix = int(np.clip(np.floor(point.x()), 0, self.image.shape[1] - 1))
        iy = int(np.clip(np.floor(point.y()), 0, self.image.shape[0] - 1))
        self.pixel_selected.emit(ix, iy)

    def _on_mouse_moved(self, scene_pos) -> None:
        if self.image is None or 1 in self.image.shape:
            return
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            self.pixel_hovered.emit(float("nan"), float("nan"))
            return
        point = self.plot.vb.mapSceneToView(scene_pos)
        self.pixel_hovered.emit(point.x(), point.y())

    def leaveEvent(self, event) -> None:
        self.pixel_hovered.emit(float("nan"), float("nan"))
        super().leaveEvent(event)
