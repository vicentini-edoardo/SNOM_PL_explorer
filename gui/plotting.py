"""Reusable pyqtgraph widgets and colormap helpers."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

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


class _OverlayLine:
    def __init__(self, plot, *, vertical: bool, pen):
        self.item = pg.PlotDataItem([], [], pen=pen)
        plot.addItem(self.item)
        self._vertical = vertical
        self._visible = False
        self._value = 0.0

    def set_line(self, value: float, lo: float, hi: float) -> None:
        self._value = float(value)
        if self._vertical:
            self.item.setData([value, value], [lo, hi])
        else:
            self.item.setData([lo, hi], [value, value])

    def show(self) -> None:
        self._visible = True
        self.item.show()

    def hide(self) -> None:
        self._visible = False
        self.item.hide()

    def isVisible(self) -> bool:
        return self._visible

    def value(self) -> float:
        return self._value


class _RowRegionOverlay:
    def __init__(self, plot):
        self.item = QtWidgets.QGraphicsRectItem()
        self.item.setBrush(pg.mkBrush(37, 99, 235, 48))
        self.item.setPen(pg.mkPen("#2563eb", width=2))
        self.item.setZValue(5)
        plot.addItem(self.item)
        self._region = (0.0, 1.0)
        self.hide()

    def setRegion(self, region: tuple[float, float]) -> None:
        self._region = (float(region[0]), float(region[1]))

    def getRegion(self) -> tuple[float, float]:
        return self._region

    def setRect(self, x: float, y: float, w: float, h: float) -> None:
        self.item.setRect(x, y, w, h)

    def show(self) -> None:
        self.item.show()

    def hide(self) -> None:
        self.item.hide()

    def isVisible(self) -> bool:
        return self.item.isVisible()


class ImagePlotWidget(pg.GraphicsLayoutWidget):
    pixel_selected = QtCore.pyqtSignal(int, int)
    pixel_hovered = QtCore.pyqtSignal(float, float)

    def __init__(self, title: str, *, default_cmap: str = "viridis", aspect_locked: bool = True, show_title: bool = True):
        super().__init__()
        self.setMinimumSize(180, 150)
        self.image: np.ndarray | None = None
        self.default_cmap = default_cmap
        self._aspect_locked = aspect_locked
        self._show_title = show_title
        self._cleaned_up = False
        self.colorbar_levels_locked = False
        self._suppress_colorbar_signal = False
        self.plot = self.addPlot(row=0, col=0, title=title if show_title else None)
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
        self.row_region = _RowRegionOverlay(self.plot)
        self.marker = pg.ScatterPlotItem(size=14, symbol="x", pen=pg.mkPen("w", width=2), brush=None)
        self.plot.addItem(self.marker)
        crosshair_pen = pg.mkPen("#8b949e", width=1, style=QtCore.Qt.PenStyle.DashLine)
        self.crosshair_v = _OverlayLine(self.plot, vertical=True, pen=crosshair_pen)
        self.crosshair_h = _OverlayLine(self.plot, vertical=False, pen=crosshair_pen)
        self.colorbar = pg.ColorBarItem(values=(0, 1), colorMap=pyqt_colormap(default_cmap), label=title, interactive=True, width=14)
        self.colorbar.setImageItem(self.item, insert_in=self.plot)
        self.colorbar.sigLevelsChanged.connect(self._on_colorbar_levels_changed)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        scene = self.scene()
        if scene is not None:
            for signal, slot in (
                (scene.sigMouseClicked, self._on_mouse_clicked),
                (scene.sigMouseMoved, self._on_mouse_moved),
            ):
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
        self.crosshair_v.hide()
        self.crosshair_h.hide()
        self.row_region.hide()
        for item in (self.crosshair_v.item, self.crosshair_h.item, self.row_region.item):
            try:
                self.plot.removeItem(item)
            except (RuntimeError, ValueError):
                pass

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
        if self._show_title:
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
        self.row_region.setRect(0.0, float(row_lo), float(self.image.shape[1]), float(row_hi - row_lo + 1))
        self.row_region.show()

    def set_crosshair(self, x: float | None, y: float | None = None) -> None:
        """Show the crosshair at view coordinates (x, y), or hide it when x is None."""
        if x is None or self.image is None or 1 in self.image.shape:
            self.crosshair_v.hide()
            self.crosshair_h.hide()
            return
        self.crosshair_v.set_line(x, 0.0, float(self.image.shape[0]))
        self.crosshair_h.set_line(y, 0.0, float(self.image.shape[1]))
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

    def closeEvent(self, event) -> None:
        self.cleanup()
        super().closeEvent(event)
