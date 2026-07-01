from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from app_model import MapSettings, SnomAppModel
from snom_pipeline import BG_HIGH_HZ, BG_LOW_HZ, HARMONICS


ROOT_DIR = Path(__file__).resolve().parent
CONTROL_LABEL_WIDTH = 92
CONTROL_FIELD_MIN_WIDTH = 110
DEMOD_LABELS = {"0w": "0omega (DC)", "1w": "1omega", "2w": "2omega", "3w": "3omega"}
COLORMAPS = ["viridis", "plasma", "magma", "inferno", "cividis", "hot", "jet", "gray"]
PHASE_COLORMAP = "CET-C1"
_CAT_PALETTE = np.array([
    [ 76, 114, 176, 255],
    [221,  95,  99, 255],
    [ 85, 168, 104, 255],
    [221, 160,  49, 255],
    [148, 103, 189, 255],
    [ 78, 195, 197, 255],
    [228, 143,  71, 255],
    [196, 142, 173, 255],
], dtype=np.uint8)


def _finite_levels(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    return (lo - 1.0, hi + 1.0) if lo == hi else (lo, hi)


def _pyqt_colormap(name: str):
    try:
        return pg.colormap.get(name)
    except Exception:
        return pg.colormap.get("viridis")


def _colormap_lut(name: str):
    return _pyqt_colormap(name).getLookupTable(0.0, 1.0, 256)


def _style_plot_item(plot_item: pg.PlotItem) -> None:
    for axis_name in ("bottom", "left", "top", "right"):
        axis = plot_item.getAxis(axis_name)
        axis.setPen(pg.mkPen("#30363d", width=1))
        axis.setTextPen(pg.mkPen("#8b949e"))
    plot_item.showGrid(x=True, y=True, alpha=0.12)


class ImagePlotWidget(pg.GraphicsLayoutWidget):
    pixel_selected = QtCore.pyqtSignal(int, int)

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
        _style_plot_item(self.plot)
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
        self.colorbar = pg.ColorBarItem(values=(0, 1), colorMap=_pyqt_colormap(default_cmap), label=title, interactive=True, width=14)
        self.colorbar.setImageItem(self.item, insert_in=self.plot)
        self.colorbar.sigLevelsChanged.connect(self._on_colorbar_levels_changed)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)

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
        image_levels = levels or _finite_levels(image)
        self.item.setImage(image, autoLevels=False)
        self.item.setLookupTable(_colormap_lut(cmap))
        self.colorbar.setColorMap(_pyqt_colormap(cmap))
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

    def _on_mouse_clicked(self, event) -> None:
        if self.image is None or not self.plot.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.plot.vb.mapSceneToView(event.scenePos())
        ix = int(np.clip(np.floor(point.x()), 0, self.image.shape[1] - 1))
        iy = int(np.clip(np.floor(point.y()), 0, self.image.shape[0] - 1))
        self.pixel_selected.emit(ix, iy)


class MapsTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.primary_map = ImagePlotWidget("Map")
        self.primary_bgsub_map = ImagePlotWidget("Background subtracted")
        self.compare_map = ImagePlotWidget("Compare map")
        self.compare_bgsub_map = ImagePlotWidget("Compare background subtracted")
        self.m1a_map = ImagePlotWidget("M1A")
        self.m1p_map = ImagePlotWidget("M1P", default_cmap=PHASE_COLORMAP)
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        widgets = [
            self.primary_map,
            self.primary_bgsub_map,
            self.compare_map,
            self.compare_bgsub_map,
            self.m1a_map,
            self.m1p_map,
        ]
        for index, widget in enumerate(widgets):
            layout.addWidget(widget, index // 2, index % 2)
            layout.setRowStretch(index // 2, 1)
            layout.setColumnStretch(index % 2, 1)


class InspectorTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.roi_plot = pg.PlotWidget(title="ROI trace")
        self.roi_plot.setMinimumHeight(90)
        _style_plot_item(self.roi_plot.getPlotItem())
        self.roi_curve = self.roi_plot.plot(pen=pg.mkPen("#1f77b4", width=2))
        self.spectrum_plot = pg.PlotWidget(title="Detector spectrum")
        self.spectrum_plot.setMinimumHeight(140)
        _style_plot_item(self.spectrum_plot.getPlotItem())
        self.spectrum_plot.addLegend(offset=(8, 8))
        self.spectrum_curve = self.spectrum_plot.plot(name="original", pen=pg.mkPen("#c9d1d9", width=2))
        self.spectrum_bgsub_curve = self.spectrum_plot.plot(name="bg-sub", pen=pg.mkPen("#d62728", width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.baseline_curve = self.spectrum_plot.plot(name="baseline", pen=pg.mkPen("#2ca02c", width=2, style=QtCore.Qt.PenStyle.DotLine))
        self.fft_plot = ImagePlotWidget("FFT", aspect_locked=False)
        self.fft_plot.setMinimumHeight(220)
        self.fft_plot.plot.setLabel("left", "Frequency (Hz)")
        self.fft_plot.plot.setLabel("bottom", "Detector px")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self.roi_plot, 1)
        layout.addWidget(self.spectrum_plot, 2)
        layout.addWidget(self.fft_plot, 3)

    def clear(self) -> None:
        for curve in (self.roi_curve, self.spectrum_curve, self.spectrum_bgsub_curve, self.baseline_curve):
            curve.clear()
        self.fft_plot.set_image(None, "FFT")


class MapsInspectorTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.maps = MapsTab()
        self.inspector = InspectorTab()
        self.inspector.setMinimumWidth(260)
        self.inspector.setMaximumWidth(360)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.maps)
        splitter.addWidget(self.inspector)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1020, 300])

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)


class LineProfileTab(QtWidgets.QWidget):
    def __init__(self, controls_group: QtWidgets.QGroupBox):
        super().__init__()
        self.controls_group = controls_group
        self.primary_preview = ImagePlotWidget("Primary map")
        self.compare_preview = ImagePlotWidget("Compare map")
        self.mechanical_preview = ImagePlotWidget("Mechanical", default_cmap=PHASE_COLORMAP)
        for preview in (self.primary_preview, self.compare_preview, self.mechanical_preview):
            preview.setMinimumHeight(140)
            preview.setMaximumHeight(200)
        self.plot = pg.PlotWidget(title="Line profile")
        self.plot.setMinimumHeight(240)
        _style_plot_item(self.plot.getPlotItem())
        self.plot.addLegend(offset=(8, 8))
        self.primary_curve = self.plot.plot(name="primary", pen=pg.mkPen("#c9d1d9", width=2))
        self.primary_bg_curve = self.plot.plot(name="primary bg-sub", pen=pg.mkPen("#d62728", width=2))
        self.compare_curve = self.plot.plot(name="compare", pen=pg.mkPen("#1f77b4", width=2))
        self.compare_bg_curve = self.plot.plot(name="compare bg-sub", pen=pg.mkPen("#ff7f0e", width=2))
        self.phase_curve = self.plot.plot(name="M1P", pen=pg.mkPen("#9467bd", width=2))
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self.controls_group, 0)
        previews = QtWidgets.QGridLayout()
        previews.setSpacing(6)
        previews.addWidget(self.primary_preview, 0, 0)
        previews.addWidget(self.compare_preview, 0, 1)
        previews.addWidget(self.mechanical_preview, 0, 2)
        for column in range(3):
            previews.setColumnStretch(column, 1)
        top.addLayout(previews, 1)
        layout.addLayout(top)
        layout.addWidget(self.plot, 1)


class DecompositionTab(QtWidgets.QWidget):
    def __init__(self, controls_group: QtWidgets.QGroupBox):
        super().__init__()
        self.controls_group = controls_group
        self.compute_button = controls_group.findChild(QtWidgets.QPushButton)
        self.category_map = ImagePlotWidget("Category map")
        self.category_map.setMaximumHeight(360)
        self.scree_plot = pg.PlotWidget(title="Scree")
        _style_plot_item(self.scree_plot.getPlotItem())
        self.scree_curve = self.scree_plot.plot(symbol="o", pen=pg.mkPen("#c9d1d9", width=2))
        self.category_spectra_plot = pg.PlotWidget(title="Category mean spectra")
        _style_plot_item(self.category_spectra_plot.getPlotItem())
        self.category_spectra_plot.addLegend(offset=(8, 8))
        self.centroids_plot = pg.PlotWidget(title="Cluster centroids")
        _style_plot_item(self.centroids_plot.getPlotItem())
        self.centroids_plot.addLegend(offset=(8, 8))
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(8)
        ctrl_col = QtWidgets.QWidget()
        ctrl_col.setFixedWidth(250)
        ctrl_vbox = QtWidgets.QVBoxLayout(ctrl_col)
        ctrl_vbox.setContentsMargins(0, 0, 0, 0)
        ctrl_vbox.setSpacing(0)
        ctrl_vbox.addWidget(self.controls_group)
        ctrl_vbox.addStretch(1)
        body.addWidget(ctrl_col)
        plots = QtWidgets.QGridLayout()
        plots.setSpacing(6)
        plots.addWidget(self.category_map, 0, 0)
        plots.addWidget(self.scree_plot, 0, 1)
        plots.addWidget(self.category_spectra_plot, 1, 0)
        plots.addWidget(self.centroids_plot, 1, 1)
        plots.setRowStretch(0, 1)
        plots.setRowStretch(1, 1)
        plots.setColumnStretch(0, 1)
        plots.setColumnStretch(1, 1)
        body.addLayout(plots, 1)
        layout.addLayout(body, 1)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, root_dir: Path = ROOT_DIR):
        super().__init__()
        self.setWindowTitle("SNOM Explorer")
        self._fit_window_to_screen()
        self.setMinimumSize(self._minimum_window_size)
        self._apply_theme()
        self.model = SnomAppModel(root_dir)
        self._loading_controls = False
        self.last_decomposition = None

        self.status_label = QtWidgets.QLabel("No scan loaded")
        self.selected_label = QtWidgets.QLabel("ix=-- iy=--")
        self.folder_combo = QtWidgets.QComboBox()
        self.file_combo = QtWidgets.QComboBox()
        self.roi_start_spin = QtWidgets.QSpinBox()
        self.roi_end_spin = QtWidgets.QSpinBox()
        self.detector_start_spin = QtWidgets.QSpinBox()
        self.detector_end_spin = QtWidgets.QSpinBox()
        self.row_start_spin = QtWidgets.QSpinBox()
        self.row_end_spin = QtWidgets.QSpinBox()
        self.harmonic_combo = QtWidgets.QComboBox()
        self.compare_combo = QtWidgets.QComboBox()
        self.target_freq_spin = QtWidgets.QDoubleSpinBox()
        self.neighbor_bins_spin = QtWidgets.QSpinBox()
        self.bg_low_spin = QtWidgets.QDoubleSpinBox()
        self.bg_high_spin = QtWidgets.QDoubleSpinBox()
        self.baseline_smooth_spin = QtWidgets.QSpinBox()
        self.background_neighbor_spin = QtWidgets.QSpinBox()
        self.avg3x3_check = QtWidgets.QCheckBox("3x3 average")
        self.fft_bgsub_check = QtWidgets.QCheckBox("FFT bg-sub")
        self.decomp_harmonic_combo = QtWidgets.QComboBox()
        self.decomp_method_combo = QtWidgets.QComboBox()
        self.decomp_components_spin = QtWidgets.QSpinBox()
        self.decomp_categorizer_combo = QtWidgets.QComboBox()
        self.decomp_clusters_spin = QtWidgets.QSpinBox()
        self.decomp_bgsub_check = QtWidgets.QCheckBox("Background subtraction")
        self.decomp_l2_check = QtWidgets.QCheckBox("L2 normalise")
        self.decomp_standardize_check = QtWidgets.QCheckBox("Standardise")
        self.decomp_normalize_spectra_check = QtWidgets.QCheckBox("Normalise spectra (0–1)")
        self.decomp_compute_btn = QtWidgets.QPushButton("Compute decomposition")
        self.export_format_combo = QtWidgets.QComboBox()
        self.export_btn = QtWidgets.QPushButton("Export images")

        self.tabs = QtWidgets.QTabWidget()
        self.explore_tab = MapsInspectorTab()
        self.maps_tab = self.explore_tab.maps
        self.inspector_tab = self.explore_tab.inspector
        self.line_profile_tab = LineProfileTab(self._build_line_profile_controls())
        self.decomposition_tab = DecompositionTab(self._build_decomposition_controls())
        self.metadata_text = QtWidgets.QPlainTextEdit()
        self.metadata_text.setReadOnly(True)
        self.tabs.addTab(self.explore_tab, "Maps + Inspector")
        self.tabs.addTab(self.line_profile_tab, "Line Profile")
        self.tabs.addTab(self.decomposition_tab, "Decomposition")
        self.tabs.addTab(self.metadata_text, "Metadata")

        splitter = QtWidgets.QSplitter()
        self.control_panel = self._build_control_panel()
        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.tabs)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([260, 1080])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._wire_signals()
        self._populate_static_controls()
        self.refresh_source_controls()

    def _fit_window_to_screen(self) -> None:
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            self._minimum_window_size = QtCore.QSize(760, 620)
            self.resize(1280, 820)
            return
        available = screen.availableGeometry()
        width = min(1360, max(760, int(available.width() * 0.94)))
        height = min(860, max(620, int(available.height() * 0.92)))
        self._minimum_window_size = QtCore.QSize(min(760, width), min(620, height))
        self.resize(width, height)

    def _apply_theme(self) -> None:
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont)
        font.setPointSize(11)
        self.setFont(font)
        pg.setConfigOptions(background="#0d1117", foreground="#8b949e", antialias=True)
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0d1117;
                color: #e6edf3;
                font-size: 11px;
            }
            QSplitter::handle {
                background: #21262d;
                width: 1px;
                height: 1px;
            }
            QTabWidget::pane {
                border: 1px solid #30363d;
                background: #0d1117;
            }
            QTabBar::tab {
                background: #161b22;
                border: 1px solid #30363d;
                border-bottom: none;
                padding: 5px 14px;
                min-width: 88px;
                color: #8b949e;
            }
            QTabBar::tab:selected {
                background: #0d1117;
                color: #e6edf3;
                font-weight: 600;
            }
            QTabBar::tab:hover:!selected {
                background: #1c2128;
                color: #c9d1d9;
            }
            QGroupBox {
                border: 1px solid #30363d;
                border-radius: 4px;
                margin-top: 10px;
                padding: 10px 6px 6px 6px;
                background: #161b22;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px;
                color: #8b949e;
                font-size: 10px;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            QLabel {
                color: #c9d1d9;
                background: transparent;
            }
            QScrollArea {
                background: #0d1117;
                border: none;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background: #0d1117;
                border: 1px solid #30363d;
                border-radius: 3px;
                min-height: 22px;
                padding: 1px 5px;
                color: #e6edf3;
            }
            QPlainTextEdit {
                background: #0d1117;
                border: 1px solid #30363d;
                border-radius: 3px;
                padding: 4px 6px;
                color: #e6edf3;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
                border: 1px solid #1f6feb;
            }
            QComboBox::drop-down {
                border: none;
                width: 18px;
            }
            QComboBox QAbstractItemView {
                background: #1c2128;
                border: 1px solid #30363d;
                color: #e6edf3;
                selection-background-color: #1f6feb;
                selection-color: #ffffff;
                outline: none;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background: #21262d;
                border: none;
                width: 14px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover,
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background: #30363d;
            }
            QPushButton {
                background: #21262d;
                border: 1px solid #30363d;
                border-radius: 4px;
                min-height: 24px;
                padding: 2px 10px;
                color: #e6edf3;
            }
            QPushButton:hover {
                background: #1c2128;
                border-color: #1f6feb;
                color: #ffffff;
            }
            QPushButton:pressed {
                background: #161b22;
            }
            QCheckBox {
                min-height: 22px;
                color: #c9d1d9;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #484f58;
                border-radius: 2px;
                background: #0d1117;
            }
            QCheckBox::indicator:checked {
                background: #1f6feb;
                border-color: #1f6feb;
            }
            QCheckBox::indicator:hover {
                border-color: #8b949e;
            }
            QScrollBar:vertical {
                background: #0d1117;
                width: 7px;
                border: none;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #30363d;
                border-radius: 3px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: #484f58;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #0d1117;
                height: 7px;
                border: none;
                margin: 0;
            }
            QScrollBar::handle:horizontal {
                background: #30363d;
                border-radius: 3px;
                min-width: 24px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #484f58;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            """
        )

    def _build_control_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(340)
        outer = QtWidgets.QVBoxLayout(panel)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        choose_root_btn = QtWidgets.QPushButton("Choose root")
        load_btn = QtWidgets.QPushButton("Load scan")
        recompute_btn = QtWidgets.QPushButton("Recompute")
        choose_root_btn.clicked.connect(self.choose_root)
        load_btn.clicked.connect(lambda: self.load_selected_scan(False))
        recompute_btn.clicked.connect(lambda: self.load_selected_scan(True))

        source_row = QtWidgets.QHBoxLayout()
        source_row.addWidget(load_btn)
        source_row.addWidget(recompute_btn)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        source_group, source_form = self._section_form("Source")
        self._add_row(source_form, "Root", choose_root_btn)
        self._add_row(source_form, "Folder", self.folder_combo)
        self._add_row(source_form, "Scan file", self.file_combo)
        source_form.addRow(source_row)
        self._add_row(source_form, "Status", self.status_label)
        self._add_row(source_form, "Selected", self.selected_label)

        demod_group, demod_form = self._section_form("Demodulation")
        self._add_row(demod_form, "ROI start", self.roi_start_spin)
        self._add_row(demod_form, "ROI end", self.roi_end_spin)
        self._add_row(demod_form, "Harmonic", self.harmonic_combo)
        self._add_row(demod_form, "Compare", self.compare_combo)
        self._add_row(demod_form, "Target Hz", self.target_freq_spin)
        self._add_row(demod_form, "Nbr bins", self.neighbor_bins_spin)
        demod_form.addRow(self.avg3x3_check)

        background_group, background_form = self._section_form("Background")
        self._add_row(background_form, "BG low Hz", self.bg_low_spin)
        self._add_row(background_form, "BG high Hz", self.bg_high_spin)
        self._add_row(background_form, "Baseline px", self.baseline_smooth_spin)
        self._add_row(background_form, "Nbr avg px", self.background_neighbor_spin)
        background_form.addRow(self.fft_bgsub_check)

        export_group, export_form = self._section_form("Export")
        self._add_row(export_form, "Format", self.export_format_combo)
        export_form.addRow(self.export_btn)

        for group in (source_group, demod_group, background_group, export_group):
            content_layout.addWidget(group)
        content_layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return panel

    def _build_line_profile_controls(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Line profile")
        form = QtWidgets.QFormLayout(group)
        form.setContentsMargins(8, 8, 8, 6)
        form.setSpacing(5)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._add_row(form, "Row start", self.row_start_spin)
        self._add_row(form, "Row end", self.row_end_spin)
        group.setMinimumWidth(220)
        group.setMaximumWidth(260)
        return group

    def _build_decomposition_controls(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Decomposition")
        form = QtWidgets.QFormLayout(group)
        form.setContentsMargins(8, 8, 8, 6)
        form.setSpacing(5)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._add_row(form, "Harmonic", self.decomp_harmonic_combo)
        self._add_row(form, "Method", self.decomp_method_combo)
        self._add_row(form, "Components", self.decomp_components_spin)
        self._add_row(form, "Categorizer", self.decomp_categorizer_combo)
        self._add_row(form, "Clusters", self.decomp_clusters_spin)
        self._add_row(form, "Det start", self.detector_start_spin)
        self._add_row(form, "Det end", self.detector_end_spin)
        form.addRow(self.decomp_bgsub_check)
        form.addRow(self.decomp_l2_check)
        form.addRow(self.decomp_standardize_check)
        form.addRow(self.decomp_normalize_spectra_check)
        form.addRow(self.decomp_compute_btn)
        return group

    def _add_row(self, form: QtWidgets.QFormLayout, label_text: str, widget: QtWidgets.QWidget) -> None:
        label = QtWidgets.QLabel(label_text)
        label.setMinimumWidth(CONTROL_LABEL_WIDTH)
        if isinstance(widget, (QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox)):
            widget.setMinimumWidth(CONTROL_FIELD_MIN_WIDTH)
        form.addRow(label, widget)

    def _section_form(self, title: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QFormLayout]:
        group = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(group)
        form.setContentsMargins(8, 8, 8, 6)
        form.setSpacing(5)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return group, form

    def _populate_static_controls(self) -> None:
        for combo in (self.harmonic_combo, self.compare_combo, self.decomp_harmonic_combo):
            combo.clear()
            for harmonic in HARMONICS:
                combo.addItem(DEMOD_LABELS[harmonic], harmonic)
        self.compare_combo.setCurrentIndex(1)
        self.decomp_harmonic_combo.setCurrentIndex(1)
        self.decomp_method_combo.addItems(["PCA", "MNF"])
        self.decomp_categorizer_combo.addItems(["kmeans", "gmm"])
        for spin in (self.roi_start_spin, self.roi_end_spin, self.detector_start_spin, self.detector_end_spin):
            spin.setRange(0, 999_999)
        for spin in (self.row_start_spin, self.row_end_spin):
            spin.setRange(0, 999_999)
        for spin in (self.neighbor_bins_spin,):
            spin.setRange(0, 100)
        self.decomp_components_spin.setRange(2, 256)
        self.decomp_components_spin.setValue(5)
        self.decomp_clusters_spin.setRange(2, 128)
        self.decomp_clusters_spin.setValue(4)
        for spin, value in ((self.bg_low_spin, BG_LOW_HZ), (self.bg_high_spin, BG_HIGH_HZ), (self.target_freq_spin, 4.0)):
            spin.setRange(0.0, 1e9)
            spin.setDecimals(6)
            spin.setValue(value)
        self.baseline_smooth_spin.setRange(1, 999)
        self.baseline_smooth_spin.setValue(1)
        self.background_neighbor_spin.setRange(1, 999)
        self.background_neighbor_spin.setSingleStep(2)
        self.background_neighbor_spin.setValue(1)
        self.avg3x3_check.setChecked(True)
        self.decomp_standardize_check.setChecked(True)
        self.decomp_normalize_spectra_check.setChecked(True)
        self.export_format_combo.addItems(["PNG", "SVG"])

    def _wire_signals(self) -> None:
        self.folder_combo.currentTextChanged.connect(self._folder_changed)
        self.decomp_compute_btn.clicked.connect(self.compute_decomposition)
        self.decomp_normalize_spectra_check.toggled.connect(self._redraw_decomposition_spectra)
        self.export_btn.clicked.connect(self.export_images)
        map_widgets = [
            self.maps_tab.primary_map,
            self.maps_tab.primary_bgsub_map,
            self.maps_tab.compare_map,
            self.maps_tab.compare_bgsub_map,
            self.maps_tab.m1a_map,
            self.maps_tab.m1p_map,
            self.line_profile_tab.primary_preview,
            self.line_profile_tab.compare_preview,
            self.line_profile_tab.mechanical_preview,
            self.decomposition_tab.category_map,
        ]
        for widget in map_widgets:
            widget.pixel_selected.connect(self.set_selected_pixel)
        for widget in [
            self.roi_start_spin,
            self.roi_end_spin,
            self.row_start_spin,
            self.row_end_spin,
            self.harmonic_combo,
            self.compare_combo,
            self.target_freq_spin,
            self.neighbor_bins_spin,
            self.bg_low_spin,
            self.bg_high_spin,
            self.baseline_smooth_spin,
            self.background_neighbor_spin,
            self.avg3x3_check,
            self.fft_bgsub_check,
        ]:
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.refresh_plots)
            elif hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self.refresh_plots)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self.refresh_plots)

    def refresh_source_controls(self) -> None:
        self._loading_controls = True
        self.folder_combo.clear()
        self.folder_combo.addItems(self.model.folder_options())
        self._folder_changed(self.folder_combo.currentText())
        self._loading_controls = False

    def _folder_changed(self, folder: str) -> None:
        current = self.file_combo.currentText()
        self.file_combo.clear()
        self.file_combo.addItems(self.model.file_options(folder))
        if current:
            index = self.file_combo.findText(current)
            if index >= 0:
                self.file_combo.setCurrentIndex(index)

    def choose_root(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose scan root", str(self.model.root_dir))
        if selected:
            self.model.set_root_dir(Path(selected))
            self.refresh_source_controls()

    def load_selected_scan(self, recompute: bool = False) -> None:
        folder = self.folder_combo.currentText()
        filename = self.file_combo.currentText()
        if not folder or not filename:
            self.status_label.setText("No scan loaded")
            return
        self.status_label.setText("Processing...")
        QtWidgets.QApplication.processEvents()
        try:
            summary = self.model.load_scan(folder, filename, recompute=recompute)
        except Exception as exc:
            self.status_label.setText(f"Load failed: {exc}")
            return
        self._reset_colorbar_locks()
        self._sync_controls_from_model()
        self.status_label.setText(f"{summary.status}: {summary.path.name}")
        self.metadata_text.setPlainText(json.dumps(summary.metadata, indent=2, sort_keys=True))
        self.refresh_plots()

    def _reset_colorbar_locks(self) -> None:
        for widget in [
            self.maps_tab.primary_map,
            self.maps_tab.primary_bgsub_map,
            self.maps_tab.compare_map,
            self.maps_tab.compare_bgsub_map,
            self.maps_tab.m1a_map,
            self.maps_tab.m1p_map,
            self.line_profile_tab.primary_preview,
            self.line_profile_tab.compare_preview,
            self.line_profile_tab.mechanical_preview,
            self.inspector_tab.fft_plot,
            self.decomposition_tab.category_map,
        ]:
            widget.reset_colorbar_lock()

    def _sync_controls_from_model(self) -> None:
        self._loading_controls = True
        det_max = self.model.detector_range[1]
        row_max = max(0, (self.model.summary.ny - 1) if self.model.summary else 0)
        for spin in (self.roi_start_spin, self.roi_end_spin, self.detector_start_spin, self.detector_end_spin):
            spin.setMaximum(det_max)
        self.roi_start_spin.setValue(self.model.roi_range[0])
        self.roi_end_spin.setValue(self.model.roi_range[1])
        self.detector_start_spin.setValue(self.model.detector_range[0])
        self.detector_end_spin.setValue(self.model.detector_range[1])
        self.row_start_spin.setMaximum(row_max)
        self.row_end_spin.setMaximum(row_max)
        self.row_start_spin.setValue(self.model.line_rows[0])
        self.row_end_spin.setValue(self.model.line_rows[1])
        self.target_freq_spin.setValue(self.model.target_frequency_hz)
        self._loading_controls = False

    def _current_settings(self) -> MapSettings:
        return MapSettings(
            harmonic=self.harmonic_combo.currentData() or "0w",
            compare_harmonic=self.compare_combo.currentData() or "1w",
            roi_range=(self.roi_start_spin.value(), self.roi_end_spin.value()),
            cmap="viridis",
            range_mode="auto",
            color_min=0.0,
            color_max=1.0,
            bg_low_hz=self.bg_low_spin.value(),
            bg_high_hz=self.bg_high_spin.value(),
            baseline_smooth_px=max(1, self.baseline_smooth_spin.value()),
            background_neighbor_px=max(1, self.background_neighbor_spin.value()),
            target_frequency_hz=self.target_freq_spin.value(),
            neighbor_bins=self.neighbor_bins_spin.value(),
            avg3x3=self.avg3x3_check.isChecked(),
            fft_bgsub=self.fft_bgsub_check.isChecked(),
        )

    def refresh_plots(self) -> None:
        if self._loading_controls or self.model.bundle is None:
            return
        settings = self._current_settings()
        maps = self.model.compute_maps(settings)
        selected = self.model.selected_pixel
        self.maps_tab.primary_map.set_image(maps["primary"], f"{settings.harmonic} map", cmap=settings.cmap, selected=selected)
        self.maps_tab.primary_bgsub_map.set_image(maps["primary_bgsub"], f"{settings.harmonic} bg-sub", cmap=settings.cmap, selected=selected)
        self.maps_tab.compare_map.set_image(maps["compare"], f"{settings.compare_harmonic} map", cmap=settings.cmap, selected=selected)
        self.maps_tab.compare_bgsub_map.set_image(maps["compare_bgsub"], f"{settings.compare_harmonic} bg-sub", cmap=settings.cmap, selected=selected)
        self.maps_tab.m1a_map.set_image(maps["m1a"], "M1A", cmap=settings.cmap, selected=selected)
        self.maps_tab.m1p_map.set_image(maps["m1p"], "M1P", selected=selected)
        self.refresh_inspector(settings)
        self.refresh_line_profile(settings, maps)

    def refresh_inspector(self, settings: MapSettings) -> None:
        data = self.model.compute_inspector(settings)
        ix, iy = self.model.selected_pixel
        self.selected_label.setText(f"ix={ix} iy={iy}")
        self.inspector_tab.roi_curve.setData(np.arange(len(data["roi_trace"])), data["roi_trace"])
        det_axis = data["det_axis"]
        self.inspector_tab.spectrum_curve.setData(det_axis, data["spectrum"])
        self.inspector_tab.spectrum_bgsub_curve.setData(det_axis, data["spectrum_bgsub"])
        self.inspector_tab.baseline_curve.setData(det_axis, data["baseline"])
        fft_data = data["fft"][1:, :]
        self.inspector_tab.fft_plot.set_image(
            fft_data,
            f"FFT ix={ix} iy={iy}",
            cmap="magma",
            selected=None,
        )
        f_axis = data["f_axis"]
        df = float(f_axis[1])
        n_det = fft_data.shape[1]
        f_start = float(f_axis[1]) - df / 2.0
        f_span = fft_data.shape[0] * df
        fft_plot = self.inspector_tab.fft_plot
        fft_plot.item.setRect(QtCore.QRectF(0.0, f_start, float(n_det), f_span))
        fft_plot.plot.setRange(xRange=(0, n_det), yRange=(f_start, f_start + f_span), padding=0.02)

    def refresh_line_profile(self, settings: MapSettings, maps: dict[str, np.ndarray] | None = None) -> None:
        if maps is None:
            maps = self.model.compute_maps(settings)
        rows = (self.row_start_spin.value(), self.row_end_spin.value())
        selected = self.model.selected_pixel
        self.line_profile_tab.primary_preview.set_image(maps["primary"], f"{settings.harmonic} map", cmap=settings.cmap, selected=selected)
        self.line_profile_tab.compare_preview.set_image(
            maps["compare"],
            f"{settings.compare_harmonic} map",
            cmap=settings.cmap,
            selected=selected,
        )
        self.line_profile_tab.mechanical_preview.set_image(maps["m1p"], "Mechanical M1P", selected=selected)
        for preview in (
            self.line_profile_tab.primary_preview,
            self.line_profile_tab.compare_preview,
            self.line_profile_tab.mechanical_preview,
        ):
            preview.set_row_region(rows)
        profile = self.model.compute_line_profile(settings, (self.row_start_spin.value(), self.row_end_spin.value()))
        x = profile["x"]
        self.line_profile_tab.primary_curve.setData(x, profile["primary"])
        self.line_profile_tab.primary_bg_curve.setData(x, profile["primary_bgsub"])
        self.line_profile_tab.compare_curve.setData(x, profile["compare"])
        self.line_profile_tab.compare_bg_curve.setData(x, profile["compare_bgsub"])
        self.line_profile_tab.phase_curve.setData(x, profile["m1p"])

    def set_selected_pixel(self, ix: int, iy: int) -> None:
        self.model.select_pixel(ix, iy)
        self.refresh_plots()

    def compute_decomposition(self) -> None:
        if self.model.bundle is None:
            return
        preprocess = []
        if self.decomp_bgsub_check.isChecked():
            preprocess.append("bgsub")
        if self.decomp_l2_check.isChecked():
            preprocess.append("l2norm")
        if self.decomp_standardize_check.isChecked():
            preprocess.append("standardize")
        settings = self._current_settings()
        result = self.model.compute_decomposition(
            harmonic=self.decomp_harmonic_combo.currentData() or "1w",
            method=self.decomp_method_combo.currentText() or "PCA",
            n_components=self.decomp_components_spin.value(),
            categorizer=self.decomp_categorizer_combo.currentText() or "kmeans",
            n_clusters=self.decomp_clusters_spin.value(),
            preprocess=preprocess,
            detector_range=(self.detector_start_spin.value(), self.detector_end_spin.value()),
            settings=settings,
        )
        self.last_decomposition = result
        n_cats = len(result.category_means)
        cat_colors = _CAT_PALETTE[np.arange(n_cats) % len(_CAT_PALETTE)]  # (n_cats, 4) uint8

        # Discrete blocked LUT: 256 entries divided into n_cats equal blocks
        cat_lut = np.zeros((256, 4), dtype=np.uint8)
        for i in range(n_cats):
            lo = i * 256 // n_cats
            hi = (i + 1) * 256 // n_cats if i < n_cats - 1 else 256
            cat_lut[lo:hi] = cat_colors[i]

        # ColorMap for colorbar (uint8 colors, N evenly-spaced stops)
        cat_cmap = pg.ColorMap(
            pos=np.linspace(0.0, 1.0, n_cats),
            color=cat_colors,
        )

        # Fixed levels: 0 → first category, n_cats-1 → last category
        cat_levels = (0.0, float(max(n_cats - 1, 1)))

        self.decomposition_tab.category_map.set_image(result.label_map, "Category map", levels=cat_levels, selected=self.model.selected_pixel)
        # setColorMap re-applies its own LUT to the image item; override after
        self.decomposition_tab.category_map.colorbar.setColorMap(cat_cmap)
        self.decomposition_tab.category_map.item.setLookupTable(cat_lut)
        self.decomposition_tab.category_map.colorbar.setLevels(cat_levels)
        self.decomposition_tab.category_map.colorbar_levels_locked = True

        self.decomposition_tab.scree_curve.setData(np.arange(1, len(result.scree_values) + 1), result.scree_values)
        self._redraw_decomposition_spectra()

    def _redraw_decomposition_spectra(self) -> None:
        result = self.last_decomposition
        if result is None:
            return
        n_cats = len(result.category_means)
        cat_colors = _CAT_PALETTE[np.arange(n_cats) % len(_CAT_PALETTE)]
        pens = [pg.mkPen(tuple(int(v) for v in cat_colors[i]), width=2) for i in range(n_cats)]
        normalize = self.decomp_normalize_spectra_check.isChecked()
        self.decomposition_tab.category_spectra_plot.clear()
        self.decomposition_tab.centroids_plot.clear()
        for idx, mean in enumerate(result.category_means):
            if normalize:
                lo, hi = np.nanmin(mean), np.nanmax(mean)
                y = (mean - lo) / (hi - lo) if hi > lo else np.zeros_like(mean)
            else:
                y = mean
            self.decomposition_tab.category_spectra_plot.plot(result.detector_axis, y, name=f"cat {idx}", pen=pens[idx])
        for idx, centroid in enumerate(result.centroids):
            self.decomposition_tab.centroids_plot.plot(np.arange(len(centroid)), centroid, name=f"cluster {idx}", pen=pens[idx])


    def _export_widget(self, widget: QtWidgets.QWidget, name: str, out_dir: Path, fmt: str) -> None:
        if fmt == "PNG":
            pixmap = widget.grab()
            pixmap.save(str(out_dir / f"{name}.png"), "PNG")
        else:
            if isinstance(widget, pg.GraphicsLayoutWidget):
                exporter = pg.exporters.SVGExporter(widget.ci)
            elif isinstance(widget, pg.PlotWidget):
                exporter = pg.exporters.SVGExporter(widget.getPlotItem())
            else:
                pixmap = widget.grab()
                pixmap.save(str(out_dir / f"{name}.png"), "PNG")
                return
            exporter.export(str(out_dir / f"{name}.svg"))

    def export_images(self) -> None:
        if self.model.summary is None:
            QtWidgets.QMessageBox.warning(self, "No scan loaded", "Load a scan before exporting.")
            return
        fmt = self.export_format_combo.currentText()
        source_path = self.model.summary.path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = source_path.parent / f"{source_path.stem}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        plots: list[tuple[QtWidgets.QWidget, str]] = [
            (self.maps_tab.primary_map, "map_primary"),
            (self.maps_tab.primary_bgsub_map, "map_primary_bgsub"),
            (self.maps_tab.compare_map, "map_compare"),
            (self.maps_tab.compare_bgsub_map, "map_compare_bgsub"),
            (self.maps_tab.m1a_map, "map_m1a"),
            (self.maps_tab.m1p_map, "map_m1p"),
            (self.inspector_tab.roi_plot, "inspector_roi_trace"),
            (self.inspector_tab.spectrum_plot, "inspector_spectrum"),
            (self.inspector_tab.fft_plot, "inspector_fft"),
            (self.line_profile_tab.primary_preview, "lineprofile_primary_preview"),
            (self.line_profile_tab.compare_preview, "lineprofile_compare_preview"),
            (self.line_profile_tab.mechanical_preview, "lineprofile_mechanical_preview"),
            (self.line_profile_tab.plot, "lineprofile_plot"),
            (self.decomposition_tab.category_map, "decomp_category_map"),
            (self.decomposition_tab.scree_plot, "decomp_scree"),
            (self.decomposition_tab.category_spectra_plot, "decomp_category_spectra"),
            (self.decomposition_tab.centroids_plot, "decomp_centroids"),
        ]

        errors: list[str] = []
        for widget, name in plots:
            try:
                self._export_widget(widget, name, out_dir, fmt)
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        settings = self._current_settings()
        ix, iy = self.model.selected_pixel
        preprocess: list[str] = []
        if self.decomp_bgsub_check.isChecked():
            preprocess.append("bgsub")
        if self.decomp_l2_check.isChecked():
            preprocess.append("l2norm")
        if self.decomp_standardize_check.isChecked():
            preprocess.append("standardize")

        export_metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source_file": str(source_path),
            "export_format": fmt,
            "settings": {
                "harmonic": settings.harmonic,
                "compare_harmonic": settings.compare_harmonic,
                "roi_range": list(settings.roi_range),
                "bg_low_hz": settings.bg_low_hz,
                "bg_high_hz": settings.bg_high_hz,
                "baseline_smooth_px": settings.baseline_smooth_px,
                "background_neighbor_px": settings.background_neighbor_px,
                "target_frequency_hz": settings.target_frequency_hz,
                "neighbor_bins": settings.neighbor_bins,
                "avg3x3": settings.avg3x3,
                "fft_bgsub": settings.fft_bgsub,
            },
            "selected_pixel": {"ix": ix, "iy": iy},
            "line_rows": [self.row_start_spin.value(), self.row_end_spin.value()],
            "decomposition": {
                "harmonic": self.decomp_harmonic_combo.currentData(),
                "method": self.decomp_method_combo.currentText(),
                "n_components": self.decomp_components_spin.value(),
                "categorizer": self.decomp_categorizer_combo.currentText(),
                "n_clusters": self.decomp_clusters_spin.value(),
                "preprocess": preprocess,
                "detector_range": [self.detector_start_spin.value(), self.detector_end_spin.value()],
            },
            "scan_metadata": self.model.summary.metadata,
        }

        with open(out_dir / "settings.json", "w") as f:
            json.dump(export_metadata, f, indent=2, sort_keys=True)

        msg = f"Exported {len(plots) - len(errors)}/{len(plots)} images to:\n{out_dir}"
        if errors:
            msg += f"\n\nFailed ({len(errors)}):\n" + "\n".join(errors[:5])
        QtWidgets.QMessageBox.information(self, "Export complete", msg)


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder="col-major")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
