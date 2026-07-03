"""Tab widgets composing the main window's pages."""
from __future__ import annotations

import math

import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from gui.plotting import ImagePlotWidget, PHASE_COLORMAP
from gui.theme import style_plot_item


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
        self.map_widgets = [
            self.primary_map,
            self.primary_bgsub_map,
            self.compare_map,
            self.compare_bgsub_map,
            self.m1a_map,
            self.m1p_map,
        ]
        for index, widget in enumerate(self.map_widgets):
            layout.addWidget(widget, index // 2, index % 2)
            layout.setRowStretch(index // 2, 1)
            layout.setColumnStretch(index % 2, 1)
            widget.pixel_hovered.connect(self._sync_crosshairs)

    def _sync_crosshairs(self, x: float, y: float) -> None:
        hide = math.isnan(x)
        for widget in self.map_widgets:
            widget.set_crosshair(None if hide else x, y)


class InspectorTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.roi_plot = pg.PlotWidget(title="ROI trace")
        self.roi_plot.setMinimumHeight(90)
        style_plot_item(self.roi_plot.getPlotItem())
        self.roi_curve = self.roi_plot.plot(pen=pg.mkPen("#1f77b4", width=2))
        self.spectrum_plot = pg.PlotWidget(title="Detector spectrum")
        self.spectrum_plot.setMinimumHeight(140)
        style_plot_item(self.spectrum_plot.getPlotItem())
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

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.maps)
        self.splitter.addWidget(self.inspector)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([1020, 300])

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)


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
        plot_item = self.plot.getPlotItem()
        style_plot_item(plot_item)
        self.plot.addLegend(offset=(8, 8))
        self.primary_curve = self.plot.plot(name="primary", pen=pg.mkPen("#c9d1d9", width=2))
        self.primary_bg_curve = self.plot.plot(name="primary bg-sub", pen=pg.mkPen("#d62728", width=2))
        self.compare_curve = self.plot.plot(name="compare", pen=pg.mkPen("#1f77b4", width=2))
        self.compare_bg_curve = self.plot.plot(name="compare bg-sub", pen=pg.mkPen("#ff7f0e", width=2))

        # Mechanical (neaSNOM) signal gets its own Y axis: units/scale differ
        # wildly from the optical curves above, so a shared axis flattens it.
        plot_item.showAxis("right")
        self.mechanical_viewbox = pg.ViewBox()
        plot_item.scene().addItem(self.mechanical_viewbox)
        plot_item.getAxis("right").linkToView(self.mechanical_viewbox)
        self.mechanical_viewbox.setXLink(plot_item)
        self.mechanical_viewbox.enableAutoRange(axis=pg.ViewBox.YAxis)

        def _sync_mechanical_viewbox() -> None:
            self.mechanical_viewbox.setGeometry(plot_item.vb.sceneBoundingRect())

        plot_item.vb.sigResized.connect(_sync_mechanical_viewbox)
        _sync_mechanical_viewbox()
        self.mechanical_curve = pg.PlotDataItem(pen=pg.mkPen("#9467bd", width=2))
        self.mechanical_viewbox.addItem(self.mechanical_curve)
        plot_item.legend.addItem(self.mechanical_curve, "M1P")

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
        style_plot_item(self.scree_plot.getPlotItem())
        self.scree_curve = self.scree_plot.plot(symbol="o", pen=pg.mkPen("#c9d1d9", width=2))
        self.category_spectra_plot = pg.PlotWidget(title="Category mean spectra")
        style_plot_item(self.category_spectra_plot.getPlotItem())
        self.category_spectra_plot.addLegend(offset=(8, 8))
        self.centroids_plot = pg.PlotWidget(title="Cluster centroids")
        style_plot_item(self.centroids_plot.getPlotItem())
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


class PeriodTab(QtWidgets.QWidget):
    def __init__(self, controls_group: QtWidgets.QGroupBox):
        super().__init__()
        self.controls_group = controls_group
        self.max_map = ImagePlotWidget("Period max")
        self.min_map = ImagePlotWidget("Period min")
        self.diff_map = ImagePlotWidget("Period max-min")
        self.spectrum_plot = pg.PlotWidget(title="Spectrum at cursor")
        spectrum_item = self.spectrum_plot.getPlotItem()
        style_plot_item(spectrum_item)
        self.spectrum_plot.addLegend(offset=(8, 8))
        self.max_curve = self.spectrum_plot.plot(name="max", pen=pg.mkPen("#d62728", width=2))
        self.min_curve = self.spectrum_plot.plot(name="min", pen=pg.mkPen("#1f77b4", width=2))

        # max-min diff gets its own Y axis: its scale can differ a lot from
        # the raw max/min curves, so a shared axis flattens it.
        spectrum_item.showAxis("right")
        self.diff_viewbox = pg.ViewBox()
        spectrum_item.scene().addItem(self.diff_viewbox)
        spectrum_item.getAxis("right").linkToView(self.diff_viewbox)
        self.diff_viewbox.setXLink(spectrum_item)
        self.diff_viewbox.enableAutoRange(axis=pg.ViewBox.YAxis)

        def _sync_diff_viewbox() -> None:
            self.diff_viewbox.setGeometry(spectrum_item.vb.sceneBoundingRect())

        spectrum_item.vb.sigResized.connect(_sync_diff_viewbox)
        _sync_diff_viewbox()
        self.diff_curve = pg.PlotDataItem(pen=pg.mkPen("#c9d1d9", width=2))
        self.diff_viewbox.addItem(self.diff_curve)
        spectrum_item.legend.addItem(self.diff_curve, "max-min")
        spectrum_item.getAxis("right").setLabel("max-min")

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
        plots.addWidget(self.max_map, 0, 0)
        plots.addWidget(self.min_map, 0, 1)
        plots.addWidget(self.diff_map, 1, 0)
        plots.addWidget(self.spectrum_plot, 1, 1)
        plots.setRowStretch(0, 1)
        plots.setRowStretch(1, 1)
        plots.setColumnStretch(0, 1)
        plots.setColumnStretch(1, 1)
        body.addLayout(plots, 1)
        layout.addLayout(body, 1)
