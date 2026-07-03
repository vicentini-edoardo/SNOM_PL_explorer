"""Main application window and entry point."""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from app_model import MapSettings, SnomAppModel
from gui.plotting import CAT_PALETTE, ImagePlotWidget
from gui.tabs import DecompositionTab, LineProfileTab, MapsInspectorTab, PeriodTab
from gui.theme import apply_theme
from gui.workers import Worker
from snom_pipeline import BG_HIGH_HZ, BG_LOW_HZ, HARMONICS, SNOM_CHANNELS

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
CONTROL_LABEL_WIDTH = 92
CONTROL_FIELD_MIN_WIDTH = 110
DEMOD_LABELS = {"0w": "0omega (DC)", "1w": "1omega", "2w": "2omega", "3w": "3omega"}
SETTINGS_DISABLE_ENV = "SNOM_PL_NO_SETTINGS"
SETTINGS_FILE_ENV = "SNOM_PL_SETTINGS_FILE"


def open_settings() -> QtCore.QSettings | None:
    """Session settings store, or None when persistence is disabled.

    SNOM_PL_NO_SETTINGS=1 disables persistence entirely (used by tests);
    SNOM_PL_SETTINGS_FILE overrides the storage location with an ini file.
    """
    if os.environ.get(SETTINGS_DISABLE_ENV):
        return None
    custom_path = os.environ.get(SETTINGS_FILE_ENV)
    if custom_path:
        return QtCore.QSettings(custom_path, QtCore.QSettings.Format.IniFormat)
    return QtCore.QSettings("SNOM_PL_explorer", "SNOM Explorer")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, root_dir: Path = ROOT_DIR):
        super().__init__()
        self.setWindowTitle("SNOM Explorer")
        self._fit_window_to_screen()
        self.setMinimumSize(self._minimum_window_size)
        apply_theme(self)
        self.model = SnomAppModel(root_dir)
        self._loading_controls = False
        self.last_decomposition = None
        self._thread_pool = QtCore.QThreadPool.globalInstance()
        self._busy = False

        self.status_label = QtWidgets.QLabel("No scan loaded")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        self.selected_label = QtWidgets.QLabel("ix=-- iy=--")
        self.folder_combo = QtWidgets.QComboBox()
        self.file_combo = QtWidgets.QComboBox()
        self.roi_start_spin = QtWidgets.QSpinBox()
        self.roi_end_spin = QtWidgets.QSpinBox()
        self.detector_start_spin = QtWidgets.QSpinBox()
        self.detector_end_spin = QtWidgets.QSpinBox()
        self.row_start_spin = QtWidgets.QSpinBox()
        self.row_end_spin = QtWidgets.QSpinBox()
        self.mechanical_combo = QtWidgets.QComboBox()
        self.period_window_spin = QtWidgets.QSpinBox()
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
        self.export_data_btn = QtWidgets.QPushButton("Export data (CSV + NPZ)")

        self.tabs = QtWidgets.QTabWidget()
        self.explore_tab = MapsInspectorTab()
        self.maps_tab = self.explore_tab.maps
        self.inspector_tab = self.explore_tab.inspector
        self.line_profile_tab = LineProfileTab(self._build_line_profile_controls())
        self.decomposition_tab = DecompositionTab(self._build_decomposition_controls())
        self.period_tab = PeriodTab(self._build_period_controls())
        self.metadata_text = QtWidgets.QPlainTextEdit()
        self.metadata_text.setReadOnly(True)
        self.tabs.addTab(self.explore_tab, "Maps + Inspector")
        self.tabs.addTab(self.line_profile_tab, "Line Profile")
        self.tabs.addTab(self.decomposition_tab, "Decomposition")
        self.tabs.addTab(self.period_tab, "Period Max/Min")
        self.tabs.addTab(self.metadata_text, "Metadata")

        self.main_splitter = QtWidgets.QSplitter()
        self.control_panel = self._build_control_panel()
        self.main_splitter.addWidget(self.control_panel)
        self.main_splitter.addWidget(self.tabs)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setSizes([260, 1080])
        self.main_splitter.setStretchFactor(1, 1)
        self.setCentralWidget(self.main_splitter)

        self._wire_signals()
        self._populate_static_controls()
        self.refresh_source_controls()
        self._restore_session(root_dir)

    def _restore_session(self, root_dir: Path) -> None:
        settings = open_settings()
        if settings is None:
            return
        geometry = settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        for key, splitter in (
            ("window/main_splitter", self.main_splitter),
            ("window/explore_splitter", self.explore_tab.splitter),
        ):
            state = settings.value(key)
            if state is not None:
                splitter.restoreState(state)
        if Path(root_dir) == ROOT_DIR:
            saved_root = settings.value("session/root_dir", "", str)
            if saved_root and Path(saved_root).is_dir():
                self.model.set_root_dir(Path(saved_root))
                self.refresh_source_controls()
        self._loading_controls = True
        try:
            for key, combo in (
                ("session/folder", self.folder_combo),
                ("session/file", self.file_combo),
                ("params/export_format", self.export_format_combo),
                ("decomp/method", self.decomp_method_combo),
                ("decomp/categorizer", self.decomp_categorizer_combo),
            ):
                index = combo.findText(settings.value(key, "", str))
                if index >= 0:
                    combo.setCurrentIndex(index)
            for key, combo in (
                ("params/harmonic", self.harmonic_combo),
                ("params/compare_harmonic", self.compare_combo),
                ("decomp/harmonic", self.decomp_harmonic_combo),
            ):
                index = combo.findData(settings.value(key, "", str))
                if index >= 0:
                    combo.setCurrentIndex(index)
            for key, spin in self._persisted_spins():
                value = settings.value(key)
                if value is not None:
                    spin.setValue(type(spin.value())(value))
            for key, check in self._persisted_checks():
                check.setChecked(settings.value(key, check.isChecked(), bool))
        finally:
            self._loading_controls = False

    def _save_session(self) -> None:
        settings = open_settings()
        if settings is None:
            return
        settings.setValue("window/geometry", self.saveGeometry())
        settings.setValue("window/main_splitter", self.main_splitter.saveState())
        settings.setValue("window/explore_splitter", self.explore_tab.splitter.saveState())
        settings.setValue("session/root_dir", str(self.model.root_dir))
        settings.setValue("session/folder", self.folder_combo.currentText())
        settings.setValue("session/file", self.file_combo.currentText())
        settings.setValue("params/export_format", self.export_format_combo.currentText())
        settings.setValue("decomp/method", self.decomp_method_combo.currentText())
        settings.setValue("decomp/categorizer", self.decomp_categorizer_combo.currentText())
        settings.setValue("params/harmonic", self.harmonic_combo.currentData())
        settings.setValue("params/compare_harmonic", self.compare_combo.currentData())
        settings.setValue("decomp/harmonic", self.decomp_harmonic_combo.currentData())
        for key, spin in self._persisted_spins():
            settings.setValue(key, spin.value())
        for key, check in self._persisted_checks():
            settings.setValue(key, check.isChecked())
        settings.sync()

    def _persisted_spins(self) -> list[tuple[str, QtWidgets.QAbstractSpinBox]]:
        return [
            ("params/neighbor_bins", self.neighbor_bins_spin),
            ("params/bg_low_hz", self.bg_low_spin),
            ("params/bg_high_hz", self.bg_high_spin),
            ("params/baseline_smooth_px", self.baseline_smooth_spin),
            ("params/background_neighbor_px", self.background_neighbor_spin),
            ("decomp/components", self.decomp_components_spin),
            ("decomp/clusters", self.decomp_clusters_spin),
            ("period/window", self.period_window_spin),
        ]

    def _persisted_checks(self) -> list[tuple[str, QtWidgets.QCheckBox]]:
        return [
            ("params/avg3x3", self.avg3x3_check),
            ("params/fft_bgsub", self.fft_bgsub_check),
            ("decomp/bgsub", self.decomp_bgsub_check),
            ("decomp/l2norm", self.decomp_l2_check),
            ("decomp/standardize", self.decomp_standardize_check),
            ("decomp/normalize_spectra", self.decomp_normalize_spectra_check),
        ]

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_session()
        super().closeEvent(event)

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

    def _build_control_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(340)
        outer = QtWidgets.QVBoxLayout(panel)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        choose_root_btn = QtWidgets.QPushButton("Choose root")
        self.load_btn = QtWidgets.QPushButton("Load scan")
        self.recompute_btn = QtWidgets.QPushButton("Recompute")
        choose_root_btn.clicked.connect(self.choose_root)
        self.load_btn.clicked.connect(lambda: self.load_selected_scan(False))
        self.recompute_btn.clicked.connect(lambda: self.load_selected_scan(True))

        source_row = QtWidgets.QHBoxLayout()
        source_row.addWidget(self.load_btn)
        source_row.addWidget(self.recompute_btn)

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
        source_form.addRow(self.progress_bar)
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
        export_form.addRow(self.export_data_btn)

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
        self._add_row(form, "Signal", self.mechanical_combo)
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

    def _build_period_controls(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Period max/min")
        form = QtWidgets.QFormLayout(group)
        form.setContentsMargins(8, 8, 8, 6)
        form.setSpacing(5)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._add_row(form, "Peak window ±", self.period_window_spin)
        group.setMinimumWidth(220)
        group.setMaximumWidth(260)
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
        self.mechanical_combo.addItems(list(SNOM_CHANNELS))
        self.mechanical_combo.setCurrentText("M1P")
        self.decomp_method_combo.addItems(["PCA", "MNF"])
        self.decomp_categorizer_combo.addItems(["kmeans", "gmm"])
        for spin in (self.roi_start_spin, self.roi_end_spin, self.detector_start_spin, self.detector_end_spin):
            spin.setRange(0, 999_999)
        for spin in (self.row_start_spin, self.row_end_spin):
            spin.setRange(0, 999_999)
        for spin in (self.neighbor_bins_spin,):
            spin.setRange(0, 100)
        self.period_window_spin.setRange(0, 20)
        self.period_window_spin.setValue(1)
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
        self.export_data_btn.clicked.connect(self.export_data)
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
            self.period_tab.max_map,
            self.period_tab.min_map,
            self.period_tab.diff_map,
        ]
        for widget in map_widgets:
            widget.pixel_selected.connect(self.set_selected_pixel)
        for widget in [
            self.roi_start_spin,
            self.roi_end_spin,
            self.row_start_spin,
            self.row_end_spin,
            self.mechanical_combo,
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
            self.period_window_spin,
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

    @property
    def is_busy(self) -> bool:
        return self._busy

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self._busy = busy
        for button in (self.load_btn, self.recompute_btn, self.decomp_compute_btn, self.export_btn, self.export_data_btn):
            button.setEnabled(not busy)
        if busy:
            self.progress_bar.setRange(0, 0)  # indeterminate until first progress report
            self.progress_bar.show()
            if message:
                self.status_label.setText(message)
        else:
            self.progress_bar.hide()

    def _on_worker_progress(self, fraction: float, message: str) -> None:
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int(fraction * 100))

    def _start_worker(self, worker: Worker, on_finished, on_error) -> None:
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)
        self._thread_pool.start(worker)

    def load_selected_scan(self, recompute: bool = False) -> None:
        if self._busy:
            return
        folder = self.folder_combo.currentText()
        filename = self.file_combo.currentText()
        if not folder or not filename:
            self.status_label.setText("No scan loaded")
            return
        logger.info("Loading scan %s/%s (recompute=%s)", folder, filename, recompute)
        self._set_busy(True, "Processing...")
        worker = Worker(self.model.load_scan, folder, filename, recompute=recompute, wants_progress=True)
        self._start_worker(worker, self._on_scan_loaded, self._on_scan_failed)

    def _on_scan_loaded(self, summary) -> None:
        self._set_busy(False)
        self._reset_colorbar_locks()
        self._sync_controls_from_model()
        self.status_label.setText(f"{summary.status}: {summary.path.name}")
        self.metadata_text.setPlainText(json.dumps(summary.metadata, indent=2, sort_keys=True))
        self.refresh_plots()

    def _on_scan_failed(self, exc: Exception) -> None:
        self._set_busy(False)
        logger.error("Scan load failed", exc_info=exc)
        self.status_label.setText("Load failed")
        QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))

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
            mechanical_channel=self.mechanical_combo.currentText() or "M1P",
            period_window=self.period_window_spin.value(),
        )

    def refresh_plots(self) -> None:
        if self._loading_controls or self._busy or self.model.bundle is None:
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
        self.refresh_period(settings)

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
        self.line_profile_tab.mechanical_preview.set_image(
            maps["mechanical"], f"Mechanical {settings.mechanical_channel}", selected=selected
        )
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
        self.line_profile_tab.mechanical_curve.setData(x, profile["mechanical"])
        plot_item = self.line_profile_tab.plot.getPlotItem()
        legend = plot_item.legend
        legend.removeItem(self.line_profile_tab.mechanical_curve)
        legend.addItem(self.line_profile_tab.mechanical_curve, settings.mechanical_channel)
        plot_item.getAxis("right").setLabel(settings.mechanical_channel)

    def refresh_period(self, settings: MapSettings) -> None:
        maps = self.model.compute_period_maps(settings)
        selected = self.model.selected_pixel
        self.period_tab.max_map.set_image(maps["max"], "Period max", selected=selected)
        self.period_tab.min_map.set_image(maps["min"], "Period min", selected=selected)
        self.period_tab.diff_map.set_image(maps["diff"], "Period max-min", selected=selected)
        spectra = self.model.compute_period_spectra(settings)
        det_axis = spectra["det_axis"]
        self.period_tab.max_curve.setData(det_axis, spectra["max"])
        self.period_tab.min_curve.setData(det_axis, spectra["min"])
        self.period_tab.diff_curve.setData(det_axis, spectra["diff"])

        trace = self.model.compute_period_trace(settings)
        self.period_tab.trace_curve.setData(trace["x"], trace["trace"])
        self.period_tab.trace_max_scatter.setData(trace["x"][trace["max_mask"]], trace["trace"][trace["max_mask"]])
        self.period_tab.trace_min_scatter.setData(trace["x"][trace["min_mask"]], trace["trace"][trace["min_mask"]])

    def set_selected_pixel(self, ix: int, iy: int) -> None:
        self.model.select_pixel(ix, iy)
        self.refresh_plots()

    def compute_decomposition(self) -> None:
        if self.model.bundle is None or self._busy:
            return
        preprocess = []
        if self.decomp_bgsub_check.isChecked():
            preprocess.append("bgsub")
        if self.decomp_l2_check.isChecked():
            preprocess.append("l2norm")
        if self.decomp_standardize_check.isChecked():
            preprocess.append("standardize")
        settings = self._current_settings()
        self._set_busy(True, "Decomposing...")
        worker = Worker(
            self.model.compute_decomposition,
            harmonic=self.decomp_harmonic_combo.currentData() or "1w",
            method=self.decomp_method_combo.currentText() or "PCA",
            n_components=self.decomp_components_spin.value(),
            categorizer=self.decomp_categorizer_combo.currentText() or "kmeans",
            n_clusters=self.decomp_clusters_spin.value(),
            preprocess=preprocess,
            detector_range=(self.detector_start_spin.value(), self.detector_end_spin.value()),
            settings=settings,
        )
        self._start_worker(worker, self._on_decomposition_done, self._on_decomposition_failed)

    def _on_decomposition_done(self, result) -> None:
        self._set_busy(False)
        if self.model.summary is not None:
            self.status_label.setText(f"Decomposition done: {self.model.summary.path.name}")
        self._show_decomposition(result)

    def _on_decomposition_failed(self, exc: Exception) -> None:
        self._set_busy(False)
        logger.error("Decomposition failed", exc_info=exc)
        self.status_label.setText("Decomposition failed")
        QtWidgets.QMessageBox.critical(self, "Decomposition failed", str(exc))

    def _show_decomposition(self, result) -> None:
        self.last_decomposition = result
        n_cats = len(result.category_means)
        cat_colors = CAT_PALETTE[np.arange(n_cats) % len(CAT_PALETTE)]  # (n_cats, 4) uint8

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
        cat_colors = CAT_PALETTE[np.arange(n_cats) % len(CAT_PALETTE)]
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

        with open(out_dir / "settings.json", "w") as f:
            json.dump(self._export_metadata(fmt), f, indent=2, sort_keys=True)

        msg = f"Exported {len(plots) - len(errors)}/{len(plots)} images to:\n{out_dir}"
        if errors:
            logger.warning("Image export finished with %d failures: %s", len(errors), errors)
            msg += f"\n\nFailed ({len(errors)}):\n" + "\n".join(errors[:5])
        QtWidgets.QMessageBox.information(self, "Export complete", msg)

    def _export_metadata(self, fmt: str) -> dict:
        settings = self._current_settings()
        ix, iy = self.model.selected_pixel
        preprocess: list[str] = []
        if self.decomp_bgsub_check.isChecked():
            preprocess.append("bgsub")
        if self.decomp_l2_check.isChecked():
            preprocess.append("l2norm")
        if self.decomp_standardize_check.isChecked():
            preprocess.append("standardize")

        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "source_file": str(self.model.summary.path),
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

    def export_data(self) -> None:
        if self.model.summary is None or self.model.bundle is None:
            QtWidgets.QMessageBox.warning(self, "No scan loaded", "Load a scan before exporting.")
            return
        source_path = self.model.summary.path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = source_path.parent / f"{source_path.stem}_data_{timestamp}"
        try:
            files = self._export_data_files(out_dir)
        except Exception as exc:
            logger.error("Data export failed", exc_info=exc)
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        QtWidgets.QMessageBox.information(
            self, "Export complete", f"Wrote {len(files)} files to:\n{out_dir}"
        )

    def _export_data_files(self, out_dir: Path) -> list[Path]:
        """Write the current maps, inspector curves and line profile as NPZ/CSV."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        settings = self._current_settings()
        files: list[Path] = []

        maps = self.model.compute_maps(settings)
        arrays = dict(maps)
        if self.last_decomposition is not None:
            arrays["category_labels"] = self.last_decomposition.label_map
        maps_path = out_dir / "maps.npz"
        np.savez_compressed(maps_path, **arrays)
        files.append(maps_path)

        inspector = self.model.compute_inspector(settings)
        roi = np.asarray(inspector["roi_trace"], dtype=np.float64)
        roi_path = out_dir / "roi_trace.csv"
        np.savetxt(
            roi_path,
            np.column_stack([np.arange(roi.size), roi]),
            delimiter=",",
            header="sample,value",
            comments="",
        )
        files.append(roi_path)

        spectrum_path = out_dir / "detector_spectrum.csv"
        np.savetxt(
            spectrum_path,
            np.column_stack(
                [
                    inspector["det_axis"],
                    inspector["spectrum"],
                    inspector["spectrum_bgsub"],
                    inspector["baseline"],
                ]
            ),
            delimiter=",",
            header="detector_px,spectrum,spectrum_bgsub,baseline",
            comments="",
        )
        files.append(spectrum_path)

        profile = self.model.compute_line_profile(
            settings, (self.row_start_spin.value(), self.row_end_spin.value())
        )
        profile_path = out_dir / "line_profile.csv"
        np.savetxt(
            profile_path,
            np.column_stack(
                [
                    profile["x"],
                    profile["primary"],
                    profile["primary_bgsub"],
                    profile["compare"],
                    profile["compare_bgsub"],
                    profile["mechanical"],
                ]
            ),
            delimiter=",",
            header=f"x,primary,primary_bgsub,compare,compare_bgsub,{settings.mechanical_channel}",
            comments="",
        )
        files.append(profile_path)

        settings_path = out_dir / "settings.json"
        with open(settings_path, "w") as f:
            json.dump(self._export_metadata("DATA"), f, indent=2, sort_keys=True)
        files.append(settings_path)
        logger.info("Exported %d data files to %s", len(files), out_dir)
        return files


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder="col-major")
    window = MainWindow()
    window.show()
    return app.exec()
