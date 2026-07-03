from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SNOM_PL_NO_SETTINGS", "1")

import pytest
import numpy as np
from PyQt6 import QtWidgets

from app import ImagePlotWidget, MainWindow
from test_app_model import _write_grid_scan


def _load_scan_and_wait(window: MainWindow, qtbot, recompute: bool = True) -> None:
    window.load_selected_scan(recompute=recompute)
    qtbot.waitUntil(lambda: not window.is_busy and window.model.bundle is not None, timeout=15000)


def _compute_decomposition_and_wait(window: MainWindow, qtbot) -> None:
    window.compute_decomposition()
    qtbot.waitUntil(lambda: not window.is_busy and window.last_decomposition is not None, timeout=15000)


@pytest.mark.usefixtures("qapp")
def test_main_window_launches_without_scan(qtbot, tmp_path):
    window = MainWindow(root_dir=tmp_path)
    qtbot.addWidget(window)

    assert window.windowTitle() == "SNOM Explorer"
    assert window.tabs.count() == 5
    assert window.tabs.tabText(0) == "Maps + Inspector"
    assert window.model.bundle is None
    assert window.status_label.text() == "No scan loaded"


@pytest.mark.usefixtures("qapp")
def test_main_window_populates_source_selectors(qtbot, tmp_path):
    _write_grid_scan(tmp_path / "mini.h5")

    window = MainWindow(root_dir=tmp_path)
    qtbot.addWidget(window)

    assert window.folder_combo.currentText() == "."
    assert window.file_combo.currentText() == "mini.h5"


@pytest.mark.usefixtures("qapp")
def test_loading_scan_initializes_controls_and_plots(qtbot, tmp_path):
    _write_grid_scan(tmp_path / "mini.h5")
    window = MainWindow(root_dir=tmp_path)
    qtbot.addWidget(window)

    _load_scan_and_wait(window, qtbot)

    assert window.model.bundle is not None
    assert window.roi_start_spin.value() == 1
    assert window.roi_end_spin.value() == 2
    assert window.detector_start_spin.value() == 0
    assert window.detector_end_spin.value() == 3
    assert window.maps_tab.primary_map.image is not None
    assert window.maps_tab.primary_map.colorbar is not None


@pytest.mark.usefixtures("qapp")
def test_left_control_panel_keeps_display_and_decomposition_options_out(qtbot, tmp_path):
    window = MainWindow(root_dir=tmp_path)
    qtbot.addWidget(window)

    group_titles = [group.title() for group in window.control_panel.findChildren(QtWidgets.QGroupBox)]

    assert "Source" in group_titles
    assert "Demodulation" in group_titles
    assert "Background" in group_titles
    assert "Display" not in group_titles
    assert "Decomposition" not in group_titles
    assert not hasattr(window, "range_mode_combo")
    assert not hasattr(window, "color_min_spin")
    assert not hasattr(window, "color_max_spin")
    assert not hasattr(window, "capture_limits_btn")


@pytest.mark.usefixtures("qapp")
def test_decomposition_options_live_in_decomposition_panel(qtbot, tmp_path):
    window = MainWindow(root_dir=tmp_path)
    qtbot.addWidget(window)

    controls = window.decomposition_tab.controls_group

    assert controls.layout().rowCount() > 0
    assert window.decomp_harmonic_combo.parent() is controls
    assert window.decomp_method_combo.parent() is controls
    assert window.decomp_components_spin.parent() is controls
    assert window.detector_start_spin.parent() is controls
    assert window.detector_end_spin.parent() is controls
    assert window.decomposition_tab.compute_button.parent() is controls
    assert window.decomposition_tab.category_map.maximumHeight() <= 360


@pytest.mark.usefixtures("qapp")
def test_line_profile_options_live_in_line_profile_panel(qtbot, tmp_path):
    window = MainWindow(root_dir=tmp_path)
    qtbot.addWidget(window)

    controls = window.line_profile_tab.controls_group
    group_titles = [group.title() for group in window.control_panel.findChildren(QtWidgets.QGroupBox)]

    assert "Line Profile" not in group_titles
    assert window.row_start_spin.parent() is controls
    assert window.row_end_spin.parent() is controls
    assert window.line_profile_tab.primary_preview.maximumHeight() <= 240


@pytest.mark.usefixtures("qapp")
def test_line_profile_previews_show_source_maps_and_selected_rows(qtbot, tmp_path):
    _write_grid_scan(tmp_path / "mini.h5")
    window = MainWindow(root_dir=tmp_path)
    qtbot.addWidget(window)

    _load_scan_and_wait(window, qtbot)
    window.row_start_spin.setValue(0)
    window.row_end_spin.setValue(1)
    window.refresh_plots()

    assert window.line_profile_tab.primary_preview.image is not None
    assert window.line_profile_tab.compare_preview.image is not None
    assert window.line_profile_tab.mechanical_preview.image is not None
    assert window.line_profile_tab.primary_preview.row_region.getRegion() == (0, 2)
    assert window.line_profile_tab.compare_preview.row_region.getRegion() == (0, 2)
    assert window.line_profile_tab.mechanical_preview.row_region.getRegion() == (0, 2)


@pytest.mark.usefixtures("qapp")
def test_period_tab_populates_maps_and_spectra(qtbot, tmp_path):
    _write_grid_scan(tmp_path / "mini.h5")
    window = MainWindow(root_dir=tmp_path)
    qtbot.addWidget(window)
    _load_scan_and_wait(window, qtbot)

    assert window.period_tab.max_map.image is not None
    assert window.period_tab.min_map.image is not None
    assert window.period_tab.diff_map.image is not None
    assert window.period_tab.max_curve.xData is not None


@pytest.mark.usefixtures("qapp")
def test_image_widget_uses_transposed_inverted_pyqtgraph_image(qtbot):
    widget = ImagePlotWidget("Map")
    qtbot.addWidget(widget)

    data = np.arange(6, dtype=float).reshape(2, 3)
    widget.set_image(data, "Map", cmap="viridis")

    assert widget.image is not None
    assert widget.item.image.shape == (3, 2)
    assert widget.plot.getViewBox().state["yInverted"] is True
    assert widget.plot.getViewBox().state["aspectLocked"] is not False
    assert widget.colorbar.isVisible()


@pytest.mark.usefixtures("qapp")
def test_image_widget_autoscales_until_colorbar_levels_are_changed(qtbot):
    widget = ImagePlotWidget("Map")
    qtbot.addWidget(widget)

    widget.set_image(np.array([[1.0, 2.0], [3.0, 4.0]]), "Map")
    assert widget.colorbar.levels() == (1.0, 4.0)

    widget.colorbar.setLevels((0.25, 0.75))
    widget._on_colorbar_levels_changed()
    assert widget.colorbar_levels_locked

    widget.set_image(np.array([[10.0, 20.0], [30.0, 40.0]]), "Map")
    assert widget.colorbar.levels() == (0.25, 0.75)


@pytest.mark.usefixtures("qapp")
def test_image_widget_uses_line_mode_for_single_axis_data(qtbot):
    widget = ImagePlotWidget("Line")
    qtbot.addWidget(widget)

    widget.set_image(np.array([[1.0, 2.0, 3.0]]), "Line")

    assert widget.line_item is not None
    assert widget.line_item.isVisible()
    assert not widget.item.isVisible()
    assert not widget.colorbar.isVisible()


@pytest.mark.usefixtures("qapp")
def test_map_pixel_signal_updates_selection_and_inspector(qtbot, tmp_path):
    _write_grid_scan(tmp_path / "mini.h5")
    window = MainWindow(root_dir=tmp_path)
    qtbot.addWidget(window)
    _load_scan_and_wait(window, qtbot)

    assert window.maps_tab.m1p_map.default_cmap == "CET-C1"
    window.maps_tab.primary_map.pixel_selected.emit(0, 1)

    assert window.model.selected_pixel == (0, 1)
    assert "ix=0 iy=1" in window.selected_label.text()
    assert window.inspector_tab.roi_curve.xData is not None


@pytest.mark.usefixtures("qapp")
def test_data_export_writes_npz_and_csv(qtbot, tmp_path):
    _write_grid_scan(tmp_path / "mini.h5")
    window = MainWindow(root_dir=tmp_path)
    qtbot.addWidget(window)
    _load_scan_and_wait(window, qtbot)

    out_dir = tmp_path / "export"
    files = window._export_data_files(out_dir)

    names = {path.name for path in files}
    assert names == {"maps.npz", "roi_trace.csv", "detector_spectrum.csv", "line_profile.csv", "settings.json"}
    with np.load(out_dir / "maps.npz") as maps:
        assert maps["primary"].shape == (window.model.summary.ny, window.model.summary.nx)
    header = (out_dir / "line_profile.csv").read_text().splitlines()[0]
    assert header == "x,primary,primary_bgsub,compare,compare_bgsub,M1P"


@pytest.mark.usefixtures("qapp")
def test_hover_crosshair_syncs_across_maps(qtbot, tmp_path):
    _write_grid_scan(tmp_path / "mini.h5")
    window = MainWindow(root_dir=tmp_path)
    qtbot.addWidget(window)
    _load_scan_and_wait(window, qtbot)

    window.maps_tab.primary_map.pixel_hovered.emit(1.5, 0.5)
    for widget in window.maps_tab.map_widgets:
        assert widget.crosshair_v.isVisible()
        assert widget.crosshair_v.value() == 1.5
        assert widget.crosshair_h.value() == 0.5

    window.maps_tab.primary_map.pixel_hovered.emit(float("nan"), float("nan"))
    for widget in window.maps_tab.map_widgets:
        assert not widget.crosshair_v.isVisible()


@pytest.mark.usefixtures("qapp")
def test_session_settings_roundtrip(qtbot, tmp_path, monkeypatch):
    monkeypatch.delenv("SNOM_PL_NO_SETTINGS", raising=False)
    monkeypatch.setenv("SNOM_PL_SETTINGS_FILE", str(tmp_path / "session.ini"))

    window = MainWindow(root_dir=tmp_path)
    qtbot.addWidget(window)
    window.bg_low_spin.setValue(12.5)
    window.harmonic_combo.setCurrentIndex(2)
    window.avg3x3_check.setChecked(False)
    window.decomp_clusters_spin.setValue(7)
    window._save_session()

    restored = MainWindow(root_dir=tmp_path)
    qtbot.addWidget(restored)

    assert restored.bg_low_spin.value() == 12.5
    assert restored.harmonic_combo.currentData() == "2w"
    assert restored.avg3x3_check.isChecked() is False
    assert restored.decomp_clusters_spin.value() == 7


@pytest.mark.usefixtures("qapp")
def test_decomposition_button_populates_category_plot(qtbot, tmp_path):
    _write_grid_scan(tmp_path / "mini.h5")
    window = MainWindow(root_dir=tmp_path)
    qtbot.addWidget(window)
    _load_scan_and_wait(window, qtbot)

    _compute_decomposition_and_wait(window, qtbot)

    assert window.last_decomposition is not None
    assert window.decomposition_tab.category_map.image is not None
