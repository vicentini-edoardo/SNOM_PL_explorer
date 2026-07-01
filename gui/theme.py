"""Dark theme: application stylesheet and pyqtgraph styling helpers."""
from __future__ import annotations

import pyqtgraph as pg
from PyQt6 import QtGui, QtWidgets

STYLESHEET = """
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
    QPushButton:disabled {
        background: #161b22;
        border-color: #21262d;
        color: #484f58;
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
    QProgressBar {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 3px;
        text-align: center;
        color: #c9d1d9;
        min-height: 14px;
        max-height: 14px;
    }
    QProgressBar::chunk {
        background: #1f6feb;
        border-radius: 2px;
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


def apply_theme(widget: QtWidgets.QWidget) -> None:
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont)
    font.setPointSize(11)
    widget.setFont(font)
    pg.setConfigOptions(background="#0d1117", foreground="#8b949e", antialias=True)
    widget.setStyleSheet(STYLESHEET)


def style_plot_item(plot_item: pg.PlotItem) -> None:
    for axis_name in ("bottom", "left", "top", "right"):
        axis = plot_item.getAxis(axis_name)
        axis.setPen(pg.mkPen("#30363d", width=1))
        axis.setTextPen(pg.mkPen("#8b949e"))
    plot_item.showGrid(x=True, y=True, alpha=0.12)
