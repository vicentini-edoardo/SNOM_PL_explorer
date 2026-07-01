"""QThreadPool worker used to keep heavy computation off the GUI thread."""
from __future__ import annotations

from typing import Callable

from PyQt6 import QtCore


class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(float, str)


class Worker(QtCore.QRunnable):
    """Run *fn* on a pool thread and report the outcome through signals.

    When *wants_progress* is true, *fn* is called with an extra keyword
    argument ``progress_cb(fraction, message)`` that forwards to the
    ``progress`` signal (safe to call from the worker thread).
    """

    def __init__(self, fn: Callable, *args, wants_progress: bool = False, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.wants_progress = wants_progress
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            if self.wants_progress:
                self.kwargs["progress_cb"] = lambda fraction, message: self.signals.progress.emit(
                    float(fraction), str(message)
                )
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # noqa: BLE001 - forwarded to the GUI thread
            self.signals.error.emit(exc)
        else:
            self.signals.finished.emit(result)
