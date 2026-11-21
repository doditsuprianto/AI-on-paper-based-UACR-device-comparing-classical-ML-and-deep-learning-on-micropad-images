"""Background workers (QThread) that keep the GUI responsive while loading the model, classifying and building PDFs."""
import traceback

from PyQt5 import QtCore


class FunctionWorker(QtCore.QThread):
    """Run `fn(*args, **kwargs)` in a separate thread.

    Signals: succeeded(object) | failed(str) ; progress(int, str) when the function accepts a `progress` kwarg.
    """
    succeeded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, str)

    def __init__(self, fn, *args, with_progress=False, parent=None, **kwargs):
        super().__init__(parent)
        self._fn, self._args, self._kwargs = fn, args, kwargs
        if with_progress:
            self._kwargs["progress"] = lambda pct, msg: self.progress.emit(int(pct), str(msg))

    def run(self):
        try:
            self.succeeded.emit(self._fn(*self._args, **self._kwargs))
        except Exception as exc:  # message to the UI; full traceback to the console
            traceback.print_exc()
            self.failed.emit(str(exc) or exc.__class__.__name__)
