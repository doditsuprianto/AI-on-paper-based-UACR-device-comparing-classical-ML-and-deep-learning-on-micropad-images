"""Interactive ROI capture flow (used by the Diagnostic and Labelling tabs)."""
from PyQt5 import QtCore, QtWidgets

import config
import frame_utils as fu
import roi_capture


def capture_roi_interactive(parent, segmenter):
    """Grab a frame -> detect -> crop ROIs. Returns a RoiCapture or None (the message has already been shown)."""
    app = QtWidgets.QApplication
    app.setOverrideCursor(QtCore.Qt.WaitCursor)
    try:
        try:
            return roi_capture.capture_roi_pair(segmenter, config.RADIUS_REDUCTION_FACTOR)
        except roi_capture.DetectionError as exc:
            app.restoreOverrideCursor()
            if not fu.ask(parent, "Spot not detected",
                          f"{exc}\n\nUse the default positions? (not recommended — the result will be flagged as "
                          "low quality)"):
                return None
            app.setOverrideCursor(QtCore.Qt.WaitCursor)
            return roi_capture.capture_roi_pair(segmenter, config.RADIUS_REDUCTION_FACTOR, allow_fallback=True)
    except roi_capture.CaptureError as exc:
        fu.warn(parent, "Capture failed", str(exc))
    except Exception as exc:  # unexpected error (camera unplugged, out of memory, etc.)
        fu.warn(parent, "Capture failed", f"Unexpected error: {exc}")
    finally:
        app.restoreOverrideCursor()
    return None
