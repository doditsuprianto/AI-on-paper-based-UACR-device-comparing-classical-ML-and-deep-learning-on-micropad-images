"""Camera acquisition + circular spot detection (YOLO) for the micropads.

Important differences with respect to the first version:
  * The camera backend is chosen according to the OS (V4L2 on Linux/Armbian; DSHOW/MSMF only on Windows).
  * `start()` returns False when the camera cannot be opened, so the application really tries the next camera
    index. Test mode (synthetic frames) is enabled explicitly through `start_test_mode()`.
  * There is no mutex deadlock any more when `start()` calls `stop()`.
  * `ultralytics` is imported safely; without the model the application still runs (without detection).
The ROI cropping logic lives in roi_capture.py.
"""
import sys

import cv2
import numpy as np
from PyQt5.QtCore import QMutex, QMutexLocker, QObject, QTimer, pyqtSignal

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - depends on the environment
    YOLO = None

PREVIEW_SIZE = (440, 241)      # size of the preview frame passed to YOLO (w, h)
PREVIEW_INTERVAL_MS = 100


def _camera_backends():
    if sys.platform.startswith("linux"):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


class CircleSegmenter(QObject):
    frame_ready = pyqtSignal(object)  # BGR preview frame (detection circles already drawn)

    def __init__(self, model_path="model/best.pt"):
        super().__init__()
        self.model = None
        if YOLO is None:
            print("Warning: the 'ultralytics' package is not installed; spot detection is disabled.")
        else:
            try:
                self.model = YOLO(str(model_path))
                print(f"Segmentation model loaded: {model_path}")
            except Exception as exc:
                print(f"Warning: the segmentation model could not be loaded ({exc})")
        self.running = False
        self.capture = None
        self.camera_index = None
        self.test_mode = False
        self.current_boxes = []
        self._mutex = QMutex()
        self._timer = QTimer(self)
        self._timer.setInterval(PREVIEW_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._test_count = 0
        self._test_frame = self._make_test_frame()

    # ------------------------------------------------------------------ camera
    def start(self, camera_index=0):
        """Open camera `camera_index` (up to 4K resolution). Returns True on success."""
        if self.running:
            self.stop()
        with QMutexLocker(self._mutex):
            cap = None
            for backend in _camera_backends():
                cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    break
                cap.release()
                cap = None
            if cap is None:
                print(f"Camera index {camera_index} could not be opened.")
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            cap.set(cv2.CAP_PROP_EXPOSURE, -4)
            print(f"Camera {camera_index}: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
                  f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ {cap.get(cv2.CAP_PROP_FPS):.0f} FPS")
            self.capture, self.camera_index = cap, camera_index
            self.test_mode, self.running = False, True
            self._timer.start()
            return True

    def start_test_mode(self):
        """No camera: show synthetic frames so the UI can still be tried out."""
        with QMutexLocker(self._mutex):
            self.capture, self.test_mode, self.running = None, True, True
            self._timer.start()

    def stop(self):
        with QMutexLocker(self._mutex):
            self._timer.stop()
            self.running = False
            if self.capture is not None:
                self.capture.release()
                self.capture = None

    @property
    def has_camera(self):
        return self.capture is not None and self.capture.isOpened()

    # ------------------------------------------------------------------ frames
    def capture_high_res_frame(self):
        """One full-resolution frame (BGR) or None."""
        if not self.running or not self.has_camera:
            return None
        with QMutexLocker(self._mutex):
            ok, frame = self.capture.read()
        return frame if ok else None

    def detect_on_high_res_frame(self, frame):
        """Detect circles on the full frame. Returns a list of xyxy boxes, highest confidence first."""
        if frame is None or self.model is None:
            return []
        try:
            results = self.model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), verbose=False, device="cpu")
            found = []
            for res in results:
                if res.boxes is not None:
                    found += list(zip(res.boxes.conf.cpu().numpy(), res.boxes.xyxy.cpu().numpy()))
            found.sort(key=lambda t: -float(t[0]))
            return [box for _, box in found]
        except Exception as exc:
            print(f"Detection failed: {exc}")
            return []

    def _tick(self):
        if not self.running:
            return
        if self.test_mode:
            self._test_count += 1
            frame = self._test_frame.copy()
            cv2.putText(frame, f"Frame: {self._test_count}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)
            self.frame_ready.emit(frame)
            return
        with QMutexLocker(self._mutex):
            ok, frame = self.capture.read() if self.has_camera else (False, None)
        if not ok:
            return
        preview = cv2.resize(frame, PREVIEW_SIZE)
        self.frame_ready.emit(self._annotate(preview))

    def _annotate(self, frame):
        """Draw the detected circles on the preview frame."""
        if self.model is None:
            return frame
        try:
            results = self.model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), verbose=False, device="cpu")
            self.current_boxes = []
            out = frame.copy()
            for res in results:
                if res.boxes is None:
                    continue
                for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
                    self.current_boxes.append((x1, y1, x2, y2))
                    cv2.circle(out, (int((x1 + x2) / 2), int((y1 + y2) / 2)), int(max(x2 - x1, y2 - y1) // 2),
                               (0, 0, 255), 2)
            return out
        except Exception as exc:
            print(f"Failed to draw the detection result: {exc}")
            return frame

    @staticmethod
    def _make_test_frame():
        f = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(f, "CAMERA NOT FOUND - TEST MODE", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.circle(f, (200, 240), 80, (0, 255, 0), 2)
        cv2.putText(f, "Albumin", (150, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(f, (440, 240), 80, (255, 0, 0), 2)
        cv2.putText(f, "Creatinine", (390, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return f
