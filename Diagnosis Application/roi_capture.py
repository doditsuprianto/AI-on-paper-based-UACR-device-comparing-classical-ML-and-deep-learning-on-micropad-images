"""Cropping of the albumin and creatinine ROIs from one full-resolution frame.

The logic is identical to the creation of the training dataset: the detected circle is shrunk
(`RADIUS_REDUCTION_FACTOR`), pixels outside the circle are whitened, and the result is cropped with 5 % padding.
The spot left of the virtual Y axis (frame width / 2) is albumin, the spot on the right is creatinine.
This module does not depend on Qt, so it can be tested on its own.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


class CaptureError(RuntimeError):
    """Acquisition error whose message is safe to show to the user."""


class DetectionError(CaptureError):
    """The albumin/creatinine spot was not detected automatically (the caller may offer the default position)."""


@dataclass
class RoiCapture:
    albumin: np.ndarray
    creatinine: np.ndarray
    frame_size: tuple
    flags: list = field(default_factory=list)   # e.g. ["albumin: default position (spot not detected)"]

    @property
    def used_fallback(self):
        return bool(self.flags)


def _circle_from_box(box, reduction, w, h):
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    r = int(max(x2 - x1, y2 - y1) // 2 * (1.0 - reduction))
    return cx, cy, max(100, min(r, min(w, h) // 4))


def find_circles(boxes, w, h, reduction, allow_fallback):
    """Select the albumin (left) and creatinine (right) circles -> ((cx,cy,r), (cx,cy,r), flags)."""
    axis = w // 2
    left = right = None
    for box in boxes:
        cx, cy, r = _circle_from_box(box, reduction, w, h)
        if cx < axis and left is None:
            left = (cx, cy, r)
        elif cx >= axis and right is None:
            right = (cx, cy, r)
        if left and right:
            break
    flags = []
    if (left is None or right is None) and not allow_fallback:
        missing = " and ".join(n for n, c in (("albumin (left)", left), ("creatinine (right)", right)) if c is None)
        raise DetectionError(f"Spot {missing} not detected. Check the micropad position/illumination and try again.")
    fallback_r = int(min(w, h) // 8 * (1.0 - reduction))
    if left is None:
        left, flags = (w // 4, h // 2, fallback_r), flags + ["albumin: default position (spot not detected)"]
    if right is None:
        right, flags = (3 * w // 4, h // 2, fallback_r), flags + ["creatinine: default position (spot not detected)"]
    return left, right, flags


def crop_circle(frame, circle):
    """Crop the circle (cx, cy, r) on a white background with 5 % padding."""
    cx, cy, r = circle
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    out = np.full_like(frame, 255)
    out[mask == 255] = frame[mask == 255]
    pad = int(r * 0.05)
    crop = out[max(0, cy - r - pad):min(h, cy + r + pad), max(0, cx - r - pad):min(w, cx + r + pad)]
    if crop.size == 0:
        raise CaptureError("The cropped ROI is empty; the spot lies outside the frame.")
    return crop


def capture_roi_pair(segmenter, reduction, allow_fallback=False):
    """Grab a full-resolution frame from `segmenter`, detect the two spots and crop their ROIs."""
    if segmenter is None or not segmenter.has_camera:
        raise CaptureError("Camera not available.")
    frame = segmenter.capture_high_res_frame()
    if frame is None:
        raise CaptureError("Failed to grab a high-resolution frame from the camera.")
    h, w = frame.shape[:2]
    left, right, flags = find_circles(segmenter.detect_on_high_res_frame(frame), w, h, reduction, allow_fallback)
    return RoiCapture(crop_circle(frame, left), crop_circle(frame, right), (w, h), flags)
