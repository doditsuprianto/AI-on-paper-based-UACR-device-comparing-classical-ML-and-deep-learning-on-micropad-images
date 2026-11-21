"""UACR category classification (NORMAL / MICRO / MACRO) from a pair of micropad ROI images.

Wraps the trained model bundle (model/uacr/) without modifying it:
  * `uacr_features.py` (v1.0, sha256-locked) -> image feature extraction,
  * `uacr_inference.UACRModel`               -> prediction + UNCERTAIN rule + quality flags.

`UACRClassifier.analyze()` takes the ROI images (BGR arrays) produced by the camera segmentation, stores them
losslessly (PNG, as for the training dataset) and returns the complete result.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

import config

sys.path.insert(0, str(config.UACR_MODEL_DIR))
from uacr_inference import UACRModel, UNCERTAIN_LABEL  # noqa: E402  (model bundle)
import uacr_features as uf                              # noqa: E402

EXPECTED_FEATURE_MODULE_SHA256 = "0dd35f167e79ae415b4b7f1dd7b58e9340462085756710554763206720c91362"

UACR_RANGE = {"NORMAL": "< 30 mg/g", "MICRO": "30 – 300 mg/g", "MACRO": "> 300 mg/g"}
UACR_STAGE = {"NORMAL": "A1", "MICRO": "A2", "MACRO": "A3"}


def imwrite_png(path, bgr):
    """Write a lossless PNG (safe for paths with spaces/unicode)."""
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"Failed to encode PNG: {path}")
    buf.tofile(str(path))


def feature_module_ok():
    """Make sure the feature module is identical to the one used in training (model integrity)."""
    sha = hashlib.sha256(Path(uf.__file__).read_bytes()).hexdigest()
    return sha == EXPECTED_FEATURE_MODULE_SHA256


class UACRClassifier(UACRModel):
    """UACRModel + cache of the last feature extraction + interface for image arrays."""

    def __init__(self, model_dir=None):
        model_dir = Path(model_dir or config.UACR_MODEL_DIR)
        super().__init__(model_dir / "uacr_model.joblib")
        self.model_dir = model_dir
        self.cfg = json.loads((model_dir / "deployment_config.json").read_text(encoding="utf-8"))
        self.version = f"{self.cfg['model']} k={self.cfg['k']} | features v{uf.MODULE_VERSION} | {self.cfg['created'][:10]}"
        self.last_extraction = None

    def _extract(self, alb, cre):
        self.last_extraction = super()._extract(alb, cre)
        return self.last_extraction

    def analyze(self, alb_bgr, cre_bgr, out_dir, stem):
        """Save the ROI images, then classify them.

        Parameters
        ----------
        alb_bgr, cre_bgr : BGR ndarrays of the auto-cropped albumin / creatinine ROIs
        out_dir          : folder in which the images are stored
        stem             : file-name prefix (e.g. 'P-001_20261115_142300')
        """
        if alb_bgr is None or cre_bgr is None:
            raise ValueError("The albumin and/or creatinine ROI image is not available yet. Run Capture first.")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        alb_path = out_dir / f"{stem}_Albumin.png"
        cre_path = out_dir / f"{stem}_Creatinin.png"
        imwrite_png(alb_path, alb_bgr)
        imwrite_png(cre_path, cre_bgr)

        result = self.predict([(alb_path, cre_path)])
        feats = self.last_extraction or {}
        result["selected_features"] = {f: float(feats.get(f, float("nan"))) for f in self.features}
        result["alb_image"] = str(alb_path)
        result["cre_image"] = str(cre_path)
        result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["model_version"] = self.version
        result["uacr_range"] = UACR_RANGE.get(result["most_likely_class"], "")
        return result

    @staticmethod
    def summary_text(result):
        """Short text for the result label in the UI."""
        probs = result["probabilities"]
        pr = "  ".join(f"{c} {100 * probs[c]:.0f}%" for c in ("NORMAL", "MICRO", "MACRO"))
        if result["decision"] == UNCERTAIN_LABEL:
            head = f"Result: UNCERTAIN (most likely {result['most_likely_class']})"
        else:
            head = f"Result: {result['decision']}  ({UACR_RANGE[result['decision']]})"
        tail = "  ⚠ check the image quality" if not result["input_quality_ok"] else ""
        return f"{head}\n{pr}{tail}"
