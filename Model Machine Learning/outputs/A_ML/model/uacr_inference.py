# uacr_inference.py - UACR category (NORMAL / MICRO / MACRO) from micropad images. RESEARCH PROTOTYPE.
#
# Usage (command line):
#     python uacr_inference.py ALB_1.png CRE_1.png [ALB_2.png CRE_2.png ...]
#         one image pair  -> single-measurement result
#         several pairs of the SAME sample -> probabilities are averaged
#
# Usage (library):
#     from uacr_inference import UACRModel
#     model = UACRModel()                                   # loads uacr_model.joblib next to this file
#     result = model.predict([("alb1.png", "cre1.png")])    # list of (albumin_image, creatinine_image) pairs
#     print(result["decision"], result["probabilities"])
#
# Decision rule: the class with the highest probability, unless that probability is below the stored threshold
# (single pair: tau_single, several pairs: tau_avg); then the decision is "UNCERTAIN" (uncertain: repeat the measurement).
# The thresholds are exploratory and must be confirmed on independent data before any clinical use.
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from joblib import load

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import uacr_features as uf  # the same image-analysis code that produced the training features

UNCERTAIN_LABEL = "UNCERTAIN"
DISCLAIMER = ("Research prototype. Not validated for clinical decisions; expected performance is limited "
              "(see model_card.md). Confirm results with the laboratory method.")


class UACRModel:
    def __init__(self, model_path=None):
        self.model_path = Path(model_path) if model_path else HERE / "uacr_model.joblib"
        self.pack = load(self.model_path)
        self.classes = list(self.pack["classes"])
        self.features = list(self.pack["features_in"])
        self.pipeline = self.pack["pipeline"]
        self.roi_params = self.pack["roi_params"]
        self.ranges = self.pack.get("train_ranges", {})
        self.tau_single = float(self.pack["tau_single"])
        self.tau_avg = float(self.pack["tau_avg"])

    def _extract(self, alb, cre):
        r = uf.process_pair(dict(path_alb=str(alb), path_cre=str(cre)), self.roi_params)
        if not r["ok"]:
            raise RuntimeError("Image pair could not be analysed: " + str(r["err"]))
        return r

    def predict(self, pairs):
        if isinstance(pairs, (tuple, list)) and len(pairs) == 2 and not isinstance(pairs[0], (tuple, list)):
            pairs = [pairs]
        rows, qc = [], []
        for alb, cre in pairs:
            r = self._extract(alb, cre)
            vals = [r.get(f, np.nan) for f in self.features]
            rows.append(vals)
            flags = []
            for role, nm in (("alb", "albumin"), ("cre", "creatinine")):
                if r.get(role + "__roi_method") == "fallback_circle":
                    flags.append(nm + " image: ROI not found automatically (fallback circle used) - check the image")
            out = [f for f, v in zip(self.features, vals) if f in self.ranges and not (self.ranges[f][0] <= v <= self.ranges[f][1])]
            if out:
                flags.append("input outside the range seen in training for: " + ", ".join(out))
            if any(not np.isfinite(v) for v in vals):
                flags.append("some feature values are missing/non-finite")
            qc.append(dict(albumin_image=str(alb), creatinine_image=str(cre), flags=flags))
        P = self.pipeline.predict_proba(np.asarray(rows, dtype=float))
        p = P.mean(axis=0)
        top = int(np.argmax(p))
        tau = self.tau_single if len(rows) == 1 else self.tau_avg
        uncertain = bool(p[top] < tau)
        any_flag = any(q["flags"] for q in qc)
        return {
            "decision": UNCERTAIN_LABEL if uncertain else self.classes[top],
            "most_likely_class": self.classes[top],
            "probabilities": {c: float(v) for c, v in zip(self.classes, p)},
            "max_probability": float(p[top]),
            "uncertain": uncertain,
            "threshold_used": tau,
            "n_pairs": len(rows),
            "per_pair_probabilities": [{c: float(v) for c, v in zip(self.classes, row)} for row in P],
            "quality_flags": [q for q in qc if q["flags"]],
            "input_quality_ok": not any_flag,
            "disclaimer": DISCLAIMER,
        }


def main(argv=None):
    ap = argparse.ArgumentParser(description="UACR category from micropad images (research prototype)")
    ap.add_argument("images", nargs="+", help="albumin and creatinine image paths, in pairs: ALB1 CRE1 [ALB2 CRE2 ...]")
    ap.add_argument("--model", default=None, help="path to uacr_model.joblib (default: next to this file)")
    a = ap.parse_args(argv)
    if len(a.images) % 2:
        ap.error("images must be given as pairs: ALB CRE [ALB CRE ...]")
    pairs = [(a.images[i], a.images[i + 1]) for i in range(0, len(a.images), 2)]
    print(json.dumps(UACRModel(a.model).predict(pairs), indent=2))


if __name__ == "__main__":
    main()
