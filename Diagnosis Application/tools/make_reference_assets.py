"""Build the reference assets for the PDF report from the training-output folder (Output_Hasil_TERPADU).

Output (written to model/uacr/):
  reference_features.csv    : reference features of the 478 tests (label, lab UACR, selected features, colour, texture)
  performance_summary.json  : nested-CV performance summary of the deployed SVM model (for the "Model Performance" section)

Usage:
    python tools/make_reference_assets.py "F:/.../Output_Hasil_TERPADU"
The assets are already included; re-run only if the model is retrained.
"""
import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "model" / "uacr"

COLOUR = [f"{r}__{c}_{s}" for r in ("alb", "cre") for c in "RGB" for s in ("mean",)]
COLOUR += [f"{r}__{c}_mean" for r in ("alb", "cre") for c in ("L", "a", "b")]
TEXTURE = [f"{r}__lbp_{i}" for r in ("alb", "cre") for i in range(10)]
TEXTURE += [f"alb__glcm_{n}" for n in ("contrast", "homogeneity", "energy", "correlation", "entropy")]
RING = [f"{r}__ring_{p}" for r in ("alb", "cre") for p in ("in", "mid", "out")]


def main(root):
    root = Path(root)
    cleaned = pd.read_csv(root / "0_Shared" / "tables" / "cleaned_dataset_all_features.csv")
    cfg = json.loads((OUT / "deployment_config.json").read_text(encoding="utf-8"))
    selected = cfg["selected_features"]
    keep = ["label", "uacr", "lux"] + selected + [c for c in COLOUR + TEXTURE + RING if c not in selected]
    keep = [c for c in dict.fromkeys(keep) if c in cleaned.columns]
    ref = cleaned[keep].copy()
    ref.to_csv(OUT / "reference_features.csv", index=False, float_format="%.6g")

    per_class = pd.read_csv(root / "A_ML" / "tables" / "TabA13_per_class_metrics.csv")
    svm = per_class[per_class["Model"] == "SVM (RBF)"]
    summary = {
        "model": cfg["model"], "hyperparameters": cfg["hyperparameters"], "k": cfg["k"],
        "classes": ["NORMAL", "MICRO", "MACRO"],
        "training": cfg["training"],
        "per_class": {r["Class"]: {"precision": r["Precision"], "recall": r["Recall"], "f1": r["F1"],
                                   "support": int(r["Support (tests)"])} for _, r in svm.iterrows()},
        # Out-of-fold confusion matrix at test level (rows = true label, columns = prediction),
        # order NORMAL, MICRO, MACRO. Source: integrated report A_ML (SVM, repeat 1).
        "confusion_test": [[65, 24, 32], [33, 28, 57], [25, 30, 184]],
        "confusion_sample": [[16, 2, 7], [9, 3, 13], [4, 4, 42]],
        "nested_cv": {
            "macro_f1": [0.510, 0.090], "balanced_accuracy": [0.523, 0.098], "mcc": [0.335, 0.133],
            "recall": {"NORMAL": [0.563, 0.202], "MICRO": [0.229, 0.101], "MACRO": [0.778, 0.080]},
            "baseline_random_macro_f1": [0.332, 0.043], "baseline_majority_macro_f1": [0.221, 0.023],
            "permutation_p": 0.010, "ece": 0.060,
        },
        "error_by_distance_to_threshold": [["<=0.1", 66.7], ["0.1-0.3", 51.9], ["0.3-0.6", 36.4], [">0.6", 22.6]],
        "tau_single": cfg["tau_single"], "tau_avg": cfg["tau_avg"],
    }
    (OUT / "performance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("OK:", OUT / "reference_features.csv", ref.shape, "|", OUT / "performance_summary.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
