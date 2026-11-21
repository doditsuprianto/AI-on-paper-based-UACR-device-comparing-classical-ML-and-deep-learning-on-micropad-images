# Diagnosis Application

PyQt5 application of the prototype point-of-care device (Armbian single-board computer). Workflow:
**Patient Data** (SQLite) → **Diagnostic** (capture the albumin and creatinine pads → spot detection and ROI crop → classification
NORMAL / MICRO / MACRO or UNCERTAIN) → **Detail Report** (scientific PDF report); **Labelling** tab to build labelled datasets.
The three main tabs are shown in Supplementary Figure S7 of the article.

> Research prototype. The deployed model reaches macro-F1 0.51 ± 0.09 in nested cross-validation and a low recall for the
> MICRO class; results must be confirmed by a laboratory method. Not a medical device.

* Installation, dependencies, Armbian notes: [`README_INSTALL.md`](README_INSTALL.md) and `requirements.txt`.
* Run: `python3 tools/verify_install.py` once (checks packages and reproduces the reference prediction), then `python3 DiagnosticsV2.py`.
* Model: `model/uacr/` — SVM (RBF) deployment bundle written by `Model Machine Learning/UACR_Pipeline.ipynb`
  (`uacr_model.joblib`, `uacr_features.py` locked by SHA-256, `deployment_config.json`, `model_card.md`).
  The bundle must be loaded with scikit-learn **1.6.1**.
* Spot detection: `model/best.pt` (YOLO weights used by `CircleSegmented.py`).
* Patient data: the first start creates `data/poct.db` with **10 dummy patients** (fictitious names, `example.com` e-mail
  addresses). No real patient data are included in this repository; `data/` is git-ignored.
* Tests: `tests/test_app_offscreen.py` (headless GUI test with sample images in `tests/sample_images/`).

## Licence notes

Own code: MIT (repository `LICENSE`). Components with their own terms: **PyMuPDF** (AGPL-3.0; optional PDF viewer — the
application falls back to `poppler-utils` when it is not installed) and **Ultralytics/YOLO** (AGPL-3.0; spot detection).
Logos in `logo/` belong to their institutions and are not covered by the MIT licence.
