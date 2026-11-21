# Intelligent POCT Diagnostics — UACR classification (clean version)

PyQt5 application for Armbian. Workflow: **Patient Data (SQLite) → Diagnostic (capture micropad → ROI segmentation/crop →
NORMAL/MICRO/MACRO classification) → Detail Report (complete scientific PDF); Labelling (labelled dataset)**.

## 1. Running
```bash
cd "Diagnosis Application"
python3 tools/verify_install.py     # check dependencies + numerical reproduction of the model (run once after installation)
python3 DiagnosticsV2.py
```
Data are created automatically inside this folder: `data/poct.db` (SQLite, containing **10 dummy patients** on first start),
`data/captures/<patient ID>/` (ROI images per examination), `data/reports/` (PDF), `dataset/` (Labelling tab).
A `data/poct.db` written by an earlier version of the application (Indonesian column names) is converted automatically to the
English schema the first time it is opened.

## 2. Model used
**RBF-kernel SVM** (C = 10, γ = 0.05; 10 image features) — the best model in nested cross-validation (macro-F1 0.510 ± 0.090;
balanced accuracy 0.523; MCC 0.335; permutation p = 0.010). Files: `model/uacr/` (`uacr_model.joblib`, `uacr_features.py` locked
by SHA-256, `deployment_config.json`, `model_card.md`). The decision is **UNCERTAIN** if the highest probability is < 0.44.
**This is a research prototype**: the recall of the MICRO class is only ≈ 0.23, so results must be confirmed in the laboratory; this
limitation is stated on the screen result and in the PDF.

## 3. NEW dependencies
| Package | Purpose | Note |
|---|---|---|
| `scikit-learn==1.6.1` | load and run the model | **must be exact**; other versions may fail to load the `.joblib` |
| `scipy`, `scikit-image`, `joblib`, `numpy` | image feature extraction | reference versions in `requirements.txt` |
| `matplotlib`, `reportlab`, `pillow` | figures & PDF report | aarch64 wheels are available |
| `pymupdf` *(or `poppler-utils`)* | show the PDF in the Detail Report tab | PyMuPDF is AGPL-licensed; the alternative is `sudo apt install poppler-utils` (the application then uses `pdftoppm` automatically) |

```bash
pip install -r requirements.txt
```
**Important on Armbian**
* Do **not** install `opencv-python-headless` / `opencv-python` if the previous application already uses the system `cv2` — pip would overwrite it and
  break the Qt plugin. The feature module only needs the `cv2` that is already present.
* If `numpy==2.1.3` conflicts with your `torch`/`ultralytics` version, use the numpy that is compatible with your torch
  (e.g. 1.26.x) — what decides is the result of `tools/verify_install.py` (the probability difference for the sample must be < 0.02 and
  scikit-learn must remain 1.6.1).
* The first matplotlib run builds the *font cache* (several seconds to minutes on an SBC); the first PDF can therefore take longer.
  A progress dialog is shown. Afterwards each report takes ±10–40 seconds depending on the device.

## 4. Changes with respect to the old code
* `Patient Data`: full CRUD to SQLite (Create/Save/Update/Delete), validation, table + latest result; the IgG/IgM fields (left over from
  the toxoplasmosis application) are hidden.
* `Diagnostic`: Patient ID combobox from SQLite; the **Classify** button runs the model (separate thread) and shows the result
  + probabilities; **Detail Report** switches to the report tab and shows the PDF. The Capture button detects the spots and rejects
  the result if no spot is found (the default position is used only after confirmation, and the result is flagged as low quality).
* `Detail Report`: history + date filter + PDF viewer (zoom, pages, open externally, regenerate).
* `Labelling`: **Sample type** selection — *Real sample* (enter the sample code) or *Artificial urine* (enter the ACR value in mg/g); lux is locked
  at capture; file-name preview; `dataset/labels_log.csv`.
  File names: `001_3401.5_117_Albumin_HighRes.PNG` (real sample) and `002_3401.5_ACR150_Albumin_HighRes.PNG` (artificial urine).
  The `ACR` prefix prevents the ACR value from being confused with the sample number; change it in `labelling_tab.ARTIFICIAL_PREFIX` if needed.
  The file-name suffix `Creatinin` follows the naming convention of the published dataset and is kept on purpose.
* `CircleSegmented.py`: V4L2 camera backend on Linux; `start()` now returns `False` if the camera fails (the old code always returned
  `True`, so camera indices 1 and 2 were never tried); test mode only if there is no camera; the mutex deadlock bug is fixed.
* `CaptureSpot.py` was removed (dead code); ROI cropping now lives in `roi_capture.py` (same logic: radius −15 %, white background, 5 % padding).
* `SimplePWM.py`: `lgpio` is imported safely; the GUI demo inside it was removed; the LED `close()` bug is fixed.
* The old fixed single-file storage of the captured images was replaced by per-examination storage in `data/captures/`.
* `mainToxoPyQt5.py` (pyuic5 output) is **not edited by hand** except for the English object names/labels; all layout adjustments are done in code.

## 5. Structure
```
DiagnosticsV2.py      main window (camera, lux sensor, tab glue)
patient_tab.py  diagnostic_tab.py  report_tab.py  labelling_tab.py      controller of each tab
patient_db.py         SQLite: patients (CRUD), diagnoses, 10 dummy records
uacr_classifier.py    model wrapper + image storage
report_generator.py   PDF (reportlab)      report_figures.py   figures (matplotlib)
pdf_viewer.py         PyQt5 PDF viewer     roi_capture.py      ROI crop   capture_ui.py  workers.py  frame_utils.py
CircleSegmented.py  SimplePWM.py  config.py  mainToxoPyQt5.py (generated UI)
model/best.pt         YOLO spot detection   model/uacr/       classification bundle + reference assets of the report
tools/verify_install.py   tools/make_reference_assets.py   tests/test_app_offscreen.py
```

## 6. Automated test (no camera/screen)
```bash
QT_QPA_PLATFORM=offscreen python3 tests/test_app_offscreen.py
```
Tests CRUD, combobox, capture (fake camera), classification, PDF creation + display in the viewer, Labelling (real sample and artificial
urine), and cascade deletion, in a temporary folder (the real data are not touched).

## 7. Important notes
* The validity of the model is limited to the same reagent, camera, illumination (≈ 3369–3705 lx) and image framing as the training
  data; the PDF contains quality checks (ROI, glare, feature and colour range, lux) that flag deviating conditions.
* Patient data are stored locally without encryption (`data/poct.db`, PDF); protect access to the device.
