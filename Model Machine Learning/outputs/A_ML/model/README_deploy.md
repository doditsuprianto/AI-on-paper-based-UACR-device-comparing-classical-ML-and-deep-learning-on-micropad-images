# Deployment bundle — UACR classifier (research prototype)

Files
* `uacr_model.joblib` – trained pipeline + metadata (selected features, ROI parameters, uncertain-zone thresholds, training ranges)
* `uacr_features.py` – image analysis (ROI detection + feature extraction); the SAME file produced the training features
* `uacr_inference.py` – image pair(s) → class, probabilities, "UNCERTAIN" flag, quality flags
* `model_card.md` – intended use, expected performance (nested CV), limitations
* `deployment_config.json`, `selected_features.csv`, `requirements.txt`, `bundle_selftest.json`

Quick start
```
pip install -r requirements.txt
python uacr_inference.py ALBUMIN.png CREATININE.png
python uacr_inference.py ALB1.png CRE1.png ALB2.png CRE2.png     # repeats of the same sample are averaged
```
```python
from uacr_inference import UACRModel
print(UACRModel().predict([("alb.png", "cre.png")]))
```
The three files `uacr_model.joblib`, `uacr_features.py`, `uacr_inference.py` are all the device needs.
