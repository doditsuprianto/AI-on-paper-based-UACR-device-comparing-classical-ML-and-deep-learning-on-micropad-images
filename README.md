# Paper-based colorimetric UACR classification — code, application and device design

Companion repository of the article **"Enabling AI on a Paper-Based Point-of-Care Device for UACR: Classical Machine Learning versus Deep Learning on Micropad Images"** (authors: Dodit Suprianto, Ahmad Luthfi Fahmi, Saptarini, Tiffany Azhar Izzuddin, Akhmad Sabarudin).

The study classifies the urine albumin-to-creatinine ratio (UACR) category — **NORMAL / MICRO / MACRO** — from a *pair* of
photographs of paper-based microfluidic (µPAD) colorimetric assays (one albumin pad, one creatinine pad), compares classical
machine learning on hand-crafted image features with convolutional neural networks under one sample-grouped nested
cross-validation, audits how much analyte information the images carry, and embeds the selected model in a prototype
point-of-care diagnosis device.

> **Research prototype.** Nothing in this repository is a validated medical device. Outputs must not be used for clinical decisions.

## Repository layout

| Folder | Content |
|---|---|
| [`Model Machine Learning/`](Model%20Machine%20Learning) | One notebook (`UACR_Pipeline.ipynb`) that runs the whole study: image pairing, ROI detection, feature extraction, EDA, cleansing, feature selection, five classical models, CNNs, nested CV, statistics, deployment bundle. |
| [`Diagnosis Application/`](Diagnosis%20Application) | PyQt5 application of the prototype device (Armbian single-board computer): patient database, image capture, ROI crop, classification with the deployed SVM bundle, PDF report. |
| [`Design Diagnosis Device/`](Design%20Diagnosis%20Device) | Mechanical design of the device (FreeCAD source, STL files, print-ready 3MF files and bill of materials). |

## Data

This repository is self-contained: it includes the image dataset and the stored results.

* `Model Machine Learning/data/images/` — albumin and creatinine pad images (≈ 3.6 GB, PNG, ≈ 1000 files)
* `Model Machine Learning/data/UACR_labels_deidentified.xlsx` — label table without direct identifiers
* `Model Machine Learning/outputs/` — figures, tables, numbers and models written by the reported run
* File-name convention and column description: [`Model Machine Learning/data/README.md`](Model%20Machine%20Learning/data/README.md)

Because of the size, clone with `git clone --depth 1 <url>` or download the archived release from Zenodo.

## Quick start

```bash
git clone https://github.com/doditsuprianto/AI-on-paper-based-UACR-device-comparing-classical-ML-and-deep-learning-on-micropad-images.git
cd AI-on-paper-based-UACR-device-comparing-classical-ML-and-deep-learning-on-micropad-images/"Model Machine Learning"
pip install -r requirements.txt
jupyter lab UACR_Pipeline.ipynb          # Run all; UACR_FAST=1 = quick smoke test only
```

## How to cite

Please cite the article and the archived release (Zenodo DOI `10.5281/zenodo.24415466`, which always resolves to the latest version); see [`CITATION.cff`](CITATION.cff).

## Licence

Code: MIT (see `LICENSE`). Images, label table, results and device design files: CC BY 4.0 (see `LICENSE-DATA.md`). Third-party components keep their own licences
(see "Licence notes" in `Diagnosis Application/README.md`).
