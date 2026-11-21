# Model Machine Learning

`UACR_Pipeline.ipynb` is the complete, executable analysis of the study — one dataset, two pipelines, one comparison.

| Part | Stages | What is done |
|---|---|---|
| I · Shared foundation | 1–9 | environment, image inventory and albumin–creatinine pairing, label table, shared image module (ROI detection + 277 features), EDA, cleansing (ICC ≥ 0.50, redundancy filter), common analysis set and the sample-grouped fold plan |
| II · Path A (classical ML) | 11–16 | measurement-quality audit, significance testing and feature selection, five models (SVM, Random Forest, Gradient Boosting, k-NN, Logistic Regression), repeated nested CV, analysis of the selected model, final training and **deployment bundle** |
| III · Path B (CNN) | 17–27 | image tensors, frozen embeddings, fine-tuned two-branch CNNs, same folds as Path A, Grad-CAM and perturbation checks |
| IV · Comparison | 28–30 | unified leaderboard, error overlap, decision rule |

Design choices that matter for reproduction: folds are **grouped by sample** (a sample is tested several times) and
class-stratified; the primary metric is **macro-F1**; both paths use *identical* outer folds; every number quoted in the
article is written by the notebook to `outputs/**/report/report_numbers*.json`.

## Requirements

* Python ≥ 3.10; packages in `requirements.txt` (the versions that were recorded when the reported results were produced).
* Parts I, II and IV run on CPU (≈ minutes to a few hours depending on the cores). Part III (fine-tuning CNNs) needs a GPU
  (a Colab T4 is sufficient). PyTorch is installed separately because the right build depends on your CUDA version:
  https://pytorch.org/get-started/locally/
* Pretrained ImageNet weights are downloaded by `torchvision` at first use (internet needed once).

## Data

Included in `data/` (see [`data/README.md`](data/README.md)). The notebook looks for the images and the label table in, in this
order: `CFG['MANUAL_ROOT']` → environment variable `UACR_DATA_DIR` → Google Drive (`MyDrive/UACR`, in Colab) → `./data`.
Results are written to `./outputs/` (or `UACR_OUT_DIR`).

## Stored results

`UACR_Pipeline.ipynb` is stored **with its outputs** (figures and tables) from the reported run of the original working notebook
(full dataset); `outputs/` holds the same figures (300 dpi PNG), tables (CSV), `report_numbers*.json`, deployment bundle and
CNN weights. Cleaned-up differences to the original: path handling, no legacy-cache or manuscript-helper cells, output folder
named `outputs/` (the original was `Output_Hasil_TERPADU`, which still appears in some stored text files). The cleaned code
itself was re-executed end-to-end only in fast smoke-test mode (`UACR_FAST=1`, 30 samples); re-run it on the full data to
regenerate the numbers. Not stored (regenerable, large): CNN fine-tuning caches. One table with free-text source notes was
removed because it contained patient identifiers.

## Running

```bash
pip install -r requirements.txt
jupyter lab UACR_Pipeline.ipynb      # Kernel > Restart & Run All
```

* **Resume:** features, nested-CV results and finished CNN tasks are cached in `outputs/*/cache`; an interrupted run continues
  from the cache. Delete the folder to recompute everything.
* **Smoke test (`UACR_FAST=1`):** trimmed grids, 64-px images, 2 epochs, no pretrained weights. It only checks that the code
  runs end-to-end on your machine; **its numbers must not be reported**.
* **Determinism:** one seed (42) controls the folds and, through derived seeds, the network initialisation and batch order.
  GPU kernels are not bit-wise deterministic, so CNN results can differ in the third decimal between runs.

## Outputs

`outputs/0_Shared` (data foundation), `outputs/A_ML` (figures, tables, **model/** = deployment bundle: `uacr_model.joblib`,
`uacr_features.py`, `uacr_inference.py`, `model_card.md`), `outputs/B_CNN`, `outputs/C_Comparison`. The bundle in
`../Diagnosis Application/model/uacr/` is the one produced by the reported run.

## Note on the label table

The class label is the column **Label Kelas** (`NORMAL`, `MIKRO`, `MAKRO`; translated to NORMAL/MICRO/MACRO when read).
Laboratory albumin, creatinine and UACR are used only for auditing and validation, **never as model features**
(the label is a direct function of UACR: target leakage).
