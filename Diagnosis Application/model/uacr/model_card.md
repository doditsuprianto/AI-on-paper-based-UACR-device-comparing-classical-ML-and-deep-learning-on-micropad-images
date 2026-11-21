# Model card — UACR category from micropad images (research prototype)

*Generated automatically by `UACR_Integrated_Pipeline.ipynb` on 2026-10-30T16:02:32. Every number below comes from that run.*

## 1. Intended use and non-use
* **Intended use:** research prototype that estimates the urine albumin-to-creatinine ratio category — **NORMAL (< 30 mg/g), MICRO (30–300 mg/g), MACRO (> 300 mg/g)** — from one pair of micropad images (albumin assay + creatinine assay), as a decision-support experiment for a diagnosis device.
* **Not for:** clinical decisions on its own, screening without laboratory confirmation, other reagents, cameras, lighting set-ups or sample matrices than those of the training data.

## 2. Model
* **Algorithm:** SVM (RBF), chosen by the pre-specified rule (highest mean outer-fold macro-F1 in nested cross-validation). Hyperparameters: `{'C': 10, 'gamma': 0.05}`.
* **Input:** 10 image features (listed in `selected_features.csv`) computed by `uacr_features.py` from the two ROI images; pipeline = median imputation → robust scaling → standard scaling → classifier.
* **Output:** class probabilities and a decision; the decision is **UNCERTAIN** (uncertain) when the highest probability is below 0.44 (one pair) or 0.53 (several pairs of the same sample averaged). These thresholds are exploratory.

## 3. Training data
100 urine samples (NORMAL: 25, MICRO: 25, MACRO: 50), 478 tests (image pairs, several per sample) after cleaning; labels from the laboratory UACR (Cobas) with KDIGO thresholds. Illumination (lux) and laboratory values were **not** used as inputs.

## 4. Expected performance (nested cross-validation; NOT the training score)
Estimated by 5×2 sample-grouped nested cross-validation, in which feature selection, k and hyperparameters were chosen without the test fold.

| Quantity (mean ± SD over 10 outer folds) | One measurement (test level) | Repeats averaged (sample level) |
|---|---|---|
| Macro-F1 | 0.510 ± 0.090 | 0.502 ± 0.092 |
| Balanced accuracy | 0.523 ± 0.098 | 0.540 ± 0.099 |
| MCC | 0.335 ± 0.133 | 0.371 ± 0.141 |
| Recall NORMAL | 0.563 ± 0.202 | 0.660 ± 0.190 |
| Recall MICRO | 0.229 ± 0.101 | 0.135 ± 0.156 |
| Recall MACRO | 0.778 ± 0.080 | 0.825 ± 0.106 |

* Chance level of macro-F1 is about 0.33; the stratified random baseline scored 0.332 ± 0.043 and the majority-class baseline 0.221 ± 0.023 on the same folds.
* Sensitivity to the random division into folds: mean macro-F1 across 7 nested-CV runs = 0.497, range 0.471 – 0.521.
* Label-permutation test (100 permutations): observed macro-F1 0.524 versus null mean 0.306 (95th percentile 0.379); p = 0.010.
* The five models were not clearly distinguishable: Friedman p = 0.0366; any pairwise difference significant after correction: False.
* Uncertain zone (exploratory, out-of-fold): at the stored single-measurement threshold 0.44, coverage = 0.89 and macro-F1 on the accepted results = 0.530 (without abstention: 0.511).

## 5. Known limitations
* Small sample (100 samples; minority classes about 25); the performance estimate has wide uncertainty and was obtained on one dataset (no external validation).
* The measurement audit (report section on the limits of image information) shows how much analyte information the images carry; classification performance cannot exceed that ceiling. MICRO (30–300 mg/g) is the weakest class.
* Samples whose laboratory UACR lies close to 30 or 300 mg/g are intrinsically ambiguous.
* Validity requires the same reagent pads, camera/lighting set-up and image framing as the training data; the device flags images whose ROI was not found or whose features fall outside the training range.
* The uncertain-zone thresholds were tuned on the same out-of-fold predictions that are used to describe them; confirm on independent data.

## 6. Reproducibility
Python 3.13.16, scikit-learn 1.6.1 (the model file must be loaded with the same major scikit-learn version; see `requirements.txt`). `uacr_features.py` SHA-256 = `0dd35f167e79ae415b4b7f1dd7b58e9340462085756710554763206720c91362`. Seeds fixed (SEED = 42). Bundle self-test passed: True.
