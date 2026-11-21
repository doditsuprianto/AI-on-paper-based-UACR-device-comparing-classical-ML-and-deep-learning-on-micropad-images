### Summary of findings (unified pipeline)

- **Data.** 100 samples (NORMAL: 25, MICRO: 25, MACRO: 50), 478 test image pairs after cleaning; 277 features generated, 163 passed the reliability filter (ICC ≥ 0.50).
- **Measurement audit (grouped CV, Random Forest).** R² for log albumin = 0.30, log creatinine = 0.16, log UACR = 0.20; RMSE of log₁₀ UACR = 0.87.
- **Nested CV, 10 outer folds, test level (macro-F1, mean ± SD):** Logistic Regression 0.495 ± 0.078; SVM (RBF) 0.510 ± 0.090; Random Forest 0.477 ± 0.077; Gradient Boosting 0.441 ± 0.088; k-NN 0.455 ± 0.075. Baselines: majority class 0.221 ± 0.023, stratified random 0.332 ± 0.043.
- **Sample level (macro-F1):** Logistic Regression 0.507 ± 0.100; SVM (RBF) 0.502 ± 0.092; Random Forest 0.513 ± 0.123; Gradient Boosting 0.460 ± 0.149; k-NN 0.462 ± 0.118.
- **Selected by the pre-specified rule:** SVM (RBF). Friedman test p = 0.0366; any pairwise difference significant after Holm correction: False.
- **MICRO recall (test level):** Logistic Regression 0.327 ± 0.116; SVM (RBF) 0.229 ± 0.101; Random Forest 0.232 ± 0.077; Gradient Boosting 0.233 ± 0.096; k-NN 0.133 ± 0.080.
- **Robustness of SVM (RBF):** across 7 nested-CV runs with different splits, macro-F1 ranges 0.471 – 0.521; label-permutation p = 0.010 (null 95th percentile 0.379).
- **Errors (SVM (RBF), sample level, OOF repeat 1):** 39 of 100 wrong; 11 two-level errors; median log₁₀ distance to the nearest threshold: wrong 0.29 vs correct 0.50.
- **Uncertain zone (exploratory):** single-measurement threshold 0.44 → coverage 0.89, macro-F1 on accepted 0.530 (without abstention 0.511).
- **Deployed model:** SVM (RBF), k = 10 features, bundle self-test passed = True.