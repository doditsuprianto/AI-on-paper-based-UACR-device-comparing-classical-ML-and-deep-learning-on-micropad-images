# Model card — CNN (efficientnet_b0)

* **Task:** NORMAL / MICRO / MACRO from albumin + creatinine micropad images (research prototype).
* **Architecture:** shared efficientnet_b0 backbone (ImageNet pre-trained = True) applied to both crops; concatenated features; 2-layer head.
* **Training data:** 100 samples / 478 image pairs; fixed recipe (no tuning on held-out data).
* **Expected performance (nested CV, test level):** macro-F1 0.508 (SD over folds 0.081); reference classical model SVM (RBF): 0.510.
* **Decision rule (CNN must be significantly better than the classical reference):** NOT MET — the classical pipeline (Path A) remains the deployed model.
* **Limitations:** very small number of independent samples; single-device images; sensitivity to illumination should be checked (see perturbation tests); not validated externally; not for clinical use.
