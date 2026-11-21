# Findings summary — CNN pipeline

*Generated 2026-10-30 16:26; fast mode: False; device: cuda.*

- **Design.** 478 image pairs from 100 samples; identical outer folds as Path A (5 folds × 2 repeats, grouped by sample; identical to the Final folds: verified).
- **Reference (classical, hand-crafted features).** SVM (RBF): macro-F1 0.510 ± 0.090; majority baseline 0.221.
- **Strategy A (frozen embeddings + LR).** Best: mobilenet_v3_large (frozen, both), macro-F1 0.488; not distinguishable from the classical reference (difference not significant after Holm correction).
- **Strategy B (fine-tuned).** Best: efficientnet_b0 (fine-tuned), macro-F1 0.508; not distinguishable from the classical reference (difference not significant after Holm correction).
- **From scratch (lower bound).** macro-F1 0.422.
- **MICRO recall (test level).** efficientnet_b0 (fine-tuned): 0.241 vs SVM (RBF): 0.229.
- **Friedman test over the finalists.** χ² = 33.21, p = 0.0001229.
- **Paired cluster bootstrap (efficientnet_b0 (fine-tuned) − SVM (RBF)).** ΔF1 = +0.024, 95 % CI -0.046 – +0.089.
- **Measurement audit with CNN embeddings.** Best cross-validated R² for log albumin = 0.33 (hand-crafted features, Path A: 0.30); CNN features do not clearly exceed the hand-crafted features (difference +0.03). Illumination (lux) is predictable from the embeddings with R² up to 0.07.
- **Shortcut check.** Border-attention ratio (Grad-CAM): albumin 1.35, creatinine 1.27; grey-scale macro-F1 0.50 vs original 0.62; brightness ±15 % changes 28 % of predictions on average.

## Decision (pre-specified rule)

**The CNN does not replace the classical pipeline (Path A).** The best CNN-based model (efficientnet_b0 (fine-tuned), macro-F1 0.508) is not distinguishable from the classical reference (difference not significant after Holm correction). With ≈ 100 independent samples this is an *inconclusive-or-negative* result, not proof of equality; the classical SVM bundle remains the primary method.

## Limitations

- Only 100 independent samples (25 / 25 / 50); the CNN recipe was fixed, not tuned, so a better-tuned CNN could differ slightly.
- The images come from one acquisition setup; no external validation set exists.
- Grad-CAM and perturbation tests are qualitative and use a single fold.
- GPU kernels are not bit-wise deterministic: re-running may change the third decimal.