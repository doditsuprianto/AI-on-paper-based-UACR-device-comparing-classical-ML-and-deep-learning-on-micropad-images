# Sensitivity of the SVM estimate to the position of the cleaning steps

`sensitivity_cleaning.py` repeats the nested cross validation of the selected SVM (RBF) with the outlier removal and the
repeatability (ICC) filter moved inside the training part of every outer fold.

* Software: scikit-learn 1.6.1, as in the main run (other versions change the inner folds and therefore the scores).
  With this version scenario A reproduces the reported fold scores exactly (maximum probability difference 0).
* Inputs: `../outputs/0_Shared/cache/features_91614d0a03ef.pkl`, `../outputs/A_ML/cache/nested_main_38b779d633.pkl`
  (frozen fold plan), `../data/UACR_labels_deidentified.xlsx`.
* Scenarios: A reported; B no outlier removal; C outliers detected in the training part only; D ICC filter computed in the
  training part only; E = C + D.
* `sensitivity_cleaning_folds.csv`: 10 frozen outer folds, scenarios A to E. `sensitivity_cleaning_seeds.csv`: five new sets of
  outer folds (seeds 2000 to 2004, 25 folds), scenarios A, B, D, E. Columns `f1_ref` (478 reference tests) and `f1_all` (all tests in the test folds).
* `prevalence_illustration.csv`: predictive values of the SVM re-weighted to assumed prevalences (pooled out-of-fold predictions of repeat 1);
  an illustration, not an estimate for any population.
* Run time: about 3 minutes on 2 CPU cores.
