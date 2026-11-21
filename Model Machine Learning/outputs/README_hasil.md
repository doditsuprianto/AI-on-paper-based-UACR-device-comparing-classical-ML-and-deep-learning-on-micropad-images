# Output_Hasil_TERPADU — map of the integrated experiment

| Folder | Content |
|---|---|
| `0_Shared/` | Part I: inventory, labels, merge, ROI module (`model/uacr_features.py`), descriptors, EDA, cleansing, **common analysis set** (`tables/cleaned_dataset_all_features.csv`) and **fold plan** (`tables/fold_plan_test_indices.csv`), workflow figure (`figures/Workflow_two_pipelines.png`) |
| `A_ML/` | Part II (Path A): audit, selection, five models, nested CV, analysis; `model/` = deployment bundle; `report/report_numbers.json` |
| `B_CNN/` | Part III (Path B): embeddings, fine-tuning, evaluation, Grad-CAM; `tables/fold_metrics_all_models_*.csv`; `report/report_numbers.json` |
| `C_Comparison/` | Part IV: leaderboard, contrasts, error overlap, `report/Findings_Summary_Integrated.md` |
| `report_numbers_all.json` | every quoted number of every Part (keys S, A, B, C) |

Figures/tables carry the scope letter: *Figure A7* = `A_ML/figures/FigA07_*.png`. Written by `UACR_Integrated_Pipeline.ipynb`; configuration in each `report/report_numbers.json` (key `config`).
