# Data

Included in this repository (CC BY 4.0, see `../../LICENSE-DATA.md`):

```
data/
├── UACR_labels_deidentified.xlsx          # sheet "Data Sampel": label table, no direct identifiers
└── images/                                # 1000 PNG files, albumin + creatinine pads
    ├── 001_3369.2_117_Albumin_HighRes.PNG
    ├── 001_3369.2_117_Creatinin_HighRes.PNG
    └── ...
```

**File names** `AAA_LLLL.L_SSS_Assay_HighRes.PNG`:
`AAA` = running number of the test, `LLLL.L` = illuminance (lux) measured at capture, `SSS` = sample number
(key to `No Sampel` in the label table), `Assay` = `Albumin` or `Creatinin`, `HighRes` = fixed suffix.
One sample is tested several times; the notebook therefore splits by *sample*, never by image.
Two duplicate files created by the file system in the working folder (names ending in `(1).PNG`) are not included.

**Label table** (`Data Sampel`): `No Sampel`, `Albumin Urin (mg/L)`, `Kreatinin Urin (mg/dL)`, `UACR Cobas (mg/g Cr)`,
`Kreatinin Serum (mg/dL)`, `eGFR (mL/menit/1,73 m2)`, `Kelompok`, `Label Kelas` (`NORMAL` / `MIKRO` / `MAKRO`).
Names, medical-record and registration numbers, ward, diagnosis text, age, sex and free-text notes were removed.

The notebook finds this folder automatically when it is started from `Model Machine Learning/`. Another location:
set the environment variable `UACR_DATA_DIR` (folder containing the label table), or `CFG['MANUAL_ROOT']` in Stage 1.
