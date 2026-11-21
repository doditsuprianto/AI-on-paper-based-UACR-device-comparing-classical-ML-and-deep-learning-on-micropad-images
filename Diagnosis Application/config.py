"""Central configuration of the Intelligent POCT Diagnostics (UACR) application.

All paths are computed relative to this application folder, so the whole folder can be
moved or copied to the Armbian device without changing the code.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------- data folders
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "poct.db"             # SQLite database: patients + diagnosis history
CAPTURE_DIR = DATA_DIR / "captures"        # ROI images per diagnosis: captures/<patient ID>/...
REPORT_DIR = DATA_DIR / "reports"          # PDF reports of the diagnoses
DATASET_DIR = BASE_DIR / "dataset"         # labelled dataset (Labelling tab)
LOGO_DIR = BASE_DIR / "logo"

# ----------------------------------------------------------------------- model
SEGMENTER_MODEL = BASE_DIR / "model" / "best.pt"       # YOLO weights for circular spot detection
UACR_MODEL_DIR = BASE_DIR / "model" / "uacr"           # SVM classification bundle (training output)

# ---------------------------------------------------------------- ROI segmentation
# Radius reduction of the detected circle (0.15 = radius reduced by 15 %).
# The same value was used when the training dataset was created.
RADIUS_REDUCTION_FACTOR = 0.15
CAMERA_INDICES = (0, 1, 2)

# ---------------------------------------------------------------------- hardware
BH1750_BUS = 3
BH1750_ADDRESS = 0x23

APP_TITLE = "Intelligent POCT Diagnostics"
APP_VERSION = "2.0"
CLASS_NAMES = ("NORMAL", "MICRO", "MACRO")


def ensure_dirs():
    """Create all required working folders (safe to call repeatedly)."""
    for folder in (DATA_DIR, CAPTURE_DIR, REPORT_DIR, DATASET_DIR):
        folder.mkdir(parents=True, exist_ok=True)
