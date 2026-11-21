"""Check the installation on the device (Armbian): dependencies, model version and numerical reproduction of the prediction.

    python3 tools/verify_install.py

Exits with code 0 if all REQUIRED checks pass. The prediction for the sample image pair must equal the reference values
obtained on the development PC (tolerance 0.02) — this proves that the model and the feature module behave identically on
the device.
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PINNED = {"sklearn": "1.6.1"}                      # must match exactly (compatibility of the .joblib model file)
RECOMMENDED = {"numpy": "2.1.3", "scipy": "1.16.3", "skimage": "0.25.2", "joblib": "1.6.0"}
REQUIRED = ["PyQt5", "cv2", "numpy", "scipy", "sklearn", "skimage", "joblib", "matplotlib", "reportlab", "PIL"]
OPTIONAL = {"pymupdf": "PDF viewer (or install poppler-utils)", "fitz": "PDF viewer (old PyMuPDF module name)",
            "ultralytics": "YOLO spot segmentation", "lgpio": "BH1750 light sensor"}
EXPECTED_PROBS = {"NORMAL": 0.0482, "MICRO": 0.2914, "MACRO": 0.6604}   # sampleAlbumin.png + sampleCreatinin.png
FEATURE_SHA256 = "0dd35f167e79ae415b4b7f1dd7b58e9340462085756710554763206720c91362"

failed = []


def say(ok, msg, required=True):
    print(("  [OK]   " if ok else ("  [FAIL] " if required else "  [info] ")) + " " + msg)
    if not ok and required:
        failed.append(msg)


print("Python", sys.version.split()[0])
print("\n1. Required packages")
for name in REQUIRED:
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", getattr(m, "PYQT_VERSION_STR", "?"))
        want = PINNED.get(name)
        say(want is None or ver == want, f"{name} {ver}" + (f" (must be {want})" if want and ver != want else ""))
        rec = RECOMMENDED.get(name)
        if rec and ver != rec:
            print(f"           advice: {name}=={rec} (installed {ver}); not required if the prediction test below passes")
    except Exception as exc:
        say(False, f"{name}: cannot be imported ({exc})")
print("\n2. Optional packages")
for name, why in OPTIONAL.items():
    try:
        importlib.import_module(name)
        say(True, f"{name} available — {why}", required=False)
    except Exception:
        say(False, f"{name} not available — {why}", required=False)

print("\n3. Model")
try:
    import uacr_classifier as uc
    import cv2
    say(uc.feature_module_ok(), "uacr_features.py is identical to the training version (SHA-256)")
    clf = uc.UACRClassifier()
    say(True, f"model loaded: {clf.version}")
    alb = cv2.imread(str(ROOT / "tests" / "sample_images" / "sampleAlbumin.png"))
    cre = cv2.imread(str(ROOT / "tests" / "sample_images" / "sampleCreatinin.png"))
    import tempfile
    res = clf.analyze(alb, cre, tempfile.mkdtemp(), "verify")
    diff = max(abs(res["probabilities"][c] - EXPECTED_PROBS[c]) for c in EXPECTED_PROBS)
    say(diff < 0.02, "prediction for the sample images: " + ", ".join(f"{c} {100 * v:.1f}%" for c, v in res["probabilities"].items())
        + f" (max. difference from the reference {diff:.4f})")
except Exception as exc:
    say(False, f"model: {exc}")

print("\nRESULT:", "ALL REQUIRED CHECKS PASSED" if not failed else f"{len(failed)} check(s) failed")
sys.exit(1 if failed else 0)
