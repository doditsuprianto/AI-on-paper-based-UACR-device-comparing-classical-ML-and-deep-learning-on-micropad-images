"""GUI integration test without screen/camera/sensor (QT_QPA_PLATFORM=offscreen).

    QT_QPA_PLATFORM=offscreen python tests/test_app_offscreen.py [sample_image_folder]

The camera is replaced by a stub that serves a synthetic frame with two spots (taken from the sample images), so the whole
flow Patient CRUD -> Capture -> Classify -> Detail Report PDF -> Labelling is tested end-to-end in a temporary folder.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PyQt5 import QtCore, QtWidgets  # noqa: E402

import config  # noqa: E402

SAMPLE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "sample_images"
SHOTS = Path(os.environ.get("SHOT_DIR", tempfile.mkdtemp(prefix="poct_shots_")))

# ---- redirect all data to a temporary folder
tmp = Path(tempfile.mkdtemp(prefix="poct_test_"))
config.DATA_DIR, config.DB_PATH = tmp / "data", tmp / "data" / "poct.db"
config.CAPTURE_DIR, config.REPORT_DIR, config.DATASET_DIR = tmp / "data" / "captures", tmp / "data" / "reports", tmp / "dataset"

import frame_utils as fu  # noqa: E402

LOG = []
fu.info = lambda p, t, m: LOG.append(("info", t, m))
fu.warn = lambda p, t, m: LOG.append(("warn", t, m))
fu.ask = lambda p, t, m: (LOG.append(("ask", t, m)), True)[1]


class StubSegmenter(QtCore.QObject):
    """Fake camera: one 1920x1080 frame with two circles containing the sample images."""
    frame_ready = QtCore.pyqtSignal(object)

    def __init__(self, alb, cre):
        super().__init__()
        self.running, self.has_camera = True, True
        self.frame = np.full((1080, 1920, 3), (60, 60, 60), np.uint8)
        self.boxes = []
        for img, cx in ((alb, 480), (cre, 1440)):
            r = 260
            patch = cv2.resize(img, (2 * r, 2 * r))
            mask = np.zeros((2 * r, 2 * r), np.uint8)
            cv2.circle(mask, (r, r), r, 255, -1)
            roi = self.frame[540 - r:540 + r, cx - r:cx + r]
            roi[mask > 0] = patch[mask > 0]
            self.boxes.append(np.array([cx - r, 540 - r, cx + r, 540 + r], float))

    def capture_high_res_frame(self):
        return self.frame.copy()

    def detect_on_high_res_frame(self, frame):
        return [self.boxes[1], self.boxes[0]]   # arbitrary order: must still be mapped to left/right correctly

    def start(self, camera_index=0):
        return True

    def start_test_mode(self):
        pass

    def stop(self):
        pass


def wait_until(cond, timeout=60.0, what="condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        QtWidgets.QApplication.processEvents()
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError(f"timeout while waiting for {what}")


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        raise SystemExit(1)


def shot(win, name):
    SHOTS.mkdir(parents=True, exist_ok=True)
    win.grab().save(str(SHOTS / f"{name}.png"))


def main():
    app = QtWidgets.QApplication(sys.argv)
    alb = cv2.imread(str(SAMPLE_DIR / "sampleAlbumin.png"))
    cre = cv2.imread(str(SAMPLE_DIR / "sampleCreatinin.png"))
    check(alb is not None and cre is not None, "sample images can be read")
    os.chdir(config.BASE_DIR)
    from DiagnosticsV2 import MainWindow
    seg = StubSegmenter(alb, cre)
    win = MainWindow(segmenter=seg, light_sensor=None, auto_start_camera=False)
    win.show()
    ui = win.ui
    app.processEvents()

    # ---------------- Patient Data
    check(ui.tblpatients.model().rowCount() == 10, "10 dummy patients shown in the table")
    check(ui.cmbpatient.count() == 11 and ui.cmbpatient.itemData(1) == "P-001", "Patient ID combobox contains the dummy records from SQLite")
    ui.tabWidget.setCurrentWidget(ui.tabpatient)
    ui.btnnewform.click()
    check(ui.txtpatientid.toPlainText() == "P-011", "Create proposes the next ID")
    ui.btnsave.click()
    check(LOG[-1][0] == "warn", "Save without a name is rejected by validation")
    ui.txtfullname.setPlainText("Test Patient")
    ui.rbmale.setChecked(True)
    ui.btnsave.click()
    check(ui.tblpatients.model().rowCount() == 11 and ui.cmbpatient.count() == 12, "Save adds a patient and the combobox follows")
    ui.txtaddress.setPlainText("Test Street 1")
    ui.btnupdate.click()
    check(win.db.get_patient("P-011")["address"] == "Test Street 1", "Update is stored in SQLite")
    shot(win, "1_patient")
    ui.btndelete.click()
    check(win.db.get_patient("P-011") is None and ui.tblpatients.model().rowCount() == 10, "Delete removes the patient")

    # ---------------- Diagnostic
    ui.tabWidget.setCurrentWidget(ui.tabdiagnostictest)
    wait_until(lambda: win.diagnostic.classifier is not None, 60, "model loaded")
    win.diagnostic.select_patient("P-001")
    ui.btnclassify.click()
    check(LOG[-1][0] == "warn", "Classify before Capture is rejected")
    ui.btncapture.click()
    check(win.diagnostic.alb is not None and win.diagnostic.cre is not None, "Capture yields the albumin & creatinine ROIs")
    check(abs(win.diagnostic.alb.mean(axis=(0, 1)) - win.diagnostic.cre.mean(axis=(0, 1))).sum() > 5,
          "Albumin/creatinine ROIs are not swapped (different colours)")
    ui.btnclassify.click()
    wait_until(lambda: win.diagnostic.last_diag_id is not None, 60, "classification result")
    txt = ui.lblresult.text()
    check("Result:" in txt and any(c in txt for c in ("NORMAL", "MICRO", "MACRO", "UNCERTAIN")), "Result text shows the classification")
    print("      ->", ui.lblresult.text().replace("<br>", " | ")[-170:])
    check(len(win.db.list_diagnoses(patient_id="P-001")) == 1, "diagnosis is stored in SQLite")
    shot(win, "2_diagnostic")

    ui.pushButtonDetailReport.click()
    check(ui.tabWidget.currentWidget() is ui.tabreporting, "Detail Report switches to the report tab")
    wait_until(lambda: win.report.viewer.page_count() > 0, 120, "PDF shown in the viewer")
    pdf = win.db.get_diagnosis(win.diagnostic.last_diag_id)["pdf_path"]
    check(Path(pdf).exists() and Path(pdf).stat().st_size > 100_000, f"PDF created ({Path(pdf).stat().st_size // 1024} KB, {win.report.viewer.page_count()} pages)")
    check(ui.tblreport.model().rowCount() == 1, "diagnosis history shown in the report table")
    app.processEvents()
    shot(win, "3_report")
    import shutil
    shutil.copy(pdf, SHOTS / "report.pdf")

    # ---------------- Labelling
    ui.tabWidget.setCurrentWidget(ui.tabLabelling)
    lab = win.labelling
    ui.textEditLuxLabel.setPlainText("3401.5")
    ui.btncaptureLabel.click()
    check(lab.alb is not None, "Labelling: capture ROIs")
    ui.pushButtonLabel.click()
    check(LOG[-1][0] == "warn", "Labelling: empty sample code is rejected")
    ui.textEditACRLabel.setPlainText("117")
    shot(win, "4_labelling")
    ui.pushButtonLabel.click()
    files = sorted(p.name for p in config.DATASET_DIR.glob("*.PNG"))
    check(files == ["001_3401.5_117_Albumin_HighRes.PNG", "001_3401.5_117_Creatinin_HighRes.PNG"], f"real sample -> {files[0]}")
    check(ui.textEditIndexLabel.toPlainText() == "002", "index increments automatically")
    lab.cmb_type.setCurrentIndex(1)
    check(ui.label_20.text().startswith("ACR"), "label changes to ACR for artificial urine")
    ui.btncaptureLabel.click()
    ui.textEditACRLabel.setPlainText("150")
    ui.pushButtonLabel.click()
    files = sorted(p.name for p in config.DATASET_DIR.glob("*.PNG"))
    check("002_3401.5_ACR150_Albumin_HighRes.PNG" in files, "artificial urine -> 002_3401.5_ACR150_Albumin_HighRes.PNG")
    chk = cv2.imread(str(config.DATASET_DIR / "001_3401.5_117_Albumin_HighRes.PNG"))
    check(chk is not None and chk.shape[0] > 100, "dataset PNG can be read back")
    log = (config.DATASET_DIR / "labels_log.csv").read_text(encoding="utf-8").strip().splitlines()
    check(len(log) == 3 and ",real,117," in log[1] and ",artificial,,150," in log[2], "labels_log.csv records the type, code and ACR")
    lab.cmb_type.setCurrentIndex(1)
    ui.textEditACRLabel.setPlainText("abc")
    ui.btncaptureLabel.click()
    ui.pushButtonLabel.click()
    check(LOG[-1][0] == "warn" and "ACR" in LOG[-1][2], "non-numeric ACR is rejected")

    # ---------------- Delete patient -> cascade
    ui.tabWidget.setCurrentWidget(ui.tabpatient)
    win.patients._select = None
    for r in range(ui.tblpatients.model().rowCount()):
        if ui.tblpatients.model().item(r, 0).text() == "P-001":
            ui.tblpatients.selectRow(r)
    ui.btndelete.click()
    check(win.db.list_diagnoses() == [] and not Path(pdf).exists(), "deleting a patient also deletes the history, images and PDF")
    win.close()
    print(f"\nALL TESTS PASSED. Screenshots: {SHOTS}")


if __name__ == "__main__":
    main()
