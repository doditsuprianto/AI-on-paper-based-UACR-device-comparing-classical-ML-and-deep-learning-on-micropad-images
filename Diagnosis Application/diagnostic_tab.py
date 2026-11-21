"""'Diagnostic' tab: select patient -> capture micropad -> segment/crop ROIs -> UACR classification -> result + report."""
from datetime import datetime
from html import escape

from PyQt5 import QtCore

import config
import frame_utils as fu
from capture_ui import capture_roi_interactive
from workers import FunctionWorker

RESULT_COLOR = {"NORMAL": "#2E8B6F", "MICRO": "#B97A00", "MACRO": "#B33A3A", "UNCERTAIN": "#5B6577"}
RANGE_TEXT = {"NORMAL": "&lt; 30 mg/g", "MICRO": "30–300 mg/g", "MACRO": "&gt; 300 mg/g"}
PLACEHOLDER = "— select patient ID —"


def _load_classifier():
    from uacr_classifier import UACRClassifier      # heavy import is deferred so the GUI appears quickly
    return UACRClassifier()


class DiagnosticTabController(QtCore.QObject):
    diagnosisSaved = QtCore.pyqtSignal(int)          # id of the diagnosis that was just saved

    def __init__(self, ui, db, host, parent=None):
        super().__init__(parent)
        self.ui, self.db, self.host = ui, db, host
        self.classifier = None
        self._loader = None
        self._worker = None
        self._load_error = None
        self.alb = self.cre = None
        self.capture_flags, self.capture_lux = [], None
        self.last_diag_id = None
        self.scene_alb = fu.make_scene(ui.gvsegmentationAlbumin_2, smooth=False)
        self.scene_cre = fu.make_scene(ui.gvsegmentationCreatinine_2, smooth=False)

        lab = ui.lblresult
        lab.setGeometry(100, 436, 500, 56)
        lab.setWordWrap(True)
        lab.setTextFormat(QtCore.Qt.RichText)
        for btn, text, tip in ((ui.btncapture, "Capture", "Capture the micropad, then segment and crop the ROIs automatically"),
                               (ui.btnclassify, "Classify", "Classify the UACR category from the ROIs")):
            btn.setToolTip(tip)
            if btn.icon().isNull():
                btn.setText(text)
        ui.btncapture.clicked.connect(self.capture)
        ui.btnclassify.clicked.connect(self.classify)
        ui.pushButtonDetailReport.clicked.connect(self.open_report)
        ui.cmbpatient.currentIndexChanged.connect(lambda _: self._on_patient_changed())
        self.reload_patients()
        self._set_status("Result: –", "Select a patient, press Capture, then Classify.")
        self.load_model_async()

    # ------------------------------------------------------------------ model
    def load_model_async(self):
        self.ui.btnclassify.setEnabled(False)
        self._loader = FunctionWorker(_load_classifier, parent=self)
        self._loader.succeeded.connect(self._on_model_ready)
        self._loader.failed.connect(self._on_model_failed)
        self._loader.start()
        self._set_status("Result: –", "Loading the classification model…")

    def _on_model_ready(self, clf):
        self.classifier = clf
        self.ui.btnclassify.setEnabled(True)
        self._set_status("Result: –", f"Model ready: {escape(clf.version)}. Select a patient, Capture, then Classify.")

    def _on_model_failed(self, msg):
        self._load_error = msg
        self.ui.btnclassify.setEnabled(True)    # so that the user gets a clear error message
        self._set_status("Result: model not available", escape(msg[:160]))

    # ------------------------------------------------------------------ patients
    def reload_patients(self):
        cmb = self.ui.cmbpatient
        current = cmb.currentData()
        cmb.blockSignals(True)
        cmb.clear()
        cmb.addItem(PLACEHOLDER, None)
        for p in self.db.list_patients():
            cmb.addItem(f"{p['patient_id']} — {p['full_name']}", p["patient_id"])
        idx = cmb.findData(current) if current else 0
        cmb.setCurrentIndex(max(idx, 0))
        cmb.blockSignals(False)

    def selected_patient_id(self):
        return self.ui.cmbpatient.currentData()

    def select_patient(self, pid):
        idx = self.ui.cmbpatient.findData(pid)
        if idx >= 0:
            self.ui.cmbpatient.setCurrentIndex(idx)

    def _on_patient_changed(self):
        if self.last_diag_id is not None:
            d = self.db.get_diagnosis(self.last_diag_id)
            if d and d["patient_id"] != self.selected_patient_id():
                self._set_status("Result: –", "Patient changed. Run Capture & Classify for this patient.")

    # ------------------------------------------------------------------ capture
    def capture(self):
        cap = capture_roi_interactive(self.ui.tabdiagnostictest, self.host.segmenter)
        if cap is None:
            return
        self.alb, self.cre, self.capture_flags = cap.albumin, cap.creatinine, cap.flags
        self.capture_lux = self.host.current_lux()
        fu.show_roi_preview(self.ui.gvsegmentationAlbumin_2, self.scene_alb, self.alb)
        fu.show_roi_preview(self.ui.gvsegmentationCreatinine_2, self.scene_cre, self.cre)
        warn = " ⚠ default position used" if cap.used_fallback else ""
        self._set_status("Result: –", f"ROIs captured — albumin {self.alb.shape[1]}×{self.alb.shape[0]}, "
                                      f"creatinine {self.cre.shape[1]}×{self.cre.shape[0]} px.{warn} Click Classify.")

    # ------------------------------------------------------------------ classification
    def classify(self):
        parent = self.ui.tabdiagnostictest
        pid = self.selected_patient_id()
        if pid is None:
            fu.warn(parent, "No patient selected", "Select a Patient ID first.")
            return
        if self.alb is None or self.cre is None:
            fu.warn(parent, "No ROI yet", "Run Capture first.")
            return
        if self.classifier is None:
            msg = (f"The model failed to load:\n{self._load_error}" if self._load_error
                   else "The model is still loading. Try again in a few seconds.")
            fu.warn(parent, "Model not ready", msg)
            return
        if self._worker is not None and self._worker.isRunning():
            return
        stem = f"{pid}_{datetime.now():%Y%m%d_%H%M%S}"
        self._busy(True)
        self._set_status("Result: analysing…", "Feature extraction and classification are running.")
        self._pending = dict(pid=pid, lux=self.capture_lux, flags=list(self.capture_flags))
        self._worker = FunctionWorker(self.classifier.analyze, self.alb.copy(), self.cre.copy(),
                                      config.CAPTURE_DIR / pid, stem, parent=self)
        self._worker.succeeded.connect(self._on_classified)
        self._worker.failed.connect(self._on_classify_failed)
        self._worker.start()

    def _on_classify_failed(self, msg):
        self._busy(False)
        self._set_status("Result: failed", escape(msg[:200]))
        fu.warn(self.ui.tabdiagnostictest, "Classification failed", msg)

    def _on_classified(self, result):
        self._busy(False)
        pend = self._pending
        if pend["flags"]:                                      # ROI from the default position -> low quality
            result["quality_flags"].append({"albumin_image": result["alb_image"],
                                            "creatinine_image": result["cre_image"], "flags": pend["flags"]})
            result["input_quality_ok"] = False
        try:
            self.last_diag_id = self.db.add_diagnosis(pend["pid"], result, pend["lux"], result["alb_image"],
                                                      result["cre_image"], result["model_version"])
        except Exception as exc:
            fu.warn(self.ui.tabdiagnostictest, "Database", f"The result could not be saved: {exc}")
            self.last_diag_id = None
        self._show_result(result)
        self.ui.pushButtonDetailReport.setEnabled(True)
        if self.last_diag_id is not None:
            self.diagnosisSaved.emit(self.last_diag_id)

    def _busy(self, on):
        for w in (self.ui.btncapture, self.ui.btnclassify, self.ui.cmbpatient, self.ui.pushButtonDetailReport):
            w.setEnabled(not on)

    # ------------------------------------------------------------------ result display
    def _show_result(self, r):
        dec, probs = r["decision"], r["probabilities"]
        color = RESULT_COLOR.get(dec, "#222")
        if dec == "UNCERTAIN":
            head = (f"Result: UNCERTAIN <span style='font-size:9pt;font-weight:400'>(most likely "
                    f"{r['most_likely_class']}, repeat the measurement)</span>")
        else:
            head = f"Result: {dec} <span style='font-size:9pt;font-weight:400'>({RANGE_TEXT[dec]})</span>"
        prob = " · ".join(f"{c} {100 * probs[c]:.0f}%" for c in ("NORMAL", "MICRO", "MACRO"))
        qc = "" if r["input_quality_ok"] else " &nbsp;⚠ check the image quality"
        self.ui.lblresult.setText(
            f"<span style='font-size:15pt;font-weight:600;color:{color}'>{head}</span><br>"
            f"<span style='font-size:8.5pt;color:#333'>{prob} &nbsp;|&nbsp; τ = {r['threshold_used']:.2f}{qc}</span>")
        self.ui.lblresult.setToolTip(r["disclaimer"])

    def _set_status(self, head, detail=""):
        self.ui.lblresult.setToolTip("")
        self.ui.lblresult.setText(
            f"<span style='font-size:13pt;font-weight:600;color:#12263F'>{head}</span><br>"
            f"<span style='font-size:8.5pt;color:#555'>{detail}</span>")

    # ------------------------------------------------------------------ report
    def open_report(self):
        pid = self.selected_patient_id()
        if pid is None:
            fu.warn(self.ui.tabdiagnostictest, "Detail Report", "Select a Patient ID first.")
            return
        diag = self.db.get_diagnosis(self.last_diag_id) if self.last_diag_id else None
        if not diag or diag["patient_id"] != pid:
            history = self.db.list_diagnoses(patient_id=pid)
            diag = history[0] if history else None
        if not diag:
            fu.warn(self.ui.tabdiagnostictest, "Detail Report",
                    "There is no classification result for this patient yet. Run Capture and then Classify first.")
            return
        self.host.show_report(diag["id"])
