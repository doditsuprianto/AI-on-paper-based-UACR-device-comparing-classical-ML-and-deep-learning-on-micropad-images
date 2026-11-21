"""'Labelling' tab: capture the albumin & creatinine ROIs as a labelled dataset.

File naming:  {index}_{lux}_{code}_{Albumin|Creatinin}_HighRes.PNG
  * Real sample      -> code = sample code/number,                 e.g. 001_3369.2_117_Albumin_HighRes.PNG
  * Artificial urine -> code = 'ACR' + ACR value (mg/g),           e.g. 002_3401.5_ACR150_Albumin_HighRes.PNG
The 'ACR' prefix prevents the ACR value from being confused with the sample number when the file name is parsed.
Every save is also recorded in dataset/labels_log.csv (separate columns for sample type, sample code and ACR value).
The file-name suffix 'Creatinin' is kept on purpose: it is the naming convention of the published dataset.
"""
import csv
import re
from datetime import datetime
from pathlib import Path

import cv2
from PyQt5 import QtCore, QtWidgets

import config
import frame_utils as fu
from capture_ui import capture_roi_interactive

SAMPLE_REAL, SAMPLE_ARTIFICIAL = 0, 1
ARTIFICIAL_PREFIX = "ACR"
CODE_RE = re.compile(r"^[A-Za-z0-9\-]{1,20}$")      # no underscore/space: '_' is the segment separator of the file name
INDEX_RE = re.compile(r"^[A-Za-z0-9\-]{1,10}$")
LOG_HEADER = ("timestamp", "index", "lux", "sample_type", "sample_code", "acr_mg_g", "albumin_file",
              "creatinine_file", "roi_albumin_wh", "roi_creatinine_wh", "fallback_roi")


def format_acr(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


class LabellingTabController(QtCore.QObject):
    def __init__(self, ui, host, parent=None):
        super().__init__(parent)
        self.ui, self.host = ui, host
        self.alb = self.cre = None
        self.flags = []
        self.capture_lux = None
        self.scene_alb = fu.make_scene(ui.gvsegmentationAlbumin, smooth=False)
        self.scene_cre = fu.make_scene(ui.gvsegmentationCreatinine, smooth=False)
        self._layout_widgets()
        ui.textEditIndexLabel.setPlainText("001")
        ui.btncaptureLabel.clicked.connect(self.capture)
        ui.btncaptureLabel.setToolTip("Capture the micropad, then segment and crop the ROIs automatically")
        if ui.btncaptureLabel.icon().isNull():
            ui.btncaptureLabel.setText("Capture")
        ui.pushButtonLabel.clicked.connect(self.save)
        self.cmb_type.currentIndexChanged.connect(self._on_type_changed)
        for w in (ui.textEditIndexLabel, ui.textEditLuxLabel, ui.textEditACRLabel):
            w.textChanged.connect(self._update_preview)
        self._on_type_changed()

    # ------------------------------------------------------------------ layout
    def _layout_widgets(self):
        ui = self.ui
        parent = ui.textEditACRLabel.parent()
        ui.gvcaptureLabel.setFixedSize(621, 375)          # leave one row for the sample type
        self.lbl_type = QtWidgets.QLabel("Sample type:", parent)
        self.lbl_type.setGeometry(10, 417, 90, 22)
        self.cmb_type = QtWidgets.QComboBox(parent)
        self.cmb_type.setGeometry(100, 413, 260, 28)
        self.cmb_type.addItems(["Real sample  (enter sample code)", "Artificial urine  (enter ACR value)"])
        self.lbl_preview = QtWidgets.QLabel("", parent)
        self.lbl_preview.setGeometry(372, 413, 340, 28)
        self.lbl_preview.setStyleSheet("color:#555; font-size:8pt;")
        self.lbl_preview.setWordWrap(True)
        for w in (self.lbl_type, self.cmb_type, self.lbl_preview):
            w.show()
        # input row: Index | Lux | Sample code / ACR | Save
        ui.label_14.setGeometry(10, 460, 50, 19)
        ui.textEditIndexLabel.setGeometry(55, 450, 61, 31)
        ui.label_19.setText("Lux:")
        ui.label_19.setGeometry(126, 460, 36, 19)
        ui.textEditLuxLabel.setGeometry(160, 450, 81, 31)
        ui.label_20.setGeometry(250, 460, 112, 19)
        ui.textEditACRLabel.setGeometry(364, 450, 112, 31)
        for w in (ui.textEditIndexLabel, ui.textEditLuxLabel, ui.textEditACRLabel):
            w.setTabChangesFocus(True)
            w.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            w.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    def _on_type_changed(self):
        artificial = self.cmb_type.currentIndex() == SAMPLE_ARTIFICIAL
        self.ui.label_20.setText("ACR (mg/g):" if artificial else "Sample code:")
        self.ui.textEditACRLabel.setToolTip("ACR value of the artificial urine, e.g. 150 or 12.5" if artificial else
                                            "Code/number of the real urine sample, e.g. 117 (letters, digits, '-' only)")
        self.ui.textEditACRLabel.clear()
        self._update_preview()

    # ------------------------------------------------------------------ live value
    def set_live_lux(self, lux):
        """Called by the sensor timer; does not overwrite a value locked at capture time or one being typed."""
        if lux is None or self.capture_lux is not None or self.ui.textEditLuxLabel.hasFocus():
            return
        self.ui.textEditLuxLabel.setPlainText(f"{lux:.1f}")

    # ------------------------------------------------------------------ capture
    def capture(self):
        cap = capture_roi_interactive(self.ui.tabLabelling, self.host.segmenter)
        if cap is None:
            return
        self.alb, self.cre, self.flags = cap.albumin, cap.creatinine, cap.flags
        lux = self.host.current_lux()
        self.capture_lux = lux if lux is not None else None
        if lux is not None:
            self.ui.textEditLuxLabel.setPlainText(f"{lux:.1f}")
        fu.show_roi_preview(self.ui.gvsegmentationAlbumin, self.scene_alb, self.alb)
        fu.show_roi_preview(self.ui.gvsegmentationCreatinine, self.scene_cre, self.cre)
        self._update_preview()

    # ------------------------------------------------------------------ file names
    def _parse_inputs(self):
        """Validate the inputs -> (index, lux_text, sample_type, code_segment, sample_code, acr). Raises ValueError."""
        ui = self.ui
        index = ui.textEditIndexLabel.toPlainText().strip()
        if not INDEX_RE.match(index):
            raise ValueError("Index is required (letters/digits/'-', max. 10 characters).")
        lux_txt = ui.textEditLuxLabel.toPlainText().strip()
        try:
            lux = float(lux_txt.replace(",", "."))
        except ValueError:
            raise ValueError("The lux value is not valid. Enter a number (e.g. 3369.2) or make sure the sensor is readable.")
        raw = ui.textEditACRLabel.toPlainText().strip()
        if self.cmb_type.currentIndex() == SAMPLE_REAL:
            if not CODE_RE.match(raw):
                raise ValueError("Sample code is required (letters/digits/'-', no spaces or '_', max. 20 characters).")
            return index, f"{lux:.1f}", "real", raw, raw, ""
        try:
            acr = float(raw.replace(",", "."))
        except ValueError:
            raise ValueError("The ACR value of the artificial urine must be a number (mg/g), e.g. 150 or 12.5.")
        if acr < 0 or acr > 100000:
            raise ValueError("The ACR value is outside the plausible range (0 – 100000 mg/g).")
        txt = format_acr(acr)
        return index, f"{lux:.1f}", "artificial", f"{ARTIFICIAL_PREFIX}{txt}", "", txt

    def _filenames(self, index, lux, code):
        return (f"{index}_{lux}_{code}_Albumin_HighRes.PNG", f"{index}_{lux}_{code}_Creatinin_HighRes.PNG")

    def _update_preview(self):
        try:
            index, lux, _, code, _, _ = self._parse_inputs()
            self.lbl_preview.setText("File name: " + self._filenames(index, lux, code)[0])
        except ValueError as exc:
            self.lbl_preview.setText(f"⚠ {exc}")

    # ------------------------------------------------------------------ save
    def save(self):
        parent = self.ui.tabLabelling
        if self.alb is None or self.cre is None:
            fu.warn(parent, "No ROI yet", "Run Capture first.")
            return
        try:
            index, lux, kind, code, sample_code, acr = self._parse_inputs()
        except ValueError as exc:
            fu.warn(parent, "Incomplete input", str(exc))
            return
        if self.capture_lux is not None:
            lux = f"{self.capture_lux:.1f}"              # lux at capture time, not when the Save button is pressed
        if self.flags and not fu.ask(parent, "Default position used",
                                     "The ROIs were taken from the default position (spot not detected). "
                                     "Saving them to the dataset is not recommended.\n\nSave anyway?"):
            return
        f_alb, f_cre = self._filenames(index, lux, code)
        out = config.DATASET_DIR
        out.mkdir(parents=True, exist_ok=True)
        p_alb, p_cre = out / f_alb, out / f_cre
        if (p_alb.exists() or p_cre.exists()) and not fu.ask(parent, "File already exists",
                                                             f"{f_alb} already exists.\n\nOverwrite?"):
            return
        params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
        ok1 = cv2.imencode(".PNG", self.alb, params)
        ok2 = cv2.imencode(".PNG", self.cre, params)
        if not (ok1[0] and ok2[0]):
            fu.warn(parent, "Failed", "The image could not be encoded as PNG.")
            return
        try:
            ok1[1].tofile(str(p_alb))
            ok2[1].tofile(str(p_cre))
            self._append_log(out / "labels_log.csv", index, lux, kind, sample_code, acr, f_alb, f_cre)
        except OSError as exc:
            fu.warn(parent, "Save failed", str(exc))
            return
        fu.info(parent, "Saved", f"Dataset saved in {out}\n\n{f_alb}\n{f_cre}\n\n"
                                     f"Albumin {self.alb.shape[1]}×{self.alb.shape[0]} px · "
                                     f"Creatinine {self.cre.shape[1]}×{self.cre.shape[0]} px")
        self._after_save(index)

    def _append_log(self, path, index, lux, kind, sample_code, acr, f_alb, f_cre):
        new = not Path(path).exists()
        if not new:
            self._upgrade_log_header(Path(path))
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(LOG_HEADER)
            w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), index, lux, kind, sample_code, acr, f_alb, f_cre,
                        f"{self.alb.shape[1]}x{self.alb.shape[0]}", f"{self.cre.shape[1]}x{self.cre.shape[0]}",
                        int(bool(self.flags))])

    @staticmethod
    def _upgrade_log_header(path):
        """Logs written by earlier versions used the spelling 'creatinin' in two column names; update the header row."""
        text = path.read_text(encoding="utf-8")
        head, sep, rest = text.partition("\n")
        new_head = head.replace("creatinin_file", "creatinine_file").replace("roi_creatinin_wh", "roi_creatinine_wh")
        if new_head != head:
            path.write_text(new_head + sep + rest, encoding="utf-8")

    def _after_save(self, index):
        self.alb = self.cre = None
        self.flags = []
        self.capture_lux = None                          # the sensor updates the lux field again
        fu.show_roi_preview(self.ui.gvsegmentationAlbumin, self.scene_alb, None)
        fu.show_roi_preview(self.ui.gvsegmentationCreatinine, self.scene_cre, None)
        if index.isdigit():
            self.ui.textEditIndexLabel.setPlainText(f"{int(index) + 1:0{max(3, len(index))}d}")
