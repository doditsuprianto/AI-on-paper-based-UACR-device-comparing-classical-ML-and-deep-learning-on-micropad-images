"""Intelligent POCT Diagnostics — UACR classification (NORMAL / MICRO / MACRO) from micropad images.

Run from this folder:   python3 DiagnosticsV2.py

Structure:
  DiagnosticsV2.py   main window: camera, lux sensor, tabs, and the glue between the controllers
  patient_tab.py     Patient Data tab   (SQLite CRUD)
  diagnostic_tab.py  Diagnostic tab     (capture -> ROI -> classification -> result)
  report_tab.py      Detail Report tab  (history + PDF viewer)
  labelling_tab.py   Labelling tab      (labelled ROI dataset)
  patient_db.py, uacr_classifier.py, report_generator.py, report_figures.py, pdf_viewer.py, ...
"""
import os
import sys

# Qt environment for Armbian/aarch64 (only if not already set; setdefault so that 'offscreen' mode can be used for tests).
if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_DEBUG_PLUGINS", "0")
    os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
    if "QT_PLUGIN_PATH" not in os.environ:
        os.environ["QT_PLUGIN_PATH"] = ":".join([
            "/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms", "/usr/lib/qt/plugins",
            os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
                         "site-packages", "cv2", "qt", "plugins")])

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

import config  # noqa: E402
import frame_utils as fu  # noqa: E402
from diagnostic_tab import DiagnosticTabController  # noqa: E402
from labelling_tab import LabellingTabController  # noqa: E402
from mainToxoPyQt5 import Ui_Form  # noqa: E402
from patient_db import PatientDB  # noqa: E402
from patient_tab import PatientTabController  # noqa: E402
from report_tab import ReportTabController  # noqa: E402

LIVE_VIEW_SIZE = {"diagnostic": (641, 341), "labelling": (621, 375)}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, segmenter=None, light_sensor=None, db=None, auto_start_camera=True):
        super().__init__()
        config.ensure_dirs()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle(config.APP_TITLE)

        self.segmenter = segmenter if segmenter is not None else self._create_segmenter()
        self.light_sensor = light_sensor if light_sensor is not None else self._create_light_sensor()
        self._last_lux = None
        self._last_frame = None
        self.db = db if db is not None else PatientDB(config.DB_PATH)

        self._setup_views()
        self._load_logos()

        self.patients = PatientTabController(self.ui, self.db, self)
        self.diagnostic = DiagnosticTabController(self.ui, self.db, self, self)
        self.report = ReportTabController(self.ui, self.db, self)
        self.labelling = LabellingTabController(self.ui, self, self)
        self.patients.patientsChanged.connect(self.diagnostic.reload_patients)
        self.patients.patientsChanged.connect(self.report.refresh)
        self.diagnostic.diagnosisSaved.connect(self._on_diagnosis_saved)

        if self.segmenter is not None:
            self.segmenter.frame_ready.connect(self._on_frame)
        self.ui.tabWidget.currentChanged.connect(lambda _: self._redraw_last_frame())
        self.ui.tabWidget.setCurrentWidget(self.ui.tabdiagnostictest)

        self._sensor_timer = QtCore.QTimer(self)
        self._sensor_timer.timeout.connect(self._read_sensor)
        self._sensor_timer.start(500)
        if auto_start_camera:
            QtCore.QTimer.singleShot(500, self.start_camera)

    # ------------------------------------------------------------------ API for the controllers
    def current_lux(self):
        """Latest lux value from the BH1750 sensor, or None if the sensor is not available."""
        return self._last_lux

    def show_report(self, diag_id):
        """Switch to the Detail Report tab and show the PDF of diagnosis `diag_id`."""
        self.ui.tabWidget.setCurrentWidget(self.ui.tabreporting)
        QtCore.QTimer.singleShot(0, lambda: self.report.show_diagnosis(diag_id))

    def _on_diagnosis_saved(self, diag_id):
        self.report.refresh(select_id=diag_id)
        self.patients.refresh()

    # ------------------------------------------------------------------ devices
    def _create_segmenter(self):
        try:
            from CircleSegmented import CircleSegmenter
            return CircleSegmenter(model_path=config.SEGMENTER_MODEL)
        except Exception as exc:
            print(f"Warning: the segmenter could not be created ({exc}).")
            return None

    @staticmethod
    def _create_light_sensor():
        try:
            from SimplePWM import BH1750Sensor
            return BH1750Sensor(bus=config.BH1750_BUS, address=config.BH1750_ADDRESS)
        except Exception as exc:
            print(f"Warning: BH1750 sensor not available ({exc}); lux will not be filled in automatically.")
            return None

    def start_camera(self):
        seg = self.segmenter
        if seg is None or seg.running:
            return
        for idx in config.CAMERA_INDICES:
            if seg.start(camera_index=idx):
                return
        seg.start_test_mode()
        fu.info(self, "Camera not found",
                "No camera was detected. The application runs in test mode (synthetic frames); "
                "Capture is not possible.")

    def _read_sensor(self):
        text = "--"
        if self.light_sensor is not None:
            try:
                self._last_lux = float(self.light_sensor.read_light())
                text = f"{self._last_lux:.1f}"
            except Exception as exc:
                self._last_lux = None
                print(f"Failed to read the sensor: {exc}")
        self.ui.lbllux.setText(text)
        self.ui.lblluxLabel.setText(text)
        self.labelling.set_live_lux(self._last_lux)

    # ------------------------------------------------------------------ camera display
    def _setup_views(self):
        ui = self.ui
        ui.gvcapturelive.setFixedSize(*LIVE_VIEW_SIZE["diagnostic"])
        ui.gvcaptureLabel.setFixedSize(*LIVE_VIEW_SIZE["labelling"])
        self.scene_live = fu.make_scene(ui.gvcapturelive)
        self.scene_label = fu.make_scene(ui.gvcaptureLabel)
        for v in (ui.gvsegmentationAlbumin_2, ui.gvsegmentationCreatinine_2, ui.gvsegmentationAlbumin,
                  ui.gvsegmentationCreatinine):
            v.setFixedSize(221, 221)

    def _on_frame(self, frame):
        if frame is None or frame.size == 0:
            return
        self._last_frame = frame
        self._draw_frame(frame)

    def _redraw_last_frame(self):
        if self._last_frame is not None:
            self._draw_frame(self._last_frame)

    def _draw_frame(self, frame):
        page = self.ui.tabWidget.currentWidget()
        try:
            if page is self.ui.tabdiagnostictest:
                fu.show_live_frame(self.ui.gvcapturelive, self.scene_live, frame, LIVE_VIEW_SIZE["diagnostic"])
            elif page is self.ui.tabLabelling:
                fu.show_live_frame(self.ui.gvcaptureLabel, self.scene_label, frame, LIVE_VIEW_SIZE["labelling"])
        except Exception as exc:
            print(f"Failed to display the frame: {exc}")

    def _load_logos(self):
        for view, name in ((self.ui.logoUB, "ub.png"), (self.ui.logoPolinema, "polinema.png"),
                           (self.ui.logoDikti, "dikti.png")):
            pm = QtGui.QPixmap(str(config.LOGO_DIR / name))
            if pm.isNull():
                print(f"Logo not found: {name}")
                continue
            pm = pm.scaled(max(view.width(), 1), max(view.height(), 1), QtCore.Qt.KeepAspectRatio,
                           QtCore.Qt.SmoothTransformation)
            scene = QtWidgets.QGraphicsScene()
            scene.addPixmap(pm)
            scene.setSceneRect(0, 0, pm.width(), pm.height())
            view.setScene(scene)
            view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            view.setAlignment(QtCore.Qt.AlignCenter)

    # ------------------------------------------------------------------ shutdown
    def closeEvent(self, event):
        self._sensor_timer.stop()
        if self.light_sensor is not None:
            try:
                self.light_sensor.close()
            except Exception:
                pass
        if self.segmenter is not None:
            self.segmenter.stop()
        event.accept()


def main():
    os.chdir(config.BASE_DIR)          # icons of the pyuic5-generated UI are loaded with relative paths ('logo/...')
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
