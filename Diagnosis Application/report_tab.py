"""'Detail Report' tab: diagnosis history (date filter) + viewer for the full PDF report."""
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

import config
import frame_utils as fu
from pdf_viewer import PdfToolbar, PdfViewer, renderer_name
from workers import FunctionWorker

HEADERS = ("Time", "Patient", "Result", "P max")
COLOR = {"NORMAL": "#2E8B6F", "MICRO": "#B97A00", "MACRO": "#B33A3A", "UNCERTAIN": "#5B6577"}


def _build_pdf(pdf_path, patient, diagnosis, progress=None):
    from report_generator import generate_report    # heavy import (matplotlib, reportlab) is deferred
    return generate_report(pdf_path, patient, diagnosis, progress=progress)


class ReportTabController(QtCore.QObject):
    def __init__(self, ui, db, parent=None):
        super().__init__(parent)
        self.ui, self.db = ui, db
        self.current_id = None
        self._worker = None
        tab = ui.tabreporting

        ui.groupBox_6.setGeometry(10, 10, 575, 71)
        ui.tblreport.setGeometry(10, 90, 360, 391)
        self.viewer = PdfViewer(tab)
        self.viewer.setGeometry(380, 90, 621, 391)
        self.toolbar = PdfToolbar(self.viewer, tab)
        self.toolbar.setGeometry(595, 14, 400, 66)
        self.toolbar.openExternal.connect(self.open_external)
        self.toolbar.regenerate.connect(self.regenerate)
        self.viewer.show()
        self.toolbar.show()

        for d in (ui.dtfrom, ui.dtto):
            d.setDisplayFormat("yyyy-MM-dd")
            d.setCalendarPopup(True)
        today = QtCore.QDate.currentDate()
        ui.dtfrom.setDate(today.addDays(-30))
        ui.dtto.setDate(today)

        self._model = QtGui.QStandardItemModel(0, len(HEADERS), self)
        self._model.setHorizontalHeaderLabels(HEADERS)
        t = ui.tblreport
        t.setModel(self._model)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        t.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setAlternatingRowColors(True)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        t.selectionModel().selectionChanged.connect(self._on_selection)
        ui.btnsearch.clicked.connect(self.refresh)
        if renderer_name() is None:
            self.viewer.show_message("No PDF rendering engine found.\nInstall:  pip install pymupdf\n"
                                     "or  sudo apt install poppler-utils")
        self.refresh()

    # ------------------------------------------------------------------ history list
    def refresh(self, select_id=None):
        ui = self.ui
        if select_id is not None and ui.dtto.date() < QtCore.QDate.currentDate():
            ui.dtto.setDate(QtCore.QDate.currentDate())        # make sure the new diagnosis is shown
        rows = self.db.list_diagnoses(ui.dtfrom.date().toString("yyyy-MM-dd"),
                                      ui.dtto.date().toString("yyyy-MM-dd"))
        self._model.removeRows(0, self._model.rowCount())
        for d in rows:
            items = [QtGui.QStandardItem(str(d["timestamp"])[:16]),
                     QtGui.QStandardItem(f"{d['patient_id']} — {d['full_name']}"),
                     QtGui.QStandardItem(d["decision"]),
                     QtGui.QStandardItem(f"{100 * d['max_prob']:.0f}%")]
            items[0].setData(d["id"], QtCore.Qt.UserRole)
            items[2].setForeground(QtGui.QBrush(QtGui.QColor(COLOR.get(d["decision"], "#222"))))
            f = items[2].font()
            f.setBold(True)
            items[2].setFont(f)
            tip = (f"Lux {d['lux']:.0f} · P: N {100 * d['p_normal']:.0f}% / Mi {100 * d['p_micro']:.0f}% / "
                   f"Ma {100 * d['p_macro']:.0f}%") if d["lux"] is not None else ""
            for it in items:
                it.setToolTip(tip)
            self._model.appendRow(items)
        self.ui.tblreport.resizeColumnsToContents()
        self.ui.tblreport.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        if select_id is not None:
            self._select_row(select_id)

    def _select_row(self, diag_id):
        for r in range(self._model.rowCount()):
            if self._model.item(r, 0).data(QtCore.Qt.UserRole) == diag_id:
                self.ui.tblreport.selectionModel().blockSignals(True)
                self.ui.tblreport.selectRow(r)
                self.ui.tblreport.selectionModel().blockSignals(False)
                return

    def _on_selection(self, *_):
        rows = self.ui.tblreport.selectionModel().selectedRows()
        if rows:
            self.show_diagnosis(self._model.item(rows[0].row(), 0).data(QtCore.Qt.UserRole))

    # ------------------------------------------------------------------ PDF
    def show_diagnosis(self, diag_id, force=False):
        """Show the PDF report of diagnosis `diag_id`; it is generated first if missing (or if force)."""
        diag = self.db.get_diagnosis(diag_id)
        if not diag:
            fu.warn(self.ui.tabreporting, "Report", "Diagnosis record not found.")
            return
        self.current_id = diag_id
        self._select_row(diag_id)
        pdf = diag.get("pdf_path")
        if not force and pdf and Path(pdf).exists():
            self._load(pdf)
        else:
            self._generate(diag)

    def regenerate(self):
        if self.current_id is None:
            fu.warn(self.ui.tabreporting, "Report", "Select a history row first.")
            return
        self.show_diagnosis(self.current_id, force=True)

    def _generate(self, diag):
        if self._worker is not None and self._worker.isRunning():
            return
        patient = self.db.get_patient(diag["patient_id"])
        pdf = config.REPORT_DIR / f"UACR-{str(diag['timestamp'])[:10].replace('-', '')}-{diag['id']:05d}_{diag['patient_id']}.pdf"
        dlg = QtWidgets.QProgressDialog("Preparing the report…", None, 0, 100, self.ui.tabreporting.window())
        dlg.setWindowTitle("Building the PDF report")
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        self._dlg = dlg
        self._worker = FunctionWorker(_build_pdf, pdf, patient, diag, with_progress=True, parent=self)
        self._worker.progress.connect(lambda p, m: (dlg.setValue(p), dlg.setLabelText(f"{m}…")))
        self._worker.succeeded.connect(lambda path, d=diag: self._on_generated(str(path), d["id"]))
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_generated(self, path, diag_id):
        self._dlg.close()
        self.db.set_report_path(diag_id, path)
        self._load(path)

    def _on_failed(self, msg):
        self._dlg.close()
        self.viewer.show_message("The report could not be created.")
        fu.warn(self.ui.tabreporting, "Report creation failed", msg)

    def _load(self, path):
        try:
            self.viewer.load(path)
        except Exception as exc:
            self.viewer.show_message(f"The PDF cannot be displayed:\n{exc}")
            fu.warn(self.ui.tabreporting, "PDF viewer", str(exc))

    def open_external(self):
        if self.viewer.path:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.viewer.path))
