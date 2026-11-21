"""'Patient Data' tab: patient CRUD in SQLite (Create / Save / Update / Delete) + patient list table."""
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

import frame_utils as fu
from patient_db import PatientError

SINGLE_LINE = ("txtpatientid", "txtfullname", "txtaddress", "txtphone", "txtemail", "txtsymptoms",
               "txtmedicalhistory", "txtphysician")
COLUMNS = ("Patient ID", "Full Name", "Birth Date", "Sex", "Phone", "Physician", "Exam Date", "Latest Result")


class _EnterToNext(QtCore.QObject):
    """Event filter: Enter/Return in a single-line QTextEdit moves the focus (no new line is inserted)."""

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.KeyPress and ev.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            obj.focusNextChild()
            return True
        return False


class PatientTabController(QtCore.QObject):
    patientsChanged = QtCore.pyqtSignal()

    def __init__(self, ui, db, parent=None):
        super().__init__(parent)
        self.ui, self.db = ui, db
        self._filter = _EnterToNext(self)
        self._selected_id = None
        self._prepare_widgets()
        self._model = QtGui.QStandardItemModel(0, len(COLUMNS), self)
        self._model.setHorizontalHeaderLabels(COLUMNS)
        t = ui.tblpatients
        t.setModel(self._model)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        t.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setAlternatingRowColors(True)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
        t.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        t.selectionModel().selectionChanged.connect(self._on_selection)
        ui.btnsave.clicked.connect(self.save)
        ui.btnupdate.clicked.connect(self.update)
        ui.btndelete.clicked.connect(self.delete)
        ui.btnnewform.clicked.connect(self.new_form)
        self.refresh()
        self.new_form()

    # ------------------------------------------------------------------ UI preparation
    def _prepare_widgets(self):
        ui = self.ui
        for name in SINGLE_LINE:
            w = getattr(ui, name)
            w.setTabChangesFocus(True)
            w.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            w.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            w.installEventFilter(self._filter)
        ui.txtnotes.setTabChangesFocus(True)
        for d in (ui.dtbirthdate, ui.dtexamdate):
            d.setDisplayFormat("yyyy-MM-dd")
            d.setCalendarPopup(True)
        ui.dtbirthdate.setMaximumDate(QtCore.QDate.currentDate())
        # IgG/IgM are left over from the toxoplasmosis application and irrelevant for UACR -> hidden.
        for name in ("label_13", "label_17", "txtigg", "txtigm"):
            getattr(ui, name).hide()
        ui.txtpatientid.setToolTip("Example: P-011 (max. 30 characters, none of / \\ : * ? \" < > |)")

    # ------------------------------------------------------------------ form <-> dict
    def read_form(self):
        ui = self.ui
        sex = "Male" if ui.rbmale.isChecked() else ("Female" if ui.rbfemale.isChecked() else "")
        return {
            "patient_id": ui.txtpatientid.toPlainText(), "full_name": ui.txtfullname.toPlainText(),
            "birth_date": ui.dtbirthdate.date().toString("yyyy-MM-dd"), "sex": sex,
            "address": ui.txtaddress.toPlainText(), "phone": ui.txtphone.toPlainText(),
            "email": ui.txtemail.toPlainText(), "exam_date": ui.dtexamdate.date().toString("yyyy-MM-dd"),
            "symptoms": ui.txtsymptoms.toPlainText(), "medical_history": ui.txtmedicalhistory.toPlainText(),
            "physician": ui.txtphysician.toPlainText(), "notes": ui.txtnotes.toPlainText(),
        }

    def fill_form(self, p):
        ui = self.ui
        ui.txtpatientid.setPlainText(p["patient_id"])
        ui.txtfullname.setPlainText(p["full_name"])
        ui.dtbirthdate.setDate(QtCore.QDate.fromString(p.get("birth_date") or "", "yyyy-MM-dd")
                              if p.get("birth_date") else QtCore.QDate(2000, 1, 1))
        self._set_sex(p.get("sex"))
        ui.txtaddress.setPlainText(p.get("address") or "")
        ui.txtphone.setPlainText(p.get("phone") or "")
        ui.txtemail.setPlainText(p.get("email") or "")
        ui.dtexamdate.setDate(QtCore.QDate.fromString(p.get("exam_date") or "", "yyyy-MM-dd")
                                 if p.get("exam_date") else QtCore.QDate.currentDate())
        ui.txtsymptoms.setPlainText(p.get("symptoms") or "")
        ui.txtmedicalhistory.setPlainText(p.get("medical_history") or "")
        ui.txtphysician.setPlainText(p.get("physician") or "")
        ui.txtnotes.setPlainText(p.get("notes") or "")

    def _set_sex(self, sex):
        for rb, label in ((self.ui.rbmale, "Male"), (self.ui.rbfemale, "Female")):
            rb.setAutoExclusive(False)
            rb.setChecked(sex == label)
            rb.setAutoExclusive(True)

    def _lock_id(self, locked):
        self.ui.txtpatientid.setReadOnly(locked)
        self.ui.txtpatientid.setStyleSheet("background:#ECEFF3;" if locked else "")

    # ------------------------------------------------------------------ actions
    def new_form(self):
        """Create: clear the form and propose the next ID."""
        self._selected_id = None
        self.ui.tblpatients.clearSelection()
        self.fill_form({"patient_id": self.db.next_patient_id(), "full_name": ""})
        self._set_sex(None)
        self.ui.dtbirthdate.setDate(QtCore.QDate(2000, 1, 1))
        self.ui.dtexamdate.setDate(QtCore.QDate.currentDate())
        self._lock_id(False)
        self.ui.txtfullname.setFocus()

    def save(self):
        """Save: store as a NEW patient."""
        if self._selected_id is not None:
            fu.info(self.ui.tabpatient, "Save",
                    "The form is showing an already saved patient.\n"
                    "Use Update to modify it, or click Create for a new patient.")
            return
        try:
            pid = self.db.create_patient(self.read_form())
        except PatientError as exc:
            fu.warn(self.ui.tabpatient, "Invalid data", str(exc))
            return
        self.refresh(select=pid)
        self.patientsChanged.emit()
        fu.info(self.ui.tabpatient, "Saved", f"Patient {pid} was saved successfully.")

    def update(self):
        if self._selected_id is None:
            fu.warn(self.ui.tabpatient, "Update", "Select a patient in the table first.")
            return
        data = self.read_form()
        data["patient_id"] = self._selected_id
        try:
            self.db.update_patient(data)
        except PatientError as exc:
            fu.warn(self.ui.tabpatient, "Invalid data", str(exc))
            return
        self.refresh(select=self._selected_id)
        self.patientsChanged.emit()
        fu.info(self.ui.tabpatient, "Updated", f"Data of patient {self._selected_id} were updated.")

    def delete(self):
        if self._selected_id is None:
            fu.warn(self.ui.tabpatient, "Delete", "Select a patient in the table first.")
            return
        pid = self._selected_id
        n = len(self.db.list_diagnoses(patient_id=pid))
        extra = f"\n\n{n} diagnosis record(s) of this patient (including images and PDFs) will also be deleted." if n else ""
        if not fu.ask(self.ui.tabpatient, "Delete patient", f"Delete patient {pid}?{extra}\n\nThis action cannot be undone."):
            return
        try:
            files = self.db.delete_patient(pid)
        except PatientError as exc:
            fu.warn(self.ui.tabpatient, "Delete", str(exc))
            return
        for f in files:                       # remove the files that belonged to the deleted history
            try:
                Path(f).unlink(missing_ok=True)
            except OSError:
                pass
        self.refresh()
        self.new_form()
        self.patientsChanged.emit()

    # ------------------------------------------------------------------ table
    def refresh(self, select=None):
        last = self.db.latest_decisions()
        self._model.removeRows(0, self._model.rowCount())
        for p in self.db.list_patients():
            res = last.get(p["patient_id"])
            cells = [p["patient_id"], p["full_name"], p.get("birth_date") or "", (p.get("sex") or "")[:1],
                     p.get("phone") or "", p.get("physician") or "", p.get("exam_date") or "",
                     f"{res[0]} ({res[1][:10]})" if res else "—"]
            self._model.appendRow([QtGui.QStandardItem(c) for c in cells])
        self.ui.tblpatients.resizeColumnsToContents()
        self.ui.tblpatients.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        if select:
            for r in range(self._model.rowCount()):
                if self._model.item(r, 0).text() == select:
                    self.ui.tblpatients.selectRow(r)
                    break

    def _on_selection(self, *_):
        rows = self.ui.tblpatients.selectionModel().selectedRows()
        if not rows:
            return
        pid = self._model.item(rows[0].row(), 0).text()
        p = self.db.get_patient(pid)
        if p:
            self._selected_id = pid
            self.fill_form(p)
            self._lock_id(True)
