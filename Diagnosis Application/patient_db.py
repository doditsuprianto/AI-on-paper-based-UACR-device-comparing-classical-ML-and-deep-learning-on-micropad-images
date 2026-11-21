"""SQLite database layer: patient records (CRUD) and diagnosis history.

This module is pure Python (built-in `sqlite3`) with no PyQt dependency, so it can be tested and reused on its own.

Tables
------
patients   : demographic and clinical data of the patient (key: patient_id)
diagnoses  : one row per examination (classification result, probabilities, image and PDF paths)

Run directly to (re)populate the 10 dummy records:
    python patient_db.py --seed            # only if the patients table is empty
    python patient_db.py --seed --force    # add the dummy records that are still missing

Databases created by earlier versions of the application (Indonesian column names such as `id_pasien`) are
migrated automatically to the English schema the first time they are opened.
"""
import argparse
import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

PATIENT_FIELDS = (
    "patient_id", "full_name", "birth_date", "sex", "address", "phone", "email",
    "exam_date", "symptoms", "medical_history", "physician", "notes",
)
SEX_VALUES = ("Male", "Female")

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS patients (
    patient_id       TEXT PRIMARY KEY,
    full_name        TEXT NOT NULL,
    birth_date       TEXT,
    sex              TEXT CHECK (sex IN ('Male', 'Female') OR sex IS NULL),
    address          TEXT,
    phone            TEXT,
    email            TEXT,
    exam_date        TEXT,
    symptoms         TEXT,
    medical_history  TEXT,
    physician        TEXT,
    notes            TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at       TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS diagnoses (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id     TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE ON UPDATE CASCADE,
    timestamp      TEXT NOT NULL,
    lux            REAL,
    decision       TEXT NOT NULL,
    most_likely    TEXT NOT NULL,
    p_normal       REAL,
    p_micro        REAL,
    p_macro        REAL,
    max_prob       REAL,
    tau            REAL,
    uncertain      INTEGER NOT NULL DEFAULT 0,
    quality_ok     INTEGER NOT NULL DEFAULT 1,
    quality_flags  TEXT,
    features_json  TEXT,
    alb_image      TEXT,
    cre_image      TEXT,
    pdf_path       TEXT,
    model_version  TEXT
);
CREATE INDEX IF NOT EXISTS idx_diagnoses_timestamp ON diagnoses(timestamp);
CREATE INDEX IF NOT EXISTS idx_diagnoses_patient ON diagnoses(patient_id);
"""

# Column names of the legacy (Indonesian) schema -> current English names.
_LEGACY_PATIENT_COLUMNS = {
    "id_pasien": "patient_id", "nama_lengkap": "full_name", "tgl_lahir": "birth_date", "jenis_kelamin": "sex",
    "alamat": "address", "telp": "phone", "email": "email", "tgl_periksa": "exam_date", "gejala": "symptoms",
    "riwayat_medis": "medical_history", "dokter": "physician", "catatan": "notes",
    "created_at": "created_at", "updated_at": "updated_at",
}
_LEGACY_DIAGNOSIS_COLUMNS = {"id_pasien": "patient_id", "waktu": "timestamp"}

# 10 DUMMY (fictitious) patients. The IgG/IgM fields (left over from the toxoplasmosis application) are no longer used.
DUMMY_PATIENTS = [
    ("P-001", "Siti Rahmawati", "1968-03-14", "Female", "Jl. Soekarno Hatta 12, Malang", "081234560001",
     "siti.rahmawati@example.com", "2026-11-02", "Foamy urine, mild leg swelling",
     "Type 2 diabetes mellitus (12 years); hypertension", "dr. Andini Pratama, Sp.PD", "DUMMY DATA"),
    ("P-002", "Budi Santoso", "1975-07-22", "Male", "Jl. Veteran 45, Malang", "081234560002",
     "budi.santoso@example.com", "2026-11-02", "Fatigue",
     "Hypertension (8 years)", "dr. Andini Pratama, Sp.PD", "DUMMY DATA"),
    ("P-003", "Dewi Lestari", "1982-11-05", "Female", "Jl. Ijen 8, Malang", "081234560003",
     "dewi.lestari@example.com", "2026-11-03", "No complaints (routine check-up)",
     "Family history of diabetes", "dr. Hendra Wijaya, Sp.PD-KGH", "DUMMY DATA"),
    ("P-004", "Ahmad Fauzi", "1959-01-30", "Male", "Jl. Mayjen Panjaitan 101, Malang", "081234560004",
     "ahmad.fauzi@example.com", "2026-11-03", "Leg edema, foamy urine",
     "Type 2 diabetes mellitus (20 years); diabetic nephropathy", "dr. Hendra Wijaya, Sp.PD-KGH",
     "DUMMY DATA"),
    ("P-005", "Ratna Kusumawati", "1990-09-18", "Female", "Jl. Bunga Merak 3, Batu", "081234560005",
     "ratna.k@example.com", "2026-11-04", "No complaints",
     "None", "dr. Andini Pratama, Sp.PD", "DUMMY DATA"),
    ("P-006", "Slamet Riyadi", "1963-05-09", "Male", "Jl. Kawi 77, Malang", "081234560006",
     "slamet.riyadi@example.com", "2026-11-05", "Frequent urination at night",
     "Hypertension; dyslipidaemia", "dr. Hendra Wijaya, Sp.PD-KGH", "DUMMY DATA"),
    ("P-007", "Nurul Hidayah", "1971-12-27", "Female", "Jl. Raya Tlogomas 20, Malang", "081234560007",
     "nurul.hidayah@example.com", "2026-11-05", "Swollen eyelids in the morning",
     "Type 2 diabetes mellitus (6 years)", "dr. Andini Pratama, Sp.PD", "DUMMY DATA"),
    ("P-008", "Yoga Pramudita", "1995-04-03", "Male", "Jl. Sigura-gura 15, Malang", "081234560008",
     "yoga.p@example.com", "2026-11-06", "No complaints (screening)",
     "None", "dr. Hendra Wijaya, Sp.PD-KGH", "DUMMY DATA"),
    ("P-009", "Endang Susilowati", "1955-08-16", "Female", "Jl. Danau Toba 6, Malang", "081234560009",
     "endang.s@example.com", "2026-11-06", "Easily tired, nausea, reduced appetite",
     "Hypertension (25 years); chronic kidney disease stage 3", "dr. Hendra Wijaya, Sp.PD-KGH",
     "DUMMY DATA"),
    ("P-010", "Teguh Wibowo", "1980-02-11", "Male", "Jl. Borobudur 31, Malang", "081234560010",
     "teguh.wibowo@example.com", "2026-11-07", "High blood pressure at follow-up visit",
     "Hypertension (3 years); obesity", "dr. Andini Pratama, Sp.PD", "DUMMY DATA"),
]


class PatientError(ValueError):
    """Validation / business-rule error for patient data (the message is safe to show to the user)."""


def _clean(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def validate_patient(data, creating=True):
    """Validate and normalise patient data. Returns a new dict; raises PatientError."""
    rec = {k: _clean(data.get(k, "")) for k in PATIENT_FIELDS}
    rec["notes"] = str(data.get("notes", "") or "").strip()      # notes may span several lines
    if not rec["patient_id"]:
        raise PatientError("Patient ID is required.")
    if len(rec["patient_id"]) > 30 or any(c in rec["patient_id"] for c in '/\\:*?"<>|'):
        raise PatientError("Patient ID may have at most 30 characters and must not contain / \\ : * ? \" < > |")
    if not rec["full_name"]:
        raise PatientError("Full name is required.")
    if rec["sex"] not in SEX_VALUES:
        raise PatientError("Please select the sex.")
    for key, label in (("birth_date", "Date of birth"), ("exam_date", "Examination date")):
        if rec[key]:
            try:
                parsed = datetime.strptime(rec[key], "%Y-%m-%d").date()
            except ValueError:
                raise PatientError(f"{label} is not valid (format YYYY-MM-DD).")
            if key == "birth_date" and parsed > date.today():
                raise PatientError("Date of birth must not be in the future.")
    if rec["email"] and ("@" not in rec["email"] or "." not in rec["email"].split("@")[-1]):
        raise PatientError("E-mail address is not valid.")
    if rec["phone"] and not all(c.isdigit() or c in "+-() " for c in rec["phone"]):
        raise PatientError("Phone number may only contain digits and + - ( ).")
    return rec


def _migrate_legacy(con):
    """Convert a database written by an earlier version (Indonesian column names) to the English schema."""
    cols = [r[1] for r in con.execute("PRAGMA table_info(patients)")]
    if "id_pasien" not in cols:
        return False
    con.commit()
    con.execute("PRAGMA foreign_keys = OFF")
    con.executescript("""
        DROP INDEX IF EXISTS idx_diagnoses_waktu;
        DROP INDEX IF EXISTS idx_diagnoses_pasien;
        ALTER TABLE diagnoses RENAME TO diagnoses_legacy;
        ALTER TABLE patients RENAME TO patients_legacy;
    """)
    con.executescript(SCHEMA)
    old_cols = {r[1] for r in con.execute("PRAGMA table_info(patients_legacy)")}
    pairs = [(o, n) for o, n in _LEGACY_PATIENT_COLUMNS.items() if o in old_cols]
    sel = ", ".join("CASE sex_old WHEN 'Laki-laki' THEN 'Male' WHEN 'Perempuan' THEN 'Female' END"
                    if n == "sex" else o for o, n in pairs).replace("sex_old", "jenis_kelamin")
    con.execute(f"INSERT INTO patients ({', '.join(n for _, n in pairs)}) SELECT {sel} FROM patients_legacy")
    d_cols = [r[1] for r in con.execute("PRAGMA table_info(diagnoses_legacy)")]
    mapped = [(_LEGACY_DIAGNOSIS_COLUMNS.get(c, c), c) for c in d_cols]
    con.execute(f"INSERT INTO diagnoses ({', '.join(n for n, _ in mapped)}) "
                f"SELECT {', '.join(o for _, o in mapped)} FROM diagnoses_legacy")
    con.executescript("DROP TABLE diagnoses_legacy; DROP TABLE patients_legacy;")
    con.execute("PRAGMA foreign_keys = ON")
    con.commit()
    return True


class PatientDB:
    """Patient CRUD + diagnosis history stored in a single SQLite file."""

    def __init__(self, path, seed_dummy=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            _migrate_legacy(con)
            con.executescript(SCHEMA)
        if seed_dummy and self.count_patients() == 0:
            self.seed_dummy()

    # ------------------------------------------------------------- connection
    @contextmanager
    def _connect(self):
        con = sqlite3.connect(str(self.path), timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # ------------------------------------------------------------- patients (CRUD)
    def create_patient(self, data):
        rec = validate_patient(data)
        if self.get_patient(rec["patient_id"]):
            raise PatientError(f"Patient ID '{rec['patient_id']}' already exists. Use Update to change it.")
        cols = ", ".join(PATIENT_FIELDS)
        marks = ", ".join("?" for _ in PATIENT_FIELDS)
        with self._connect() as con:
            con.execute(f"INSERT INTO patients ({cols}) VALUES ({marks})", [rec[k] for k in PATIENT_FIELDS])
        return rec["patient_id"]

    def get_patient(self, patient_id):
        with self._connect() as con:
            row = con.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
        return dict(row) if row else None

    def list_patients(self):
        with self._connect() as con:
            rows = con.execute("SELECT * FROM patients ORDER BY patient_id").fetchall()
        return [dict(r) for r in rows]

    def update_patient(self, data):
        rec = validate_patient(data, creating=False)
        if not self.get_patient(rec["patient_id"]):
            raise PatientError(f"Patient ID '{rec['patient_id']}' was not found.")
        sets = ", ".join(f"{k} = ?" for k in PATIENT_FIELDS if k != "patient_id")
        values = [rec[k] for k in PATIENT_FIELDS if k != "patient_id"] + [rec["patient_id"]]
        with self._connect() as con:
            con.execute(f"UPDATE patients SET {sets}, updated_at = datetime('now', 'localtime') "
                        "WHERE patient_id = ?", values)

    def delete_patient(self, patient_id):
        """Delete a patient together with the whole diagnosis history (ON DELETE CASCADE).
        Returns the list of file paths (images/PDF) that belonged to the deleted diagnoses."""
        files = []
        for d in self.list_diagnoses(patient_id=patient_id):
            files += [d.get("alb_image"), d.get("cre_image"), d.get("pdf_path")]
        with self._connect() as con:
            cur = con.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
            if cur.rowcount == 0:
                raise PatientError(f"Patient ID '{patient_id}' was not found.")
        return [f for f in files if f]

    def count_patients(self):
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]

    def next_patient_id(self, prefix="P-"):
        """Next unused ID, e.g. P-011."""
        numbers = []
        for p in self.list_patients():
            pid = p["patient_id"]
            if pid.startswith(prefix) and pid[len(prefix):].isdigit():
                numbers.append(int(pid[len(prefix):]))
        return f"{prefix}{(max(numbers) + 1 if numbers else 1):03d}"

    def seed_dummy(self, force=False):
        """Insert the 10 dummy patients. force=True adds the missing ones without touching other data."""
        if not force and self.count_patients() > 0:
            return 0
        added = 0
        for row in DUMMY_PATIENTS:
            rec = dict(zip(PATIENT_FIELDS, row))
            if not self.get_patient(rec["patient_id"]):
                self.create_patient(rec)
                added += 1
        return added

    # ------------------------------------------------------------- diagnoses
    def add_diagnosis(self, patient_id, result, lux, alb_image, cre_image, model_version=""):
        """Store a classification result (dict from UACRClassifier.analyze). Returns the diagnosis id."""
        probs = result["probabilities"]
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO diagnoses (patient_id, timestamp, lux, decision, most_likely, p_normal, p_micro, p_macro,"
                " max_prob, tau, uncertain, quality_ok, quality_flags, features_json, alb_image, cre_image,"
                " model_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (patient_id, result.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"), lux,
                 result["decision"], result["most_likely_class"],
                 probs.get("NORMAL"), probs.get("MICRO"), probs.get("MACRO"),
                 result["max_probability"], result["threshold_used"], int(result["uncertain"]),
                 int(result["input_quality_ok"]), json.dumps(result.get("quality_flags", []), ensure_ascii=False),
                 json.dumps(result.get("selected_features", {})), str(alb_image), str(cre_image), model_version))
            return cur.lastrowid

    def get_diagnosis(self, diagnosis_id):
        with self._connect() as con:
            row = con.execute(
                "SELECT d.*, p.full_name FROM diagnoses d JOIN patients p USING (patient_id) WHERE d.id = ?",
                (diagnosis_id,)).fetchone()
        return dict(row) if row else None

    def list_diagnoses(self, date_from=None, date_to=None, patient_id=None):
        """Diagnosis history (newest first). date_from/date_to use the format 'YYYY-MM-DD' (inclusive)."""
        sql = ("SELECT d.*, p.full_name FROM diagnoses d JOIN patients p USING (patient_id) WHERE 1=1")
        args = []
        if date_from:
            sql += " AND date(d.timestamp) >= date(?)"
            args.append(date_from)
        if date_to:
            sql += " AND date(d.timestamp) <= date(?)"
            args.append(date_to)
        if patient_id:
            sql += " AND d.patient_id = ?"
            args.append(patient_id)
        sql += " ORDER BY d.timestamp DESC, d.id DESC"
        with self._connect() as con:
            return [dict(r) for r in con.execute(sql, args).fetchall()]

    def latest_decisions(self):
        """{patient_id: (decision, timestamp)} for the most recent diagnosis of each patient."""
        sql = ("SELECT d.patient_id, d.decision, d.timestamp FROM diagnoses d JOIN "
               "(SELECT patient_id, MAX(id) AS mid FROM diagnoses GROUP BY patient_id) m ON d.id = m.mid")
        with self._connect() as con:
            return {r["patient_id"]: (r["decision"], r["timestamp"]) for r in con.execute(sql).fetchall()}

    def set_report_path(self, diagnosis_id, pdf_path):
        with self._connect() as con:
            con.execute("UPDATE diagnoses SET pdf_path = ? WHERE id = ?", (str(pdf_path), diagnosis_id))

    def delete_diagnosis(self, diagnosis_id):
        with self._connect() as con:
            con.execute("DELETE FROM diagnoses WHERE id = ?", (diagnosis_id,))


if __name__ == "__main__":
    import config
    ap = argparse.ArgumentParser(description="Patient database utility")
    ap.add_argument("--seed", action="store_true", help="insert the 10 dummy patient records")
    ap.add_argument("--force", action="store_true", help="add the dummy records even if the table is not empty")
    args = ap.parse_args()
    db = PatientDB(config.DB_PATH, seed_dummy=False)
    if args.seed:
        print(f"{db.seed_dummy(force=args.force)} dummy patients added.")
    print(f"Total patients in {config.DB_PATH}: {db.count_patients()}")
