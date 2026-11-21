"""PDF report generator for the UACR diagnosis (reportlab + matplotlib).

The report is rebuilt deterministically from (a) the patient data, (b) the diagnosis row in SQLite and
(c) the two stored ROI images; the features are re-extracted with the same feature module as in training,
so a report from the history can always be reproduced.

Usage:
    from report_generator import generate_report
    generate_report("report.pdf", patient_dict, diagnosis_dict)
"""
import io
from datetime import date, datetime
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, Image, KeepTogether, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

import config
import report_figures as rf
import uacr_classifier as uc

NAVY, GOLD = colors.HexColor("#12263F"), colors.HexColor("#B08D57")
PAPER, MIST, INK = colors.HexColor("#F6F3EC"), colors.HexColor("#E9ECF1"), colors.HexColor("#222222")
CLASS_RGB = {"NORMAL": colors.HexColor("#2E8B6F"), "MICRO": colors.HexColor("#E0A030"),
             "MACRO": colors.HexColor("#B33A3A"), "UNCERTAIN": colors.HexColor("#5B6577")}
KDIGO_TEXT = {"NORMAL": "A1 — normal to mildly increased", "MICRO": "A2 — moderately increased (microalbuminuria)",
              "MACRO": "A3 — severely increased (macroalbuminuria)"}
TRAIN_LUX = (3369.2, 3705.0)  # illumination range of the training data
REPORT_TITLE = "UACR Micropad Analysis Report"

# --------------------------------------------------------------------------------------------- font
_FONTS_READY = False


def _register_fonts():
    global _FONTS_READY
    if _FONTS_READY:
        return
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, fn in (("Sans", "DejaVuSans.ttf"), ("Sans-B", "DejaVuSans-Bold.ttf"),
                     ("Sans-I", "DejaVuSans-Oblique.ttf"), ("Sans-BI", "DejaVuSans-BoldOblique.ttf"),
                     ("Serif", "DejaVuSerif.ttf"), ("Serif-B", "DejaVuSerif-Bold.ttf"),
                     ("Serif-I", "DejaVuSerif-Italic.ttf"), ("Serif-BI", "DejaVuSerif-BoldItalic.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(ttf / fn)))
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-BI")
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-BI")
    _FONTS_READY = True


def _styles():
    return {
        "body": ParagraphStyle("body", fontName="Serif", fontSize=9, leading=13.2, alignment=TA_JUSTIFY,
                               textColor=INK, spaceAfter=4),
        "small": ParagraphStyle("small", fontName="Sans", fontSize=7.4, leading=10, textColor=colors.HexColor("#4B5563")),
        "cap": ParagraphStyle("cap", fontName="Serif-I", fontSize=7.8, leading=10.5, alignment=TA_JUSTIFY,
                              textColor=colors.HexColor("#374151"), spaceBefore=2, spaceAfter=9),
        "h1": ParagraphStyle("h1", fontName="Sans-B", fontSize=12.5, leading=16, textColor=NAVY, spaceBefore=10,
                             spaceAfter=3, keepWithNext=1),
        "h2": ParagraphStyle("h2", fontName="Sans-B", fontSize=9.6, leading=13, textColor=GOLD, spaceBefore=6,
                             spaceAfter=2, keepWithNext=1),
        "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8, leading=10.5, textColor=INK),
        "cellb": ParagraphStyle("cellb", fontName="Sans-B", fontSize=8, leading=10.5, textColor=NAVY),
        "cellw": ParagraphStyle("cellw", fontName="Sans-B", fontSize=8, leading=10.5, textColor=colors.white),
        "bullet": ParagraphStyle("bullet", fontName="Serif", fontSize=9, leading=13, alignment=TA_JUSTIFY,
                                 leftIndent=12, bulletIndent=2, textColor=INK, spaceAfter=2),
        "center": ParagraphStyle("center", fontName="Sans", fontSize=8, alignment=TA_CENTER, textColor=INK),
    }


def P(text, style):
    return Paragraph(text, style)


def esc(v):
    return escape("" if v is None else str(v))


# --------------------------------------------------------------------------------------------- numbered canvas
class _NumberedCanvas(rl_canvas.Canvas):
    """Two-pass canvas so that the footer can show 'Page X of Y'."""
    meta = {"id": "", "model": ""}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self._decorate(total)
            super().showPage()
        super().save()

    def _decorate(self, total):
        w, h = A4
        page = self._pageNumber
        # watermark
        self.saveState()
        self.setFont("Sans-B", 54)
        self.setFillColor(colors.Color(0.07, 0.15, 0.25, alpha=0.045))
        self.translate(w / 2, h / 2)
        self.rotate(38)
        self.drawCentredString(0, 0, "RESEARCH PROTOTYPE")
        self.restoreState()
        # header (pages > 1)
        if page > 1:
            self.setFillColor(NAVY)
            self.rect(0, h - 1.15 * cm, w, 1.15 * cm, stroke=0, fill=1)
            self.setFillColor(GOLD)
            self.rect(0, h - 1.15 * cm - 1.2, w, 1.2, stroke=0, fill=1)
            self.setFillColor(colors.white)
            self.setFont("Sans-B", 8.2)
            self.drawString(2 * cm, h - 0.72 * cm, REPORT_TITLE)
            self.setFont("Sans", 7.6)
            self.drawRightString(w - 2 * cm, h - 0.72 * cm, self.meta["id"])
        # footer
        self.setStrokeColor(GOLD)
        self.setLineWidth(0.6)
        self.line(2 * cm, 1.55 * cm, w - 2 * cm, 1.55 * cm)
        self.setFillColor(colors.HexColor("#4B5563"))
        self.setFont("Sans", 6.8)
        self.drawString(2 * cm, 1.15 * cm, "Research prototype — not validated for clinical decisions; "
                                           "confirm with the laboratory method.")
        self.setFont("Sans-B", 7.4)
        self.drawRightString(w - 2 * cm, 1.15 * cm, f"Page {page} of {total}")
        self.setFont("Sans", 6.4)
        self.drawString(2 * cm, 0.8 * cm, self.meta["model"])


# --------------------------------------------------------------------------------------------- util
def _age(dob):
    try:
        d = datetime.strptime(dob, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return "-"
    t = date.today()
    return f"{t.year - d.year - ((t.month, t.day) < (d.month, d.day))} years"


def _img(png, width_cm=17.0):
    im = PILImage.open(io.BytesIO(png))
    w, h = im.size
    return Image(io.BytesIO(png), width=width_cm * cm, height=width_cm * cm * h / w)


def _logo(path, height_mm=12):
    try:
        im = PILImage.open(path)
        w, h = im.size
        return Image(str(path), width=height_mm * mm * w / h, height=height_mm * mm)
    except Exception:
        return Spacer(1, 1)


def _table(rows, widths, header=True, zebra=True, extra=None, align_right_cols=()):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3.2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2), ("LEFTPADDING", (0, 0), (-1, -1), 5),
          ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, -1), (-1, -1), 0.8, NAVY)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("LINEBELOW", (0, 0), (-1, 0), 1.2, GOLD)]
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if (i % 2) == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), PAPER))
    for c in align_right_cols:
        st.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(st + (extra or [])))
    return t


# --------------------------------------------------------------------------------------------- QC
def _qc_checks(preps, feats, ref, lux, extra_flags):
    rows = []

    def add(name, status, detail):
        rows.append((name, status, detail))

    for role, nm in (("alb", "Albumin"), ("cre", "Creatinine")):
        m = preps[role]["method"]
        add(f"{nm} ROI detection", "PASS" if m != "fallback_circle" else "WARN",
            f"method = {m}" + ("" if m != "fallback_circle" else " (ROI not located automatically)"))
    for role, nm in (("alb", "Albumin"), ("cre", "Creatinine")):
        sf = float(feats[f"{role}__spec_frac"])
        add(f"{nm} specular glare", "PASS" if sf < 0.05 else "WARN", f"glare fraction = {100 * sf:.1f}% of ROI core")
    for role, nm in (("alb", "Albumin"), ("cre", "Creatinine")):
        side = min(preps[role]["bgr"].shape[:2])
        add(f"{nm} ROI resolution", "PASS" if side >= 200 else "WARN", f"{side} px short side (≥ 200 px required)")
    out = []
    for f in ref.selected:
        v = feats[f]
        if not np.isfinite(v) or not (ref.col(f).min() <= v <= ref.col(f).max()):
            out.append(rf.pretty(f))
    add("Selected features in training range", "PASS" if not out else "WARN",
        "10/10 inside the range seen in training" if not out else f"outside range: {', '.join(out)}")
    cout = []
    for role in ("alb", "cre"):
        for k in ("L", "a", "b"):
            key = f"{role}__{k}_mean"
            if not (ref.col(key).min() <= feats[key] <= ref.col(key).max()):
                cout.append(f"{role}-{k}*")
    add("ROI colour in training colour range", "PASS" if not cout else "WARN",
        "L*, a*, b* of both pads within training range" if not cout else f"outside: {', '.join(cout)} "
        "(image colour differs from the training set: lighting, camera white balance or reagent lot)")
    if lux is None or not np.isfinite(lux):
        add("Illumination (lux)", "N/A", "no lux reading stored")
    else:
        ok = TRAIN_LUX[0] - 100 <= lux <= TRAIN_LUX[1] + 100
        add("Illumination (lux)", "PASS" if ok else "WARN",
            f"{lux:.0f} lx (training {TRAIN_LUX[0]:.0f}–{TRAIN_LUX[1]:.0f} lx, ±100 lx tolerance)")
    if extra_flags:
        add("Model input-quality flags", "WARN", "; ".join(extra_flags))
    return rows


# --------------------------------------------------------------------------------------------- narrative
def _interpretation(decision, likely, pmax, tau, qc_ok, S):
    out = []
    if decision == "UNCERTAIN":
        out.append(P(f"The highest probability is only <b>{100 * pmax:.1f}%</b> (most likely class: <b>{likely}</b>), "
                     f"below the decision threshold τ = {tau:.2f}. The system <b>does not give a definitive category</b> "
                     "because the evidence from the image is not conclusive enough. The measurement should be "
                     "<b>repeated</b> with a new micropad and illumination according to the procedure, or the sample "
                     "should be confirmed with a laboratory method (quantitative UACR).", S["body"]))
    elif decision == "NORMAL":
        out.append(P("The estimated UACR category is <b>NORMAL (&lt; 30 mg/g; KDIGO A1)</b>. The image does not show "
                     "a colour pattern consistent with albuminuria. In patients with risk factors (diabetes, "
                     "hypertension, family history of kidney disease) periodic screening remains recommended according "
                     "to the KDIGO 2024 guideline, because the detection performance for this class is still limited in the prototype.", S["body"]))
    elif decision == "MICRO":
        out.append(P("The estimated UACR category is <b>MICRO (30–300 mg/g; KDIGO A2)</b>, i.e. moderately "
                     "increased albuminuria. According to KDIGO, a finding of albuminuria must be <b>confirmed</b> by "
                     "a repeat measurement (first-morning UACR) before it is regarded as persistent (≥ 3 months). "
                     "Further evaluation — glycaemic and blood-pressure control and cardiovascular risk "
                     "factors — is the responsibility of the physician. Note that MICRO is the class "
                     "with the lowest recall of this model (≈ 0.23).", S["body"]))
    else:
        out.append(P("The estimated UACR category is <b>MACRO (&gt; 300 mg/g; KDIGO A3)</b>, i.e. severely "
                     "increased albuminuria. This result must be <b>confirmed promptly in the laboratory</b> and followed up "
                     "by a physician (assessment of kidney function/eGFR, aetiology, and the need for nephrology referral). "
                     "MACRO is the class best recognised by the model (recall ≈ 0.78), but "
                     "about 1 in 4 MACRO samples can still be misclassified.", S["body"]))
    if not qc_ok:
        out.append(P("<b>Quality note:</b> one or more image-quality checks returned WARN (see "
                     "Section 9). The classification result must be interpreted with great caution, and a "
                     "repeat measurement is recommended.", S["body"]))
    return out


# --------------------------------------------------------------------------------------------- main builder
def generate_report(pdf_path, patient, diagnosis, model_dir=None, progress=None):
    """Build the complete PDF report. Returns the PDF path.

    patient   : dict of a row of the `patients` table
    diagnosis : dict of a row of the `diagnoses` table (needs: id, timestamp, lux, decision, most_likely, p_normal, p_micro,
                p_macro, max_prob, tau, quality_ok, quality_flags, alb_image, cre_image, model_version)
    progress  : optional callback progress(percent: int, message: str)
    """
    _register_fonts()
    model_dir = Path(model_dir or config.UACR_MODEL_DIR)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    note = progress or (lambda p, m: None)
    S = _styles()

    note(3, "Loading reference data")
    ref = rf.Reference(model_dir)
    roi_params = ref.cfg["roi_params"]
    alb_path, cre_path = diagnosis["alb_image"], diagnosis["cre_image"]
    note(8, "Extracting image features")
    feats = uc.uf.process_pair(dict(path_alb=alb_path, path_cre=cre_path), roi_params)
    if not feats["ok"]:
        raise RuntimeError(f"The image could not be analysed: {feats['err']}")
    preps = {"alb": uc.uf.prepare_roi(alb_path, roi_params), "cre": uc.uf.prepare_roi(cre_path, roi_params)}

    probs = {"NORMAL": diagnosis["p_normal"], "MICRO": diagnosis["p_micro"], "MACRO": diagnosis["p_macro"]}
    decision, likely = diagnosis["decision"], diagnosis["most_likely"]
    tau, pmax = float(diagnosis["tau"]), float(diagnosis["max_prob"])
    result = {"probabilities": probs, "threshold_used": tau, "most_likely_class": likely, "decision": decision}
    lux = diagnosis.get("lux")
    lux = float(lux) if lux is not None else None
    flags = []
    try:
        import json
        for q in json.loads(diagnosis.get("quality_flags") or "[]"):
            flags += q.get("flags", []) if isinstance(q, dict) else [str(q)]
    except Exception:
        pass
    report_id = f"UACR-{str(diagnosis['timestamp'])[:10].replace('-', '')}-{int(diagnosis['id']):05d}"
    _NumberedCanvas.meta = {"id": f"{report_id}  |  {patient['patient_id']} — {patient['full_name']}",
                            "model": f"Model: {diagnosis.get('model_version') or ref.cfg['model']}"}

    figs = {}
    steps = [("prob", "Probability figure", lambda: rf.fig_probabilities(result)),
             ("band", "KDIGO category figure", lambda: rf.fig_uacr_band(result, ref)),
             ("roi", "ROI & radial profile figure", lambda: rf.fig_roi(preps, ref)),
             ("hist", "RGB histogram figure", lambda: rf.fig_rgb_hist(preps)),
             ("col", "Colour analysis figure", lambda: rf.fig_colour(feats, ref)),
             ("box", "Feature boxplot figure", lambda: rf.fig_feature_boxplots(feats, ref)),
             ("z", "Z-score heatmap figure", lambda: rf.fig_zheat(feats, ref)),
             ("pca", "PCA & neighbours figure", lambda: rf.fig_pca_neighbours(feats, ref)),
             ("tex", "Texture figure", lambda: rf.fig_texture(feats, ref)),
             ("perf", "Model performance figure", lambda: rf.fig_performance(ref))]
    for i, (key, msg, fn) in enumerate(steps):
        note(12 + int(60 * i / len(steps)), msg)
        figs[key] = fn()
    zpng, z = figs["z"]
    ppng, comp, med = figs["pca"]
    figs["z"], figs["pca"] = zpng, ppng

    note(76, "Assembling the document")
    fig_no = [0]

    def figure(png, caption, width_cm=17.0):
        fig_no[0] += 1
        return KeepTogether([_img(png, width_cm),
                             P(f"<b>Figure {fig_no[0]}.</b> {caption}", S["cap"])])

    # ---- concordance evidence
    zmean = np.abs(z).mean(1)
    z_cls = rf.CLASSES[int(np.argmin(zmean))]
    de_sum = {c: rf._delta_e(feats, ref, "alb", c) + rf._delta_e(feats, ref, "cre", c) for c in rf.CLASSES}
    col_cls = min(de_sum, key=de_sum.get)
    knn_cls = max(comp, key=comp.get)
    votes = [likely, knn_cls, z_cls, col_cls]
    agree = sum(1 for v in votes[1:] if v == likely)

    qc_rows = _qc_checks(preps, feats, ref, lux, flags)
    qc_ok = all(r[1] in ("PASS", "N/A") for r in qc_rows)

    story = []
    W = A4[0] - 4 * cm

    # ---------------- Page 1: letterhead & summary
    logos = [_logo(config.LOGO_DIR / n) for n in ("ub.png", "polinema.png", "dikti.png")]
    t = Table([logos], colWidths=[W / 3] * 3)
    t.setStyle(TableStyle([("ALIGN", (0, 0), (0, 0), "LEFT"), ("ALIGN", (1, 0), (1, 0), "CENTER"),
                           ("ALIGN", (2, 0), (2, 0), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story += [t, Spacer(1, 5)]
    band = Table([[P("<font color='white' size=17><b>UACR MICROPAD ANALYSIS REPORT</b></font>", ParagraphStyle(
        "t", fontName="Serif-B", leading=22, alignment=TA_LEFT))],
        [P("<font color='#E3C98F' size=9>Image-based classification of the urine albumin-to-creatinine ratio "
           "(NORMAL · MICRO · MACRO) — Intelligent POCT Diagnostics</font>", ParagraphStyle(
            "t2", fontName="Sans", leading=12))]], colWidths=[W])
    band.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 12),
                              ("TOPPADDING", (0, 0), (0, 0), 11), ("BOTTOMPADDING", (0, 1), (0, 1), 10),
                              ("LINEBELOW", (0, -1), (-1, -1), 2.4, GOLD)]))
    story += [band, Spacer(1, 7)]

    meta = [[P("<b>Report ID</b>", S["cell"]), P(report_id, S["cell"]),
             P("<b>Examination date</b>", S["cell"]), P(esc(diagnosis["timestamp"]), S["cell"])],
            [P("<b>Patient</b>", S["cell"]), P(esc(f"{patient['patient_id']} — {patient['full_name']}"), S["cell"]),
             P("<b>Illumination at capture</b>", S["cell"]),
             P("-" if lux is None else f"{lux:.1f} lx", S["cell"])]]
    story.append(_table(meta, [3 * cm, 6.2 * cm, 3.8 * cm, 4 * cm], header=False, zebra=False,
                        extra=[("BACKGROUND", (0, 0), (0, -1), MIST), ("BACKGROUND", (2, 0), (2, -1), MIST)]))
    story.append(Spacer(1, 8))

    # result banner
    bcol = CLASS_RGB.get(decision, NAVY)
    head = "UNCERTAIN" if decision == "UNCERTAIN" else decision
    sub = (f"Most likely: {likely} ({100 * pmax:.1f}%) &lt; τ {tau:.2f} — repeat the measurement / confirm in the lab"
           if decision == "UNCERTAIN" else f"UACR {uc.UACR_RANGE[decision]} · KDIGO {KDIGO_TEXT[decision]}")
    ban = Table([[P(f"<font color='white' size=9>CLASSIFICATION RESULT</font><br/><font color='white' size=26>"
                    f"<b>{head}</b></font>", ParagraphStyle("b", fontName="Sans", leading=30)),
                  P(f"<font color='white' size=8.5>{sub}<br/><br/>Probability: "
                    f"NORMAL {100 * probs['NORMAL']:.1f}% · MICRO {100 * probs['MICRO']:.1f}% · "
                    f"MACRO {100 * probs['MACRO']:.1f}%<br/>Image quality: "
                    f"{'good (all checks PASS)' if qc_ok else 'NEEDS REVIEW (WARN present)'}</font>",
                    ParagraphStyle("b2", fontName="Sans", leading=12))]], colWidths=[6.2 * cm, W - 6.2 * cm])
    ban.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bcol), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                             ("LEFTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 9),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("LINEBELOW", (0, 0), (-1, -1), 2.4, GOLD)]))
    story += [ban, Spacer(1, 6)]

    story.append(P("Executive Summary", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=GOLD, spaceAfter=4))
    ex = (f"On {esc(diagnosis['timestamp'])} the micropad image (albumin and creatinine spots) of patient "
          f"<b>{esc(patient['full_name'])}</b> ({esc(patient['patient_id'])}) was analysed. Ten selected image features "
          f"were computed from the two ROIs and classified by the <b>{esc(ref.cfg['model'])}</b> model into a "
          "UACR category. ")
    if decision == "UNCERTAIN":
        ex += (f"The model <b>did not give a definitive decision</b> (highest probability {100 * pmax:.1f}% for class "
               f"{likely}, below τ = {tau:.2f}). ")
    else:
        ex += (f"The model decided on category <b>{decision}</b> with a probability of {100 * pmax:.1f}% "
               f"(threshold τ = {tau:.2f}). ")
    ex += (f"Of the three independent lines of evidence based on reference data, <b>{agree} of 3</b> agree with class "
           f"{likely} (nearest neighbours: {knn_cls}; z-distance: {z_cls}; colour proximity: {col_cls}). "
           "All values are <i>research estimates</i> and must be confirmed with a laboratory method.")
    story.append(P(ex, S["body"]))
    story.append(figure(figs["prob"], "Class probabilities of the SVM-RBF model. The dashed line is the decision "
                        "threshold τ; if the highest probability lies below it, the decision is reported as UNCERTAIN."))

    # ---------------- 1. Patient data
    story.append(P("1. Patient and Examination Data", S["h1"]))
    pr = lambda k, v: [P(f"<b>{k}</b>", S["cell"]), P(esc(v) or "-", S["cell"])]
    left = [pr("Patient ID", patient["patient_id"]), pr("Full name", patient["full_name"]),
            pr("Date of birth / age", f"{patient.get('birth_date') or '-'}  ({_age(patient.get('birth_date'))})"),
            pr("Sex", patient.get("sex")), pr("Address", patient.get("address")),
            pr("Phone / e-mail", f"{patient.get('phone') or '-'} / {patient.get('email') or '-'}")]
    right = [pr("Examination date (registration)", patient.get("exam_date")), pr("Physician", patient.get("physician")),
             pr("Symptoms", patient.get("symptoms")), pr("Medical history", patient.get("medical_history")),
             pr("Notes", (patient.get("notes") or "").replace("\n", " ")), pr("Model", diagnosis.get("model_version"))]
    rows = [l + r for l, r in zip(left, right)]
    story.append(_table(rows, [2.7 * cm, 5.5 * cm, 3 * cm, 5.8 * cm], header=False,
                        extra=[("BACKGROUND", (0, 0), (0, -1), MIST), ("BACKGROUND", (2, 0), (2, -1), MIST),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))

    # ---------------- 2. Methods
    story.append(P("2. Summary of the Method", S["h1"]))
    story.append(P(
        "The micropad contains two reagent zones: an <b>albumin</b> zone and a <b>creatinine</b> zone. After the urine "
        "sample has been applied and the colour reaction is complete, the camera captures a high-resolution image. A circle detector (YOLO) "
        "finds both spots; each spot is cut out with a circular mask (radius reduced by "
        f"{int(config.RADIUS_REDUCTION_FACTOR * 100)}% to avoid the edge) to give the ROI image. From each ROI "
        "118 quantitative features are computed (RGB/HSV/CIELAB colour statistics, chromatic ratios, GLCM and LBP texture, "
        "gradient, radial profile, asymmetry). In the training stage, the 10 features with the best significance were selected "
        "(ANOVA-F, Kruskal-Wallis ε², Spearman correlation, mutual information with FDR correction, and selection "
        "frequency in nested cross-validation). "
        "The selected features are processed by a pipeline of median imputation → <i>robust scaling</i> → <i>standard scaling</i> → "
        "RBF-kernel SVM with probability output. The final decision is the class with the highest probability, "
        "<b>unless</b> that probability is below the threshold τ, in which case the result is reported as <b>UNCERTAIN</b>.",
        S["body"]))
    sel = [[P("<b>#</b>", S["cellw"]), P("<b>Feature</b>", S["cellw"]), P("<b>Patient value</b>", S["cellw"]),
            P("<b>NORMAL</b>", S["cellw"]), P("<b>MICRO</b>", S["cellw"]), P("<b>MACRO</b>", S["cellw"]),
            P("<b>Patient percentile*</b>", S["cellw"])]]
    for i, f in enumerate(ref.selected, 1):
        allv = ref.col(f)
        pct = 100.0 * float((allv < feats[f]).mean())
        sel.append([P(str(i), S["cell"]), P(rf.pretty(f), S["cell"]), P(f"{feats[f]:.4f}", S["cellb"])] +
                   [P(f"{ref.mean(f, c):.4f}", S["cell"]) for c in rf.CLASSES] + [P(f"{pct:.0f}", S["cell"])])
    story.append(KeepTogether([
        _table(sel, [0.8 * cm, 4.2 * cm, 2.6 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.2 * cm],
               align_right_cols=(2, 3, 4, 5, 6)),
        P("<b>Table 1.</b> The ten selected features: patient value and mean of each class in the training data "
          "(478 tests). *Percentile of the patient value with respect to all training data.", S["cap"])]))

    # ---------------- 3. Acquisition & ROI
    story.append(P("3. ROI Images and Radial Profile", S["h1"]))
    story.append(P(
        "The following figure shows the ROIs obtained by automatic segmentation together with the contour of the core mask (gold) and the "
        "square analysis window (dashed) used for the texture features. The radial profile describes the change in "
        "intensity from the centre to the edge of the spot; a decrease towards the edge is common because of the reagent distribution and "
        "illumination vignetting. Gold diamonds mark the mean grey intensity of the three rings (inner, middle, outer) "
        "in the training data as a reference.", S["body"]))
    story.append(figure(figs["roi"], "Albumin and creatinine ROIs from the segmentation (top) and radial intensity "
                        "profile per channel (bottom)."))

    # ---------------- 4. Colour
    story.append(P("4. Colour Analysis", S["h1"]))
    story.append(P(
        "The colorimetric reaction produces a colour change that is quantified through the pixel distribution and the CIELAB "
        "colour space. The colour difference ΔE*ab with respect to the mean of each class (CIE76) measures how close the "
        f"patient colour is to the reference classes. In this case the smallest combined ΔE belongs to class <b>{col_cls}</b> "
        f"(Σ ΔE = {de_sum[col_cls]:.1f}). Large ΔE values (&gt; ±10) on both spots indicate that the "
        "image lies far from the domain of the training data, so the model prediction is less reliable.", S["body"]))
    story.append(figure(figs["hist"], "Distribution of the RGB pixel intensities inside the ROI (mean μ and standard deviation σ "
                        "in the legend)."))
    story.append(figure(figs["col"], "Top: mean ROI colour of the patient compared with the mean of the training classes "
                        "together with ΔE*ab. Bottom: position of the patient in the a*–b* chromaticity plane relative to the distribution "
                        "of the 478 training tests."))

    # ---------------- 5. Features
    story.append(P("5. Feature-Based Evidence", S["h1"]))
    story.append(P(
        "The boxplots show the distribution of each selected feature by class; the dashed line is the patient value. "
        "The z-score heat map measures how far the patient value lies from the mean of each class in units of standard deviation; "
        f"the smallest mean |z| belongs to class <b>{z_cls}</b> ({zmean.min():.2f}).", S["body"]))
    story.append(figure(figs["box"], "Patient value against the training distribution for the ten selected features."))
    story.append(figure(figs["z"], "Z-score of the patient value with respect to the mean and standard deviation of each class (left) and "
                        "summary of the mean |z| (right). Blue/red = below/above the class mean."))
    story.append(P(
        f"In the 10-feature space (standardised), the {sum(comp.values())} nearest training tests consist of "
        f"NORMAL {comp['NORMAL']}, MICRO {comp['MICRO']}, MACRO {comp['MACRO']}; the median laboratory UACR "
        f"of these neighbours is <b>{med:.0f} mg/g</b>. This figure is exploratory: one sample "
        "has several tests, so neighbours may originate from the same sample, and the median "
        "is not a calibrated estimate.", S["body"]))
    story.append(figure(figs["pca"], "Left: PCA projection of the selected features (ellipse = 95% covariance per class; "
                        "black circles = the 15 nearest training tests). Right: laboratory UACR values of "
                        "these 15 neighbours on a log scale with the KDIGO boundaries."))

    # ---------------- 6. Texture
    story.append(P("6. Texture Analysis", S["h1"]))
    story.append(P(
        "The surface texture of the spot (homogeneity of the colour deposition) is quantified with the <i>Local Binary Pattern</i> "
        "(LBP, Ojala et al., 2002) and the <i>Gray-Level Co-occurrence Matrix</i> (GLCM, Haralick et al., 1973). Three "
        "GLCM/entropy features and three LBP features are among the ten selected features, so texture makes a "
        "large contribution to the model decision.", S["body"]))
    story.append(figure(figs["tex"], "LBP histogram (bars = patient; lines = class mean) for albumin and "
                        "creatinine, and z-scores of the albumin GLCM features with respect to all training data."))

    # ---------------- 7. Results
    story.append(P("7. Classification Result and Concordance of Evidence", S["h1"]))
    story.append(P(
        f"Decision: <b>{esc(decision)}</b>. The most likely class is <b>{likely}</b> with a probability "
        f"of <b>{100 * pmax:.1f}%</b>; the decision threshold for a single image pair is τ = <b>{tau:.2f}</b> "
        f"(<i>{'greater' if pmax >= tau else 'smaller'} than τ</i>). The following diagram places the model "
        "probabilities on the KDIGO categories and the laboratory UACR of the training data.", S["body"]))
    story.append(figure(figs["band"], "Probability mass of the model over the three UACR categories (coloured bars). The dots at the "
                        "bottom are the laboratory UACR of the 478 training tests, shown for context."))
    conc = [[P("<b>Evidence source</b>", S["cellw"]), P("<b>Class</b>", S["cellw"]), P("<b>Basis</b>", S["cellw"]),
             P("<b>Agrees?</b>", S["cellw"])],
            [P("Model SVM-RBF", S["cell"]), P(f"<b>{likely}</b>", S["cell"]),
             P(f"highest probability {100 * pmax:.1f}%", S["cell"]), P("—", S["cell"])],
            [P("Nearest neighbours (k = 15)", S["cell"]), P(knn_cls, S["cell"]),
             P(f"N {comp['NORMAL']} / Mi {comp['MICRO']} / Ma {comp['MACRO']}", S["cell"]),
             P("Yes" if knn_cls == likely else "No", S["cell"])],
            [P("Smallest z-distance", S["cell"]), P(z_cls, S["cell"]),
             P(" / ".join(f"{c[:2]} {zmean[i]:.2f}" for i, c in enumerate(rf.CLASSES)), S["cell"]),
             P("Yes" if z_cls == likely else "No", S["cell"])],
            [P("Colour proximity (Σ ΔE)", S["cell"]), P(col_cls, S["cell"]),
             P(" / ".join(f"{c[:2]} {de_sum[c]:.1f}" for c in rf.CLASSES), S["cell"]),
             P("Yes" if col_cls == likely else "No", S["cell"])]]
    story.append(KeepTogether([_table(conc, [4.6 * cm, 2.4 * cm, 7.2 * cm, 2.8 * cm]),
                               P("<b>Table 2.</b> Concordance of the model with three approaches based on reference data. "
                                 "They are not fully independent (they use the same data) and serve only "
                                 "as a plausibility check.", S["cap"])]))

    # ---------------- 8. Performance
    story.append(P("8. Model Performance and Reliability", S["h1"]))
    nc = ref.perf["nested_cv"]
    story.append(P(
        "The performance estimate was obtained from 5×2 nested cross-validation grouped by sample "
        "(feature selection and hyperparameters were chosen without looking at the test fold), on 100 urine samples / 478 tests. "
        f"Macro-F1 = <b>{nc['macro_f1'][0]:.3f} ± {nc['macro_f1'][1]:.3f}</b> (chance level ≈ "
        f"{nc['baseline_random_macro_f1'][0]:.3f}; permutation test p = {nc['permutation_p']:.3f}), balanced accuracy "
        f"{nc['balanced_accuracy'][0]:.3f}, MCC {nc['mcc'][0]:.3f}, ECE {nc['ece']:.3f}. Recall per class: "
        f"NORMAL {nc['recall']['NORMAL'][0]:.2f}, MICRO <b>{nc['recall']['MICRO'][0]:.2f}</b>, MACRO "
        f"{nc['recall']['MACRO'][0]:.2f}. This means the model is better than random guessing, but it is <b>not yet "
        "adequate as a stand-alone clinical decision tool</b>, especially for the MICRO class. Errors are concentrated in "
        "samples whose UACR is close to the 30 and 300 mg/g boundaries (error rate 67% at a distance ≤ 0.1 log10).", S["body"]))
    story.append(figure(figs["perf"], "Out-of-fold performance of the SVM-RBF at test level: precision/recall/F1 per "
                        "class, confusion matrix (row-normalised; N = NORMAL, Mi = MICRO, Ma = MACRO), and "
                        "error rate by the distance of the laboratory UACR to the nearest category boundary."))

    # ---------------- 9. QC
    story.append(P("9. Image Quality Control", S["h1"]))
    qrows = [[P("<b>Check</b>", S["cellw"]), P("<b>Status</b>", S["cellw"]), P("<b>Detail</b>", S["cellw"])]]
    extra = []
    for i, (n, s_, d) in enumerate(qc_rows, 1):
        qrows.append([P(esc(n), S["cell"]), P(f"<b>{s_}</b>", S["cell"]), P(esc(d), S["cell"])])
        extra.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor("#2E8B6F") if s_ == "PASS" else
                      (colors.HexColor("#B33A3A") if s_ == "WARN" else colors.grey)))
    story.append(KeepTogether([_table(qrows, [5.2 * cm, 1.8 * cm, 10 * cm], extra=extra),
                               P("<b>Table 3.</b> Automatic quality checks. WARN does not invalidate the result but "
                                 "lowers its reliability.", S["cap"])]))

    # ---------------- 10. Interpretation
    story.append(P("10. Interpretation and Recommendations", S["h1"]))
    story += _interpretation(decision, likely, pmax, tau, qc_ok, S)
    for b in ("This result is <b>not a diagnosis</b>. Establishing albuminuria requires a quantitative UACR measurement "
              "(e.g. Cobas) and, according to KDIGO 2024, confirmation on a repeat urine sample within 3 months.",
              "Repeating the measurement with several micropads from the same sample and averaging the probabilities "
              f"(mean τ = {ref.cfg['tau_avg']:.2f}) can improve reliability.",
              "Validity is limited to the same reagent, camera, illumination (± 3369–3705 lx) and image framing "
              "as the training data; there is no external validation yet."):
        story.append(Paragraph(b, S["bullet"], bulletText="•"))

    # ---------------- 11. Limitations + references + signature
    story.append(P("11. Main Limitations", S["h1"]))
    for b in ("Small sample size (100 samples; minority classes ≈ 25) and a single data set; the uncertainty interval "
              "of the performance estimate is wide.",
              "The UNCERTAIN threshold (τ) is exploratory and was tuned on the same out-of-fold predictions "
              "that are used to describe it.",
              "The features are image-based; the analyte information contained in the image limits the maximum "
              "classification performance.",
              "Samples with a laboratory UACR close to 30 or 300 mg/g are inherently ambiguous."):
        story.append(Paragraph(b, S["bullet"], bulletText="•"))

    story.append(P("References", S["h1"]))
    refs = [
        "KDIGO CKD Work Group. KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of "
        "Chronic Kidney Disease. <i>Kidney Int.</i> 2024;105(4S):S117–S314.",
        "Cortes C, Vapnik V. Support-vector networks. <i>Mach Learn.</i> 1995;20:273–297.",
        "Pedregosa F, et al. Scikit-learn: Machine learning in Python. <i>J Mach Learn Res.</i> 2011;12:2825–2830.",
        "Ojala T, Pietikäinen M, Mäenpää T. Multiresolution gray-scale and rotation invariant texture "
        "classification with local binary patterns. <i>IEEE Trans Pattern Anal Mach Intell.</i> 2002;24(7):971–987.",
        "Haralick RM, Shanmugam K, Dinstein I. Textural features for image classification. <i>IEEE Trans Syst Man "
        "Cybern.</i> 1973;SMC-3(6):610–621.",
        "van der Walt S, et al. scikit-image: image processing in Python. <i>PeerJ.</i> 2014;2:e453.",
    ]
    for i, r in enumerate(refs, 1):
        story.append(Paragraph(r, ParagraphStyle("ref", parent=S["body"], fontSize=8, leading=11, leftIndent=14,
                                                 firstLineIndent=-14, alignment=TA_LEFT), bulletText=f"[{i}]"))

    story.append(Spacer(1, 12))
    sig = Table([[P("Operator / Examiner", S["center"]), P("Responsible physician", S["center"])],
                 [Spacer(1, 38), Spacer(1, 38)],
                 [P("(................................)", S["center"]),
                  P(f"({esc(patient.get('physician') or '................................')})", S["center"])]],
                colWidths=[W / 2, W / 2])
    story.append(KeepTogether([sig, Spacer(1, 6), P(
        "<b>Disclaimer.</b> Research prototype. Not validated for clinical decisions; the expected performance is limited "
        "(see the model card). Confirm the result with a laboratory method.", S["small"])]))

    note(88, "Rendering the PDF")
    doc = BaseDocTemplate(str(pdf_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.9 * cm,
                          bottomMargin=2.0 * cm, title=f"{REPORT_TITLE} — {report_id}",
                          author="Intelligent POCT Diagnostics", subject="UACR micropad classification")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f", leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame])])
    doc.build(story, canvasmaker=_NumberedCanvas)
    note(100, "Done")
    return pdf_path
