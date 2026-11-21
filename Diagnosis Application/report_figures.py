"""Scientific figures for the UACR PDF report (matplotlib, Agg backend).

All figure labels are in English (publication convention); the report narrative is in
report_generator.py. Each `fig_*` function returns PNG bytes ready to be inserted.
"""
import csv
import io
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

NAVY, GOLD, INK, GRID = "#12263F", "#B08D57", "#2B2B2B", "#D9DCE1"
CLASS_COLOR = {"NORMAL": "#2E8B6F", "MICRO": "#E0A030", "MACRO": "#B33A3A"}
CLASSES = ("NORMAL", "MICRO", "MACRO")
BAND = {"NORMAL": (1.0, 30.0), "MICRO": (30.0, 300.0), "MACRO": (300.0, 3000.0)}
CM = 1 / 2.54
FIG_W = 17.0 * CM  # full width of the text area (A4, 2 cm margins)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8, "axes.edgecolor": "#6B7280", "axes.linewidth": 0.7, "axes.titlecolor": NAVY,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 200,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
})

_PRETTY = {
    "alb__V_p10": "Alb V p10", "alb__lbp_6": "Alb LBP-6", "alb__glcm_homogeneity": "Alb GLCM homog.",
    "alb__R_std": "Alb R std", "alb__lbp_2": "Alb LBP-2", "alb__glcm_entropy": "Alb GLCM entropy",
    "alb__gray_entropy": "Alb gray entropy", "alb__grad_std": "Alb grad std", "alb__gray_p90": "Alb gray p90",
    "cre__lbp_8": "Cre LBP-8",
}


def pretty(name):
    if name in _PRETTY:
        return _PRETTY[name]
    role, _, rest = name.partition("__")
    return f"{'Alb' if role == 'alb' else 'Cre'} {rest.replace('_', ' ')}"


def to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------------------------- reference
class Reference:
    """Training reference data (478 tests) + model performance summary."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        with open(model_dir / "reference_features.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.label = np.array([r["label"] for r in rows])
        self.data = {c: np.array([float(r[c]) for r in rows]) for c in rows[0] if c != "label"}
        self.perf = json.loads((model_dir / "performance_summary.json").read_text(encoding="utf-8"))
        self.cfg = json.loads((model_dir / "deployment_config.json").read_text(encoding="utf-8"))
        self.selected = list(self.cfg["selected_features"])

    def col(self, name, cls=None):
        v = self.data[name]
        return v if cls is None else v[self.label == cls]

    def mean(self, name, cls):
        return float(np.mean(self.col(name, cls)))

    def matrix(self):
        return np.column_stack([self.data[f] for f in self.selected])

    def zscores(self, feats):
        """z-scores of the patient's selected features with respect to each class -> array (3, k)."""
        out = np.zeros((3, len(self.selected)))
        for i, c in enumerate(CLASSES):
            for j, f in enumerate(self.selected):
                v = self.col(f, c)
                out[i, j] = (feats[f] - v.mean()) / (v.std(ddof=1) + 1e-12)
        return out

    def knn(self, feats, k=15):
        """k nearest neighbours in the (standardised) selected-feature space."""
        X = self.matrix()
        mu, sd = X.mean(0), X.std(0) + 1e-12
        z = (X - mu) / sd
        q = (np.array([feats[f] for f in self.selected]) - mu) / sd
        d = np.sqrt(((z - q) ** 2).sum(1))
        idx = np.argsort(d)[:k]
        return idx, d[idx]


def _cls_legend(ax, loc="best", extra=None):
    handles = [plt.Line2D([], [], marker="o", ls="", color=CLASS_COLOR[c], label=c, ms=5) for c in CLASSES]
    if extra:
        handles += extra
    ax.legend(handles=handles, loc=loc)


# --------------------------------------------------------------------------------------------- Fig: probabilities
def fig_probabilities(result):
    probs = result["probabilities"]
    tau = result["threshold_used"]
    fig, ax = plt.subplots(figsize=(FIG_W, 2.15))
    ys = np.arange(3)[::-1]
    for y, c in zip(ys, CLASSES):
        p = probs[c]
        ax.barh(y, p, color=CLASS_COLOR[c], height=0.55, alpha=1.0 if c == result["most_likely_class"] else 0.55)
        ax.text(p + 0.012, y, f"{100 * p:.1f}%", va="center", fontsize=9, fontweight="bold", color=INK)
    ax.axvline(tau, color=NAVY, ls="--", lw=1.1)
    ax.text(tau, 2.52, f" decision threshold τ = {tau:.2f}", color=NAVY, fontsize=7.5, va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c}\n({_rng(c)})" for c in CLASSES])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, 2.9)
    ax.set_xlabel("Predicted class probability (SVM-RBF, Platt-calibrated)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Class probabilities for the analysed micropad pair", loc="left")
    return to_png(fig)


def _rng(c):
    return {"NORMAL": "<30 mg/g", "MICRO": "30–300 mg/g", "MACRO": ">300 mg/g"}[c]


# --------------------------------------------------------------------------------------------- Fig: UACR band
def fig_uacr_band(result, ref):
    fig, ax = plt.subplots(figsize=(FIG_W, 2.6))
    for c in CLASSES:
        lo, hi = BAND[c]
        ax.axvspan(lo, hi, color=CLASS_COLOR[c], alpha=0.10, lw=0)
        ax.add_patch(Rectangle((lo, 0), hi - lo, result["probabilities"][c], fc=CLASS_COLOR[c], alpha=0.75,
                               ec="white", lw=1.2))
        ax.text(math.sqrt(lo * hi), result["probabilities"][c] + 0.025, f"{100 * result['probabilities'][c]:.0f}%",
                ha="center", fontsize=8.5, fontweight="bold")
    rng = np.random.default_rng(7)
    u = ref.data["uacr"]
    for c in CLASSES:
        m = ref.label == c
        ax.scatter(np.clip(u[m], 1.2, 2900), -0.06 + rng.uniform(-0.025, 0.025, m.sum()), s=7, alpha=0.45,
                   color=CLASS_COLOR[c], lw=0)
    ax.set_xscale("log")
    ax.set_xlim(1, 3000)
    ax.set_ylim(-0.2, 1.08)
    ax.axvline(30, color=NAVY, lw=0.9)
    ax.axvline(300, color=NAVY, lw=0.9)
    ax.set_xlabel("Laboratory UACR (mg/g, log scale) — KDIGO category boundaries at 30 and 300 mg/g")
    ax.set_ylabel("Model probability")
    ax.text(1.05, -0.17, "dots: laboratory UACR of the 478 training tests", fontsize=6.8, color="#6B7280")
    ax.set_title("Probability mass over the KDIGO albuminuria categories", loc="left")
    return to_png(fig)


# --------------------------------------------------------------------------------------------- Fig: ROI
def _radial_profile(prep, bins=18):
    bgr, m = prep["bgr"], prep["mask"]
    yy, xx = np.indices(m.shape)
    cx, cy = prep["center"]
    rho = np.hypot(xx - cx, yy - cy) / max(prep["r_core"], 1.0)
    edges = np.linspace(0, 1.0, bins + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full((3, bins), np.nan)
    for b in range(bins):
        sel = m & (rho >= edges[b]) & (rho < edges[b + 1])
        if sel.sum() > 10:
            for ch in range(3):
                prof[ch, b] = bgr[..., ch][sel].mean() / 255.0
    return mid, prof[::-1]  # R, G, B


def fig_roi(preps, ref=None):
    """preps = {'alb': prepare_roi(...), 'cre': prepare_roi(...)}."""
    fig = plt.figure(figsize=(FIG_W, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.28, wspace=0.18)
    names = {"alb": "Albumin assay ROI", "cre": "Creatinine assay ROI"}
    for col, role in enumerate(("alb", "cre")):
        p = preps[role]
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(cv2.cvtColor(p["bgr"], cv2.COLOR_BGR2RGB))
        cs = cv2.findContours(p["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        for cnt in cs:
            c = cnt[:, 0, :]
            ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=GOLD, lw=1.4)
        cx, cy = p["center"]
        half = p["r_core"] / math.sqrt(2) * 0.98
        ax.add_patch(Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False, ec=NAVY, lw=1.1, ls="--"))
        ax.plot([cx], [cy], "+", color=NAVY, ms=8)
        ax.set_title(f"{names[role]}  [{p['method']}]", loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.5, -0.045, f"{p['bgr'].shape[1]}×{p['bgr'].shape[0]} px | gold: ROI core, dashed: analysis window",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.4, color="#6B7280")
        axp = fig.add_subplot(gs[1, col])
        mid, prof = _radial_profile(p)
        for ch, (nm, colr) in enumerate((("R", "#C0392B"), ("G", "#2E8B57"), ("B", "#2C6FBB"))):
            axp.plot(mid, prof[ch], color=colr, lw=1.6, label=nm)
        axp.plot(mid, 0.299 * prof[0] + 0.587 * prof[1] + 0.114 * prof[2], color=INK, lw=1.4, ls="--", label="gray")
        if ref is not None:
            for k, key in enumerate(("in", "mid", "out")):
                v = ref.data.get(f"{role}__ring_{key}")
                if v is not None:
                    axp.scatter([(2 * k + 1) / 6], [v.mean()], marker="D", s=24, color=GOLD, ec=INK, lw=0.5,
                                zorder=5, label="training gray mean (3 rings)" if k == 0 else None)
        axp.set_xlabel("Normalised radius ρ (0 = centre, 1 = ROI edge)")
        axp.set_ylabel("Mean intensity (0–1)")
        axp.set_title("Radial intensity profile", loc="left")
        lo_, hi_ = axp.get_ylim()
        axp.set_ylim(lo_, hi_ + 0.28 * (hi_ - lo_))
        axp.legend(ncol=3, loc="upper center", fontsize=6.6, columnspacing=1.0, handlelength=1.6)
    return to_png(fig)


# --------------------------------------------------------------------------------------------- Fig: RGB histogram
def fig_rgb_hist(preps):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.7), sharey=False)
    for ax, role, title in zip(axes, ("alb", "cre"), ("Albumin ROI", "Creatinine ROI")):
        bgr, m = preps[role]["bgr"], preps[role]["mask"]
        for ch, nm, colr in ((2, "R", "#C0392B"), (1, "G", "#2E8B57"), (0, "B", "#2C6FBB")):
            v = bgr[..., ch][m]
            h, e = np.histogram(v, bins=64, range=(0, 255), density=True)
            ax.plot(0.5 * (e[1:] + e[:-1]), h, color=colr, lw=1.5, label=f"{nm}  μ={v.mean():.0f}  σ={v.std():.1f}")
            ax.fill_between(0.5 * (e[1:] + e[:-1]), h, color=colr, alpha=0.12)
        ax.set_title(f"{title}: pixel-intensity distribution", loc="left")
        ax.set_xlabel("Pixel value (8-bit)")
        ax.set_ylabel("Density")
        ax.legend()
    fig.tight_layout()
    return to_png(fig)


# --------------------------------------------------------------------------------------------- Fig: swatches + Lab
def _delta_e(feats, ref, role, cls):
    d = [feats[f"{role}__{k}_mean"] - ref.mean(f"{role}__{k}_mean", cls) for k in ("L", "a", "b")]
    return math.sqrt(sum(x * x for x in d))


def fig_colour(feats, ref):
    fig = plt.figure(figsize=(FIG_W, 5.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.5, wspace=0.22)
    for col, (role, title) in enumerate((("alb", "Albumin pad"), ("cre", "Creatinine pad"))):
        ax = fig.add_subplot(gs[0, col])
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1.5)
        ax.axis("off")
        swatches = [("Patient", tuple(np.clip([feats[f"{role}__{c}_mean"] for c in "RGB"], 0, 1)), None)]
        for c in CLASSES:
            swatches.append((c, tuple(np.clip([ref.mean(f"{role}__{k}_mean", c) for k in "RGB"], 0, 1)),
                             _delta_e(feats, ref, role, c)))
        for i, (nm, rgb, de) in enumerate(swatches):
            ax.add_patch(Rectangle((i + 0.06, 0.45), 0.88, 0.95, fc=rgb, ec=NAVY if i == 0 else "#9CA3AF",
                                   lw=2.0 if i == 0 else 0.8))
            ax.text(i + 0.5, 0.33, nm, ha="center", fontsize=7.5, fontweight="bold")
            ax.text(i + 0.5, 0.13, "measured" if de is None else f"ΔE {de:.1f}", ha="center", fontsize=7,
                    color="#4B5563")
            ax.text(i + 0.5, 0.92, "#%02X%02X%02X" % tuple(int(round(255 * v)) for v in rgb), ha="center", va="center",
                    fontsize=7, color="white" if sum(rgb) < 1.6 else INK)
        ax.set_title(f"{title}: ROI colour", loc="left")
    for col, (role, title) in enumerate((("alb", "Albumin"), ("cre", "Creatinine"))):
        ax = fig.add_subplot(gs[1, col])
        for c in CLASSES:
            ax.scatter(ref.col(f"{role}__a_mean", c), ref.col(f"{role}__b_mean", c), s=9, alpha=0.4,
                       color=CLASS_COLOR[c], lw=0)
            ax.scatter(ref.mean(f"{role}__a_mean", c), ref.mean(f"{role}__b_mean", c), marker="X", s=60,
                       color=CLASS_COLOR[c], ec="white", lw=0.8, zorder=4)
        ax.scatter(feats[f"{role}__a_mean"], feats[f"{role}__b_mean"], marker="*", s=190, color=NAVY, ec=GOLD,
                   lw=1.0, zorder=6, label="Patient")
        ax.set_xlabel("a* (green–red)")
        ax.set_ylabel("b* (blue–yellow)")
        ax.set_title(f"{title}: CIELAB chromaticity plane", loc="left")
        _cls_legend(ax, loc="upper right", extra=[
            plt.Line2D([], [], marker="*", ls="", color=NAVY, mec=GOLD, ms=10, label="Patient"),
            plt.Line2D([], [], marker="X", ls="", color="#6B7280", ms=7, label="class mean")])
    return to_png(fig)


# --------------------------------------------------------------------------------------------- Fig: feature boxplots
def fig_feature_boxplots(feats, ref):
    sel = ref.selected
    fig, axes = plt.subplots(2, 5, figsize=(FIG_W, 4.5))
    for ax, f in zip(axes.ravel(), sel):
        data = [ref.col(f, c) for c in CLASSES]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.1), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
        for patch, c in zip(bp["boxes"], CLASSES):
            patch.set(facecolor=CLASS_COLOR[c], alpha=0.55, lw=0.7)
        ax.axhline(feats[f], color=NAVY, lw=1.3, ls="--")
        ax.scatter([2], [feats[f]], marker="*", s=70, color=NAVY, ec=GOLD, lw=0.6, zorder=6)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["N", "Mi", "Ma"])
        ax.set_title(pretty(f), fontsize=8, loc="left")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Patient value (dashed line / star) against the training distribution of each selected feature "
                 "(N = NORMAL, Mi = MICRO, Ma = MACRO)", fontsize=8, color=NAVY, y=1.0, x=0.01, ha="left")
    fig.tight_layout()
    return to_png(fig)


# --------------------------------------------------------------------------------------------- Fig: heatmap z-score
def fig_zheat(feats, ref):
    z = ref.zscores(feats)
    fig = plt.figure(figsize=(FIG_W, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[5.2, 0.12, 1.1], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(z, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(ref.selected)))
    ax.set_xticklabels([pretty(f) for f in ref.selected], rotation=35, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(CLASSES)
    for i in range(3):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:+.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(z[i, j]) > 1.8 else INK)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Patient distance from each class mean (z-score)", loc="left")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1]))
    cb.ax.tick_params(labelsize=7)
    ax2 = fig.add_subplot(gs[2])
    mz = np.abs(z).mean(1)
    ax2.barh(range(3), mz, color=[CLASS_COLOR[c] for c in CLASSES], height=0.6)
    ax2.set_ylim(2.5, -0.5)
    ax2.set_yticks([])
    for i, v in enumerate(mz):
        ax2.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=8, fontweight="bold")
    ax2.set_xlim(0, max(mz) * 1.35)
    ax2.set_title("mean |z|", fontsize=8.5)
    ax2.grid(False)
    return to_png(fig), z


# --------------------------------------------------------------------------------------------- Fig: PCA + neighbours
def _cov_ellipse(ax, pts, color):
    if len(pts) < 5:
        return
    c = np.cov(pts.T)
    w, v = np.linalg.eigh(c)
    ang = math.degrees(math.atan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse(pts.mean(0), 2 * math.sqrt(5.991 * w[1]), 2 * math.sqrt(5.991 * w[0]), angle=ang,
                         fill=False, ec=color, lw=1.1, ls="-", alpha=0.9))


def fig_pca_neighbours(feats, ref, k=15):
    X = ref.matrix()
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    q = (np.array([feats[f] for f in ref.selected]) - mu) / sd
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    P = Z @ Vt[:2].T
    pq = q @ Vt[:2].T
    var = S ** 2 / np.sum(S ** 2)
    idx, dist = ref.knn(feats, k)
    fig = plt.figure(figsize=(FIG_W, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    for c in CLASSES:
        m = ref.label == c
        ax.scatter(P[m, 0], P[m, 1], s=11, alpha=0.4, color=CLASS_COLOR[c], lw=0)
        _cov_ellipse(ax, P[m], CLASS_COLOR[c])
    ax.scatter(P[idx, 0], P[idx, 1], s=26, facecolors="none", edgecolors=INK, lw=0.6, label=f"{k} nearest tests")
    ax.scatter(pq[0], pq[1], marker="*", s=230, color=NAVY, ec=GOLD, lw=1.1, zorder=7, label="Patient")
    ax.set_xlabel(f"PC1 ({100 * var[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100 * var[1]:.1f}% variance)")
    ax.set_title("PCA of the 10 selected features", loc="left")
    _cls_legend(ax, extra=ax.get_legend_handles_labels()[0], loc="best")
    ax2 = fig.add_subplot(gs[1])
    rng = np.random.default_rng(3)
    for c in CLASSES:
        lo, hi = BAND[c]
        ax2.axhspan(lo, hi, color=CLASS_COLOR[c], alpha=0.09, lw=0)
    comp = {c: 0 for c in CLASSES}
    for i in idx:
        c = str(ref.label[i])
        comp[c] += 1
        ax2.scatter(rng.uniform(-0.3, 0.3), min(max(ref.data["uacr"][i], 1.2), 2900), s=28, color=CLASS_COLOR[c],
                    alpha=0.85, ec="white", lw=0.5)
    med = float(np.median(ref.data["uacr"][idx]))
    ax2.axhline(med, color=NAVY, ls="--", lw=1.1)
    ax2.text(0.42, med * 1.08, f"median {med:.0f} mg/g", fontsize=7.5, color=NAVY, ha="right")
    ax2.set_yscale("log")
    ax2.set_ylim(1, 3000)
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_xticks([])
    ax2.grid(axis="x", visible=False)
    ax2.set_ylabel("Laboratory UACR of neighbours (mg/g)")
    ax2.set_title(f"Reference UACR of the {k} nearest tests", loc="left")
    txt = "  ".join(f"{c[:2]} {comp[c]}" for c in CLASSES)
    ax2.text(0.0, -0.04, txt, transform=ax2.transAxes, ha="center", va="top", fontsize=7.5)
    return to_png(fig), comp, med


# --------------------------------------------------------------------------------------------- Fig: texture
def fig_texture(feats, ref):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.85), gridspec_kw=dict(width_ratios=[1, 1, 1.05], wspace=0.5))
    for ax, role, title in zip(axes[:2], ("alb", "cre"), ("Albumin", "Creatinine")):
        x = np.arange(10)
        ax.bar(x, [feats[f"{role}__lbp_{i}"] for i in x], color=NAVY, alpha=0.85, width=0.6, label="Patient")
        for c in CLASSES:
            ax.plot(x, [ref.mean(f"{role}__lbp_{i}", c) for i in x], marker="o", ms=3.2, lw=1.0,
                    color=CLASS_COLOR[c], label=c)
        ax.set_xticks(x)
        ax.set_xlabel("LBP uniform pattern bin")
        ax.set_ylabel("Relative frequency")
        ax.set_title(f"{title}: LBP", loc="left")
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.35 * (hi_ - lo_))
        ax.legend(ncol=3, fontsize=6.2, loc="upper center", columnspacing=0.8, handlelength=1.2)
    ax = axes[2]
    names = ["contrast", "homogeneity", "energy", "correlation", "entropy"]
    allv = {n: ref.data[f"alb__glcm_{n}"] for n in names}
    y = np.arange(len(names))[::-1]
    for c in CLASSES:
        zc = [(ref.mean(f"alb__glcm_{n}", c) - allv[n].mean()) / (allv[n].std() + 1e-12) for n in names]
        ax.scatter(zc, y, s=26, color=CLASS_COLOR[c], label=c, zorder=3)
    zp = [(feats[f"alb__glcm_{n}"] - allv[n].mean()) / (allv[n].std() + 1e-12) for n in names]
    ax.scatter(zp, y, marker="*", s=110, color=NAVY, ec=GOLD, lw=0.8, zorder=5, label="Patient")
    ax.axvline(0, color="#9CA3AF", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("z-score vs all training tests")
    ax.set_title("Albumin: GLCM", loc="left")
    ax.set_ylim(-0.6, len(names) + 0.9)
    ax.legend(ncol=3, fontsize=6.2, loc="upper center", columnspacing=0.8, handlelength=1.2)
    return to_png(fig)


# --------------------------------------------------------------------------------------------- Fig: model performance
def fig_performance(ref):
    perf = ref.perf
    fig = plt.figure(figsize=(FIG_W, 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 0.95], wspace=0.6)
    ax = fig.add_subplot(gs[0])
    x = np.arange(3)
    for k, (m, colr) in enumerate((("precision", "#7A8CA5"), ("recall", GOLD), ("f1", NAVY))):
        vals = [perf["per_class"][c][m] for c in CLASSES]
        ax.bar(x + (k - 1) * 0.26, vals, 0.25, color=colr, label=m.capitalize() if m != "f1" else "F1")
        for xi, v in zip(x + (k - 1) * 0.26, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6, rotation=90, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-class performance", loc="left")
    ax.legend(ncol=3, loc="upper left", fontsize=6.5)
    ax.grid(axis="x", visible=False)
    ax = fig.add_subplot(gs[1])
    cm = np.array(perf["confusion_test"], dtype=float)
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n(n={int(cm[i, j])})", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["N", "Mi", "Ma"])
    ax.set_yticks(range(3))
    ax.set_yticklabels(["N", "Mi", "Ma"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix", loc="left")
    ax.grid(False)
    ax = fig.add_subplot(gs[2])
    labs = [d[0] for d in perf["error_by_distance_to_threshold"]]
    vals = [d[1] for d in perf["error_by_distance_to_threshold"]]
    ax.bar(range(4), vals, color=[GOLD, "#C9A56B", "#9AA7B8", NAVY][::-1], width=0.65)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"{v:.0f}%", ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs, rotation=20)
    ax.set_xlabel("|log10 UACR − log10 threshold|")
    ax.set_ylabel("Error rate (%)")
    ax.set_ylim(0, 85)
    ax.set_title("Errors near cut-offs", loc="left")
    ax.grid(axis="x", visible=False)
    return to_png(fig)
