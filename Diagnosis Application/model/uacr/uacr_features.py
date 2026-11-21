"""
uacr_features.py - image analysis shared by the research notebook and the deployed device.

ONE source of truth: the notebook UACR_Integrated_Pipeline.ipynb writes this file, imports it, and uses it to extract
the features for the study; the very same file is shipped in the deployment bundle and used by uacr_inference.py.

Content
  read_image / detect_roi_mask / prepare_roi : circular-ROI detection and preprocessing
  image_features                              : 118 quantitative features of one ROI image
  process_pair                                : features of an albumin + creatinine image pair
                                                (alb__*, cre__*, and cross-image x__* features)
"""
import hashlib
import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy import stats as sps
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

MODULE_VERSION = "1.0"
EPS = 1e-6

DEFAULT_PARAMS = dict(BG_DIST=18.0, ROI_ERODE_FRAC=0.10, MAX_SIDE=512, SPEC_V=0.985, SPEC_S=0.10)


def _P(params):
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: params[k] for k in DEFAULT_PARAMS if k in params})
    return p


# ----------------------------------------------------------------------------------------------------------------
# ROI detection
# ----------------------------------------------------------------------------------------------------------------
def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    alpha = None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha, img = img[..., 3], img[..., :3]
    return np.ascontiguousarray(img), alpha


def _largest_component(m):
    n, lab_, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return lab_ == k


def detect_roi_mask(bgr, alpha=None, params=None):
    P = _P(params)
    h, w = bgr.shape[:2]
    method, mask = None, None
    if alpha is not None and (alpha < 10).mean() > 0.02:
        mask, method = alpha > 127, "alpha"
    if mask is None:
        b = 5
        border = np.concatenate([bgr[:b].reshape(-1, 3), bgr[-b:].reshape(-1, 3),
                                 bgr[:, :b].reshape(-1, 3), bgr[:, -b:].reshape(-1, 3)]).astype(np.float32)
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(bgr.astype(np.float32) - bg, axis=2)
        m = (dist > P["BG_DIST"]).astype(np.uint8)
        k = max(3, int(min(h, w) * 0.01) | 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
        m = ndi.binary_fill_holes(_largest_component(m.astype(bool)))
        mask, method = m, "background"
    frac = mask.mean()
    if frac < 0.05 or frac > 0.97:
        yy, xx = np.ogrid[:h, :w]
        r = min(h, w) / 2.0
        mask = (yy - (h - 1) / 2.0) ** 2 + (xx - (w - 1) / 2.0) ** 2 <= r ** 2
        method = "fallback_circle"
    return mask.astype(bool), method


def prepare_roi(path, params=None, keep_full=False):
    P = _P(params)
    bgr0, alpha = read_image(path)
    mask0, method = detect_roi_mask(bgr0, alpha, P)
    area = mask0.sum()
    r_eq = math.sqrt(area / math.pi)
    dt = cv2.distanceTransform(mask0.astype(np.uint8), cv2.DIST_L2, 5)
    core = dt > P["ROI_ERODE_FRAC"] * r_eq
    if core.sum() < 200:
        core = mask0
    hsv0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2HSV)
    spec = (hsv0[..., 2] > P["SPEC_V"] * 255) & (hsv0[..., 1] < P["SPEC_S"] * 255)
    spec_frac = float(spec[core].mean())
    if spec_frac < 0.25:
        core = core & ~spec
    ys, xs = np.where(core)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, m = bgr0[y0:y1, x0:x1], core[y0:y1, x0:x1]
    s = P["MAX_SIDE"] / max(crop.shape[:2])
    if s < 1:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    cy, cx = [float(v.mean()) for v in np.where(m)]
    r_core = math.sqrt(m.sum() / math.pi)
    half = int(r_core / math.sqrt(2) * 0.98)
    if half >= 8:
        ya, yb = max(0, int(cy) - half), min(crop.shape[0], int(cy) + half)
        xa, xb = max(0, int(cx) - half), min(crop.shape[1], int(cx) + half)
        square = crop[ya:yb, xa:xb]
    else:
        square = crop
    out = dict(bgr=crop, mask=m, square=square, method=method, r_eq=float(r_eq), area_frac=float(area / mask0.size),
               spec_frac=spec_frac, center=(cx, cy), r_core=r_core)
    if keep_full:
        out.update(orig=bgr0, full_mask=mask0, core_full=core)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Feature extraction (118 features per image)
# ----------------------------------------------------------------------------------------------------------------
def _stats(v, p, out):
    v = np.asarray(v, dtype=np.float64)
    q = np.percentile(v, [10, 25, 50, 75, 90])
    sd = v.std()
    out[f"{p}_mean"] = v.mean(); out[f"{p}_std"] = sd; out[f"{p}_median"] = q[2]
    out[f"{p}_p10"] = q[0]; out[f"{p}_p90"] = q[4]; out[f"{p}_iqr"] = q[3] - q[1]
    out[f"{p}_skew"] = float(sps.skew(v)) if sd > 1e-9 else 0.0
    out[f"{p}_kurt"] = float(sps.kurtosis(v)) if sd > 1e-9 else 0.0


def image_features(d):
    bgr, m, sq = d["bgr"], d["mask"], d["square"]
    f32 = bgr.astype(np.float32) / 255.0
    rgb = f32[..., ::-1]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(f32, cv2.COLOR_BGR2Lab)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    chan = {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2], "S": hsv[..., 1] / 255.0, "V": hsv[..., 2] / 255.0,
            "L": lab[..., 0], "a": lab[..., 1], "b": lab[..., 2], "gray": gray / 255.0}
    out = {}
    for nm, arr in chan.items():
        _stats(arr[m], nm, out)
    # circular hue
    ang = np.deg2rad(hsv[..., 0][m].astype(np.float64) * 2.0)
    hc, hs = np.cos(ang).mean(), np.sin(ang).mean()
    out["hue_cos"], out["hue_sin"], out["hue_R"] = hc, hs, math.hypot(hc, hs)
    # colour indices
    mR, mG, mB = out["R_mean"], out["G_mean"], out["B_mean"]
    tot = mR + mG + mB + EPS
    out.update(chr_r=mR / tot, chr_g=mG / tot, chr_b=mB / tot,
               ratio_RG=mR / (mG + EPS), ratio_GB=mG / (mB + EPS), ratio_RB=mR / (mB + EPS),
               abs_R=-math.log10(max(mR, 1e-4)), abs_G=-math.log10(max(mG, 1e-4)), abs_B=-math.log10(max(mB, 1e-4)),
               abs_gray=-math.log10(max(out["gray_mean"], 1e-4)), exg=2 * mG - mR - mB)
    La, aa, bb = out["L_mean"], out["a_mean"], out["b_mean"]
    ch = math.hypot(aa, bb)
    out.update(chroma=ch, lab_hue_sin=bb / (ch + EPS), lab_hue_cos=aa / (ch + EPS),
               dE_white=math.sqrt((100 - La) ** 2 + aa ** 2 + bb ** 2))
    Lp, ap, bp = lab[..., 0][m], lab[..., 1][m], lab[..., 2][m]
    out["dE_spread"] = float(np.sqrt((Lp - La) ** 2 + (ap - aa) ** 2 + (bp - bb) ** 2).mean())
    out["gray_cv"] = out["gray_std"] / (out["gray_mean"] + EPS)
    # texture on the inscribed square
    g8 = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
    q = (g8 // 8).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    for pr in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        val = np.nan_to_num(graycoprops(glcm, pr), nan=1.0 if pr == "correlation" else 0.0)
        out[f"glcm_{pr}"] = float(val.mean())
    ents = []
    for di in range(glcm.shape[2]):
        for ai in range(glcm.shape[3]):
            g = glcm[:, :, di, ai]; g = g[g > 0]
            ents.append(float(-(g * np.log2(g)).sum()))
    out["glcm_entropy"] = float(np.mean(ents))
    hist, _ = np.histogram(gray[m], bins=64, range=(0, 255)); p = hist / max(hist.sum(), 1); p = p[p > 0]
    out["gray_entropy"] = float(-(p * np.log2(p)).sum())
    lbp = local_binary_pattern(g8, 8, 1, "uniform")
    lh = np.bincount(lbp.astype(int).ravel(), minlength=10)[:10].astype(float); lh /= max(lh.sum(), 1)
    for i, v in enumerate(lh):
        out[f"lbp_{i}"] = float(v)
    gf = g8.astype(np.float32)
    gx, gy = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.hypot(gx, gy)
    out["grad_mean"], out["grad_std"] = float(gm.mean()), float(gm.std())
    out["lap_var"] = float(cv2.Laplacian(gf, cv2.CV_32F).var())
    # radial profile & asymmetry
    yy, xx = np.indices(m.shape)
    cx, cy = d["center"]
    rho = np.hypot(xx - cx, yy - cy) / max(d["r_core"], 1.0)
    gn = gray / 255.0
    rings = []
    for lo, hi in [(0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)]:
        mm = m & (rho >= lo) & (rho < hi)
        rings.append(float(gn[mm].mean()) if mm.sum() > 20 else float(gn[m].mean()))
    out["ring_in"], out["ring_mid"], out["ring_out"] = rings
    out["ring_grad"] = rings[2] - rings[0]
    lm, rm = m & (xx < cx), m & (xx >= cx)
    tm, bm = m & (yy < cy), m & (yy >= cy)
    out["asym_lr"] = float(gn[lm].mean() - gn[rm].mean()) if lm.sum() > 20 and rm.sum() > 20 else 0.0
    out["asym_tb"] = float(gn[tm].mean() - gn[bm].mean()) if tm.sum() > 20 and bm.sum() > 20 else 0.0
    return out


POS_KEYS = ["R_mean", "G_mean", "B_mean", "S_mean", "V_mean", "L_mean", "gray_mean", "chr_r", "chr_g", "chr_b",
            "abs_R", "abs_G", "abs_B", "abs_gray", "chroma", "dE_white", "gray_std", "glcm_contrast", "glcm_homogeneity"]
SIGNED_KEYS = ["a_mean", "b_mean", "exg"]


def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def process_pair(row, params=None):
    """Features of one test (albumin image + creatinine image). `row` needs path_alb and path_cre
    (seq, sample, lux are copied to the result when present)."""
    res = dict(seq=row.get("seq", 0), sample=row.get("sample", 0), lux=row.get("lux", float("nan")), ok=True, err="")
    feats = {}
    try:
        for role, key in (("alb", "path_alb"), ("cre", "path_cre")):
            d = prepare_roi(row[key], params)
            f = image_features(d)
            feats[role] = f
            res[f"{role}__roi_method"] = d["method"]; res[f"{role}__roi_area"] = d["area_frac"]
            res[f"{role}__spec_frac"] = d["spec_frac"]; res[f"md5_{role}"] = md5_file(row[key])
        for role in ("alb", "cre"):
            for k, v in feats[role].items():
                res[f"{role}__{k}"] = v
        for k in POS_KEYS:
            res[f"x__{k}_diff"] = feats["alb"][k] - feats["cre"][k]
            res[f"x__{k}_lr"] = math.log(feats["alb"][k] + EPS) - math.log(feats["cre"][k] + EPS)
        for k in SIGNED_KEYS:
            res[f"x__{k}_diff"] = feats["alb"][k] - feats["cre"][k]
    except Exception as e:
        res["ok"], res["err"] = False, repr(e)
    return res
