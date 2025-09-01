
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple


import numpy as np
from pathlib import Path
from typing import Literal, Tuple

# ───────────────── CONFIG ────────────────────────────────
USE_NORMALISED: bool = True          # flip to True for scale‑free metrics
NORMALISE_RMS_MODE: Literal["none", "range", "stdev"] = "range"  # ignored if USE_NORMALISED=False
RELATIVE_EUCLIDEAN: bool = True      # True ⇒ ||Δ|| / ||ref||
EPS: float = 1e-12                    # tolerance for safe division
# ─────────────────────────────────────────────────────────

# ─── 1)  Mecke’s published coefficients (descending degree) ─────
MECKE_PV_DOTS   = np.array([-0.82,    0.397,  -1.59,    -0.45])      # cubic
MECKE_PS_DOTS   = np.array([ 0.36,   -0.18,    0.43,    0.14,    0.83]) # quartic
MECKE_PCHI_DOTS = np.array([ 0.033,  -0.025,  0.056,   0.024])      # cubic

MECKE_PV_STRIPES   = np.array([-1.09,    0.03,   -1.39,    0.0096])
MECKE_PS_STRIPES   = np.array([-0.062,   0.035,   0.599,   -0.021,   0.55])
MECKE_PCHI_STRIPES = np.array([ 0.031,  -1.3e-05, 0.024,  -2.0e-04])

# ─── 2)  Loader for your fit‐.npz files ───────────────────────────
def load_fit_coeffs(npz_path: str | Path):
    """Load one .npz produced by weighted_polyfit_range()."""
    data = np.load(npz_path, allow_pickle=False)
    return {
        'pv':         data['poly_pv_coeffs'],
        'pv_sigma':   data['poly_pv_sigma'],
        'ps':         data['poly_ps_coeffs'],
        'ps_sigma':   data['poly_ps_sigma'],
        'pchi':       data['poly_pchi_coeffs'],
        'pchi_sigma': data['poly_pchi_sigma'],
    }

# ─── 3)  Euclidean distance and percent‐difference ───────────────
def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray, *, relative: bool = RELATIVE_EUCLIDEAN):
    dist = np.linalg.norm(vec1 - vec2)
    if not relative:
        return dist
    ref_norm = np.linalg.norm(vec2)
    return np.nan if ref_norm < EPS else dist / ref_norm

def percent_difference(vec1: np.ndarray, vec2: np.ndarray):
    denom = np.where(np.abs(vec2) < EPS, np.nan, np.abs(vec2))
    return 100.0 * np.abs(vec1 - vec2) / denom

# ─── 4)  RMS difference between two polynomials over [rmin, rmax] ─

def _curve_span(vals: np.ndarray, mode: str):
    if mode == "range":
        return np.ptp(vals)  # max-min
    if mode == "stdev":
        return np.std(vals)
    raise ValueError("mode must be 'range' or 'stdev'")

def compute_rms(coeffs1: np.ndarray, coeffs2: np.ndarray, *, rmin: float = -1.0, rmax: float = 1.0,
                num: int = 200, normalise: str = "none"):
    r = np.linspace(rmin, rmax, num)
    y1, y2 = np.polyval(coeffs1, r), np.polyval(coeffs2, r)
    rms_raw = np.sqrt(np.mean((y1 - y2) ** 2))
    if normalise == "none":
        return rms_raw
    span = _curve_span(y2, normalise)
    return np.nan if span < EPS else rms_raw / span

# ─── 5)  Orientation agreement (sign & Pearson corr) ─────────────

def orientation_agreement(coeffs_dots, coeffs_stripes, mecke_dots, mecke_stripes,
                          *, rmin=-1.0, rmax=1.0, num=500):
    r = np.linspace(rmin, rmax, num)
    ours_diff   = np.polyval(coeffs_dots, r) - np.polyval(coeffs_stripes, r)
    mecke_diff  = np.polyval(mecke_dots,  r) - np.polyval(mecke_stripes,  r)

    # 1) sign agreement
    same_sign = np.sign(ours_diff) == np.sign(mecke_diff)
    frac_same = np.count_nonzero(same_sign) / num

    # 2) Pearson correlation (zero‑meaned)
    ours_dm, mecke_dm = ours_diff - ours_diff.mean(), mecke_diff - mecke_diff.mean()
    denom = np.std(ours_dm) * np.std(mecke_dm)
    corr = np.nan if denom < EPS else float(np.dot(ours_dm, mecke_dm) / (denom * num))
    return frac_same, corr

# ─── 6)  Main routine ─────────────────────────────────────────────

def analysis():
    # 6a) Paths (modify as needed)
    dots_path = Path("data/stats/dots/s1/curves_s1_20runs_fit.npz")
    stripes_path = Path("data/stats/stripes/s1/curves_s1_20runs_fit.npz")

    dots = load_fit_coeffs(dots_path)
    stripes = load_fit_coeffs(stripes_path)

    # Helper to pick RMS normalisation mode
    rms_mode = NORMALISE_RMS_MODE if USE_NORMALISED else "none"

    print("=== Euclidean distances (coefficients) ===")
    for name, ref_d, ref_s in [
        ("pv", MECKE_PV_DOTS, MECKE_PV_STRIPES),
        ("ps", MECKE_PS_DOTS, MECKE_PS_STRIPES),
        ("pchi", MECKE_PCHI_DOTS, MECKE_PCHI_STRIPES),
    ]:
        dist_d = euclidean_distance(dots[name], ref_d)
        dist_s = euclidean_distance(stripes[name], ref_s)
        dist_ds = euclidean_distance(dots[name], stripes[name])
        label = "rel" if RELATIVE_EUCLIDEAN else "abs"
        print(f"{name}: {label} dist   dots→Mecke = {dist_d:.4f} | stripes→Mecke = {dist_s:.4f} | dots→stripes = {dist_ds:.4f}")

    print("\n=== RMS differences of entire curves [r ∈ –1..+1] ===")
    for name, ref_d, ref_s in [
        ("pv", MECKE_PV_DOTS, MECKE_PV_STRIPES),
        ("ps", MECKE_PS_DOTS, MECKE_PS_STRIPES),
        ("pchi", MECKE_PCHI_DOTS, MECKE_PCHI_STRIPES),
    ]:
        rms_d = compute_rms(dots[name], ref_d, normalise=rms_mode)
        rms_s = compute_rms(stripes[name], ref_s, normalise=rms_mode)
        rms_ds = compute_rms(dots[name], stripes[name], normalise=rms_mode)
        tag = {"none": "raw", "range": "norm‑rng", "stdev": "norm‑std"}[rms_mode]
        print(f"{name}: {tag} RMS  dots→Mecke = {rms_d:.4f} | stripes→Mecke = {rms_s:.4f} | dots→stripes = {rms_ds:.4f}")

    # Orientation agreement
    print("\n=== Orientation agreement (dots vs stripes w.r.t. Mecke) ===")
    for name, ref_d, ref_s in [
        ("pv", MECKE_PV_DOTS, MECKE_PV_STRIPES),
        ("ps", MECKE_PS_DOTS, MECKE_PS_STRIPES),
        ("pchi", MECKE_PCHI_DOTS, MECKE_PCHI_STRIPES),
    ]:
        frac, corr = orientation_agreement(dots[name], stripes[name], ref_d, ref_s)
        print(f"{name}: frac_same_sign = {frac:.3f}, corr_coef = {corr:.3f}")

    # Mecke intrinsic comparison
    print("\n=== Mecke’s intrinsic (dots vs stripes) ===")
    for name, ref_d, ref_s in [
        ("pv", MECKE_PV_DOTS, MECKE_PV_STRIPES),
        ("ps", MECKE_PS_DOTS, MECKE_PS_STRIPES),
        ("pchi", MECKE_PCHI_DOTS, MECKE_PCHI_STRIPES),
    ]:
        frac, corr = orientation_agreement(ref_d, ref_s, ref_d, ref_s)
        print(f"{name}: frac_same_sign = {frac:.3f}, corr_coef = {corr:.3f}")

if __name__ == "__main__":
    analysis()
