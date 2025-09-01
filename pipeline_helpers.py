"""
=== Pipeline Helpers for Dots vs Stripes Analysis ===
Common routines for configuring, running, and plotting the dots-vs-stripes
pipelines. 
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import analysis_utils as au
import image_processing as ip
from tqdm import tqdm
from plotcode import save_raw_data_plots, plot_ds_vs_mecke
import seaborn as sns
import re
import os
from dataclasses import replace


def config_dots_stripes(analysis_params):
    if analysis_params.blur_type.lower() not in ("gaussian", "median", "mecke"):
        raise ValueError("blur_type must be 'gaussian', 'median' or 'mecke'")
    
    if analysis_params.blur_type.lower() == "mecke":
        folder_name = f"f{analysis_params.fwhm_px}_g{analysis_params.gamma}"

    elif analysis_params.blur_type.lower() == "gaussian":
        folder_name = f"s{analysis_params.sigma}"

    elif analysis_params.blur_type.lower() == "median":
        folder_name = f"k{analysis_params.ksize}"    

    dots_cfg = replace(
        analysis_params,
        input_stack = r"data/stats/dots/sigma0/images_stack_20runs_June4.npz",
        folder_name = "dots/" + folder_name,  # e.g. "dots/f1.00"
    )

    stripes_cfg = replace(
        analysis_params,
        input_stack = r"data/stats/stripes/sigma0/images_stack_20runs_June6.npz",
        folder_name = "stripes/" + folder_name,
    )

    return dots_cfg, stripes_cfg



def load_as_list(input_file):
    """
    Load either
        • .npz with key "images" (stack or single 2-D image), or
        • .npy file (single 2-D array),
    and always return a list of 2-D float32 arrays in [0,1].
    """
    p = Path(input_file)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix == ".npz":
        data = np.load(p)
        if "images" not in data:
            raise KeyError(f"{p} does not contain key 'images'")
        arr = data["images"]
    elif p.suffix == ".npy":
        arr = np.load(p)
    else:
        raise ValueError("input_file must be .npz or .npy")

    if arr.ndim == 3:                     # stack
        images_norm = [a.astype(np.float32) for a in arr]
    elif arr.ndim == 2:                   # single
        images_norm = [arr.astype(np.float32)]
    else:
        raise ValueError(f"Expected 2-D or 3-D array, got shape {arr.shape}")

    # Ensure [0,1] range (safety)
    images_norm = [np.clip(im, 0.0, 1.0) for im in images_norm]
    return images_norm


def analyse_curves_once(stack_npz, *, levels=256):
    """
    Given a processed (blurred) stack, run just the parts of the
    pipeline that create *_stats.npz and return its path.
    """
    tmp_dir = Path(stack_npz).parent
    images_norm = load_as_list(stack_npz)
    curves = au.process_images(images_norm, levels=levels,
                               outdir=tmp_dir, tag="auto")
    stats  = au.random_error(curves)
    return stats


def collect_pipeline_paths(pipeline_dir: str) -> dict:
    """
    Scan a finished pipeline directory and return the same dict structure
    produced by run_pipeline(): {blurred, curves, stats, fits}.
    Raises FileNotFoundError if any piece is missing.
    """
    d = Path(pipeline_dir)

    # blurred_*.npz  (take the first match)
    blurred = next(d.glob("blurred_*.npz"), None)
    if blurred is None:
        raise FileNotFoundError("no blurred_*.npz in", d)

    # threshold_curves_*.npz
    curves = next(d.glob("curves_*.npz"), None)
    if curves is None:
        raise FileNotFoundError("no curves_*.npz in", d)

    # derive expected stems
    stem     = curves.stem                   # e.g. curves_s1_20runs
    stats    = d / f"{stem}_stats.npz"
    fits     = d / f"{stem}_fit.npz"

    for p in (stats, fits):
        if not p.exists():
            raise FileNotFoundError(p)

    return dict(blurred=blurred, curves=curves, stats=stats, fits=fits)


def run_pipeline(input_stack,
                 folder_name : str,
                 *,
                 blur_type : str="gaussian",
                 sigma : float=1, # for "gaussian" blur
                 ksize : int=3, # for "median" blur
                 fwhm_px : float=2.5,  # for "mecke" blur
                 gamma : float=0.6,    # for "mecke" blur
                 levels : int =256,
                 outroot="data/stats/",
                 tag=None):
    """
    Input stack can be a single image or a stack of images, in .npy or .npz 
    format.

    This function runs the full pipeline on a given input stack:
    (1) blur the stack   →  blurred_{tag}.npz
    (2) threshold curves →  curves_{tag}.npz v, s, chi for each run
    (3) stats            →  curves_{tag}_stats.npz
    (4) polynomial fits  →  curves_{tag}_fit.npz
    Returns a dict with all four paths.

    MEDIAN BLUR:
    -----------
    Median blur takes an odd integer kernel size (ksize) ≥ 3 and a median filter
    is applied to each image. It replaces each pixel with the median value
    within the ksize×ksize square neighbourhood centered on that pixel. This
    ensures that output pixel is an existing value, and can reduce noise. In 
    this case, results thus far are less effective than Gaussian blur.


    MECKE BLUR:
    -----------
    Mecke blur parameters (fwhm_px, gamma) are an experimental addition. This
    allows for tuning blur using full width at half maximum (FWHM) in pixels,
    and gamma controls the shape of the kernel (0.5 < gamma < 1.0). Results 
    thus far are poorer than a simple Gaussian blur.
    """
    outroot = Path(outroot) / folder_name
    outroot.mkdir(parents=True, exist_ok=True)

    # 1) BLUR  ------------------------------------------------------------
    if blur_type.lower() == "gaussian":
        blur_tag = f"s{sigma}"          # e.g. σ=1   → “s1”
    elif blur_type.lower() == "median":
        blur_tag = f"k{ksize}"          # e.g. ksize=3 → “k3”
    elif blur_type.lower() == "mecke":
        blur_tag = f"f{fwhm_px}_g{gamma}"  
        # e.g. fwhm_px=2.5, gamma=0.6 → “f2.50_g0.60”}"

    else: # unknown blur type
        raise ValueError("blur_type must be 'gaussian', 'median' or 'mecke'")
    
    bname = f"blurred_{(tag + '_') if tag else ''}{blur_tag}.npz"
    blurred_npz = ip.blur_image_stack(
        input_stack,
        outroot / bname,
        blur_type=blur_type,
        sigma=sigma,
        ksize=ksize
    )

    # 2) CURVES -----------------------------------------------------------
    images_norm = load_as_list(blurred_npz)
    curves_npz  = au.process_images(
        images_norm,
        levels=levels,
        outdir=outroot,
        tag=f"{(tag + '_') if tag else ''}{blur_tag}"
    )

    # 3) STATS  -----------------------------------------------------------
    stats_npz = au.random_error(curves_npz)

    # 4) FITS   -----------------------------------------------------------
    fit_npz   = au.fit_polynomials(stats_npz)

    return dict(blurred=blurred_npz,
                curves=curves_npz,
                stats =stats_npz,
                fits  =fit_npz)


    
def run_dots_vs_stripes_plots(dots_results,        # dict or path – MAY be None
                              stripes_results=None,# ← DEFAULT None
                              *,
                              n_points: int = 100,
                              subdir: str = "comparison_plots"):
    """
    Build plot PNGs from one or two pipelines.

    * If **both** are provided → dots-vs-stripes comparison plots.
    * If **only one** is provided → plots for that dataset alone.
    """

    # ---------- Normalise inputs to dicts ------------------------------
    if dots_results is not None and not isinstance(dots_results, dict):
        dots_results = collect_pipeline_paths(dots_results)
    if stripes_results is not None and not isinstance(stripes_results, dict):
        stripes_results = collect_pipeline_paths(stripes_results)

    if dots_results is None and stripes_results is None:
        raise ValueError("Give at least one pipeline (dots or stripes)")

    # ---------- Where to save the PNGs ---------------------------------
    if dots_results and stripes_results:             # COMPARISON MODE
        dots_dir,    stripes_dir = (Path(dots_results["stats"]).parent,
                                    Path(stripes_results["stats"]).parent)

        common_parent = Path(os.path.commonpath([dots_dir, stripes_dir]))
        exp_tag = (dots_dir.name if dots_dir.name == stripes_dir.name
                   else f"{dots_dir.name}_vs_{stripes_dir.name}")
        plot_dir = common_parent / f"{subdir}_{exp_tag}"

    else:                                            # SINGLE DATASET MODE
        single = dots_results or stripes_results
        root   = Path(single["stats"]).parent
        plot_dir = root / subdir
        exp_tag = root.name                        # ← “sigma0”, “s1”, etc.
        plot_dir = root / f"{subdir}_{exp_tag}"

    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Palette ------------------------------------------------
    red   = sns.color_palette("Set1", 4)[3]
    green = sns.color_palette("Set1", 4)[2]

    dots_stats   = dots_results["stats"] if dots_results else None
    dots_fit     = dots_results["fits"]  if dots_results else None
    stripes_stats = stripes_results["stats"] if stripes_results else None
    stripes_fit   = stripes_results["fits"]  if stripes_results else None

    # ---------- RAW v, s, χ -------------------------------------------
    owner = "dots" if dots_results else "stripes"
    for metric in ("v", "s", "chi"):
        png = plot_dir / f"raw_{metric}.png"
        save_raw_data_plots(n_points, metric, png,
                            dots_stats, dots_fit,
                            stripes_stats, stripes_fit,
                            lcol1=red if owner == "dots" else green,
                            lcol2=green)

    # ---------- Functionals + Mecke  (only if both datasets) ----------
    if dots_results and stripes_results:
        for func in ("pv", "ps", "pchi"):
            png = plot_dir / f"{func}_mecke.png"
            plot_ds_vs_mecke(func, png,
                             dots_stats, dots_fit,
                             stripes_stats, stripes_fit,
                             n_points=n_points)

    print(f"✔ plots  →  {plot_dir}")