"""
=== Analysis Utilities ===
This module contains functions for calculating errors, saving run data in data
folder and performing the Minkowski analysis on image stacks.
"""

from datetime import datetime
import numpy as np
from pathlib import Path
import csv
from scipy.ndimage import label
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import plotcode as plot
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class PipelineConfig:
    # depends on pattern type
    input_stack: str
    folder_name:      str

    # unchanged defaults
    blur_type: str = "mecke"
    sigma:     float = 0
    ksize:     int   = 3
    fwhm_px:   float = 0.2
    gamma:     float = 0.95
    tag:       Optional[str] = None


def count_components(mask, struct):
    """
    Counts the number of connected components in a binary mask while ignoring
    components that touch the border of the mask."""
    labels, n = label(mask, structure=struct)
    if n == 0:
        return 0
    border_labels = np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ]))
    border_labels = border_labels[border_labels != 0]  
    return n - len(border_labels)


def minkowski_metric(binary_field):
    """
    Returns (non-normalized) Minkowski metrics for a binary field; 
    White pixel = True, black pixel = false.

    White area (v)          : # of white pixels
    Boundary length (s)     : # of pairs of black/white pixels
    Euler characteristic (χ): # of white components - # of black components

    White component: is defined as a region of white pixels connected by a path
    of either nearest-neighbors (up, left, down, right) or second nearest-
    neighbors (diagonal neighbors). 
    
    A black component is defined as a region of black pixels connected by a path
    of only nearest-neighbors.

    Parameters
    ----------
    binary_field : np.ndarray
        binary field with values either 1 or 0
    
    Returns
    -------
    white_area : float
        white area
    bdry_length : float
        boundary length
    euler_char : float
        euler characteristic
    """
    field = np.array(binary_field, dtype=bool)
    
    # --- WHITE AREA ---
    white_area = int(binary_field.sum())

    # --- BOUNDARY LENGTH ---
    horiz_pairs = field[:, :-1] != field[:, 1:]
    vert_pairs  = field[:-1, :] != field[1:, :]
    bdry_length = np.sum(horiz_pairs) + np.sum(vert_pairs)

    # --- EULER CHARACTERISTIC ---
    eight_connect   = np.array([[1,1,1],
                                [1,1,1],
                                [1,1,1]], dtype=bool)
    
    four_connect    = np.array([[0,1,0],
                                [1,1,1],
                                [0,1,0]], dtype=bool)
    
    n_white = count_components(field, eight_connect)
    n_black = count_components(~field, four_connect)
    euler_char   = n_white - n_black


    return white_area, bdry_length, euler_char


def threshold_curves(arr, levels=256):
    """
    Computes Minkowski curves over a range of thresholds. The incoming pixel 
    array is normalized to a maximum value of 255 (default 0, ..., 255).
    At each threshold, each "gray" value is reset to white or black depending on
    if the original value is higher or lower than the threshold. With 256 black
    and white images, Minkowski metrics can be plotted vs threshold to show 
    distinct pattern behavior. 

    Normalized.

    """
    N = arr.size
    
    v_arr = np.empty(levels)
    s_arr = np.empty(levels)
    chi_arr = np.empty(levels)

    # normalise the field_array to the maximum threshold level of 255,
    low, high = np.percentile(arr, (0.1, 99.9))  # clip 0.1% tails
    arr_8bit   = np.clip((arr - low)/(high - low), 0, 1)*255
    thresholds = np.arange(256)                  # 0 … 255 inclusive

    for i, r in enumerate(thresholds):
        mask = arr_8bit > r
        v, s, chi = minkowski_metric(mask)
        v_arr[i] = v
        s_arr[i] = s
        chi_arr[i] = chi

    return v_arr / N, s_arr / N, chi_arr / N


def plot_threshold_curves(v_arr, s_arr, chi_arr, levels=256):
    """Plots curves for v, s and chi"""
    x_axis = np.arange(0, levels)
    plt.plot(x_axis, v_arr)
    plt.ylabel(r"$v$")
    plt.show()
    plt.plot(x_axis, s_arr)
    plt.ylabel(r"$s$")
    plt.show()
    plt.plot(x_axis, chi_arr)
    plt.ylabel(r"$\chi$")
    plt.show()
    

def transform_curves(v_arr, s_arr, chi_arr):
    """
    Transform Minkowski metrics to functions p_v, p_s and p_chi as described in 
    Mecke 1996.
    """
    # clipping v_arr no nan outputs from arctanh
    eps = 1e-10
    v_clip = np.clip(v_arr, 0 + eps, 1 - eps)
    p_v = np.arctanh(2 * v_clip - 1)
    # p_s = 4 * s_arr * (np.cosh(p_v))**2
    p_s = s_arr / (v_arr * (1-v_arr))

    p_chi = chi_arr / (s_arr + eps) 
    return p_v, p_s, p_chi


def plot_transform_curves(p_v, p_s, p_chi, levels=256):
    """Plots the transformed curves"""
    x_axis = np.linspace(-1, 1, levels)
    plt.plot(x_axis, p_v)
    plt.ylabel(r"$p_v$")
    plt.show()
    plt.plot(x_axis, p_s)
    plt.ylabel(r"$p_s$")
    plt.show()
    plt.plot(x_axis, p_chi)
    plt.ylabel(r"$p_\chi$")
    plt.show()


def stack_curves(field_stack, levels=256):
    M = field_stack.shape[0]
    
    v_all, s_all, chi_all = [], [], []

    for arr in field_stack:
        v, s, chi = threshold_curves(arr, levels)
        v_all.append(v); s_all.append(s), chi_all.append(chi)

    v_all = np.vstack(v_all) #(M, levels)
    s_all = np.vstack(s_all)
    chi_all= np.vstack(chi_all)

    mean   = lambda a: a.mean(axis=0)
    stderr = lambda a: a.std(axis=0, ddof=1)/np.sqrt(M)

    return (mean(v_all),  stderr(v_all)), \
           (mean(s_all),  stderr(s_all)), \
           (mean(chi_all),stderr(chi_all))


def process_images(images_norm,
                   *,
                   levels=256,
                   outdir="data/stats/",
                   tag=None):
    """
    Compute v, s, χ threshold curves for a *list* of 2-D float arrays in [0,1].
    NO BLURRING IS DONE HERE ANYMORE.
    A single .npz is written:
        curves_{tag-} {N}runs.npz
    Returns that file’s Path.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_runs = len(images_norm)
    thresholds = np.arange(levels, dtype=np.uint8)

    v_list, s_list, chi_list = [], [], []

    preview_dir = outdir / f"previews_{tag or 'untagged'}"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for i, I_norm in enumerate(images_norm):
        I_u8 = (np.clip(I_norm, 0.0, 1.0) * 255).round().astype(np.uint8)
        Image.fromarray(I_u8).save(preview_dir / f"frame_{i:03d}.png")

        v, s, chi = threshold_curves(I_u8, levels)
        v_list.append(v.astype(np.float32))
        s_list.append(s.astype(np.float32))
        chi_list.append(chi.astype(np.float32))

    v_stack   = np.stack(v_list,   axis=0)
    s_stack   = np.stack(s_list,   axis=0)
    chi_stack = np.stack(chi_list, axis=0)

    stem = f"curves_{(tag + '_') if tag else ''}{n_runs}runs"
    curves_path = outdir / f"{stem}.npz"
    np.savez_compressed(curves_path,
                        thresholds=thresholds,
                        v=v_stack, s=s_stack, chi=chi_stack)
    print("✔ curves written →", curves_path)
    return curves_path


def random_error(curves_file: str | Path):
    """
    Load a .npz file with keys "v", "s", "chi" (as created by process_images()),
    compute the mean and standard error of the mean (SEM) for each curve,
    and save the results in a new .npz file with keys:
        v_mean, v_stderr, s_mean, s_stderr, chi_mean, chi_stderr,
        pv_mean, pv_stderr, ps_mean, ps_stderr, pchi_mean, pchi_stderr.
    """
    cf = Path(curves_file)
    d  = np.load(cf)
    v, s, χ = d["v"], d["s"], d["chi"]
    n = v.shape[0]

    r = np.arange(v.shape[1])
    mean   = lambda a: a.mean(axis=0)
    if n == 1:
        # Single run: show “no error” in plots but keep a finite weight
        stderr = lambda a: np.full_like(a[0], 1e-6, dtype=np.float64)
    else:
        stderr = lambda a: a.std(axis=0, ddof=1) / np.sqrt(n)

    stats = {
        'v_mean':  mean(v),  'v_stderr':  stderr(v),
        's_mean':  mean(s),  's_stderr':  stderr(s),
        'chi_mean':mean(χ),  'chi_stderr':stderr(χ),
    }

    eps = 1e-12
    p_v_runs = np.arctanh(np.clip(2*v - 1, -1+eps, 1-eps))
    p_s_runs = s / (v*(1-v))
    p_χ_runs = χ / (s + eps)

    stats.update({
        'pv_mean': mean(p_v_runs),   'pv_stderr': stderr(p_v_runs),
        'ps_mean': mean(p_s_runs),   'ps_stderr': stderr(p_s_runs),
        'pchi_mean': mean(p_χ_runs), 'pchi_stderr': stderr(p_χ_runs),
    })

    out = cf.with_name(cf.stem + "_stats.npz")
    np.savez_compressed(out, thresholds=r, **stats)
    print("✔ stats   →", out)
    return out


def weighted_polyfit_range(x, y, se, *, deg,
                           x_min=None, x_max=None, mask=None):
    """
    Weighted least-squares polynomial fit that can ignore the troublesome
    tails of the curve.

    Parameters
    ----------
    x, y, se : 1-D arrays of equal length
        Abscissa, ordinate and 1 σ errors.
    deg : int
        Polynomial degree (3 for p_v and p_chi, 4 for p_s).
    x_min, x_max : float or None
        Only points with x_min <= x <= x_max are used.
    mask : 1-D bool array or None
        Alternative arbitrary selection mask (True = keep).
        If given, x_min / x_max are ignored.

    Returns
    -------
    coeffs  : array length (deg+1), highest power first 
    perr    : 1-D array of same length with 1 σ uncertainties.
    """
 
    if mask is None:
        mask = np.isfinite(y) & np.isfinite(se) & (se > 0)
        if x_min is not None:
            mask &= x >= x_min
        if x_max is not None:
            mask &= x <= x_max
    else:
        mask = mask.copy()           

    if mask.sum() <= deg:
        raise ValueError("Not enough valid points for the requested degree")
    
    mask &= (se > 1e-8) #eliminate small pts

    w          = 1 / se[mask]         # weights = 1/σ\
    coeffs, cov = np.polyfit(x[mask], y[mask], deg, w=w, cov=True)
    perr       = np.sqrt(np.diag(cov))
    return coeffs, perr


def fit_polynomials(stats_file,
                    *,
                    rmin_v= 2,
                    rmax_v= 254,
                    rmin_s= 2,
                    rmax_s= 254,
                    rmin_chi= 2,
                    rmax_chi= 254,
                    deg_v=3, deg_s=4, deg_chi=3,
                    suffix=None):
    """
    Read *_stats.npz, fit p_v, p_s, p_chi on the chosen x-range,
    and save the coefficients + errors into a companion file.

    The stats file must already contain pv_mean, pv_stderr, … as created
    by the earlier analyse_curves() step.
    """

    pf = Path(stats_file)
    st = np.load(pf)

    r = st["thresholds"]  # thresholds, shape (256,)
    xmax = float(r.max())
    x = 2*(r/xmax) - 1 # scaled [-1,1] domain

    def bounds(rmin, rmax):
        lo = -1.0 if rmin is None else 2*(rmin/xmax)-1
        hi =  1.0 if rmax is None else 2*(rmax/xmax)-1
        return lo, hi

    # unique (x_min, x_max)
    coeffs = {}
    coeffs["pv"], σpv = weighted_polyfit_range(
        x, st["pv_mean"], st["pv_stderr"], deg=deg_v, 
        x_min=bounds(rmin_v,rmax_v)[0], x_max=bounds(rmin_v,rmax_v)[1])
    coeffs["ps"], σps = weighted_polyfit_range(
        x, st["ps_mean"], st["ps_stderr"], deg=deg_s, 
        x_min=bounds(rmin_s,rmax_s)[0], x_max=bounds(rmin_s,rmax_s)[1])
    coeffs["pchi"], σpc = weighted_polyfit_range(
        x, st["pchi_mean"], st["pchi_stderr"], deg=deg_chi, 
        x_min=bounds(rmin_chi,rmax_chi)[0], x_max=bounds(rmin_chi,rmax_chi)[1])

    out = pf.with_name(pf.stem.replace("_stats", "_fit") + ".npz")
    np.savez_compressed(out,
        poly_pv_coeffs=coeffs["pv"],   poly_pv_sigma=σpv,
        poly_ps_coeffs=coeffs["ps"],   poly_ps_sigma=σps,
        poly_pchi_coeffs=coeffs["pchi"], poly_pchi_sigma=σpc
    )
    print("✔ poly fit →", out)
    return out


def print_fit_results(fit_file):
    """
    Load a .npz file produced by `fit_polynomials` and print its contents
    (raw ranges, degrees, coefficients, and uncertainties) in a readable format.
    PRINTS IN REVERSE ORDER
    
    Parameters
    ----------
    fit_file : str or Path
        Path to the .npz file created by `fit_polynomials`.
    """
    pf = Path(fit_file)
    data = np.load(pf, allow_pickle=True)
    
    # Format polynomial series
    def format_poly(coeffs, sigmas):
        lines = []
        for power, (c, s) in enumerate(zip(coeffs, sigmas)):
            lines.append(f"  c_{power:<2d} = {c: .6e}  ± {s: .6e}")
        return "\n".join(lines)
    
    print("\n=== Fit Summary ===\n")
    
    # 1) Raw threshold ranges
    def safe_get(key):
        return data[key] if key in data.files else None

    rmin_v = safe_get("rmin_v")
    rmax_v = safe_get("rmax_v")
    print(f"  p_v  : rmin = {rmin_v}   rmax = {rmax_v}")

    rmin_s = safe_get("rmin_s")
    rmax_s = safe_get("rmax_s")
    print(f"  p_s  : rmin = {rmin_s}   rmax = {rmax_s}")

    rmin_chi = safe_get("rmin_chi")
    rmax_chi = safe_get("rmax_chi")
    print(f"  p_chi: rmin = {rmin_chi} rmax = {rmax_chi}\n")
        
    # 2) Polynomial degrees
    print("Polynomial degrees:")
    print(f"  p_v  degree = {data['deg_v']}")
    print(f"  p_s  degree = {data['deg_s']}")
    print(f"  p_chi degree = {data['deg_chi']}\n")
    
    # 3) Coefficients and uncertainties for each polynomial
    print("p_v coefficients and 1σ uncertainties:")
    coeffs_pv = data['poly_pv_coeffs']
    sigmas_pv = data['poly_pv_sigma']
    print(format_poly(coeffs_pv, sigmas_pv))
    print("\n" + "-"*40 + "\n")
    
    print("p_s coefficients and 1σ uncertainties:")
    coeffs_ps = data['poly_ps_coeffs']
    sigmas_ps = data['poly_ps_sigma']
    print(format_poly(coeffs_ps, sigmas_ps))
    print("\n" + "-"*40 + "\n")
    
    print("p_chi coefficients and 1σ uncertainties:")
    coeffs_pchi = data['poly_pchi_coeffs']
    sigmas_pchi = data['poly_pchi_sigma']
    print(format_poly(coeffs_pchi, sigmas_pchi))
    print("\n=== End of Summary ===\n")


def save_data(U_snap, V_snap, params, dt, dx, steps, 
              save_interval, base_dir = "data/test_runs"): 
    """Saves simulation data in an .npz file and creates a png of the final
    snapshot."""
    # --- CREATE DIRECTORY AND FILENAME ---
    base_dir = Path(base_dir)
    runs_dir = base_dir 
    runs_dir.mkdir(parents=True, exist_ok=True)

    time_stamp = datetime.now().strftime("%m‑%d_%H%M%S")

    run_id = (
        f"{time_stamp}"
        + f"_F{params.F}"
        + f"_k{params.k}"
        + f"_S{steps}" # S - steps
        + f"_SI{save_interval}" # SI - save interval
    )

    run_path = runs_dir / run_id
    run_path.mkdir()


    # --- SAVE RUN SNAPSHOTS TO NPZ ---
    np.savez_compressed(run_path / "snapshots.npz", U=U_snap, V=V_snap)
    # U, V: (n_snapshots, N, N) 

    # --- SAVE U/V COMPOSITE PNG OF FINAL SNAPSHOT TO FOLDER ---
    u = U_snap[-1]
    v = V_snap[-1]

    # U is red and V is blue
    plt.imshow(u); plt.axis('off')
    plt.savefig(run_path / 'final_snapshot.png', 
                bbox_inches='tight', pad_inches=0)
    plt.close()


    # --- APPEND TO master_log.csv : OVERVIEW OF ALL RUNS IN BASE DIRECTORY ---
    master_log = base_dir / "master_log.csv"
    write_header = not master_log.exists()
    with master_log.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["timestamp","Du","Dv","F","k","dt","dx","steps","path"]
            )
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            params.Du, params.Dv, params.F, params.k,
            dt, dx, steps,
            str(run_path),
        ])
