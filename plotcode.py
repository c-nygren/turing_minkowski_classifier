"""
=== Plotting Code ===
This module contains functions for plotting results from the simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import matplotlib.animation as animation
from pathlib import Path
import seaborn as sns
import imageio.v2 as imageio
from natsort import natsorted  # for natural sorting like: img1, img2, ..., img10


def extract_sim_data(subhint, base_dir="data/simulations/runs"):
    """
    Extracts sim data from npz file using a subhint. I suggest using the time of 
    the simulation run as the subhint, e.g. "162215".

    Parameters
    ----------
    subhint : string
        portion of filename that will be searched for
    base_dir : string
        base directory

    Returns
    -------
    U_snap : np.ndarray
        shape (n_frames, N, N), dtype float32
        Contains snapshots of U values
    V_snap : np.ndarray
        shape (n_frames, N, N), dtype float32
        Contains snapshots of V values
    """
    # Find folder containing subhint
    all_dirs = [d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))]
    matches = [d for d in all_dirs
               if subhint.lower() in d.lower()]

    if not matches:
        raise ValueError\
            (f"No folders under {base_dir!r} contain '{subhint}'")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous hint {subhint!r}, matches: {matches!r}")
    
    sweep_dir = os.path.join(base_dir, matches[0])
    
    # Find an .npz file inside the matched directory
    npz_files = [f for f in os.listdir(sweep_dir) if f.endswith('.npz')]
    if not npz_files:
        raise FileNotFoundError(f"No .npz file found in {sweep_dir}")
    if len(npz_files) > 1:
        raise ValueError(f"Multiple .npz files found in {sweep_dir}: {npz_files!r}")

    npz_path = os.path.join(sweep_dir, npz_files[0])
    with np.load(npz_path) as data:
        if "U" not in data or "V" not in data:
            raise KeyError(f"Expected keys 'U' and 'V' in {npz_path}, got {data.files}")
        U_snap = data["U"]    # shape (n_frames, N, N), dtype float32
        V_snap = data["V"]

    return U_snap, V_snap


def extract_mean_data(raw_path = "data/stats/dots/curves_20runs_May31_transform_stats.npz", 
                      coeffs_path = "data/stats/dots/fits/curves_20runs_May31_transform_fit_WITHCUTOFF.npz"):

    raw_data = np.load(Path(raw_path))
    coeffs_data = np.load(Path(coeffs_path))

    return {
        # --- raw arrays ---
        "thresholds": raw_data["thresholds"],
        "v_mean":     raw_data["v_mean"],
        "v_stderr":   raw_data["v_stderr"],  
        "s_mean":     raw_data["s_mean"],
        "s_stderr":   raw_data["s_stderr"],
        "chi_mean":   raw_data["chi_mean"],
        "chi_stderr": raw_data["chi_stderr"],
        "pv_mean":    raw_data["pv_mean"],
        "pv_stderr":  raw_data["pv_stderr"],
        "ps_mean":    raw_data["ps_mean"],
        "ps_stderr":  raw_data["ps_stderr"],
        "pchi_mean":  raw_data["pchi_mean"],
        "pchi_stderr":raw_data["pchi_stderr"],

        # --- polynomial fits ---
        "poly_pv_coeffs":   coeffs_data["poly_pv_coeffs"],
        "poly_ps_coeffs":   coeffs_data["poly_ps_coeffs"],
        "poly_pchi_coeffs": coeffs_data["poly_pchi_coeffs"],
        "poly_pv_sigma":    coeffs_data["poly_pv_sigma"],
        "poly_ps_sigma":    coeffs_data["poly_ps_sigma"],
        "poly_pchi_sigma":  coeffs_data["poly_pchi_sigma"],
    }





def pearson_arr(*,U_snap=None, V_snap=None, subhint=None, black=(1.0, 0.0), 
                white=(0.3, 0.25), base_dir="data/simulations/runs"):
    """
    Map each (U,V) pair to a gray level in [0,1] according to Pearson:
      U=1,V=0   → 0.0 (black)
      U=0.3,V=0.25 → 1.0 (white)
    Everything else is linearly interpolated along that line and clamped.

    Parameters
    ----------
    subhint : str
        A string containing the lookup hint to find the correct file. 
    black : tuple, optional (default=(1.0,0.0))
        For contrast definition? # TODO
    white : tuple, optional (default=(0.3,0.25))
        For contrast definition? # TODO

    Returns
    -------
    # TODO
    """
    if subhint:
        U_snap, V_snap = extract_sim_data(subhint, base_dir=base_dir)
    sb = np.asarray(black, dtype=float)
    sw = np.asarray(white, dtype=float)
    delta = sw - sb                 
    D = np.dot(delta, delta)            

    # compute dot((U,V) - sb, delta) for every pixel
    num = (U_snap - sb[0]) * delta[0] + (V_snap - sb[1]) * delta[1]
    I = num / D
    return np.clip(I, 0.0, 1.0)


def pearson_plot(clipped_array, id_number, base_dir="data/test_images"):
    """
    Plots blue-red with yellow as intermediate color. 
    """
    phi = 1.0 - clipped_array[-1]
    
    cmap_byr = LinearSegmentedColormap.from_list(
    'red-yellow-blue',
    [
        (0.0, 'red'),
        (0.5, 'yellow'),
        (1.0, 'blue'),
    ])
    
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    save_id = f"value_{id_number}.png"
    save_path = base_dir / save_id

    plt.figure(figsize=(6,6))
    plt.imshow(phi, cmap=cmap_byr, origin='lower')
    plt.axis('off')
    plt.savefig(save_path, 
                bbox_inches='tight', pad_inches=0)
    plt.close()

def save_raw_data_plots_extended(
    n_points: int,
    metric: str,
    filename: str,
    pearson_dots_stats_path: str,
    pearson_dots_fit_path: str,
    our_dots_stats_path: str,
    our_dots_fit_path: str,
    pearson_stripes_stats_path: str,
    pearson_stripes_fit_path: str,
    our_stripes_stats_path: str,
    our_stripes_fit_path: str,
    lcols: dict = None
):
    """
    Saves raw data plots for Pearson's and our dots and stripes,
    comparing the specified metric (v, s, chi).
    """
    if lcols is None:
        lcols = {
            "Pearson dots": sns.color_palette("Set1", 4)[0],   # red
            "Our dots": sns.color_palette("Set1", 4)[1],       # blue
            "Pearson stripes": sns.color_palette("Set1", 4)[2],# green
            "Our stripes": sns.color_palette("Set1", 4)[3],    # purple
        }

    def limited_points(n_points, raw_mean, raw_stderr):
        N = len(raw_mean)
        idx = np.linspace(0, N - 1, n_points).astype(int)
        return raw_mean[idx], raw_stderr[idx]

    # Extract data using extract_mean_data for all four cases
    data = {
        "Pearson dots": extract_mean_data(pearson_dots_stats_path, pearson_dots_fit_path),
        "Our dots": extract_mean_data(our_dots_stats_path, our_dots_fit_path),
        "Pearson stripes": extract_mean_data(pearson_stripes_stats_path, pearson_stripes_fit_path),
        "Our stripes": extract_mean_data(our_stripes_stats_path, our_stripes_fit_path),
    }

    # Map metric key
    if metric == "v":
        key_mean, key_stderr = "v_mean", "v_stderr"
    elif metric == "s":
        key_mean, key_stderr = "s_mean", "s_stderr"
    elif metric == "chi":
        key_mean, key_stderr = "chi_mean", "chi_stderr"
    else:
        raise ValueError("Metric must be 'v', 's', or 'chi'.")
    
    # Plot each dataset
    for label, dataset in data.items():
        raw_mean = dataset[key_mean]
        raw_stderr = dataset[key_stderr]
        mean_sub, stderr_sub = limited_points(n_points, raw_mean, raw_stderr)
        quick_plot(mean_sub, stderr_sub, label=f"{metric} {label}", lcol=lcols[label])

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=250, bbox_inches='tight')
    plt.close()

def save_raw_data_plots(n_points,
                        metric: str,
                        filename: str,
                        dots_stats_path: str,
                        dots_fit_path: str,
                        stripes_stats_path: str | None = None,
                        stripes_fit_path: str | None = None,
                        lcol1=None, lcol2=None):
    """
    Saves raw data plots for dots and stripes, comparing the specified metric.
    """
    if lcol1 is None:
        lcol1 = sns.color_palette("Set1", 4)[3]  # red
    if lcol2 is None:
        lcol2 = sns.color_palette("Set1", 4)[2]  # green

    dot_data = extract_mean_data(raw_path=dots_stats_path, \
                                 coeffs_path=dots_fit_path)
    str_data = (extract_mean_data(raw_path=stripes_stats_path,
                                  coeffs_path=stripes_fit_path)
                if stripes_stats_path and stripes_fit_path else None)

    def limited_points(n_points, raw_mean, raw_stderr):
        N = len(raw_mean)
        idx = np.linspace(0, N - 1, n_points).astype(int)
        return raw_mean[idx], raw_stderr[idx]
    
    if metric == "v":
        data_mean_dot = dot_data["v_mean"]
        data_stderr_dot = dot_data["v_stderr"]

        if str_data is not None:
            data_mean_str = str_data["v_mean"]
            data_stderr_str = str_data["v_stderr"]

    elif metric == "s":
        data_mean_dot = dot_data["s_mean"]
        data_stderr_dot = dot_data["s_stderr"]

        if str_data is not None:
            data_mean_str = str_data["s_mean"]
            data_stderr_str = str_data["s_stderr"]
    
    elif metric == "chi":
        data_mean_dot = dot_data["chi_mean"]
        data_stderr_dot = dot_data["chi_stderr"]

        if str_data is not None:
            data_mean_str = str_data["chi_mean"]
            data_stderr_str = str_data["chi_stderr"]

    dot_mean_sub, dot_stderr_sub = limited_points(n_points, data_mean_dot,
                                                        data_stderr_dot)
    


    quick_plot(dot_mean_sub, dot_stderr_sub,
               label=r"$v$ dots", lcol=lcol1)

    if str_data is not None:
        str_mean_sub, str_stderr_sub = limited_points(n_points, data_mean_str,
                                                    data_stderr_str)
        str_mean_sub, str_stderr_sub = limited_points(n_points,
                                                      data_mean_str,
                                                      data_stderr_str)
        quick_plot(str_mean_sub, str_stderr_sub,
                   label=r"$v$ stripes", lcol=lcol2)

    plt.savefig(filename, dpi=250, bbox_inches='tight')
    plt.close()


def quick_plot(mean, err, label, lcol : str, 
               xmin=None, xmax=None, orig_range=(0, 255)):
    # Create x-axis in [-1, 1] (normalized)
    x = np.linspace(-1, 1, mean.size)
    
    orig_min, orig_max = orig_range
    if xmin is not None:
        xmin_norm = (xmin - orig_min) / (orig_max - orig_min) * 2 - 1
    else:
        xmin_norm = -1
    if xmax is not None:
        xmax_norm = (xmax - orig_min) / (orig_max - orig_min) * 2 - 1
    else:
        xmax_norm = 1

    # Create mask in [-1, 1] normalized space
    mask = (x >= xmin_norm) & (x <= xmax_norm)

    x_plot = x[mask]
    mean_plot = mean[mask]
    err_plot = err[mask]

    plt.scatter(x_plot, mean_plot, s=5, c=lcol, zorder=5, label=f'{label} mean')
    plt.errorbar(x_plot, mean_plot, yerr=err_plot,
                  fmt='none', ecolor='red', elinewidth=1,
                  capsize=0, zorder=6)


    # plt.ylabel(f'{label}')
    plt.xlim(-1, 1)  # Always show full normalized axis
    # plt.ylim(-0.1, 0.1)
    # plt.legend()
    plt.tight_layout()


def poly_fit_plot(raw_mean : np.ndarray, 
                  raw_stderr : np.ndarray,
                  poly_coeffs : np.ndarray,
                  label : str, xmin : float, xmax : float, 
                  n_points : int = 100,
                  lcol1 : str = "C0", lcol2 : str = "black"):
    r = np.linspace(-1.0, +1.0, 1000)

    deg = len(poly_coeffs) - 1
    fit = np.polyval(poly_coeffs, r)

    N = len(raw_mean)
    idx = np.linspace(0, N - 1, n_points).astype(int)
    mean_sub   = raw_mean[idx]
    stderr_sub = raw_stderr[idx]
    quick_plot(mean_sub, stderr_sub, label="_nolegend_", lcol=lcol2, xmin=xmin, xmax=xmax)
    
    plt.plot(r, fit, "-", color=lcol1, linewidth=2, label=label + " fit")


def mecke_plot(mecke_coeffs : np.ndarray, lcol="C2", lbl : str = None):
    deg = len(mecke_coeffs) - 1
    if deg not in (3, 4):
        raise ValueError(f"mecke_coeffs must have length 4 or 5 \
                         (got {len(mecke_coeffs)}).")

    r = np.linspace(-1.0, +1.0, 1000)
    fit = np.polyval(mecke_coeffs, r)
    
    plt.plot(r, fit, '--', color=lcol, linewidth=1, label=lbl)


# TODO - needs to take in different data sources
def plot_ds_vs_mecke(
    functional: str,
    filename: str,
    dots_stats_path: str,
    dots_fit_path: str,
    stripes_stats_path: str,
    stripes_fit_path: str,
    n_points: int = 100,
):
    filename = Path(filename)

    green = sns.color_palette("Set1", 4)[2]  # green

    data = extract_mean_data(raw_path=dots_stats_path, 
                             coeffs_path=dots_fit_path)
    data_stripes = extract_mean_data(raw_path=stripes_stats_path, 
                                     coeffs_path=stripes_fit_path)

    mecke_pv   = [-0.82,   0.397,  -1.59,   -0.45]
    mecke_ps   = [ 0.36,  -0.18,    0.43,   0.14,   0.83]
    mecke_pchi = [ 0.033, -0.025,  0.056,  0.024]
    
    mecke_pv_stripes   = [-1.09, 0.03, -1.39, 0.0096]
    mecke_ps_stripes   = [-0.062, 0.035, 0.599, -0.021, 0.55]
    mecke_pchi_stripes = [0.031, -0.000013, 0.024, -0.0002]

    if functional == "pv":
        dots_mean = data["pv_mean"]
        dots_stderr = data["pv_stderr"]
        stripes_mean = data_stripes["pv_mean"]
        stripes_stderr = data_stripes["pv_stderr"]
        dots_coeffs = data["poly_pv_coeffs"]
        stripes_coeffs = data_stripes["poly_pv_coeffs"]

        mecke_dots = mecke_pv
        mecke_stripes = mecke_pv_stripes

        ylabel = r"$p_{v}$"

    elif functional == "ps":
        dots_mean = data["ps_mean"]
        dots_stderr = data["ps_stderr"]
        stripes_mean = data_stripes["ps_mean"]
        stripes_stderr = data_stripes["ps_stderr"]
        dots_coeffs = data["poly_ps_coeffs"]
        stripes_coeffs = data_stripes["poly_ps_coeffs"]

        mecke_dots = mecke_ps
        mecke_stripes = mecke_ps_stripes

        ylabel = r"$p_{s}$"
    
    elif functional == "pchi":
        dots_mean = data["pchi_mean"]
        dots_stderr = data["pchi_stderr"]
        stripes_mean = data_stripes["pchi_mean"]
        stripes_stderr = data_stripes["pchi_stderr"]
        dots_coeffs = data["poly_pchi_coeffs"]
        stripes_coeffs = data_stripes["poly_pchi_coeffs"]

        mecke_dots = mecke_pchi
        mecke_stripes = mecke_pchi_stripes

        ylabel = r"$p_{\chi}$"
    
    else:
        raise ValueError("Invalid functional type. Choose from 'pv', 'ps', or 'pchi'.")

    # --- Plotting the raw data curves ---
    poly_fit_plot(dots_mean, dots_stderr, dots_coeffs,
                        "Dots", xmin=1, xmax=255, n_points=n_points,
                        lcol1="gray", lcol2="black"
                        )
    
    poly_fit_plot(stripes_mean, stripes_stderr, stripes_coeffs,
                        "Stripes", xmin=1, xmax=255, n_points=n_points,
                        lcol1="lightgreen", lcol2="seagreen"
                        )

    # --- Plotting the Mecke curves as dashed lines ---
    mecke_plot(mecke_dots, lbl="_nolegend", lcol="gray")
    mecke_plot(mecke_stripes, lbl="_nolegend_", lcol="mediumseagreen")

    plt.ylabel(ylabel)
    plt.xlabel('Threshold (normalized)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=250, bbox_inches='tight') # MAKE SURE THIS IS SAVING A PIC
    plt.close()

def plot_functional_comparison(
    functional: str,
    filename: str,
    stats_pearson_dots: str,
    fit_pearson_dots: str,
    stats_ours_dots: str,
    fit_ours_dots: str,
    stats_pearson_stripes: str,
    fit_pearson_stripes: str,
    stats_ours_stripes: str,
    fit_ours_stripes: str,
    n_points: int = 100,
):
    from pathlib import Path
    filename = Path(filename)

    # Load all data sets
    data_pd = extract_mean_data(stats_pearson_dots, fit_pearson_dots)
    data_od = extract_mean_data(stats_ours_dots, fit_ours_dots)
    data_ps = extract_mean_data(stats_pearson_stripes, fit_pearson_stripes)
    data_os = extract_mean_data(stats_ours_stripes, fit_ours_stripes)

    # Functional-specific labels and keys
    if functional == "pv":
        ylabel = r"$p_{v}$"
        key_mean, key_stderr, key_coeffs = "pv_mean", "pv_stderr", "poly_pv_coeffs"
    elif functional == "ps":
        ylabel = r"$p_{s}$"
        key_mean, key_stderr, key_coeffs = "ps_mean", "ps_stderr", "poly_ps_coeffs"
    elif functional == "pchi":
        ylabel = r"$p_{\chi}$"
        key_mean, key_stderr, key_coeffs = "pchi_mean", "pchi_stderr", "poly_pchi_coeffs"
    else:
        raise ValueError("Invalid functional type. Choose from 'pv', 'ps', or 'pchi'.")

    # Plot each curve with its own label and color
    poly_fit_plot(
        data_pd[key_mean], data_pd[key_stderr], data_pd[key_coeffs],
        label="Pearson's dots", xmin=1, xmax=255, n_points=n_points,
        lcol1="lightgray", lcol2="gray"
    )

    poly_fit_plot(
        data_od[key_mean], data_od[key_stderr], data_od[key_coeffs],
        label="Our dots", xmin=1, xmax=255, n_points=n_points,
        lcol1="lightblue", lcol2="steelblue"
    )

    poly_fit_plot(
        data_ps[key_mean], data_ps[key_stderr], data_ps[key_coeffs],
        label="Pearson's stripes", xmin=1, xmax=255, n_points=n_points,
        lcol1="lightgreen", lcol2="mediumseagreen"
    )

    poly_fit_plot(
        data_os[key_mean], data_os[key_stderr], data_os[key_coeffs],
        label="Our stripes", xmin=1, xmax=255, n_points=n_points,
        lcol1="peachpuff", lcol2="orangered"
    )

    # Final plot formatting
    plt.ylabel(ylabel)
    plt.xlabel("Threshold (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=250, bbox_inches="tight")
    plt.close()
    

def animate_snapshots(subhint, save_name, interval=100, 
                      save_path="data/time_evolution_pics/", 
                      base_dir="data/simulations"):
    """
    Animate the evolution of a field ('U' or 'V') from a stored .npz snapshot.

    Parameters
    ----------
    subhint : str
        Identifier to locate the run folder (e.g. timestamp).
    field : str
        'U' or 'V' — which field to animate.
    interval : int
        Delay between frames in ms.
    base_dir : str
        Base path to simulation data folders.
    save_path : str or Path
        If given, save the animation as GIF.
    """
    data = pearson_arr(subhint=subhint, base_dir=base_dir)

    cmap_byr = LinearSegmentedColormap.from_list(
        'red-yellow-blue',
        [
            (0.0, 'blue'),
            (0.5, 'yellow'),
            (1.0, 'red'),
        ])

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(data[0], cmap=cmap_byr, animated=True, origin='lower')
    ax.axis('off')

    def update(i):
        im.set_array(data[i])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(data), interval=interval, blit=True
    )

    save_path = save_path + save_name
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer='pillow')
        # if we rather prefer .mp4 files (save_path.suffix == ".mp4") we use:
        # ani.save(save_path, writer='ffmpeg')
       
    plt.show() 

def make_gif_from_folder(folder_path, 
                         output_path="data/phaseimages/animations/animation.gif", 
                         fps=3):
    """
    Creates a GIF animation from a sequence of image files in a folder.

    This function reads all .png or .jpg files from the specified folder,
    sorts them in natural order (e.g., img1, img2, ..., img10), and compiles
    them into a single GIF file.

    Parameters
    ----------
    folder_path (str): Path to the folder containing the image files.
    output_path (str): Path to save the output GIF. Defaults to 'animation.gif'.
    fps (int): Frames per second for the animation. Higher values result in faster playback.

    Returns
    -------
    None. The GIF is saved to disk.
    """
    # Collect image filenames
    files = [f for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")]
    files = natsorted(files)  # Sort naturally by number

    images = []
    for file in files:
        image_path = os.path.join(folder_path, file)
        images.append(imageio.imread(image_path))

    imageio.mimsave(output_path, images, fps=fps)
    print(f"GIF saved to {output_path}")