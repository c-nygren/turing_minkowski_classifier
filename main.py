"""
=== Main script ===
This script runs the main simulation and any analysis/plotting functions.
"""
import scipy.ndimage
import projectcode as proj
import plotcode as plot
import numpy as np
import analysis_utils as au
import main_helpers as mh
import matplotlib.pyplot as plt
from pathlib import Path
import scipy
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import seaborn as sns # pip install seaborn
import image_processing as ip
from dataclasses import replace

def main():
    # nr_images=100
    # k_vals = np.full(nr_images, 0.061)
    # F_vals = np.linspace(0.02,0.06,nr_images)
    # proj.pattern_sweep_simulation(k_array=k_vals, F_array=F_vals, 
    #                               base_dir="data/phaseimages/k0061")

    # --- DEFINE PARAMETERS ---
    params = proj.ModelParams(
        Du=2e-5, 
        Dv=1e-5, 
        F=0.048,
        k=0.063, # F0.037, k0.065 - dots, lambda, F0.048, k0.063 stripes, kappa
        dt=1,
        steps=35000,
        N=256
        )

    # # --- RUN SIMULATION ---
    # usnap, vsnap, u, v, = \
    #     proj.simulation(p=params, base_dir = "data/simulations", save=False) 
    # clipped_arr = plot.pearson_arr(U_snap=usnap, V_snap=vsnap)
    # plot.pearson_plot(clipped_array=clipped_arr, id_number=2)

    # plot.animate_snapshots("222735", save_name="Test_miscDots_k0051_F0024.gif")
    # print("successful run")

    # --- ANIMATION PLOTS ---
    # plot.pearson_plot(hint, base_dir="data/simulations/runs/along_phase_diagram") 
    # plot.animate_snapshots("194029", field='U',
    #                         base_dir="data/simulations/runs/along_phase_diagram", 
    #                         save_path="data/time_evolution_pic/f04_k06.gif")
    # plot.animate_snapshots("194029", field='U',
    #                         base_dir="data/simulations/runs/along_phase_diagram", 
    #                         save_path="data/time_evolution_pic/f04_k06.gif")

    # --- COLLECT IMAGES --- #TODO try just U_snap instead of pearson_arr
    # proj.collect_images(
    #     params=params,
    #     n_runs=20,
    #     outdir="data/stats/stripes/sigma0",
    #     tag="June6",
    #     test=False
    # )

    # --- ANALYSIS PARAMETERS ---
    # Base set parameters defined.

    # I was testing other types of blurs, but gaussian sigma 1 will do
    analysis_params = au.PipelineConfig(
        input_stack = "",       # gets overwritten in config_dots_stripes()
        folder_name = "",       # same
        blur_type   = "gaussian", # three types: "gaussian", "median", "mecke"
        sigma       = 2.5,
        ksize       = 3,
        fwhm_px     = 0.05,    # for "mecke" blur
        gamma       = 2,      # for "mecke" blur
        tag         = None,
    )

    # this automatically creates the configs for dots and stripes, with the
    # parameters defined above.
    # dots_cfg, stripes_cfg = mh.config_dots_stripes(analysis_params)

   # --- RUN ANALYSIS PIPELINE ---
   # This runs the full analysis pipeline for both dots and stripes, we always
   # use the sigma 0 image_stack (already set up in dots_cfg, stripes_cfg) 
   # as the base, and then can apply different blurs

    # dots_results = paths = mh.run_pipeline(
    #     **dots_cfg.__dict__ 
    # )

    # print("\nAll output files:")
    # for k, v in paths.items():
    #     print(f"  {k:7s} : {v}")

    # stripes_results = paths = mh.run_pipeline(
    #     **stripes_cfg.__dict__ 
    # )

    # print("\nAll output files:")
    # for k, v in paths.items():
    #     print(f"  {k:7s} : {v}")


    # --- RUN PIPELINE FOR SINGLE IMAGE ---
    # this runs the full analysis pipeline for a single image, for pearson
    # once you have the data for both, and the comparison plots to mecke
    # it would be good to make separate plots comparing our results (in stats)
    # to pearson. Use sigma 1 for our results (s1) 
    # Data found in data\stats\dots\s1 and data\stats\stripes\s1


    # Use original image (array that was converted to grayscale), and this
    # applies the blur to it. We will use sigma 1 for both.
    dots_results = mh.run_pipeline("data/journal/final_presentation/pearson_converted/NEW_pearson_dots_array.npy",
                 folder_name = "pearson/dots/s1",
                 blur_type="gaussian", # make sure this is gaussian
                 sigma=1, # applying sigma 1 to original image
                 levels=256,
                 outroot="data/stats/",
                 tag=None)
    
    # setup for stripes, once you have saved a good grayscale array
    
    stripes_results = mh.run_pipeline("data/journal/final_presentation/pearson_converted/NEW_pearson_stripes_array.npy",
                 folder_name = "pearson/stripes/s1",
                 blur_type="gaussian",
                 sigma=1,
                 levels=256,
                 outroot="data/stats/",
                 tag=None)
    
    # --- RUN PLOTTING PIPELINE ---
    # this runs the entire plotting pipeline for dots vs stripes
    # once you have the data for both, and the comparison plots to mecke
    # mh.run_dots_vs_stripes_plots(
    #     pearson_results,        # ← can also put in folder paths here instead
    #     stripes_results=pearson_stripes, # once you have both you can run the full pipeline
    #     n_points=150 # choose how many pts. feel free to change colors etc
    #     )
    red   = sns.color_palette("Set1", 4)[3]
    green = sns.color_palette("Set1", 4)[2]
    
    dots_stats   = "data/stats/pearson/dots/s1/curves_s1_1runs_stats.npz"
    dots_fit     = "data/stats/pearson/dots/s1/curves_s1_1runs_fit.npz"
    stripes_stats = "data/stats/pearson/stripes/s1/curves_s1_1runs_stats.npz"
    stripes_fit   = "data/stats/pearson/stripes/s1/curves_s1_1runs_fit.npz"
    ours_dots_stats = "data/stats/dots/s1/curves_s1_20runs_stats.npz"
    ours_dots_fit = "data/stats/dots/s1/curves_s1_20runs_fit.npz"
    ours_stripes_stats = "data/stats/stripes/s1/curves_s1_20runs_stats.npz"
    ours_stripes_fit = "data/stats/stripes/s1/curves_s1_20runs_fit.npz"

    plot_dir = Path("data/journal/final_presentation/comparison_PvsUs_sigma1")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("v", "s", "chi"):
        png = plot_dir/f"pearson_dots_{metric}.png"

        plot.save_raw_data_plots_extended(150, metric, png, dots_stats, dots_fit,
                                        ours_dots_stats, ours_dots_fit,
                                        stripes_stats, stripes_fit, 
                                        ours_stripes_stats, ours_stripes_fit,)



    for func in ("pv", "ps", "pchi"):
        png = plot_dir/f"pearson_vs_us_{func}.png"
        plot.plot_functional_comparison(func, png,
                            dots_stats, dots_fit,
                            ours_dots_stats, ours_dots_fit,
                            stripes_stats, stripes_fit, 
                            ours_stripes_stats, ours_stripes_fit,
                            n_points=100)
        



    # --- WIP: PEARSON DOTS AND STRIPES ---
    # pearson_dots = ip.convert_img_to_grayscale_NEW("data/pearson/original_images/pearson_dots.png")
    # pearson_dots_blurred = scipy.ndimage.gaussian_filter(pearson_dots, sigma=1)
    # # # print(pearson_dots.max())
    # # # np.save("pearson_dots_array.npy", pearson_dots)
    # # plt.imshow(pearson_stripes, cmap='gray')
    # # plt.axis("off")
    # # plt.show()

    # np.save("NEW_pearson_dots_array.npy", pearson_dots)
    # v, s, chi = au.threshold_curves(pearson_dots_blurred)
    # au.plot_threshold_curves(v,s,chi)
    # pv, ps, pchi = au.transform_curves(v, s, chi)
    # au.plot_transform_curves(pv, ps, pchi)


    # --- COLOR TEST ---
    # grey = au.convert_img_to_grayscale("data\stats\dots\sigma0\img\I_raw_seed0.png",
    #                             median_blur=3)

    # print("grey.shape =", grey.shape) 
    # plt.imshow(grey, cmap='gray')
    # plt.axis("off")
    # plt.show()
    # v, s, chi = au.threshold_curves(grey)
    # au.plot_threshold_curves(v, s, chi)
    # pv, ps, pchi = au.transform_curves(v, s, chi)
    # au.plot_transform_curves(pv, ps, pchi)

    

    
if __name__ == "__main__":
    main()
