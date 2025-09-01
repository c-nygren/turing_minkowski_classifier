"""
=== Project Code ===
This module contains functions for simulating a reaction-diffusion system using
the Gray-Scott model. 
"""

from dataclasses import dataclass
import numpy as np
import analysis_utils as au
import plotcode as plot
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import scipy
from scipy.ndimage import convolve, gaussian_filter
import plotcode as plot
import multiprocessing as mp


@dataclass
class ModelParams:
    """Parameters of the Gray-Scott model."""

    Du: float   # Diffusion coefficient of activator U
    Dv: float   # Diffusion coefficient of inhibitor V
    F: float    # Feed rate
    k: float    # Kill rate
    dt: float   # Timestep
    steps: int  # Simulation steps
    N : int     # Grid size

def initial_grid(N):
    """
    Return initial concentration fields (u, v). By default the entire domain 
    starts in steady state (u=1, v=0).

    Parameters
    ----------
    N : int
        The number of points along a grid length. 
    
    Returns
    -------
    u : np.ndarray, shape (N,N)
        Concentration field as a numpy grid initialized with ones.
    v : np.ndarray, shape (N,N)
        Concentration field as a numpy grid initialized with zeroes.
    """
    u = np.ones((N,N))
    v = np.zeros((N,N))
    return u,v


def collect_images(params,
                   n_runs,
                   *,
                   outdir="data/stats",
                   tag=None,
                   test=False):
    """
    Runs the simulation n_runs times (or a dummy test), saves the *raw*, so non-
    blured, final‐frame images (normalized to [0,255] uint8) into outdir/"img", 
    and returns a list of 2D float‐arrays normalized to [0,1].

    - If test=True, we just generate random 256×256 patterns.
    - Otherwise, we actually run `simulation(p=params, seed=…)`, take the last 
    frame,
      convert to a “photo” via plot.pearson_arr, normalize to [0,1], convert to 
      uint8 *only for saving* but keep the float‐[0,1] array in memory.

    Returns:
        images_norm:  list of length n_runs, each element is a 2D float32 array 
        in [0,1].
    """
    outdir = Path(outdir)
    img_dir = outdir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    images_norm = []
    seeds = list(range(n_runs))

    if test:
        print("Running in test‐mode: generating random images.")
        for seed in seeds:
            # generate a random uint8 image; then also keep a float version in [0,1]
            test_dir = Path("data/stats/test")
            I_orig_uint8 = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)
            Image.fromarray(I_orig_uint8).save(test_dir / f"I_raw_seed{seed}.png")

            # create a float version in [0,1]
            I_norm = I_orig_uint8.astype(np.float32) / 255.0
            images_norm.append(I_norm)

    else:
        print("Running real simulations and collecting final‐frame images.")
        for seed in seeds:
            #  simulation
            U, V, *_ = simulation(p=params, save=False, seed=seed)

            # last time‐slice of U and V
            last_U = U[-1]    # 2D float array
            last_V = V[-1]    # 2D float array

            # "pseudo‐photo” via plotting function
            I_orig = plot.pearson_arr(U_snap=last_U, V_snap=last_V)
            #     → this yields a 2D float array in some arbitrary range

            # TODO try using just last_U
            # min–max normalize I_orig to [0,1]
            I_min, I_max = float(np.min(I_orig)), float(np.max(I_orig))
            if I_max > I_min:
                I_norm = (I_orig - I_min) / (I_max - I_min)
            else:
                I_norm = np.zeros_like(I_orig, dtype=np.float32)

            # save a uint8 copy for debugging
            I_uint8 = (I_norm * 255.0).round().astype(np.uint8)
            Image.fromarray(I_uint8).save(img_dir / f"I_raw_seed{seed}.png")

            images_norm.append(I_norm.astype(np.float32))

        image_stack = np.stack(images_norm, axis=0)  # shape = (n_runs, height, width)
        tag_suffix = f"_{tag}" if tag else ""
        stack_filename = outdir / f"images_stack_{n_runs}runs{tag_suffix}.npz"
        np.savez_compressed(stack_filename, images=image_stack)
        print(f"Saved full image stack to:\n    {stack_filename}")

    # Return a list of float32 arrays, each in [0,1].
    return images_norm
    

def simulation(p: ModelParams, save_interval=500, save=True, 
         base_dir="data/simulations", seed=None):
    """
    Runs a simulation to create Turing patterns by solving the Grey-Scott 
    reaction-diffusion equations for a system of two concentrations (u,v). 

    Parameters
    ----------
    p : ModelParams
        Class of parameters for the system to solve and simulate. Consists of
        the two diffusion coefficients, Du (activator of u) and Dv (inhibitor
        of v), the feed rate (F), kill rate (k), timestep (dt), simulation 
        steps (steps), and the grid size (N).
    save_interval : int, optional (default=500)
        The interval size at which snapshots (data) of the system is saved.
    save : bool, optional (default=True)
        Decides whether data from a simulation run should be saved or not.
    base_dir : str, optional (default="data/simulations")
        The path of the place to save data.
    
    Returns
    -------
    U_snap : np.ndarray
        Storage array of snapshots of u.
    V_snap : np.ndarray
        Storage array of snapshots of u. 
    u : np.ndarray, shape (N,N)
        Final result of u.
    v : np.ndarray, shape (N,N)
        Final result of v.
    """
    # Define spatial scale to match Pearson 1993
    dx = 2.5 / p.N

    # Defining the 5-point stencil kernel
    kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])

    u, v = initial_grid(p.N)

    # Add sqaure pertubation
    r = 20
    u[p.N//2 - r:p.N//2 + r, p.N//2 - r:p.N//2 + r] = 0.5
    v[p.N//2 - r:p.N//2 + r, p.N//2 - r:p.N//2 + r] = 0.25

    # Add +- 1% multiplicative random noise, as Pearson
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(2025)
    level = 0.01 

    u *= 1 + level * (2 * rng.random(u.shape) - 1)
    v *= 1 + level * (2 * rng.random(v.shape) - 1)

    # keep concentrations within range
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    U_list: list[np.ndarray] = []
    V_list: list[np.ndarray] = []

    for i in tqdm(range(p.steps)):
    # for i in range(p.steps):

        # Compute periodic Laplacian using 5-point stencil
        Lu = convolve(u, kernel, mode='wrap') / dx**2
        Lv = convolve(v, kernel, mode='wrap') / dx**2
        
        # Perform one Euler step
        u += p.dt * (p.Du * Lu - u * v * v + p.F * (1 - u))
        v += p.dt * (p.Dv * Lv + u * v * v - (p.F + p.k) * v)
   
        if i % save_interval == 0:
            U_list.append(u.copy())
            V_list.append(v.copy())
    
    U_list.append(u.copy())
    V_list.append(v.copy())

    # stack into shape (n_snapshots, N, N) - float 32 to half storage space
    U_snap = np.stack(U_list, axis=0).astype(np.float32)
    V_snap = np.stack(V_list, axis=0).astype(np.float32)

    if save:
        print("--- saving data ---")
        au.save_data(
            U_snap, V_snap,
            params=p,
            dt=p.dt,
            dx=1,
            steps=p.steps,
            save_interval=save_interval,
            base_dir=base_dir      
        )

    return U_snap, V_snap, u, v

def one_pattern_sim(F, k, save_interval, save, base_dir):
    # Define parameters
    params = ModelParams(
        Du=2e-5, 
        Dv=1e-5, 
        F=F,
        k=k,  
        dt=1,
        steps=35000,
        N=256
    )

    # Run simulation
    usnap, vsnap, u, v = simulation(
        p=params, 
        save_interval=save_interval, 
        save=save, 
        base_dir=base_dir
    )

    # Plot and save
    merged_arr = plot.pearson_arr(U_snap=usnap, V_snap=vsnap)
    plot.pearson_plot(merged_arr, id_number=F, base_dir=base_dir)


def one_pattern_sim_wrapper(args):
    """Unpack arguments for multiprocessing."""
    return one_pattern_sim(*args)

def pattern_sweep_simulation(k_array, F_array, save_interval=500, save=False, 
         base_dir="data/phaseimages/"):
    
    M = len(k_array)

    # Create tuples of parameters to pass to each process
    param_tuples = [
        (F_array[i], k_array[i], save_interval, save, base_dir) 
        for i in range(M)
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(one_pattern_sim_wrapper, param_tuples), total=M):
            pass