"""
=== Image Processing Utilities ===
Functions for converting images to grayscale, blurring image stacks,
and simulating CCD imaging effects. These are needed to prepare images for the
Minkowski analysis in analysis_utils.py.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import scipy.ndimage
import skimage.io as io
import warnings

def convert_img_to_grayscale_NEW(filename="data/test_images/pearson_dots.png", median_blur=None, size=(256, 256)):
    """
    Convert a BGR image to grayscale, optionally apply median blur, then resize to `size`.
    """
    img = cv2.imread(filename)

    if median_blur:
        # Standard grayscale conversion and optional blur
        gray_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.medianBlur(gray_bgr, ksize=median_blur)
        resized = cv2.resize(gray_blurred, size, interpolation=cv2.INTER_AREA)
        return resized  # shape: (256, 256)

    else:
        # HSV-based grayscale conversion via hue channel
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H = hsv[..., 0].astype(np.float32)

        hue_gray = (H / 120.0 * 255.0).clip(0, 255).astype(np.uint8)

        # Clean speckles
        kernel = np.ones((8, 8), np.uint8)
        cleaner_gray = cv2.morphologyEx(hue_gray, cv2.MORPH_OPEN, kernel, iterations=1)

        # Normalize brightness
        stretched = cv2.normalize(cleaner_gray, None, alpha=0, beta=255,
                                  norm_type=cv2.NORM_MINMAX).astype(np.uint8)

        resized = cv2.resize(stretched, size, interpolation=cv2.INTER_AREA)
        return resized  # shape: (256, 256)


def convert_img_to_grayscale(filename="data/test_images/pearson_dots.png", median_blur=None):
    img = cv2.imread(filename)

    if median_blur:
        """
        Convert apply median blur to grayscale img if specified.
        """
        gray_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        # 2) Then median‐blur that gray image
        gray_blurred = cv2.medianBlur(gray_bgr, ksize=median_blur)
        return gray_blurred   # now shape is (H, W)
    
    else:
        """
        Convert an image to grayscale from BGR to HSV, then extract the hue channel
        and normalize it to a grayscale image.
        """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0].astype(np.float32)  
    
    # takes hue where blue is max and red is min
    hue_gray = (H / 120.0 * 255.0).clip(0, 255).astype(np.uint8)

    # removes white speckled pixels using a kernel
    kernel = np.ones((8, 8), np.uint8)
    cleaner_gray = cv2.morphologyEx(hue_gray, cv2.MORPH_OPEN, kernel, iterations=1)

    # normalizes to 0 - 255 range
    stretched = cv2.normalize(cleaner_gray, None, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return stretched


def convert_png_to_array(file_name):
    """
    Convert grayscale image to array with signal strength values.
    """
    img = Image.open(file_name)
    img_gray = img.convert('L')
    img_gray = img_gray.resize((633, 641), Image.Resampling.LANCZOS)
    arr = np.array(img_gray)
    return arr


def blur_image_stack(input_path,
                     output_path,
                     *,
                     blur_type="gaussian",
                     sigma=1,
                     ksize=3,
                     fwhm_px=2.5,   
                     gamma=0.6):
    """
    Apply a Gaussian (σ) or median (ksize) blur to every 2-D slice in a stack.
    Writes a *new* .npz with key "images" (float32 in [0,1]) and a preview folder.
    """
    input_path, output_path = Path(input_path), Path(output_path)

    if input_path.suffix == ".npz":
        data = np.load(input_path)
        if "images" in data:
            arr = data["images"]
        else:                       # allow an un-keyed single array in .npz
            arr = data[list(data.files)[0]]
    else:                           # .npy assumed
        arr = np.load(input_path)

    arr = arr.astype(np.float32)        # ensure float32
    if arr.max() > 1.0:                 # normalise uint8 or 0..255
        arr /= 255.0

    # ── 2. ensure 3-D (n, H, W) ----------------------------------------
    if arr.ndim == 2:               # single pattern → pretend stack of one
        images = arr[np.newaxis, ...].astype(np.float32)
    elif arr.ndim == 3:
        images = arr.astype(np.float32)
    else:
        raise ValueError("array must be 2-D or 3-D")

    # always keep floats in [0,1]
    images = np.clip(images, 0.0, 1.0)


    if blur_type.lower() == "gaussian":
        blurred = scipy.ndimage.gaussian_filter(images, sigma=(0, sigma, sigma))

    elif blur_type.lower() == "median":
        if not (isinstance(ksize, int) and ksize >= 3 and ksize % 2 == 1):
            raise ValueError("ksize must be an odd integer ≥3")
        n, H, W = images.shape
        blurred = np.empty_like(images, dtype=np.float32)
        for i in range(n):
            im_u8 = (images[i] * 255).round().astype(np.uint8)
            med   = cv2.medianBlur(im_u8, ksize=ksize)
            blurred[i] = med.astype(np.float32) / 255.0
    
    elif blur_type.lower() == "mecke":
        rng = np.random.default_rng(0)
        n, H, W = images.shape
        blurred_u8 = np.empty((n, H, W), dtype=np.uint8)
        for i in range(n):
            blurred_u8[i] = mimic_mecke(images[i],
                                         fwhm_px=fwhm_px,
                                         gamma=gamma,
                                         rng=rng)
        blurred = blurred_u8.astype(np.float32) / 255.0   # back to [0,1]

    else:
        raise ValueError("blur_type must be 'gaussian', 'median' or 'mecke'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, images=blurred)

    print("✔ blurred stack  →", output_path)
    
    return output_path      # so caller knows where the blurred .npz is


def mimic_mecke(im, *, fwhm_px=2.5, gamma=0.6,
                 add_poisson=False, rng=np.random.default_rng()):
    """Return a uint8 image that looks like a CCD photo of *im* (float32 [0,1])."""
    sigma = fwhm_px / (2*np.sqrt(2*np.log(2)))          # FWHM → σ
    blur  = scipy.ndimage.gaussian_filter(im, sigma=sigma, mode="reflect")
    if add_poisson:
        blur = rng.poisson(blur*255) / 255.0             # shot noise
    out = np.clip(blur, 0, 1) ** gamma
    return (out*255).round().astype(np.uint8)



def process_stack(input_path, output_path, *,
                  fwhm_px=2.5, gamma=0.6, add_poisson=True,
                  preview_png=False, seed=None):
    """
    Apply `mimic_mecke()` to every slice in an .npz stack and write a *new*
    .npz with the uint8 images under key "images".
    """
    rng  = np.random.default_rng(seed)
    inp  = Path(input_path)
    outp = Path(output_path)

    data = np.load(inp)
    if "images" not in data:
        raise KeyError(f"No key 'images' in {inp}")
    raw  = data["images"]          # shape (n, H, W), float32 [0,1]

    proc = np.empty_like(raw, dtype=np.uint8)
    for i, frame in enumerate(raw):
        proc[i] = mimic_mecke(frame, fwhm_px=fwhm_px,
                              gamma=gamma, add_poisson=add_poisson, rng=rng)

    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, images=proc)
    print(f"✔ imaging model → {outp}")

    # optional quick-look PNGs ----------------------------------------------
    if preview_png:
        pdir = outp.with_suffix("")       # same basename, just a dir
        pdir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(proc):
            io.imsave(pdir / f"slice_{i:03d}.png", fr, check_contrast=False)
        print(f"  previews      → {pdir} ({proc.shape[0]} files)")

    return outp

