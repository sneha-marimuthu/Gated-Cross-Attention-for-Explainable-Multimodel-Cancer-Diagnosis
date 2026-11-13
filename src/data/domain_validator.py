"""
Histopathology Domain Validator.
Rigorous, physically-grounded input validation for brightfield H&E histopathology.
Distinguishes authentic H&E tissue sections from:
  1. Terminal screenshots (PowerShell, Command Prompt, Bash)
  2. Document screenshots (PDFs, papers, browser UI)
  3. General photographs (landscapes, people, animals, objects)
  4. Blank / empty / saturated slides (pure white, pure black, uniform noise)

Physical & Optical Principles:
  In Brightfield H&E microscopy:
  - Both Hematoxylin (nuclear dye) and Eosin (cytoplasmic counterstain) absorb Green wavelengths.
    Therefore, pixels where Green is dominant (G > R and G > B with G > 0.15) are biologically
    forbidden in H&E tissue and represent green foliage, grass, or outdoor scenes.
  - Hematoxylin is a violet/purple dye with substantial red transmission (R >= 0.15, R/B >= 0.35).
    Pixels with high blue and near-zero red (B > 0.20 and R < 0.10) represent non-biological
    monochromatic cobalt/cyan colors, such as Windows PowerShell blue (#012456).
  - Blank slides and digital text documents have high white fractions (>85%) or minimal
    tissue stain fraction (<8%).
"""

import cv2
import numpy as np
from PIL import Image

def validate_histopathology_domain(input_image):
    """
    Validates whether an input PIL Image belongs to the H&E histopathology domain.
    
    Returns:
        is_valid (bool): True if image passes all histopathology domain checks.
        domain_type (str): Classification of input ("histopathology", "terminal_screenshot",
                           "document_screenshot", "general_photo", "blank_or_uninformative").
        rejection_reason (str): Detailed clinical/technical reason for rejection.
        metrics (dict): Quantitative optical and chromatic measurements.
    """
    if input_image is None:
        return False, "null_input", "No image provided.", {}

    img = input_image.convert("RGB")
    img_rgb = np.array(img).astype(np.float32) / 255.0
    h, w, _ = img_rgb.shape
    total_pixels = h * w

    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # 1. Blank, pure white, or pure black slide detection
    is_white = (r > 0.92) & (g > 0.92) & (b > 0.92)
    is_black = (r < 0.08) & (g < 0.08) & (b < 0.08)
    white_ratio = float(np.mean(is_white))
    black_ratio = float(np.mean(is_black))

    if white_ratio > 0.85:
        return False, "blank_or_uninformative", (
            f"Blank / Empty Slide Detected: {white_ratio*100:.1f}% of pixels are uninformative white background. "
            "Please upload a slide region containing biological tissue."
        ), {"white_ratio": white_ratio, "black_ratio": black_ratio, "tissue_fraction": 0.0, "he_fraction": 0.0, "forbidden_fraction": 0.0}

    if black_ratio > 0.85:
        return False, "blank_or_uninformative", (
            f"Over-Dark / Black Image Detected: {black_ratio*100:.1f}% of pixels are black or underexposed sensor noise. "
            "Please upload a properly illuminated brightfield pathology specimen."
        ), {"white_ratio": white_ratio, "black_ratio": black_ratio, "tissue_fraction": 0.0, "he_fraction": 0.0, "forbidden_fraction": 0.0}

    # 2. Tissue Saturation & Content Detection
    max_c = np.max(img_rgb, axis=-1)
    min_c = np.min(img_rgb, axis=-1)
    delta = max_c - min_c

    sat = np.zeros_like(max_c)
    non_zero = max_c > 0
    sat[non_zero] = delta[non_zero] / max_c[non_zero]

    tissue_mask = (sat > 0.10) & (~is_white) & (~is_black)
    tissue_pixel_count = int(np.sum(tissue_mask))
    tissue_fraction = float(tissue_pixel_count / total_pixels)

    if tissue_fraction < 0.08:
        return False, "document_screenshot", (
            f"Non-Histology Specimen Rejected: Insufficient stained tissue content "
            f"({tissue_fraction*100:.1f}% detected, minimum 8.0% required). "
            "Detected text document, screenshot, or unstained slide."
        ), {
            "white_ratio": round(white_ratio, 4),
            "black_ratio": round(black_ratio, 4),
            "tissue_fraction": round(tissue_fraction, 4),
            "he_fraction": 0.0,
            "forbidden_fraction": 0.0,
            "unnatural_blue_ratio": 0.0,
            "green_dominant_ratio": 0.0
        }

    # 3. Non-Biological Spectrum & H&E Chromophore Physics
    # A. Unnatural Cobalt Blue (PowerShell default background, Windows interface)
    unnatural_blue = (b > 0.20) & (r < 0.10)
    unnatural_blue_ratio = float(np.mean(unnatural_blue))

    # B. Green Dominance (Foliage, vegetation, general outdoor photography)
    green_dominant = (g > r) & (g > b) & (g > 0.15)
    green_dominant_ratio = float(np.mean(green_dominant))

    # C. Dark Slate UI Background (e.g., dark-mode browser, terminal, IDE themes)
    dark_slate = (gray < 0.22) & (r < 0.28) & (g < 0.28)
    dark_slate_ratio = float(np.mean(dark_slate))

    # D. H&E Chromophore Concordance among stained pixels (Hematoxylin violet + Eosin pink absorb Green)
    is_he_pixel = (
        tissue_mask
        & (r > 0.12)
        & ((r > g + 0.02) | (b > g + 0.02))
        & ~((g > r) & (g > b))
        & ~unnatural_blue
    )
    he_fraction = float(np.sum(is_he_pixel) / max(1, tissue_pixel_count))
    forbidden_fraction = float(min(1.0, unnatural_blue_ratio + green_dominant_ratio + dark_slate_ratio))

    # E. Structural Texture, Digital Flatness & Focus Sharpness
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    flat_ratio = float(np.mean((dx < 0.002) & (dy < 0.002)))
    gray_224 = cv2.resize((gray * 255.0).astype(np.uint8), (224, 224))
    laplacian_var = float(cv2.Laplacian(gray_224, cv2.CV_64F).var())
    gy, gx = np.gradient(gray)
    mean_grad = float(np.mean(np.sqrt(gx**2 + gy**2)))

    metrics = {
        "white_ratio": round(white_ratio, 4),
        "black_ratio": round(black_ratio, 4),
        "tissue_fraction": round(tissue_fraction, 4),
        "he_fraction": round(he_fraction, 4),
        "forbidden_fraction": round(forbidden_fraction, 4),
        "unnatural_blue_ratio": round(unnatural_blue_ratio, 4),
        "green_dominant_ratio": round(green_dominant_ratio, 4),
        "dark_slate_ratio": round(dark_slate_ratio, 4),
        "flat_ratio": round(flat_ratio, 4),
        "laplacian_var": round(laplacian_var, 2),
        "mean_gradient": round(mean_grad, 4),
    }

    # Evaluation of validation gates
    if unnatural_blue_ratio > 0.12:
        return False, "terminal_screenshot", (
            f"Terminal / Digital Interface Screenshot Rejected: {unnatural_blue_ratio*100:.1f}% of image consists of "
            "unnatural monochromatic cobalt blue (e.g., Windows PowerShell or console UI). "
            "H&E histopathology does not emit pure cobalt light without red/eosin transmission."
        ), metrics

    if green_dominant_ratio > 0.18:
        return False, "general_photo", (
            f"General Non-Medical Photograph Rejected: {green_dominant_ratio*100:.1f}% of image contains dominant "
            "green wavelengths (e.g., foliage, grass, or natural scene). "
            "In H&E brightfield microscopy, green light is absorbed by both counterstains."
        ), metrics

    if dark_slate_ratio > 0.35:
        return False, "terminal_screenshot", (
            f"Dark-Mode UI / Unilluminated Scene Rejected: {dark_slate_ratio*100:.1f}% dark slate/low-luminance pixels detected. "
            "Brightfield H&E microscopy requires transmitted slide illumination."
        ), metrics

    if flat_ratio > 0.55:
        return False, "document_screenshot", (
            f"Synthetic Graphic / UI Screenshot Rejected: {flat_ratio*100:.1f}% spatially flat rendered pixels detected "
            "(characteristic of digital UIs, plots, or text screenshots rather than cellular tissue)."
        ), metrics

    if laplacian_var < 3.5:
        return False, "blank_or_uninformative", (
            f"Severely Blurred / Featureless Image Rejected: focus sharpness metric ({laplacian_var:.2f}) is below "
            "minimum brightfield cellular resolution threshold (3.50)."
        ), metrics

    if he_fraction < 0.78:
        return False, "general_photo", (
            f"Non-H&E Color Spectrum Rejected: only {he_fraction*100:.1f}% of stained pixels match hematoxylin-eosin "
            "absorption physics (minimum 78.0% required)."
        ), metrics

    # Image passed all histopathology domain checks
    return True, "histopathology", "Valid Brightfield H&E Histopathology Specimen Verified.", metrics
