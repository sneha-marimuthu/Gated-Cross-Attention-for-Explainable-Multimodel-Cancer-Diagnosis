"""
Complete Audit Test Suite for GCAEMMCD Hierarchical Histopathology Diagnostic System.
Tests all 10 required scenarios:
1. Valid Breast Histopathology
2. Valid Lung Histopathology
3. Valid Colorectal Histopathology
4. Normal Tissue (Colon)
5. Benign Tissue (Breast)
6. Malignant Tissue (Lung)
7. Terminal Screenshot (Windows PowerShell)
8. General Photograph (Landscape: Sky & Foliage)
9. Blank Image (Pure White Slide)
10. Unsupported Histopathology Tissue (Oral Mucosa Biopsy)
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.data.domain_validator import validate_histopathology_domain
from src.ui.level1_app import hierarchical_inference_pipeline

os.makedirs("data/test_cases", exist_ok=True)

# Generate synthetic test files for non-histological test cases:
# 1. Terminal Screenshot (Windows PowerShell Dark Blue + Text)
ps_img = Image.new("RGB", (300, 200), (1, 36, 86))
draw_ps = ImageDraw.Draw(ps_img)
draw_ps.text((10, 10), "Windows PowerShell", fill=(255, 255, 255))
draw_ps.text((10, 30), "Copyright (C) Microsoft Corporation. All rights reserved.", fill=(204, 204, 204))
draw_ps.text((10, 60), "PS C:\\Users\\Administrator> python diagnose.py", fill=(255, 255, 255))
ps_path = "data/test_cases/powershell_terminal.png"
ps_img.save(ps_path)

# 2. General Photograph (Landscape: Blue Sky + Green Foliage)
photo_img = Image.new("RGB", (256, 256))
for y in range(256):
    for x in range(256):
        if y < 120:
            photo_img.putpixel((x, y), (135, 206, 235))  # Sky Blue
        else:
            photo_img.putpixel((x, y), (34, 139, 34))    # Forest Green
photo_path = "data/test_cases/general_photo.png"
photo_img.save(photo_path)

# 3. Blank Image (Pure White Slide)
blank_img = Image.new("RGB", (224, 224), (255, 255, 255))
blank_path = "data/test_cases/blank_slide.png"
blank_img.save(blank_path)

# 4. Unsupported Histopathology Tissue: use existing oral biopsy
oral_path = "/tmp/gradio/893bd77ca1dd2ac9124a979c148bad5825e1544a08d56e2793bf8501a0fb4231/oral1.jpg"

test_definitions = [
    ("Valid Breast Histopathology", "data/sample_images/breast/breast_00.png"),
    ("Valid Lung Histopathology", "data/sample_images/lung/lung_00.png"),
    ("Valid Colorectal Histopathology", "data/sample_images/colorectal/colorectal_00.png"),
    ("Normal Tissue (Colon)", "data/sample_images/normal/normal_colon_00.png"),
    ("Benign Tissue (Breast)", "data/sample_images/benign/benign_breast_00.png"),
    ("Malignant Tissue (Lung)", "data/sample_images/lung/lung_01.png"),
    ("Terminal Screenshot (PowerShell)", ps_path),
    ("General Photograph (Landscape)", photo_path),
    ("Blank Image (White Slide)", blank_path),
    ("Unsupported Histopathology Tissue (Oral Biopsy)", oral_path)
]

print("="*100)
print(f"{'TEST CASE':<32} | {'VALID':<5} | {'INFERENCE':<9} | {'ORGAN RESULT':<22} | {'STATUS RESULT':<16} | {'SUBTYPE':<16} | {'GRAD-CAM'}")
print("="*100)

results = []

for name, path in test_definitions:
    img = Image.open(path).convert("RGB")
    is_valid, domain, reason, _ = validate_histopathology_domain(img)
    
    # Run full pipeline
    res = hierarchical_inference_pipeline(img)
    organ_text, organ_probs, status_text, status_probs, subtype_text, subtype_probs, audit_text, overlay, cam_caption = res
    
    inference_executed = is_valid and ("Blocked" not in status_text and "BLOCKED" not in status_text)
    
    # Extract Organ
    if not is_valid:
        organ_res = f"REJECTED ({domain})"
        status_res = "BLOCKED"
        subtype_res = "BLOCKED"
        cam_gen = "NO (Suppressed)"
    else:
        if "Unsupported Organ" in organ_text or "Out-of-Distribution" in organ_text:
            organ_res = "OOD / Unsupported"
            status_res = "BLOCKED"
            subtype_res = "BLOCKED"
            cam_gen = "NO (Suppressed)"
        else:
            # Parse organ
            if "Breast" in organ_text:
                organ_res = "Breast"
            elif "Lung" in organ_text:
                organ_res = "Lung"
            elif "Colorectal" in organ_text:
                organ_res = "Colorectal"
            else:
                organ_res = "Unknown"
                
            # Parse status
            if "Normal" in status_text:
                status_res = "Normal"
            elif "Benign" in status_text:
                status_res = "Benign"
            elif "Malignant" in status_text:
                status_res = "Malignant"
            else:
                status_res = "Blocked"
                
            # Parse subtype
            if subtype_probs is not None:
                sub_names = ["Breast Carcinoma", "Lung Carcinoma", "Colorectal Carcinoma"]
                # get highest
                sub_top = max(subtype_probs.items(), key=lambda x: x[1])[0]
                subtype_res = sub_top
            elif "Uncertain" in subtype_text:
                subtype_res = "Uncertain"
            elif "Not Applicable" in subtype_text:
                subtype_res = "Not Applicable"
            else:
                subtype_res = "None"
                
            cam_gen = "YES (Overlay Generated)" if ("Grad-CAM Target" in cam_caption) else "NO"
            
    val_str = "PASS" if is_valid else "FAIL"
    inf_str = "YES" if inference_executed else "NO"
    
    print(f"{name:<32} | {val_str:<5} | {inf_str:<9} | {organ_res:<22} | {status_res:<16} | {subtype_res:<16} | {cam_gen}")
    results.append({
        "case": name,
        "input_validation": val_str,
        "inference_executed": inf_str,
        "organ": organ_res,
        "status": status_res,
        "subtype": subtype_res,
        "gradcam": cam_gen,
        "card_organ": organ_text[:120].replace("\n", " "),
        "card_status": status_text[:100].replace("\n", " "),
        "audit": audit_text[:120].replace("\n", " ")
    })

print("="*100)
