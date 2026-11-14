import os
import sys
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
from src.ui.aisight_app import (
    SLIDE_PRESETS,
    process_aisight_case,
    _map_radio_binary,
    _map_radio_int
)

def test_inputs_and_outputs():
    print("=" * 60)
    print("STEP 1: INSPECTING PRESET INPUTS (S-01 to S-05)")
    print("=" * 60)
    
    for key, p in SLIDE_PRESETS.items():
        img_path = p["img_path"]
        assert os.path.exists(img_path), f"Image file missing for {key}: {img_path}"
        text = p["text"]
        tab = p["tab"]
        assert len(tab) == 16, f"Expected 16 biomarkers, got {len(tab)} for {key}"
        print(f"[{key}] {p['id']}: Image={os.path.basename(img_path)}, Text='{text[:45]}...', 16 Biomarkers: Age={tab[0]}, Sex={tab[1]}, CEA={tab[4]}, Ki67={tab[7]}")
        
    print("\n" + "=" * 60)
    print("STEP 2: RUNNING INFERENCE ON ALL 5 PRESETS & CHECKING 4-TASK OUTPUTS")
    print("=" * 60)
    
    for key, p in SLIDE_PRESETS.items():
        im = Image.open(p["img_path"])
        tb = p["tab"]
        outs = process_aisight_case(
            im, p["text"], True, *tb,
            active_view_layer="🔥 Faithful Grad-CAM Saliency",
            current_slide_id=p["id"],
            current_badge_text=p["badge_text"],
            patient_id=p.get("patient_id", "PAT-001"),
            tumor_size=p.get("tumor_size", "3.2 cm")
        )
        
        card_html, viewport_img, faith_badge, shap_img, shap_badge, meta = outs
        
        print(f"\n--- Output verification for {p['id']} ({key}) ---")
        print(f"1. Viewport image shape: {np.array(viewport_img).shape if viewport_img is not None else None}")
        print(f"2. Card HTML length: {len(card_html)} chars")
        print(f"3. Faithfulness Badge: {faith_badge.strip()[:60]}...")
        
        # Check Sections 01 to 06 in card_html
        has_sec1 = "01. CANCER CLASSIFICATION" in card_html
        has_sec2 = "02. TNM STAGING" in card_html
        has_sec3 = "03. SURVIVAL ESTIMATION" in card_html
        has_sec4 = "04. TREATMENT RECOMMENDATION" in card_html
        has_sec5 = "05. EXPLAINABLE AI ANALYSIS" in card_html
        has_sec6 = "06. MODEL & INFERENCE INFORMATION" in card_html
        
        print(f"4. Sections in card_html:")
        print(f"   - Section 01 (Classification): {has_sec1}")
        print(f"   - Section 02 (TNM Staging): {has_sec2}")
        print(f"   - Section 03 (Survival Estimation): {has_sec3}")
        print(f"   - Section 04 (Treatment): {has_sec4}")
        print(f"   - Section 05 (XAI): {has_sec5}")
        print(f"   - Section 06 (Model Info): {has_sec6}")
        
        print(f"5. SHAP chart generated: {shap_img is not None}")

    print("\n" + "=" * 60)
    print("STEP 3: TESTING CUSTOM UPLOAD INPUT GATE & OOD VALIDATOR")
    print("=" * 60)
    white_dummy = Image.new("RGB", (224, 224), color=(255, 255, 255))
    dummy_outs = process_aisight_case(
        white_dummy, "Test note", True, 50, "Female", 0, "0", 1.0, 10.0, 10.0, 10.0,
        "Negative (0)", "Negative (0)", "Negative (0)", "Wildtype", "Wildtype", 13.0, 6.0, 250.0
    )
    print(f"Blank image ingestion test: {dummy_outs[0][:80]}...")
    print(f"Algorithm stopped: {'OPTICAL DOMAIN CHECK FAILED' in dummy_outs[0]}")

if __name__ == "__main__":
    test_inputs_and_outputs()
