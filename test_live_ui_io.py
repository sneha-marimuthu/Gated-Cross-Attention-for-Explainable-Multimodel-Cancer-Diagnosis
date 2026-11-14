import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/frappe/.gemini/antigravity/scratch/gcaemmcd_demo")
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
            active_view_layer="Faithful Grad-CAM Saliency",
            current_slide_id=p["id"],
            current_badge_text=p["badge_text"]
        )
        
        main_viewer_img, minimap_view, badge_component, accept_display, algo_display, align_display, faith_display, text_html_view, shap_thumbnail = outs
        
        print(f"\n--- Output verification for {p['id']} ({key}) ---")
        print(f"1. Viewport image shape: {np.array(main_viewer_img).shape}")
        print(f"2. Minimap image shape: {np.array(minimap_view).shape}")
        print(f"3. Badge HTML: {badge_component.strip()[:60]}...")
        print(f"4. Acceptability (Stage 0): {'VERIFIED' if 'VERIFIED' in accept_display else 'REJECTED'}")
        
        # Check Task 1 to 4 in algo_display
        has_task1 = "Task 1: Neoplasm Classification" in algo_display
        has_task2 = "Task 2: TNM Pathological Cancer Staging" in algo_display
        has_task3 = "Task 3: Multimodal Treatment Recommendation" in algo_display
        has_task4 = "Task 4: 5-Year Survival Risk" in algo_display
        
        print(f"5. Tasks in algo_display:")
        print(f"   - Task 1 (Classification): {has_task1}")
        print(f"   - Task 2 (TNM Staging): {has_task2}")
        print(f"   - Task 3 (Treatment Ranking): {has_task3}")
        print(f"   - Task 4 (5-Year Survival): {has_task4}")
        
        print(f"6. BioBERT Token Saliency HTML length: {len(text_html_view)} chars (contains spans: {'<span' in text_html_view})")
        print(f"7. SHAP chart generated: {shap_thumbnail is not None}")

    print("\n" + "=" * 60)
    print("STEP 3: TESTING CUSTOM UPLOAD INPUT GATE & OOD VALIDATOR")
    print("=" * 60)
    white_dummy = Image.new("RGB", (224, 224), color=(255, 255, 255))
    dummy_outs = process_aisight_case(
        white_dummy, "Test note", True, 50, "Female", 0, "0", 1.0, 10.0, 10.0, 10.0,
        "Negative (0)", "Negative (0)", "Negative (0)", "Wildtype", "Wildtype", 13.0, 6.0, 250.0
    )
    print(f"Blank image ingestion test: {dummy_outs[3][:80]}...")
    print(f"Algorithm stopped: {'Execution Halted' in dummy_outs[4]}")

if __name__ == "__main__":
    test_inputs_and_outputs()
