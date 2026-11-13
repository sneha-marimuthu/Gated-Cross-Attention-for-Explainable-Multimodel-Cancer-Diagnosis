"""
Comprehensive Integration Test Suite for GCAEMMCD Clinical Workstation.
Verifies:
  1. GCAEMMCD Theme & HTML Generators (Zero external vendor strings).
  2. Tissue macro minimap overview generation.
  3. Specimen S-01 (Colorectal Adenocarcinoma) full pipeline execution, Grad-CAM, SHAP, and BioBERT saliency.
  4. Specimen S-04 (Non-Neoplastic Emphysema) execution and status gate.
  5. Pathology viewport layer switching (Grad-CAM, Clean H&E, Masked Saliency, SHAP Waterfall).
  6. Optical domain rejection gate on non-histopathology input.
  7. Gradio app interface structural integrity.
"""

import os
import sys
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config

from src.ui.aisight_app import (
    process_aisight_case, SLIDE_PRESETS, create_overview_minimap
)
from src.ui.aisight_theme import (
    get_top_navbar_html, get_sub_header_html, get_viewer_toolbar_html,
    get_slide_badge_html, get_footer_disclaimer_html
)


def run_tests():
    print("=" * 80)
    print("RUNNING GCAEMMCD CLINICAL WORKSTATION INTEGRATION TESTS")
    print("=" * 80)

    # 1. Test Theme HTML Generators
    print("\n--- [Test 1: GCAEMMCD Theme & HTML Generation] ---")
    top_nav = get_top_navbar_html()
    assert "GCAEMMCD" in top_nav and "Clinical Pathology Workstation" in top_nav, "Top navbar missing GCAEMMCD branding!"
    assert "AISight" not in top_nav and "PathAI" not in top_nav, "Found forbidden external vendor branding in top navbar!"
    print("[PASS] Top global navigation bar verified (Clean GCAEMMCD branding).")

    sub_header = get_sub_header_html(specimen_id="Specimen S-01", status="Ready")
    assert "Specimen S-01" in sub_header and "Diagnostic Session" in sub_header, "Sub-header missing session metadata!"
    assert "Rishi" not in sub_header and "Dana-Farber" not in sub_header, "Found forbidden external names in sub-header!"
    print("[PASS] Sub-header specimen metadata bar verified.")

    toolbar = get_viewer_toolbar_html(specimen_id="Specimen S-01")
    assert "Pathology Micrograph Viewport" in toolbar and "1.84 mm²" in toolbar, "Digital pathology toolbar missing!"
    print("[PASS] Digital pathology toolbar verified.")

    badge = get_slide_badge_html("H&E Slide S-01", "🔴 Malignant Neoplasm")
    assert "status-pill-malignant" in badge, "Malignant status pill class missing!"
    print("[PASS] Dynamic viewport status badge verified.")

    disclaimer = get_footer_disclaimer_html()
    assert "GCAEMMCD" in disclaimer and "Investigational & Research" in disclaimer, "Disclaimer missing!"
    print("[PASS] Clinical disclaimer footer verified.")

    # 2. Test Minimap Macro Overview Generator
    print("\n--- [Test 2: Tissue Macro Minimap Overview] ---")
    p1 = SLIDE_PRESETS["slide_1"]
    im1 = Image.open(p1["img_path"])
    m_img = create_overview_minimap(im1)
    assert m_img.size == (140, 100), f"Minimap dimensions unexpected: {m_img.size}"
    print(f"[PASS] Tissue macro minimap generated successfully with dimensions {m_img.size}.")

    # 3. Test Slide 1: Colorectal Carcinoma
    print("\n--- [Test 3: Specimen S-01 Colorectal Adenocarcinoma Full Pipeline] ---")
    tb1 = p1["tab"]
    res1 = process_aisight_case(
        im1, p1["text"], True, *tb1,
        active_view_layer="🔥 Faithful Grad-CAM Saliency",
        current_slide_id="H&E Slide S-01",
        current_badge_text="🔴 Malignant Neoplasm"
    )
    v_img, mini_img, badge_html, accept_md, algo_md, align_md, faith_md, text_html, shap_img = res1

    assert v_img is not None, "Viewport image is None!"
    assert "VERIFIED H&E HISTOPATHOLOGY" in accept_md, "Acceptability check failed on Slide 1!"
    assert "Colorectal" in algo_md or "Malignant" in algo_md, "Malignancy diagnosis not found in algorithm results!"
    assert "FAITHFUL" in faith_md, "Faithfulness verification missing in faith_md!"
    assert shap_img is not None, "SHAP chart not generated!"
    assert "<span" in text_html, "BioBERT narrative attribution HTML not generated!"
    print("[PASS] Specimen S-01 Colorectal Ca: Stage 0-3 + GCA + SHAP + BioBERT verified.")

    # 4. Test Slide 4: Non-Neoplastic Lung
    print("\n--- [Test 4: Specimen S-04 Non-Neoplastic Lung Resection] ---")
    p4 = SLIDE_PRESETS["slide_4"]
    im4 = Image.open(p4["img_path"])
    tb4 = p4["tab"]
    res4 = process_aisight_case(
        im4, p4["text"], True, *tb4,
        active_view_layer="🖼️ Clean H&E Micrograph",
        current_slide_id="H&E Slide S-04",
        current_badge_text="🟢 Non-Neoplastic"
    )
    assert "Non-Neoplastic" in res4[2], f"Status badge did not reflect non-neoplastic: {res4[2]}"
    print("[PASS] Specimen S-04 Non-Neoplastic: Negative status gating verified.")

    # 5. Test Layer Mode Switching
    print("\n--- [Test 5: Pathology Viewport Layer Toggles] ---")
    # Clean H&E
    res_clean = process_aisight_case(im1, p1["text"], True, *tb1, active_view_layer="🖼️ Clean H&E Micrograph")
    assert res_clean[0].size == (224, 224) or res_clean[0].size == (256, 256) or res_clean[0].size == im1.size, "Clean view error"

    # Saliency Masked
    res_mask = process_aisight_case(im1, p1["text"], True, *tb1, active_view_layer="🎭 Causal Saliency Masking (Top 20% Blurred)")
    assert res_mask[0] is not None, "Saliency masked view error"

    # SHAP Waterfall
    res_shap = process_aisight_case(im1, p1["text"], True, *tb1, active_view_layer="📊 SHAP Biomarker Waterfall")
    assert res_shap[0] is not None, "SHAP waterfall view error"
    print("[PASS] Viewport layer switcher verified (Grad-CAM, Clean H&E, Masked, SHAP).")

    # 6. Test Ingestion Gate Rejection (Dark UI Screenshot)
    print("\n--- [Test 6: Ingestion Optical Domain Gatekeeper] ---")
    dark_screenshot = Image.new("RGB", (300, 300), color=(18, 24, 38))
    res_rej = process_aisight_case(dark_screenshot, "Clinical text", False, *tb1)
    assert "REJECTED AT INGESTION GATE" in res_rej[3], "Non-tissue image was not rejected!"
    assert "Algorithm Execution Halted" in res_rej[4], "Model execution was not halted!"
    print("[PASS] Optical domain rejection gate successfully prevented non-tissue processing.")

    print("\n" + "=" * 80)
    print("ALL GCAEMMCD CLINICAL WORKSTATION INTEGRATION TESTS PASSED (100%)")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
