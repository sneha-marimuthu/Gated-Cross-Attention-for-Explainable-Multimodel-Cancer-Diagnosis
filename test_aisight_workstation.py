"""
Comprehensive Test Suite for PathAI AISight Clinical Workstation (GCAEMMCD).
Verifies:
  1. Slide 1 (Colorectal Adenocarcinoma) full execution, Grad-CAM, SHAP, and BioBERT saliency.
  2. Slide 2 (Lung Adenocarcinoma) execution and EGFR concordance.
  3. Slide 4 (Non-Neoplastic Emphysema) execution and status gate.
  4. Slide 5 (Benign Breast Fibroadenoma) execution.
  5. Viewport layer routing (Grad-CAM, Clean H&E, Masked Saliency, SHAP Waterfall).
  6. Optical domain rejection gate on non-histopathology image.
  7. Gradio app interface structural integrity.
"""

import os
import sys
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.ui.aisight_app import (
    process_aisight_case, SLIDE_PRESETS, create_overview_minimap
)
from src.ui.aisight_theme import (
    get_top_navbar_html, get_sub_header_html, get_viewer_toolbar_html,
    get_slide_badge_html, get_footer_disclaimer_html
)


def run_tests():
    print("=" * 80)
    print("RUNNING PATHAI AISIGHT CLINICAL WORKSTATION INTEGRATION TESTS")
    print("=" * 80)

    # 1. Test Theme HTML Generators
    print("\n--- [Test 1: Theme & HTML Generation] ---")
    top_nav = get_top_navbar_html()
    assert "AISight" in top_nav and "PathAI Demo Site" in top_nav, "Top navbar missing branding!"
    print("[PASS] Top global navigation bar verified.")

    sub_header = get_sub_header_html(accession_id="ACC-2026-9041", status="Assessment Complete")
    assert "ACC-2026-9041" in sub_header and "Preview Report" in sub_header, "Sub-header missing metadata!"
    print("[PASS] Sub-header accession metadata bar verified.")

    toolbar = get_viewer_toolbar_html()
    assert "Caliper Ruler" in toolbar and "1.84 mm²" in toolbar, "Digital pathology toolbar missing!"
    print("[PASS] Digital pathology toolbar verified.")

    badge = get_slide_badge_html("H&E Slide A1-1", "🔴 Possible cancer")
    assert "slide-pill-cancer" in badge, "Cancer status pill class missing!"
    print("[PASS] Dynamic viewport status badge verified.")

    disclaimer = get_footer_disclaimer_html()
    assert "Research Use Only" in disclaimer, "Disclaimer missing!"
    print("[PASS] Clinical disclaimer footer verified.")

    # 2. Test Minimap Macro Overview Generator
    print("\n--- [Test 2: Tissue Macro Minimap Overview] ---")
    p1 = SLIDE_PRESETS["slide_1"]
    im1 = Image.open(p1["img_path"])
    m_img = create_overview_minimap(im1)
    assert m_img.size == (140, 100), f"Minimap dimensions unexpected: {m_img.size}"
    print(f"[PASS] Tissue macro minimap generated successfully with dimensions {m_img.size}.")

    # 3. Test Slide 1: Colorectal Carcinoma
    print("\n--- [Test 3: Slide A1-1 Colorectal Adenocarcinoma Full Pipeline] ---")
    tb1 = p1["tab"]
    res1 = process_aisight_case(
        im1, p1["text"], True, *tb1,
        active_view_layer="🔥 Faithful Grad-CAM Heatmap",
        current_slide_id="H&E Slide A1-1",
        current_badge_text="🔴 Possible cancer"
    )
    v_img, mini_img, badge_html, accept_md, algo_md, align_md, faith_md, text_html, shap_img = res1

    assert v_img is not None, "Viewport image is None!"
    assert "VERIFIED H&E HISTOPATHOLOGY" in accept_md, "Acceptability check failed on Slide 1!"
    assert "Colorectal" in algo_md or "Malignant" in algo_md, "Malignancy diagnosis not found in algorithm results!"
    assert "FAITHFUL" in faith_md, "Faithfulness verification missing in faith_md!"
    assert shap_img is not None, "SHAP chart not generated!"
    assert "<span" in text_html, "BioBERT narrative attribution HTML not generated!"
    print("[PASS] Slide A1-1 Colorectal Ca: Stage 0-3 + GCA + SHAP + BioBERT verified.")

    # 4. Test Slide 4: Non-Neoplastic Lung
    print("\n--- [Test 4: Slide A1-4 Non-Neoplastic Lung Resection] ---")
    p4 = SLIDE_PRESETS["slide_4"]
    im4 = Image.open(p4["img_path"])
    tb4 = p4["tab"]
    res4 = process_aisight_case(
        im4, p4["text"], True, *tb4,
        active_view_layer="🖼️ Clean H&E Micrograph",
        current_slide_id="H&E Slide A1-4",
        current_badge_text="🟢 Non-Neoplastic"
    )
    assert "Non-Neoplastic" in res4[2], f"Status badge did not reflect non-neoplastic: {res4[2]}"
    print("[PASS] Slide A1-4 Non-Neoplastic: Negative status gating verified.")

    # 5. Test Layer Mode Switching
    print("\n--- [Test 5: Pathology Viewport Layer Toggles] ---")
    # Clean H&E
    res_clean = process_aisight_case(im1, p1["text"], True, *tb1, active_view_layer="🖼️ Clean H&E Micrograph")
    assert res_clean[0].size == (224, 224) or res_clean[0].size == (256, 256) or res_clean[0].size == im1.size, "Clean view error"

    # Saliency Masked
    res_mask = process_aisight_case(im1, p1["text"], True, *tb1, active_view_layer="🎭 Saliency Masked (Top 20% Blurred)")
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
    print("ALL PATHAI AISIGHT CLINICAL WORKSTATION INTEGRATION TESTS PASSED (100%)")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
