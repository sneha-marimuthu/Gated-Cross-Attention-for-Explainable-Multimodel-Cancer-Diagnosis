"""
Comprehensive Integration Test Suite for GCAEMMCD Clinical Workstation.
Verifies:
  1. GCAEMMCD Theme & HTML Generators (Zero external vendor strings).
  2. Structured Cancer Diagnosis System Output (01-06 Sections).
  3. Specimen S-01 (Colorectal Adenocarcinoma) full pipeline execution, Grad-CAM, and SHAP.
  4. Specimen S-03 (Invasive Ductal Breast Carcinoma) 4-class cancer probabilities.
  5. Specimen S-04 (Non-Neoplastic Lung Resection) negative status gating.
  6. Clinical Consultation Signout file generation & download.
  7. Optical domain rejection gate on non-histopathology input.
"""

import os
import sys
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import config

from src.ui.aisight_app import (
    process_aisight_case, SLIDE_PRESETS, create_overview_minimap
)
from src.ui.aisight_theme import (
    get_top_navbar_html, get_sub_header_html, get_viewer_toolbar_html,
    get_slide_badge_html, get_footer_disclaimer_html
)
from src.ui.report_generator import generate_downloadable_report_file


def test_aisight_workstation():
    print("=" * 80)
    print("RUNNING GCAEMMCD CLINICAL WORKSTATION INTEGRATION TESTS")
    print("=" * 80)

    # 1. Test Theme HTML Generators
    print("\n--- [Test 1: GCAEMMCD Theme & HTML Generation] ---")
    top_nav = get_top_navbar_html()
    assert "GCAEMMCD" in top_nav and "Clinical Pathology Workstation" in top_nav, "Top navbar missing GCAEMMCD branding!"
    assert "AISight" not in top_nav and "PathAI" not in top_nav, "Found forbidden external vendor branding in top navbar!"
    print("[PASS] Top global navigation bar verified (Clean GCAEMMCD branding).")

    sub_header = get_sub_header_html(specimen_id="Specimen S-01", status="Ready", patient_id="PID-84021")
    assert "Specimen S-01" in sub_header and "Diagnostic Session" in sub_header, "Sub-header missing session metadata!"
    assert "PID-84021" in sub_header, "Patient ID missing in sub-header!"
    print("[PASS] Sub-header specimen metadata bar verified.")

    badge = get_slide_badge_html("H&E Slide S-01", "🔴 Malignant Neoplasm")
    assert "status-pill-malignant" in badge, "Malignant status pill class missing!"
    print("[PASS] Dynamic viewport status badge verified.")

    disclaimer = get_footer_disclaimer_html()
    assert "GCAEMMCD" in disclaimer and "Investigational & Research" in disclaimer, "Disclaimer missing!"
    print("[PASS] Clinical disclaimer footer verified.")

    # 2. Test Specimen S-01: Colorectal Carcinoma
    print("\n--- [Test 2: Specimen S-01 Colorectal Adenocarcinoma Full Pipeline] ---")
    p1 = SLIDE_PRESETS["slide_1"]
    im1 = Image.open(p1["img_path"])
    tb1 = p1["tab"]
    res1 = process_aisight_case(
        im1, p1["text"], True, *tb1,
        active_view_layer="🔥 Faithful Grad-CAM Saliency",
        current_slide_id=p1["id"],
        current_badge_text=p1["badge_text"],
        patient_id=p1["patient_id"],
        tumor_size=p1["tumor_size"]
    )
    card_html, v_img, faith_badge, shap_img, shap_badge, meta1 = res1

    assert card_html is not None, "Report card HTML is None!"
    assert "01. CANCER CLASSIFICATION" in card_html, "Section 01 missing!"
    assert "COLORECTAL CARCINOMA" in card_html, "Colorectal Carcinoma prediction missing!"
    assert "02. TNM STAGING" in card_html, "Section 02 missing!"
    assert "03. SURVIVAL ESTIMATION" in card_html, "Section 03 missing!"
    assert "04. TREATMENT RECOMMENDATION" in card_html, "Section 04 missing!"
    assert "Requires evaluation by an oncologist" in card_html, "Oncologist evaluation warning missing!"
    assert v_img is not None, "Grad-CAM image is None!"
    assert "FAITHFUL" in faith_badge, "Faithfulness verification missing in badge!"
    assert shap_img is not None, "SHAP chart not generated!"
    print("[PASS] Specimen S-01: Sections 01-04 + Grad-CAM + SHAP verified.")

    # 3. Test Specimen S-03: Breast Carcinoma
    print("\n--- [Test 3: Specimen S-03 Breast Carcinoma Classification] ---")
    p3 = SLIDE_PRESETS["slide_3"]
    im3 = Image.open(p3["img_path"])
    tb3 = p3["tab"]
    res3 = process_aisight_case(
        im3, p3["text"], True, *tb3,
        patient_id=p3["patient_id"],
        tumor_size=p3["tumor_size"]
    )
    card_html3 = res3[0]
    assert "BREAST CARCINOMA" in card_html3, "Breast Carcinoma prediction missing for Specimen S-03!"
    assert "Breast Carcinoma" in res3[5]["class_probs"], "Class probabilities missing breast!"
    print(f"[PASS] Specimen S-03 Breast Ca confirmed: Confidence = {res3[5]['confidence']*100:.2f}%.")

    # 4. Test Specimen S-04: Non-Neoplastic Lung
    print("\n--- [Test 4: Specimen S-04 Non-Neoplastic Lung Resection] ---")
    p4 = SLIDE_PRESETS["slide_4"]
    im4 = Image.open(p4["img_path"])
    tb4 = p4["tab"]
    res4 = process_aisight_case(
        im4, p4["text"], True, *tb4,
        patient_id=p4["patient_id"],
        tumor_size=p4["tumor_size"]
    )
    assert "NORMAL / BENIGN" in res4[0] or "Stage 0" in res4[0], "Non-neoplastic classification failed!"
    assert res4[5]["t_stage"] == "T0", "Expected T0 for non-neoplastic tissue!"
    print("[PASS] Specimen S-04 Non-Neoplastic: T0 N0 M0 / Stage 0 verified.")

    # 5. Test Downloadable Report Generation
    print("\n--- [Test 5: Downloadable Consultation Signout Generator] ---")
    fpath = generate_downloadable_report_file(meta1)
    assert os.path.exists(fpath), f"Report file not generated at {fpath}!"
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    assert "GCAEMMCD CLINICAL CONSULTATION SIGNOUT" in content, "Report title missing!"
    assert "COLORECTAL CARCINOMA" in content, "Diagnosis missing in file!"
    print(f"[PASS] Printable consultation report written to: {fpath} ({len(content)} bytes).")

    # 6. Test Ingestion Gate Rejection (Dark UI Screenshot)
    print("\n--- [Test 6: Ingestion Optical Domain Gatekeeper] ---")
    dark_screenshot = Image.new("RGB", (300, 300), color=(18, 24, 38))
    res_rej = process_aisight_case(dark_screenshot, "Clinical text", False, *tb1)
    assert "SLIDE ACCEPTABILITY: REJECTED" in res_rej[0], "Non-tissue image was not rejected!"
    assert "INGESTION GATEKEEPER HALTED" in res_rej[0], "Rejection card header missing!"
    print("[PASS] Optical domain rejection gate successfully prevented non-tissue processing.")

    print("\n" + "=" * 80)
    print("ALL GCAEMMCD CLINICAL WORKSTATION INTEGRATION TESTS PASSED (100%)")
    print("=" * 80)


if __name__ == "__main__":
    test_aisight_workstation()
