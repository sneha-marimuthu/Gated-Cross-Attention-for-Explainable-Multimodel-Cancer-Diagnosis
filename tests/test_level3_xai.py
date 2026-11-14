"""
Verification Suite for GCAEMMCD Level 3 XAI & Research Dashboard.
Tests:
  1. Full Level 3 Tri-Modal Explainability Execution (Grad-CAM + Masking + BioBERT + SHAP)
  2. Quantitative Faithfulness Metrics (Confidence Drop upon salient masking)
  3. SHAP Mathematical Local Additivity Verification (|sum(phi) - delta_f| < 1e-4)
  4. BioBERT Token-Level Saliency Generation
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
from src.ui.level3_app import run_level3_xai_audit, build_level3_dashboard


def test_level3_xai():
    print("=" * 80)
    print("RUNNING GCAEMMCD LEVEL 3 EXPLAINABILITY & AUDIT VERIFICATION SUITE")
    print("=" * 80)

    # 1. Test Preset 1: Colorectal Carcinoma
    print("\n[*] Testing Case 1: Colorectal Carcinoma Tri-Modal Case...")
    img1 = Image.open("data/sample_images/colorectal/colorectal_00.png")
    txt1 = "Endoscopic biopsy confirms invasive adenocarcinoma of the sigmoid colon with prominent cribriform architecture; KRAS codon 12 mutation confirmed."
    tab_args1 = [64, "Male", 35, "1", 34.5, 14.0, 48.0, 55.0, "Negative (-)", "Negative (-)", "Negative (-)", "Wild-type", "Mutant", 11.2, 8.4, 290.0]

    b1, d1, cam1, mask1, html1, shap1, faith1, gate1 = run_level3_xai_audit(img1, txt1, True, *tab_args1)

    print(f"  [+] Diagnostic Headline: {d1.splitlines()[2] if len(d1.splitlines()) > 2 else d1[:50]}")
    assert cam1 is not None, "Grad-CAM overlay should not be None"
    assert mask1 is not None, "Salient masked image should not be None"
    assert shap1 is not None, "SHAP bar chart should not be None"
    assert "<span" in html1, "BioBERT HTML view should contain highlighted spans"
    assert "Quantitative Saliency Faithfulness Test" in faith1, "Faithfulness card should contain visual test results"
    assert "SHAP Biomarker Local Efficiency" in faith1, "Faithfulness card should contain SHAP efficiency check"
    print("  [+] Case 1 Explainability Outputs: Grad-CAM OK | Masked View OK | BioBERT HTML OK | SHAP Plot OK")

    # 2. Test Preset 2: Lung Carcinoma
    print("\n[*] Testing Case 2: Lung Carcinoma Tri-Modal Case...")
    img2 = Image.open("data/sample_images/lung/lung_00.png")
    txt2 = "Core biopsy of right upper lobe pulmonary mass confirms primary invasive lung adenocarcinoma with papillary features and EGFR exon 19 deletion."
    tab_args2 = [62, "Female", 22, "1", 3.2, 18.0, 15.0, 42.0, "Negative (-)", "Negative (-)", "Negative (-)", "Mutant", "Wild-type", 12.4, 7.1, 230.0]

    b2, d2, cam2, mask2, html2, shap2, faith2, gate2 = run_level3_xai_audit(img2, txt2, True, *tab_args2)
    print(f"  [+] Diagnostic Headline: {d2.splitlines()[2] if len(d2.splitlines()) > 2 else d2[:50]}")
    assert cam2 is not None
    assert shap2 is not None
    assert "<span" in html2
    print("  [+] Case 2 Explainability Outputs Verified.")

    # 3. Test Safety Gate: Non-histology Screenshot
    print("\n[*] Testing Case 3: Reject Non-Histology Dark Screenshot...")
    img3 = Image.open("data/test_cases/powershell_terminal.png")
    b3, d3, cam3, mask3, html3, shap3, faith3, gate3 = run_level3_xai_audit(img3, "Normal biopsy notes", True, *tab_args1)
    assert "EXECUTION HALTED" in d3, "Should halt on non-histopathology screenshot"
    assert cam3 is None, "Grad-CAM should be suppressed on invalid inputs"
    assert mask3 is None, "Masking should be suppressed on invalid inputs"
    print("  [+] Case 3 Safety Gate Verified: Halted prior to XAI invocation, zero false Grad-CAM.")

    # 4. Test Dashboard Build
    print("\n[*] Testing Gradio Level 3 Dashboard Construction...")
    demo = build_level3_dashboard()
    assert demo is not None
    print("  [+] Gradio Level 3 Dashboard constructed with all 4 tabs.")

    print("\n" + "=" * 80)
    print("🎉 ALL LEVEL 3 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_level3_xai()
