"""
Comprehensive Level 2 Real-World & Edge-Case Verification Suite.
Tests authentic histopathology slides, out-of-domain / screenshot / blurred / blank images,
unseen clinical narratives (including negated history), overlapping clinical biomarkers,
missing/invalid modality states, cross-modal discordance, and Grad-CAM gating.
Saves generated Grad-CAM overlays directly into `gradcam_outputs/`.
"""

import json
import os
import sys
import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import config
from src.data.multimodal_dataset import generate_biomarker_vector
from src.ui.level2_app import run_multimodal_pipeline
from scripts.save_gradcam_images import save_gradcam_array


def make_tab_args(vec=None, mask_ihc=False):
    if vec is None:
        return [
            None, "Unknown / Not Tested", 0.0, "Unknown / Not Tested",
            None, None, None, None,
            "Unknown / Not Tested", "Unknown / Not Tested", "Unknown / Not Tested",
            "Unknown / Not Tested", "Unknown / Not Tested", None, None, None,
        ]
    return [
        round(float(vec[0]), 1),
        "Female" if vec[1] >= 0.5 else "Male",
        round(float(vec[2]), 1),
        str(int(round(vec[3]))),
        round(float(vec[4]), 1),
        round(float(vec[5]), 1),
        round(float(vec[6]), 1),
        "Not Tested / Unknown" if mask_ihc else round(float(vec[7]), 1),
        "Unknown / Not Tested" if mask_ihc else ("Positive (+)" if vec[8] >= 0.5 else "Negative (-)"),
        "Unknown / Not Tested" if mask_ihc else ("Positive (+)" if vec[9] >= 0.5 else "Negative (-)"),
        "Unknown / Not Tested" if mask_ihc else ("Positive (+)" if vec[10] >= 0.5 else "Negative (-)"),
        "Unknown / Not Tested" if mask_ihc else ("Mutant" if vec[11] >= 0.5 else "Wild-type"),
        "Unknown / Not Tested" if mask_ihc else ("Mutant" if vec[12] >= 0.5 else "Wild-type"),
        round(float(vec[13]), 1),
        round(float(vec[14]), 1),
        round(float(vec[15]), 1),
    ]


def build_test_cases():
    # Held-out test split samples (unseen during training)
    img_breast = Image.open(os.path.join(config.IMAGE_DIR, "breast", "breast_00.png")).convert("RGB")
    img_lung = Image.open(os.path.join(config.IMAGE_DIR, "lung", "lung_06.png")).convert("RGB")
    img_colorectal = Image.open(os.path.join(config.IMAGE_DIR, "colorectal", "colorectal_03.png")).convert("RGB")
    img_benign = Image.open(os.path.join(config.IMAGE_DIR, "benign", "benign_breast_00.png")).convert("RGB")
    img_normal_lung = Image.open(os.path.join(config.IMAGE_DIR, "normal", "normal_lung_00.png")).convert("RGB")

    # Vectors
    v_bm = generate_biomarker_vector("breast_malignant", seed=42)
    v_bm[0], v_bm[1], v_bm[2], v_bm[3] = 56.0, 1.0, 0.0, 1.0

    v_lm = generate_biomarker_vector("lung_malignant", seed=42)
    v_lm[0], v_lm[1], v_lm[2], v_lm[3] = 66.0, 0.0, 32.0, 1.0

    v_cm = generate_biomarker_vector("colorectal_malignant", seed=42)
    v_cm[0], v_cm[1], v_cm[2], v_cm[3] = 64.0, 0.0, 12.0, 1.0

    v_bb = generate_biomarker_vector("breast_benign", seed=42)
    v_bb[0], v_bb[1], v_bb[2], v_bb[3] = 35.0, 1.0, 0.0, 0.0

    v_ln = generate_biomarker_vector("lung_normal", seed=42)
    v_ln[0], v_ln[1], v_ln[2], v_ln[3] = 48.0, 0.0, 0.0, 0.0

    v_imp = v_lm.copy()
    v_imp[0], v_imp[2] = 20.0, 60.0  # Biological impossibility: 20yo with 60 pack-years

    v_unit = v_cm.copy()
    v_unit[0], v_unit[13], v_unit[15] = 0.0, 125.0, 250000.0  # Age 0, g/L, /mm^3

    ui_screenshot_path = "/home/frappe/.gemini/antigravity/brain/0ed6296f-b1fc-4214-a2f2-24348269df2b/.user_uploaded/media_1794301479562_df9b4da5.jpg"
    ui_screenshot = Image.open(ui_screenshot_path).convert("RGB") if os.path.exists(ui_screenshot_path) else Image.new("RGB", (224, 224), (15, 23, 42))

    cases = [
        # 1-5: Valid Authentic Histopathology Cases (Tri-modal & Unimodal)
        {
            "name": "01_Breast_Malignant_TriModal",
            "img": img_breast,
            "text": "56-year-old female, lifelong non-smoker, irregular BI-RADS 4C pectoral mass; needle core shows malignant mammary ducts.",
            "include_tab": True,
            "tab_args": make_tab_args(v_bm),
            "expect_gradcam": True,
        },
        {
            "name": "02_Lung_Malignant_TriModal",
            "img": img_lung,
            "text": "66yo male (32 pack-years) with hemoptysis and hilar opacity; core biopsy establishes invasive bronchogenic adenocarcinoma.",
            "include_tab": True,
            "tab_args": make_tab_args(v_lm),
            "expect_gradcam": True,
        },
        {
            "name": "03_Colorectal_Malignant_NegatedPriorHx",
            "img": img_colorectal,
            "text": "64yo male, no prior history of breast cancer, colonoscopy shows invasive sigmoid colon adenocarcinoma with dirty necrosis.",
            "include_tab": True,
            "tab_args": make_tab_args(v_cm),
            "expect_gradcam": True,
        },
        {
            "name": "04_Breast_Benign_TriModal",
            "img": img_benign,
            "text": "35yo female, non-smoker, smooth mobile breast lump; excisional histology confirms benign fibroadenoma negative for carcinoma.",
            "include_tab": True,
            "tab_args": make_tab_args(v_bb),
            "expect_gradcam": True,
        },
        {
            "name": "05_Normal_Lung_TriModal",
            "img": img_normal_lung,
            "text": "48yo male non-smoker, bronchoscopic biopsy shows intact alveolar walls and normal parenchyma, zero evidence of carcinoma.",
            "include_tab": True,
            "tab_args": make_tab_args(v_ln),
            "expect_gradcam": True,
        },
        {
            "name": "06_ImageOnly_Colorectal",
            "img": img_colorectal,
            "text": "   ",
            "include_tab": False,
            "tab_args": make_tab_args(v_bm),  # must be ignored!
            "expect_gradcam": True,
        },
        # 7-13: Domain Validator & OOD Rejection Cases (MUST halt & suppress Grad-CAM)
        {
            "name": "07_Reject_PowerShell_Screenshot",
            "img": Image.open(os.path.join(config.DATA_DIR, "test_cases", "powershell_terminal.png")).convert("RGB"),
            "text": "",
            "include_tab": False,
            "tab_args": make_tab_args(None),
            "expect_gradcam": False,
        },
        {
            "name": "08_Reject_General_Landscape_Photo",
            "img": Image.open(os.path.join(config.DATA_DIR, "test_cases", "general_photo.png")).convert("RGB"),
            "text": "",
            "include_tab": False,
            "tab_args": make_tab_args(None),
            "expect_gradcam": False,
        },
        {
            "name": "09_Reject_DarkMode_UI_Screenshot",
            "img": ui_screenshot,
            "text": "",
            "include_tab": False,
            "tab_args": make_tab_args(None),
            "expect_gradcam": False,
        },
        {
            "name": "10_Reject_Blank_White_Slide",
            "img": Image.open(os.path.join(config.DATA_DIR, "test_cases", "blank_slide.png")).convert("RGB"),
            "text": "",
            "include_tab": False,
            "tab_args": make_tab_args(None),
            "expect_gradcam": False,
        },
        {
            "name": "11_Reject_Dark_Slate_Terminal_Block",
            "img": Image.new("RGB", (224, 224), (24, 28, 36)),
            "text": "",
            "include_tab": False,
            "tab_args": make_tab_args(None),
            "expect_gradcam": False,
        },
        {
            "name": "12_Reject_Blurred_Histology",
            "img": img_breast.filter(ImageFilter.GaussianBlur(radius=6)),
            "text": "",
            "include_tab": False,
            "tab_args": make_tab_args(None),
            "expect_gradcam": False,
        },
        {
            "name": "13_Reject_LowRes_24x24",
            "img": img_breast.resize((24, 24)),
            "text": "",
            "include_tab": False,
            "tab_args": make_tab_args(None),
            "expect_gradcam": False,
        },
        # 14-18: Clinical Text, Tabular & Concordance Edge Cases
        {
            "name": "14_TextOnly_Negated_Benign",
            "img": None,
            "text": "42yo female non-smoker. Breast screening biopsy fibroadenoma; negative for breast carcinoma and no evidence of malignancy.",
            "include_tab": False,
            "tab_args": make_tab_args(None),
            "expect_gradcam": False,
        },
        {
            "name": "15_TabularOnly_PreBiopsy_Masked_IHC",
            "img": None,
            "text": "",
            "include_tab": True,
            "tab_args": make_tab_args(v_cm, mask_ihc=True),
            "expect_gradcam": False,
        },
        {
            "name": "16_DualModal_Unsupported_ImgPlusText",
            "img": img_breast,
            "text": "56-year-old female with irregular mammary lesion showing invasive ductal features.",
            "include_tab": False,
            "tab_args": make_tab_args(None),
            "expect_gradcam": True,
        },
        {
            "name": "17_CrossModal_Identity_And_Organ_Mismatch",
            "img": img_colorectal,
            "text": "58-year-old female non-smoker with invasive breast carcinoma.",
            "include_tab": True,
            "tab_args": make_tab_args(v_lm),
            "expect_gradcam": True,
        },
        {
            "name": "18_Reject_Tabular_Biological_Impossibility",
            "img": None,
            "text": "",
            "include_tab": True,
            "tab_args": make_tab_args(v_imp),
            "expect_gradcam": False,
        },
        {
            "name": "19_Reject_Tabular_Unit_And_Zero_Error",
            "img": None,
            "text": "",
            "include_tab": True,
            "tab_args": make_tab_args(v_unit),
            "expect_gradcam": False,
        },
        {
            "name": "20_Reject_OffTopic_Text",
            "img": None,
            "text": "The weather outside is sunny today and coffee tastes great.",
            "include_tab": False,
            "tab_args": make_tab_args(None),
            "expect_gradcam": False,
        },
    ]
    return cases


def main():
    out_dir = os.path.join(PROJECT_ROOT, "gradcam_outputs")
    os.makedirs(out_dir, exist_ok=True)
    # Remove any stale files in gradcam_outputs before fresh run
    for old_f in os.listdir(out_dir):
        if old_f.endswith(".png"):
            os.remove(os.path.join(out_dir, old_f))

    cases = build_test_cases()
    results = []

    print("\n" + "=" * 100)
    print(f"{'CASE':<44} | {'VAL/EXEC':<14} | {'GRAD-CAM':<10} | {'VERDICT / HEADLINE'}")
    print("=" * 100)

    for c in cases:
        badges, diag, p_stat, p_site, ablat, gates, cam_img, cam_cap = run_multimodal_pipeline(
            c["img"], c["text"], c["include_tab"], *c["tab_args"]
        )
        gradcam_generated = cam_img is not None
        gradcam_path = None
        if gradcam_generated:
            gradcam_path = save_gradcam_array(cam_img, c["name"], out_dir)

        halted = "EXECUTION HALTED" in diag
        discordant = "CROSS-MODAL DISCORDANCE" in diag
        unsupported = "Insufficient Modality Support" in diag
        state_lbl = "HALTED" if halted else ("DISCORDANT" if discordant else ("DUAL_UNSUPP" if unsupported else "EXECUTED"))

        # First meaningful content line of diag card
        headline = ""
        for line in diag.splitlines():
            s = line.strip()
            if s and not s.startswith("###"):
                headline = s[:70]
                break

        cam_ok = "MATCH" if gradcam_generated == c["expect_gradcam"] else "MISMATCH!"
        print(f"{c['name']:<44} | {state_lbl:<14} | {str(gradcam_generated):<5}({cam_ok}) | {headline}")

        results.append({
            "name": c["name"],
            "execution_state": state_lbl,
            "gradcam_generated": gradcam_generated,
            "expect_gradcam": c["expect_gradcam"],
            "gradcam_path": gradcam_path,
            "status_probs": p_stat,
            "site_probs": p_site,
            "modality_badge": badges,
            "diagnostic_card": diag,
            "ablation_table": ablat,
            "gates_and_attribution": gates,
            "gradcam_caption": cam_cap,
        })

    out_dir_res = os.path.join(PROJECT_ROOT, "tests", "results")
    os.makedirs(out_dir_res, exist_ok=True)
    out_json = os.path.join(out_dir_res, "extended_test_results_summary.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 100)
    print(f"[+] Saved extended test summary -> {out_json}")
    print(f"[+] Saved {sum(1 for r in results if r['gradcam_path'])} Grad-CAM PNGs -> {out_dir}")


def test_extended_cases():
    main()


if __name__ == "__main__":
    main()
