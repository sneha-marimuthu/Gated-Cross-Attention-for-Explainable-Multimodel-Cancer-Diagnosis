"""
Professional External Validation Benchmark for GCAEMMCD (Level 1 & Level 2).
Tests 12 genuine external specimens sourced from Wikimedia Commons (CC0 / CC-BY / Public Domain)
recorded in `data/external_validation/provenance_manifest.json`.
Evaluates:
  1. Stage 0 Domain Validation (`validate_histopathology_domain`)
  2. Stage 1 Phikon Calibrated Organ & OOD Gatekeeper (`max_cosine_distance = 0.50`)
  3. Level 2 Image-Only Generalization (pure external visual performance)
  4. Level 2 Tri-Modal GCA Generalization (with independent clinical vignettes & biomarkers)
  5. Level 1 Hierarchical Compatibility
"""

import json
import os
import sys
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.data.domain_validator import validate_histopathology_domain
from src.data.image_transforms import preprocess_multicrop
from src.data.multimodal_dataset import generate_biomarker_vector
from src.ui.level1_app import hierarchical_inference_pipeline
from src.ui.level2_app import run_multimodal_pipeline, organ_extractor, organ_head
from save_gradcam_images import save_gradcam_array


def make_tab_args(vec=None):
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
        round(float(vec[7]), 1),
        "Positive (+)" if vec[8] >= 0.5 else "Negative (-)",
        "Positive (+)" if vec[9] >= 0.5 else "Negative (-)",
        "Positive (+)" if vec[10] >= 0.5 else "Negative (-)",
        "Mutant" if vec[11] >= 0.5 else "Wild-type",
        "Mutant" if vec[12] >= 0.5 else "Wild-type",
        round(float(vec[13]), 1),
        round(float(vec[14]), 1),
        round(float(vec[15]), 1),
    ]


# External clinical narratives & biomarker vignettes (written independently of training templates)
EXTERNAL_CLINICAL_PAIRS = {
    "ext_breast_malignant.jpg": (
        "52-year-old woman referred after diagnostic mammogram showed clustered pleomorphic calcifications; "
        "excision demonstrates cords of hyperchromatic neoplastic cells infiltrating mammary adipocytes.",
        "breast_malignant",
        (52.0, 1.0, 0.0, 1.0),
    ),
    "ext_breast_benign.jpg": (
        "28-year-old woman with rubbery well-circumscribed breast mass unchanged over 12 months; "
        "histology shows stromal proliferation compressing slit-like benign ducts without atypia.",
        "breast_benign",
        (28.0, 1.0, 0.0, 0.0),
    ),
    "ext_lung_adenocarcinoma.png": (
        "67-year-old male former smoker (28 pack-years) with peripheral right upper lobe ground-glass nodule; "
        "resection confirms invasive pulmonary adenocarcinoma with predominant acinar architecture.",
        "lung_malignant",
        (67.0, 0.0, 28.0, 1.0),
    ),
    "ext_lung_squamous_ca.jpg": (
        "71-year-old male smoker (45 pack-years) presenting with central hilar mass and hemoptysis; "
        "endobronchial biopsy demonstrates poorly differentiated non-small cell lung carcinoma.",
        "lung_malignant",
        (71.0, 0.0, 45.0, 2.0),
    ),
    "ext_lung_emphysema_non_ca.jpg": (
        "61-year-old male smoker (25 pack-years) with COPD evaluated after wedge resection for bullous emphysema; "
        "alveolar septa show airspace enlargement and congestion without evidence of neoplasia.",
        "lung_normal",
        (61.0, 0.0, 25.0, 1.0),
    ),
    "ext_colon_malignant.jpg": (
        "69-year-old man with occult GI bleeding and microcytic anemia; colonoscopy biopsy of fungating cecal mass "
        "reveals complex malignant colonic glands invading submucosa.",
        "colorectal_malignant",
        (69.0, 0.0, 8.0, 1.0),
    ),
    "ext_colon_normal.jpg": (
        "54-year-old woman undergoing routine average-risk screening colonoscopy; "
        "biopsies demonstrate evenly spaced tubular crypts rich in goblet cells, unremarkable mucosa.",
        "colorectal_normal",
        (54.0, 1.0, 0.0, 0.0),
    ),
}


def extract_headline(diag_md):
    for line in diag_md.splitlines():
        s = line.strip()
        if s and not s.startswith("###") and not s.startswith("- "):
            return s
    return diag_md[:100].replace("\n", " ")


def main():
    manifest_path = os.path.join(PROJECT_ROOT, "data", "external_validation", "provenance_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    cam_dir = os.path.join(PROJECT_ROOT, "data", "external_validation", "gradcam_outputs")
    os.makedirs(cam_dir, exist_ok=True)
    empty_tab = make_tab_args(None)

    report = []
    print("\n" + "=" * 120)
    print(f"{'EXTERNAL SPECIMEN':<32} | {'EXPECTED':<22} | {'DOM':<5} | {'PHIKON (d_cos)':<22} | {'L2 IMAGE-ONLY':<34} | {'CAM'}")
    print("=" * 120)

    for idx, item in enumerate(manifest):
        fname = item["filename"]
        img = Image.open(item["file"]).convert("RGB")

        # 1. Domain & Phikon raw diagnostics
        dom_ok, dom_type, dom_reason, dom_metrics = validate_histopathology_domain(img)
        with torch.no_grad():
            feats = organ_extractor(preprocess_multicrop(img).to(config.DEVICE))
            ph_pred, ph_conf, _, ph_ood, ph_unc, d_cos = organ_head.predict_with_calibration(feats)

        # 2. Level 2 Image-Only pipeline
        b_im, d_im, ps_im, po_im, ab_im, _, cam_im, cap_im = run_multimodal_pipeline(
            img, "", False, *empty_tab
        )
        im_verdict = (
            f"REJECTED ({dom_type})"
            if ("EXECUTION HALTED" in d_im and not dom_ok)
            else ("OOD_REJECTED" if "EXECUTION HALTED" in d_im else extract_headline(d_im))
        )

        # 3. Level 1 pipeline check
        l1_org, _, l1_st, _, l1_sub, _, _, l1_cam, l1_cap = hierarchical_inference_pipeline(img)
        l1_summary = extract_headline(l1_org) + " / " + extract_headline(l1_st)

        # 4. Level 2 Tri-Modal pipeline (for in-scope histopathology cases)
        tri_summary = None
        tri_cam_path = None
        if fname in EXTERNAL_CLINICAL_PAIRS:
            txt, cat_key, (age, sex, smoke, ecog) = EXTERNAL_CLINICAL_PAIRS[fname]
            vec = generate_biomarker_vector(cat_key, seed=100 + idx)
            vec[0], vec[1], vec[2], vec[3] = age, sex, smoke, ecog
            _, d_tri, ps_tri, po_tri, ab_tri, g_tri, cam_tri, cap_tri = run_multimodal_pipeline(
                img, txt, True, *make_tab_args(vec)
            )
            tri_summary = {
                "headline": extract_headline(d_tri),
                "discordant": "CROSS-MODAL DISCORDANCE" in d_tri,
                "halted": "EXECUTION HALTED" in d_tri,
                "status_probs": ps_tri,
                "site_probs": po_tri,
                "gradcam_generated": cam_tri is not None,
            }
            if cam_tri is not None:
                tri_cam_path = save_gradcam_array(cam_tri, f"tri_{fname}", cam_dir)

        im_cam_path = save_gradcam_array(cam_im, f"img_{fname}", cam_dir) if cam_im is not None else None
        ph_str = f"{'OOD' if ph_ood else ph_pred} ({d_cos:.3f})"
        exp_str = f"{item['expected_organ']}:{item['expected_status']}"[:22]
        print(f"{fname:<32} | {exp_str:<22} | {str(dom_ok):<5} | {ph_str:<22} | {im_verdict[:34]:<34} | {cam_im is not None}")

        report.append({
            "filename": fname,
            "source_title": item["source_title"],
            "source_url": item["source_url"],
            "license": item["license"],
            "expected_organ": item["expected_organ"],
            "expected_status": item["expected_status"],
            "in_supported_scope": item["in_supported_scope"],
            "domain_ok": dom_ok,
            "domain_type": dom_type,
            "domain_metrics": dom_metrics,
            "phikon_pred": ph_pred,
            "phikon_d_cos": round(float(d_cos), 4),
            "phikon_ood": bool(ph_ood),
            "level2_image_only": {
                "verdict": im_verdict,
                "status_probs": ps_im,
                "site_probs": po_im,
                "gradcam_generated": cam_im is not None,
                "gradcam_path": im_cam_path,
            },
            "level2_trimodal": tri_summary,
            "level1_summary": l1_summary,
        })

    out_path = os.path.join(PROJECT_ROOT, "data", "external_validation", "external_validation_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("=" * 120)
    print(f"[+] Full external benchmark saved -> {out_path}")


if __name__ == "__main__":
    main()
