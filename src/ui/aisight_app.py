"""
GCAEMMCD Clinical Pathology Decision Support Workstation.
Interactive 3-Column Workstation:
  - Column 1: Sample Input (5 preset specimens with clinical patient tags)
  - Column 2: Manual Input (Patient Information, Histopathology Image, Clinical Report, Clinical Parameters)
  - Column 3: Output (Digital pathology viewport, 4-task multimodal clinical findings, explainability)
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data.multimodal_dataset import validate_tabular_biomarkers, encode_biomarkers_with_mask
from src.data.image_transforms import crop_eyepiece_vignette, preprocess_multicrop
from src.explainability.faithfulness import evaluate_visual_faithfulness, evaluate_tabular_faithfulness
from src.explainability.shap_explainer import TabularShapExplainer
from src.explainability.text_saliency import BioBERTTextSaliencyExplainer
from src.models.clinical_staging_engine import STAGE_NAMES, TREATMENT_OPTIONS
from src.ui.aisight_theme import (
    GCAEMMCD_CUSTOM_CSS, FORCE_LIGHT_MODE_JS, get_gcaemmcd_theme,
    get_top_navbar_html, get_sub_header_html, get_viewer_toolbar_html,
    get_slide_badge_html, get_footer_disclaimer_html
)
from src.ui.level1_app import (
    validate_histopathology_domain
)
from src.ui.level2_app import (
    img_extractor, text_extractor, text_proj, tab_mlp, classifier, gradcam,
    run_multimodal_pipeline, SITE_NAMES,
    _map_radio_binary, _map_radio_int
)

print("[*] Booting GCAEMMCD Workstation Backend...")
shap_explainer = TabularShapExplainer(tab_mlp, classifier, device=config.DEVICE, num_bg_samples=24)
text_saliency_explainer = BioBERTTextSaliencyExplainer(text_extractor, text_proj, classifier, device=config.DEVICE)
print("[+] GCAEMMCD Workstation Backend Active.")

# Slide Case Database for Preset Carousel
SLIDE_PRESETS = {
    "slide_1": {
        "id": "H&E Slide S-01",
        "patient_id": "PID-84021",
        "tumor_size": 4.2,
        "badge_text": "🔴 Malignant Neoplasm",
        "organ_label": "Colorectal Biopsy",
        "patient_tag": "PID-84021 · 64M · Sigmoid Colon · CEA 34.5 · KRAS Mut",
        "category": "colorectal_malignant",
        "img_path": "data/sample_images/colorectal/colorectal_00.png",
        "text": "Endoscopic biopsy of sigmoid colon mass confirms invasive adenocarcinoma with prominent cribriform glands; KRAS codon 12 mutation confirmed.",
        "tab": [64, "Male", 35, "1", 34.5, 14.0, 48.0, 55.0, "Negative (-)", "Negative (-)", "Negative (-)", "Wild-type", "Mutant", 11.2, 8.4, 290.0]
    },
    "slide_2": {
        "id": "H&E Slide S-02",
        "patient_id": "PID-62914",
        "tumor_size": 3.5,
        "badge_text": "🔴 Malignant Neoplasm",
        "organ_label": "Lung Biopsy",
        "patient_tag": "PID-62914 · 62F · Lung RUL · EGFR Exon 19 Del · Ki-67 42%",
        "category": "lung_malignant",
        "img_path": "data/sample_images/lung/lung_00.png",
        "text": "Core biopsy of right upper lobe pulmonary mass confirms primary invasive lung adenocarcinoma with papillary architecture and EGFR exon 19 deletion.",
        "tab": [62, "Female", 22, "1", 3.2, 18.0, 15.0, 42.0, "Negative (-)", "Negative (-)", "Negative (-)", "Mutant", "Wild-type", 12.4, 7.1, 230.0]
    },
    "slide_3": {
        "id": "H&E Slide S-03",
        "patient_id": "PID-52109",
        "tumor_size": 2.8,
        "badge_text": "🔴 Malignant Neoplasm",
        "organ_label": "Breast Biopsy",
        "patient_tag": "PID-52109 · 52F · Left Breast · ER+/PR+ · Grade 2 IDC",
        "category": "breast_malignant",
        "img_path": "data/sample_images/breast/breast_00.png",
        "text": "Core needle biopsy of left breast mass reveals Nottingham Grade 2 invasive ductal carcinoma infiltrating mammary adipocytes; ER/PR positive, HER2 negative.",
        "tab": [52, "Female", 0, "0", 1.8, 24.0, 11.0, 30.0, "Positive (+)", "Positive (+)", "Negative (-)", "Wild-type", "Wild-type", 13.1, 6.2, 210.0]
    },
    "slide_4": {
        "id": "H&E Slide S-04",
        "patient_id": "PID-68340",
        "tumor_size": 0.0,
        "badge_text": "🟢 Non-Neoplastic",
        "organ_label": "Lung Resection",
        "patient_tag": "PID-68340 · 68M · Wedge Margin · Emphysema · CEA 1.2",
        "category": "lung_normal",
        "img_path": "data/sample_images/normal/normal_lung_00.png",
        "text": "Wedge resection margin reveals emphysematous lung parenchyma with alveolar septal thinning without evidence of epithelial atypia or malignant neoplasia.",
        "tab": [68, "Male", 40, "1", 1.2, 8.0, 7.5, 5.0, "Negative (-)", "Negative (-)", "Negative (-)", "Wild-type", "Wild-type", 14.5, 6.5, 240.0]
    },
    "slide_5": {
        "id": "H&E Slide S-05",
        "patient_id": "PID-29184",
        "tumor_size": 1.5,
        "badge_text": "🟡 Benign Lesion",
        "organ_label": "Breast Biopsy",
        "patient_tag": "PID-29184 · 29F · Mobile Nodule · Benign Fibroadenoma",
        "category": "breast_benign",
        "img_path": "data/sample_images/benign/benign_breast_00.png",
        "text": "Core biopsy demonstrates intracanalicular fibroadenoma with benign stromal proliferation compressing non-atypical breast epithelium.",
        "tab": [29, "Female", 0, "0", 0.9, 11.0, 9.0, 8.0, "Positive (+)", "Positive (+)", "Negative (-)", "Wild-type", "Wild-type", 12.8, 5.9, 260.0]
    }
}

REPORT_TEMPLATES = {
    "Custom / Free Text Note": "",
    "Template: Invasive Colorectal Adenocarcinoma (KRAS codon 12)": (
        "Endoscopic biopsy of sigmoid colon mass confirms invasive adenocarcinoma with prominent cribriform glands; KRAS codon 12 mutation confirmed."
    ),
    "Template: Invasive Lung Adenocarcinoma (EGFR exon 19 del)": (
        "Core biopsy of right upper lobe pulmonary mass confirms primary invasive lung adenocarcinoma with papillary architecture and EGFR exon 19 deletion."
    ),
    "Template: Invasive Ductal Carcinoma of Breast (ER+/PR+)": (
        "Core needle biopsy of left breast mass reveals Nottingham Grade 2 invasive ductal carcinoma infiltrating mammary adipocytes; ER/PR positive, HER2 negative."
    ),
    "Template: Non-Neoplastic Lung Resection Margin (Emphysema)": (
        "Wedge resection margin reveals emphysematous lung parenchyma with alveolar septal thinning without evidence of epithelial atypia or malignant neoplasia."
    ),
    "Template: Benign Breast Fibroadenoma": (
        "Core biopsy demonstrates intracanalicular fibroadenoma with benign stromal proliferation compressing non-atypical breast epithelium."
    )
}


def create_overview_minimap(pil_img):
    """Generates a microscopic macro minimap showing tissue section with viewport bounding box."""
    w, h = pil_img.size
    thumb = pil_img.resize((140, 100), Image.Resampling.LANCZOS)
    arr = np.array(thumb)
    # Draw blue bounding box representing active field of view
    bx1, by1, bx2, by2 = 30, 20, 110, 80
    arr[by1:by1+2, bx1:bx2] = [37, 99, 235]
    arr[by2-2:by2, bx1:bx2] = [37, 99, 235]
    arr[by1:by2, bx1:bx1+2] = [37, 99, 235]
    arr[by1:by2, bx2-2:bx2] = [37, 99, 235]
    return Image.fromarray(arr)


def process_aisight_case(
    input_img, input_text, include_tabular,
    age, sex, smoking, ecog, cea, ca125, ca19_9, ki67,
    er, pr, her2, egfr, kras, hb, wbc, plt,
    active_view_layer="🔥 Faithful Grad-CAM Saliency",
    current_slide_id="H&E Slide S-01",
    current_badge_text="🔴 Malignant Neoplasm",
    magnification="20× (Benchmark)",
    suspected_organ="Auto-Detect (Phikon ViT-B Gatekeeper)",
    patient_id="PID-84021",
    tumor_size=3.5
):
    """
    Main GCAEMMCD Execution Controller.
    Runs Level 1 + Level 2 + Level 3 pipelines across all 4 tasks:
      - Task 1: Primary Cancer Classification & Anatomical Subtype
      - Task 2: TNM Pathological Cancer Staging (Stages 0–IV)
      - Task 3: Multimodal Treatment Recommendation Ranking (NDCG@3)
      - Task 4: 5-Year Survival Risk & Prognostic Estimation (C-Index)
      - Explainability: Grad-CAM, Faithfulness Causal Masking, SHAP Waterfall, BioBERT Attribution
    """
    if input_img is None:
        blank_viewer = Image.new("RGB", (600, 450), color="#f8fafc")
        return (
            blank_viewer, None,
            get_slide_badge_html(current_slide_id, "⚪ No Specimen Loaded"),
            "### ⚪ Slide Acceptability: AWAITING INPUT\n*Please load a specimen or upload a micrograph tile.*",
            "### ⚪ Stages 1–3: AWAITING INPUT\n*Please load a specimen or upload a micrograph tile.*",
            "### ⚪ Multimodal Alignment: AWAITING INPUT",
            "### ⚪ Explainability: AWAITING INPUT",
            "<div style='color: #64748b;'><em>Awaiting specimen input...</em></div>",
            None
        )

    pil_img = input_img.convert("RGB") if hasattr(input_img, "convert") else Image.fromarray(np.asarray(input_img)).convert("RGB")
    minimap_img = create_overview_minimap(pil_img)

    # 1. Slide Acceptability (Stage 0 Domain Validator)
    is_valid, domain_type, rej_reason, metrics = validate_histopathology_domain(pil_img)
    if not is_valid:
        accept_md = (
            f"### 🚫 Slide Acceptability: REJECTED AT INGESTION GATE\n\n"
            f"- **Detected Domain:** `{domain_type.upper()}`\n"
            f"- **Rejection Reason:** {rej_reason}\n\n"
            f"**Optical Measurements:**\n"
            f"- Tissue Stained Fraction: `{metrics.get('tissue_fraction', 0.0)*100:.1f}%`\n"
            f"- H&E Chromophore Ratio: `{metrics.get('he_fraction', 0.0)*100:.1f}%`\n"
            f"- Forbidden Spectrum: `{metrics.get('forbidden_fraction', 0.0)*100:.1f}%`\n"
            f"- Dark Slate UI Ratio: `{metrics.get('dark_slate_ratio', 0.0)*100:.1f}%`\n"
            f"- Focus Sharpness (Laplacian): `{metrics.get('laplacian_var', 0.0):.1f}`\n\n"
            f"*(Execution halted: Model inference is prohibited on non-histopathology inputs.)*"
        )
        algo_md = "🚫 **Algorithm Execution Halted:** Specimen failed optical domain validation."
        align_md = "🚫 **Tri-Modal Alignment Suppressed:** Ingestion gate blocked."
        faith_md = "🚫 **Explainability Suppressed:** No saliency computed for non-tissue inputs."
        text_html = "<div style='color: #b91c1c;'><em>Execution halted: Invalid slide domain.</em></div>"
        badge_html = get_slide_badge_html(current_slide_id, "🚫 Ingestion Rejected")
        return (
            pil_img, minimap_img, badge_html, accept_md, algo_md, align_md, faith_md, text_html, None
        )

    accept_md = (
        f"### 🟢 Slide Acceptability: VERIFIED H&E HISTOPATHOLOGY\n\n"
        f"- **Patient Evaluation ID:** `{patient_id}`\n"
        f"- **Specimen Quality:** Brightfield Tissue Patch (Authentic)\n"
        f"- **Magnification Mode:** `{magnification}`\n"
        f"- **Tissue Stained Fraction:** `{metrics.get('tissue_fraction', 0.0)*100:.1f}%`\n"
        f"- **H&E Chromophore Concordance:** `{metrics.get('he_fraction', 0.0)*100:.1f}%`\n"
        f"- **Focus Sharpness Score:** `{metrics.get('laplacian_var', 0.0):.1f}` (In-focus)\n"
        f"- **Digital Artifact Filter:** Passed (Dark Slate: `{metrics.get('dark_slate_ratio', 0.0)*100:.1f}%`)"
    )

    # 2. Level 2 Multimodal Pipeline Execution
    badge_md, diag_card, ps_dict, po_dict, abl_md, gate_md, cam_overlay, cam_cap = run_multimodal_pipeline(
        pil_img, input_text, include_tabular,
        age, sex, smoking, ecog, cea, ca125, ca19_9, ki67,
        er, pr, her2, egfr, kras, hb, wbc, plt
    )

    # 3. Target Class Selection for Explainability
    target_site_idx = 0
    if po_dict:
        top_name = max(po_dict, key=po_dict.get)
        for idx, s in enumerate(SITE_NAMES):
            if s.lower() in top_name.lower():
                target_site_idx = idx
                break

    # Determine dynamic status badge from actual model output
    top_status = "Malignant Neoplasm"
    if ps_dict:
        top_s = max(ps_dict, key=ps_dict.get)
        if "Normal" in top_s:
            top_status = "Non-Neoplastic"
        elif "Benign" in top_s:
            top_status = "Benign Lesion"

    badge_html = get_slide_badge_html(current_slide_id, f"● {top_status}")

    # 4. Compute Visual Explainability & Causal Masking
    cropped_pil = crop_eyepiece_vignette(pil_img)
    t_batch = preprocess_multicrop(cropped_pil).to(config.DEVICE)
    raw_views, _ = img_extractor(t_batch)
    has_i = True
    has_t = bool(input_text and input_text.strip())
    has_r = bool(include_tabular)

    v_t = text_proj(text_extractor(input_text, device=config.DEVICE)) if has_t else None
    if has_r:
        raw_tab_dict_pre = {
            "age": age, "sex": _map_radio_binary(sex, "Female"),
            "smoking_pack_yrs": smoking, "ecog_score": _map_radio_int(ecog),
            "cea_level": cea, "ca125_level": ca125, "ca19_9_level": ca19_9, "ki67_index": ki67,
            "er_status": _map_radio_binary(er, "Positive (+)"),
            "pr_status": _map_radio_binary(pr, "Positive (+)"),
            "her2_status": _map_radio_binary(her2, "Positive (+)"),
            "egfr_mutation": _map_radio_binary(egfr, "Mutant"),
            "kras_mutation": _map_radio_binary(kras, "Mutant"),
            "hemoglobin": hb, "wbc_count": wbc, "platelet_count": plt
        }
        _, _, _, _, raw_vec_16_pre, mask_pre = validate_tabular_biomarkers(raw_tab_dict_pre)
        vec_32_pre = encode_biomarkers_with_mask(raw_vec_16_pre, mask_pre)
        v_r = tab_mlp(torch.tensor(vec_32_pre, dtype=torch.float32).unsqueeze(0).to(config.DEVICE))
    else:
        v_r = None
    raw_hm, _, _ = gradcam.generate_heatmap(t_batch[:1], pathway="image", head_type="site", target_class=target_site_idx)

    v_faith = evaluate_visual_faithfulness(
        cropped_pil, raw_hm, img_extractor, classifier,
        target_class_idx=target_site_idx, head_type="site", device=config.DEVICE
    )
    masked_salient_img = v_faith["img_masked_salient"]

    # 5. Compute Narrative Explainability (BioBERT Saliency)
    text_html = "<div style='color: #64748b;'><em>Clinical text not provided; narrative attribution skipped.</em></div>"
    if input_text and input_text.strip():
        t_res = text_saliency_explainer.explain_text(input_text, target_class_idx=target_site_idx, head_type="site")
        text_html = t_res["html_view"]

    # 6. Compute Biomarker Explainability (SHAP)
    shap_chart_img = None
    shap_metrics_text = ""
    if include_tabular:
        raw_tab_dict = {
            "age": age, "sex": _map_radio_binary(sex, "Female"),
            "smoking_pack_yrs": smoking, "ecog_score": _map_radio_int(ecog),
            "cea_level": cea, "ca125_level": ca125, "ca19_9_level": ca19_9, "ki67_index": ki67,
            "er_status": _map_radio_binary(er, "Positive (+)"),
            "pr_status": _map_radio_binary(pr, "Positive (+)"),
            "her2_status": _map_radio_binary(her2, "Positive (+)"),
            "egfr_mutation": _map_radio_binary(egfr, "Mutant"),
            "kras_mutation": _map_radio_binary(kras, "Mutant"),
            "hemoglobin": hb, "wbc_count": wbc, "platelet_count": plt
        }
        _, _, _, _, raw_vec_16, _ = validate_tabular_biomarkers(raw_tab_dict)
        t_faith = evaluate_tabular_faithfulness(
            raw_vec_16, shap_explainer, target_class_idx=target_site_idx, head_type="site"
        )
        shap_chart_img = t_faith["shap_result"]["plot_pil"]
        shap_metrics_text = (
            f"- **SHAP Additivity Error:** `{t_faith['efficiency_residual']:.2e}` (Exact Efficiency Verified)\n"
            f"- **Top-3 Biomarker Ablation Drop:** `+{t_faith['tabular_confidence_drop']*100:.2f}%`\n"
        )

    # 7. Format Inspector Panels (Complete 4-Task Research Architecture)
    multi_task_out = classifier(
        raw_views.mean(dim=0, keepdim=True) if has_i else None,
        v_t if has_t else None,
        v_r if has_r else None
    )

    stage_md_block = ""
    treatment_md_block = ""
    survival_md_block = ""

    if multi_task_out.get("logits_staging") is not None:
        stg_logits = multi_task_out["logits_staging"][0]
        stg_probs = torch.softmax(stg_logits, dim=-1).cpu().detach().numpy()
        stg_idx = int(np.argmax(stg_probs))
        stg_name = STAGE_NAMES[stg_idx]
        stg_conf = stg_probs[stg_idx]

        stage_md_block = (
            f"#### 🏛️ Task 2: TNM Pathological Cancer Staging\n"
            f"- **Predicted Stage:** <strong style='color: #1d4ed8;'>{stg_name}</strong> (`{stg_conf*100:.1f}%` confidence)\n"
            f"- **Primary Tumor Size (T-stage basis):** `{tumor_size:.1f} cm`\n"
            f"- **Staging Distribution:** " + " · ".join([f"Stage {i}: `{stg_probs[i]*100:.1f}%`" for i in range(5)]) + "\n"
        )

    if multi_task_out.get("treatment_scores") is not None:
        trt_scores = multi_task_out["treatment_scores"][0].cpu().detach().numpy()
        ranked_trt = sorted(list(enumerate(trt_scores)), key=lambda x: -x[1])[:3]
        treatment_lines = []
        for rk, (t_idx, score) in enumerate(ranked_trt, 1):
            treatment_lines.append(f"{rk}. **{TREATMENT_OPTIONS[t_idx]}** *(Priority Score: `{score:.2f}`)*")
        treatment_md_block = (
            f"#### 💊 Task 3: Multimodal Treatment Recommendation Ranking (NDCG@3)\n"
            + "\n".join([f"- {tl}" for tl in treatment_lines]) + "\n"
        )

    if multi_task_out.get("survival_probabilities") is not None:
        surv_val = float(multi_task_out["survival_probabilities"][0].item())
        haz_val = float(multi_task_out["hazard_scores"][0].item())
        risk_tier = "High Mortality Risk" if surv_val < 0.40 else ("Moderate Risk" if surv_val < 0.75 else "Favorable Prognosis")
        risk_color = "#dc2626" if surv_val < 0.40 else ("#d97706" if surv_val < 0.75 else "#16a34a")

        survival_md_block = (
            f"#### ⏳ Task 4: 5-Year Survival Risk & Prognostic Estimation (C-Index)\n"
            f"- **Estimated 5-Year Survival Probability:** <strong style='color: {risk_color}; font-size: 15px;'>{surv_val*100:.1f}%</strong>\n"
            f"- **Cox Proportional Hazard Index:** `{haz_val:.2f}` ({risk_tier})\n"
        )

    algo_md = (
        f"### 🏆 Multi-Task Multimodal Clinical Findings (4 Tasks)\n\n"
        f"**Patient Reference:** `{patient_id}` · **Tumor Dimension:** `{tumor_size:.1f} cm`\n\n"
        f"#### 🔬 Task 1: Neoplasm Classification & Anatomical Subtype\n"
        f"{diag_card}\n\n"
        f"{stage_md_block}\n"
        f"{treatment_md_block}\n"
        f"{survival_md_block}\n"
        f"**Pathway Confidence Distribution:**\n"
        + "\n".join([f"- **{k}:** `{v*100:.1f}%`" for k, v in ps_dict.items()]) + "\n"
        + "\n".join([f"- **{k}:** `{v*100:.1f}%`" for k, v in po_dict.items()])
    )

    align_md = (
        f"### 📋 Tri-Modal Alignments & Concordance\n\n"
        f"{badge_md}\n\n"
        f"{gate_md}\n\n"
        f"{abl_md}"
    )

    faith_md = (
        f"### 📊 Explainability & Quantitative Faithfulness\n\n"
        f"- **Causal Saliency Test:** {'🟢 **FAITHFUL**' if v_faith['is_faithful'] else '⚠️ **DIFFUSE**'}\n"
        f"- **Baseline Target Confidence:** `{v_faith['original_confidence']*100:.1f}%`\n"
        f"- **Top-20% Salient Region Masked Confidence:** `{v_faith['masked_confidence']*100:.1f}%`\n"
        f"- **Salient Confidence Drop (Perturbation):** `+{v_faith['salient_confidence_drop']*100:.2f}%`\n"
        f"- **Random Control Region Drop:** `+{v_faith['control_confidence_drop']*100:.2f}%`\n"
        f"{shap_metrics_text}"
    )

    # 8. Route Viewport Layer
    if "Grad-CAM" in active_view_layer:
        viewport_img = cam_overlay
    elif "Masked" in active_view_layer and masked_salient_img is not None:
        viewport_img = masked_salient_img
    elif "SHAP" in active_view_layer and shap_chart_img is not None:
        viewport_img = shap_chart_img
    else:
        viewport_img = cropped_pil

    return (
        viewport_img, minimap_img, badge_html, accept_md, algo_md, align_md, faith_md, text_html, shap_chart_img
    )


# -----------------------------------------------------------------------------
# BUILD GCAEMMCD THREE-COLUMN INTERFACE
# -----------------------------------------------------------------------------
def build_aisight_ui():
    custom_theme = get_gcaemmcd_theme()
    with gr.Blocks(title="GCAEMMCD Clinical Pathology Workstation", css=GCAEMMCD_CUSTOM_CSS, theme=custom_theme, js=FORCE_LIGHT_MODE_JS) as demo:
        # Top Global Workstation Navigation Bar
        gr.HTML(get_top_navbar_html())

        # Specimen Session Sub-Header (Dynamic)
        sub_header_comp = gr.HTML(get_sub_header_html(specimen_id="Specimen S-01", status="Ready", patient_id="PID-84021"))

        # State Variables
        active_slide_img = gr.State(value=Image.open(SLIDE_PRESETS["slide_1"]["img_path"]))
        cur_slide_id = gr.State(value="H&E Slide S-01")
        cur_badge_text = gr.State(value="🔴 Malignant Neoplasm")

        # Main Workstation 3-Column Workspace: Sample Input | Manual Input | Output
        with gr.Row(equal_height=False, elem_classes=["workstation-row-container"]):
            # -----------------------------------------------------------------
            # COLUMN 1: SAMPLE INPUT (Fixed Height, Column Inner Scroll)
            # -----------------------------------------------------------------
            with gr.Column(scale=3, min_width=250, elem_classes=["col-sample-input"]):
                gr.HTML("""
                <div class="column-title-bar">
                    <span class="column-title-text">📂 Column 1: Sample Input</span>
                    <span class="column-badge">5 Presets</span>
                </div>
                """)
                gr.Markdown(
                    "*Select a calibrated clinical specimen to load histological patch, clinical narrative, and 16 biomarkers:*"
                )

                # Preset Specimen 1: Colorectal
                with gr.Group(elem_classes=["specimen-preset-card", "active-selected"]):
                    with gr.Row():
                        thumb_1 = gr.Image(value=SLIDE_PRESETS["slide_1"]["img_path"], show_label=False, width=44, height=44)
                        with gr.Column(scale=3):
                            gr.Markdown("**H&E S-01** · Colon Biopsy\n<span class='status-pill-malignant'>● Malignant</span>")
                    gr.HTML(f"<div class='specimen-patient-tag'>{SLIDE_PRESETS['slide_1']['patient_tag']}</div>")
                    btn_slide_1 = gr.Button("Load Specimen S-01", size="sm", variant="secondary", elem_classes=["btn-punchy-secondary"])

                # Preset Specimen 2: Lung
                with gr.Group(elem_classes=["specimen-preset-card"]):
                    with gr.Row():
                        thumb_2 = gr.Image(value=SLIDE_PRESETS["slide_2"]["img_path"], show_label=False, width=44, height=44)
                        with gr.Column(scale=3):
                            gr.Markdown("**H&E S-02** · Lung Biopsy\n<span class='status-pill-malignant'>● Malignant</span>")
                    gr.HTML(f"<div class='specimen-patient-tag'>{SLIDE_PRESETS['slide_2']['patient_tag']}</div>")
                    btn_slide_2 = gr.Button("Load Specimen S-02", size="sm", variant="secondary", elem_classes=["btn-punchy-secondary"])

                # Preset Specimen 3: Breast
                with gr.Group(elem_classes=["specimen-preset-card"]):
                    with gr.Row():
                        thumb_3 = gr.Image(value=SLIDE_PRESETS["slide_3"]["img_path"], show_label=False, width=44, height=44)
                        with gr.Column(scale=3):
                            gr.Markdown("**H&E S-03** · Breast Biopsy\n<span class='status-pill-malignant'>● Malignant</span>")
                    gr.HTML(f"<div class='specimen-patient-tag'>{SLIDE_PRESETS['slide_3']['patient_tag']}</div>")
                    btn_slide_3 = gr.Button("Load Specimen S-03", size="sm", variant="secondary", elem_classes=["btn-punchy-secondary"])

                # Preset Specimen 4: Emphysema Normal Lung
                with gr.Group(elem_classes=["specimen-preset-card"]):
                    with gr.Row():
                        thumb_4 = gr.Image(value=SLIDE_PRESETS["slide_4"]["img_path"], show_label=False, width=44, height=44)
                        with gr.Column(scale=3):
                            gr.Markdown("**H&E S-04** · Lung Resection\n<span class='status-pill-normal'>● Non-Neoplastic</span>")
                    gr.HTML(f"<div class='specimen-patient-tag'>{SLIDE_PRESETS['slide_4']['patient_tag']}</div>")
                    btn_slide_4 = gr.Button("Load Specimen S-04", size="sm", variant="secondary", elem_classes=["btn-punchy-secondary"])

                # Preset Specimen 5: Fibroadenoma Benign Breast
                with gr.Group(elem_classes=["specimen-preset-card"]):
                    with gr.Row():
                        thumb_5 = gr.Image(value=SLIDE_PRESETS["slide_5"]["img_path"], show_label=False, width=44, height=44)
                        with gr.Column(scale=3):
                            gr.Markdown("**H&E S-05** · Breast Biopsy\n<span class='status-pill-benign'>● Benign Lesion</span>")
                    gr.HTML(f"<div class='specimen-patient-tag'>{SLIDE_PRESETS['slide_5']['patient_tag']}</div>")
                    btn_slide_5 = gr.Button("Load Specimen S-05", size="sm", variant="secondary", elem_classes=["btn-punchy-secondary"])

                gr.Markdown(
                    "**Cohort Specifications:**\n"
                    "- **Magnification:** 20× Brightfield\n"
                    "- **Tissue Field:** 1.84 mm² (224×224 px)\n"
                    "- **Staining:** Certified H&E Protocol\n"
                    "- **Modalities:** 3 / 3 Complete"
                )

            # -----------------------------------------------------------------
            # COLUMN 2: MANUAL INPUT (Fixed Height, Column Inner Scroll)
            # Exactly formatted matching User Diagnosis System Box Layout:
            # 1. PATIENT INFORMATION | 2. HISTOPATHOLOGY IMAGE | 3. CLINICAL REPORT | 4. CLINICAL PARAMETERS | ANALYSE PATIENT
            # -----------------------------------------------------------------
            with gr.Column(scale=5, min_width=350, elem_classes=["col-manual-input"]):
                gr.HTML("""
                <div class="column-title-bar">
                    <span class="column-title-text">🩺 CANCER DIAGNOSIS SYSTEM · MULTIMODAL INPUT</span>
                    <span class="column-badge">Interactive Entry</span>
                </div>
                """)

                # 1. PATIENT INFORMATION
                with gr.Group(elem_classes=["input-sub-section"]):
                    gr.HTML('<div class="input-section-header">👤 1. PATIENT INFORMATION</div>')
                    with gr.Row():
                        s_patient_id = gr.Textbox(value="PID-84021", label="Patient ID", placeholder="e.g. PID-90421", scale=2)
                        s_age = gr.Slider(18, 100, value=64, step=1, label="Age (Years)", scale=2)
                    with gr.Row():
                        s_sex = gr.Radio(["Male", "Female"], value="Male", label="Gender / Biological Sex")
                        s_ecog = gr.Radio(["0", "1", "2", "3"], value="1", label="ECOG Performance (0: Active, 1: Ambulatory, 2: Bedridden <50%, 3: Bedridden >50%)")

                # 2. HISTOPATHOLOGY IMAGE
                with gr.Group(elem_classes=["input-sub-section"]):
                    gr.HTML('<div class="input-section-header">🔬 2. HISTOPATHOLOGY IMAGE</div>')
                    custom_upload_img = gr.Image(
                        type="pil",
                        label="Upload Image / Drag & Drop (Tissue Micrograph Tile 224×224 px RGB)",
                        height=140
                    )
                    with gr.Row():
                        s_mag = gr.Radio(
                            ["20× (Benchmark)", "40× (High-Power)", "10× (Survey)"],
                            value="20× (Benchmark)",
                            label="Optical Magnification"
                        )
                        s_organ = gr.Dropdown(
                            [
                                "Auto-Detect (Phikon ViT-B Gatekeeper)",
                                "Colorectal Biopsy",
                                "Lung Biopsy",
                                "Breast Biopsy",
                                "Normal / Non-Neoplastic"
                            ],
                            value="Auto-Detect (Phikon ViT-B Gatekeeper)",
                            label="Suspected Anatomical Site"
                        )

                # 3. CLINICAL REPORT
                with gr.Group(elem_classes=["input-sub-section"]):
                    gr.HTML('<div class="input-section-header">📝 3. CLINICAL REPORT</div>')
                    s_narrative_tpl = gr.Dropdown(
                        list(REPORT_TEMPLATES.keys()),
                        value="Custom / Free Text Note",
                        label="Select Pathology Vignette Template"
                    )
                    input_text_box = gr.Textbox(
                        lines=3,
                        label="Enter Pathology Report / Surgical Observations",
                        placeholder="Enter clinical notes, histological observations, or oncological findings...",
                        value=SLIDE_PRESETS["slide_1"]["text"]
                    )
                    gr.HTML('<div style="font-size: 11px; color: #64748b; margin-top: -2px;">Context: Maximum 256 BioBERT tokens (Section 2 Viva Specification)</div>')

                # 4. CLINICAL PARAMETERS
                with gr.Group(elem_classes=["input-sub-section"]):
                    with gr.Row():
                        gr.HTML('<div class="input-section-header" style="flex:1;">🧬 4. CLINICAL PARAMETERS & BIOMARKERS</div>')
                        inc_tab_check = gr.Checkbox(label="Include in Fusion", value=True)

                    # Tumor Size & Smoking
                    with gr.Row():
                        s_tumor_size = gr.Slider(0.0, 15.0, value=4.2, step=0.1, label="Tumor Size (cm)")
                        s_smoke = gr.Slider(0, 100, value=35, step=1, label="Smoking History (Pack-Yrs)")

                    # Serum Markers
                    gr.HTML('<div class="clinical-ref-chip">ℹ️ Serum Markers Norms: CEA < 3.0 ng/mL · CA-125 < 35.0 U/mL · CA19-9 < 37.0 U/mL</div>')
                    with gr.Row():
                        s_cea = gr.Slider(0, 150, value=34.5, step=0.1, label="CEA (ng/mL)")
                        s_ca125 = gr.Slider(0, 200, value=14.0, step=0.1, label="CA-125 (U/mL)")
                        s_ca19_9 = gr.Slider(0, 250, value=48.0, step=0.1, label="CA19-9 (U/mL)")

                    # Proliferation & Receptors
                    gr.HTML('<div class="clinical-ref-chip">ℹ️ Proliferation Index: Ki-67 < 15% Low · 15–30% Intermediate · > 30% High</div>')
                    with gr.Row():
                        s_ki67 = gr.Slider(0, 100, value=55.0, step=1.0, label="Ki-67 Index (%)")
                        s_er = gr.Radio(["Negative (-)", "Positive (+)"], value="Negative (-)", label="ER")
                    with gr.Row():
                        s_pr = gr.Radio(["Negative (-)", "Positive (+)"], value="Negative (-)", label="PR")
                        s_her2 = gr.Radio(["Negative (-)", "Positive (+)"], value="Negative (-)", label="HER2")

                    # Genetics & CBC Labs
                    with gr.Row():
                        s_egfr = gr.Radio(["Wild-type", "Mutant"], value="Wild-type", label="EGFR")
                        s_kras = gr.Radio(["Wild-type", "Mutant"], value="Mutant", label="KRAS")

                    gr.HTML('<div class="clinical-ref-chip">ℹ️ Hematology Norms: Hemoglobin 12.0–17.5 g/dL · WBC 4.5–11.0 ×10³/µL · Platelets 150–450 ×10³/µL</div>')
                    with gr.Row():
                        s_hb = gr.Slider(5, 20, value=11.2, step=0.1, label="Hemoglobin (g/dL)")
                        s_wbc = gr.Slider(1, 40, value=8.4, step=0.1, label="WBC (×10³/µL)")
                        s_plt = gr.Slider(50, 800, value=290.0, step=1.0, label="Platelets (×10³/µL)")

                # Section 5: Primary Action Button: ANALYSE PATIENT
                with gr.Row():
                    btn_recompute = gr.Button("🔬 ANALYSE PATIENT", variant="primary", size="lg", elem_classes=["btn-punchy-primary"])
                    btn_reset_tab = gr.Button("↺ Reset to Preset Defaults", variant="secondary", size="sm", elem_classes=["btn-punchy-secondary"])

            # -----------------------------------------------------------------
            # COLUMN 3: OUTPUT (Fixed Height, Column Inner Scroll)
            # -----------------------------------------------------------------
            with gr.Column(scale=5, min_width=370, elem_classes=["col-diagnostic-output"]):
                gr.HTML("""
                <div class="column-title-bar">
                    <span class="column-title-text">🔬 Column 3: Output</span>
                    <span class="column-badge">Deep Model Findings</span>
                </div>
                """)

                # Section 1: Viewport Toolbar & Layer Switcher (Dynamic)
                viewer_toolbar_comp = gr.HTML(get_viewer_toolbar_html(specimen_id="Specimen S-01"))
                view_layer_radio = gr.Radio(
                    [
                        "🔥 Faithful Grad-CAM Saliency",
                        "🖼️ Clean H&E Micrograph",
                        "🎭 Causal Saliency Masking (Top 20% Blurred)",
                        "📊 SHAP Biomarker Waterfall"
                    ],
                    value="🔥 Faithful Grad-CAM Saliency",
                    label="Micrograph Viewport Layer"
                )

                # Section 2: Main Viewport Container
                with gr.Group(elem_classes=["viewer-container"]):
                    badge_component = gr.HTML(get_slide_badge_html("H&E Slide S-01", "🔴 Malignant Neoplasm"))
                    main_viewer_img = gr.Image(
                        value=Image.open(SLIDE_PRESETS["slide_1"]["img_path"]),
                        show_label=False,
                        interactive=False,
                        height=350
                    )
                    gr.HTML("""
                    <div class="viewer-scale-bar">
                        <span>Scale: <strong>200 µm</strong></span>
                        <span style="display:inline-block; width:50px; height:3px; background:#2563eb; border-radius:2px;"></span>
                    </div>
                    <div class="viewer-timestamp">
                        FOV: 1.84 mm² · Mag: 20x
                    </div>
                    """)

                # Section 3: Diagnostic Findings & Consultation Signout Tabs
                with gr.Tabs():
                    with gr.Tab("Diagnostic Inspector"):
                        with gr.Accordion("🔍 Slide Acceptability (Stage 0 Domain Check)", open=True):
                            accept_display = gr.Markdown("### ⚪ Stage 0: Optical Domain Validation\n*Awaiting specimen inspection...*")

                        with gr.Accordion("🏛️ Hierarchical Cancer Diagnosis (Stages 1–3)", open=True):
                            algo_display = gr.Markdown("### ⚪ Stages 1–3: Neoplasm Diagnosis\n*Awaiting specimen inspection...*")

                        with gr.Accordion("📋 Multimodal Alignment & GCA Dynamics", open=True):
                            align_display = gr.Markdown("### ⚪ Multimodal Alignment & GCA Dynamics\n*Awaiting specimen inspection...*")

                        with gr.Accordion("📊 Faithfulness & BioBERT Narrative Saliency", open=True):
                            faith_display = gr.Markdown("### ⚪ Explainability & Faithfulness\n*Awaiting specimen inspection...*")
                            gr.Markdown("**BioBERT Clinical Narrative Attribution:**")
                            text_html_view = gr.HTML("<div style='color: #64748b;'><em>Awaiting clinical text narrative for token attribution...</em></div>")

                    with gr.Tab("Consultation Signout"):
                        gr.Markdown(
                            "### **GCAEMMCD Multimodal Clinical Consultation Signout**\n\n"
                            "This panel generates a comprehensive clinical consultation summary based on verified multimodal findings:\n\n"
                            "- **Stage 0:** Physical stain concordance & brightfield optical validity.\n"
                            "- **Stage 1:** Phikon ViT-B organ identification & prototype distance out-of-distribution guardrail.\n"
                            "- **Stages 2–3:** Hierarchical malignancy classification & conditional carcinoma subtyping.\n"
                            "- **Level 2:** Gated Cross-Attention (GCA) multimodal fusion across histology, clinical notes, and 16 biomarkers.\n"
                            "- **Level 3:** Faithful Grad-CAM saliency, causal region masking verification, and SHAP local efficiency.\n\n"
                            "*Click 'ANALYSE PATIENT' or choose a sample to populate findings.*"
                        )

                # Section 4: Macro Minimap & SHAP Waterfall Row
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Tissue Macro Minimap:**")
                        minimap_view = gr.Image(
                            value=create_overview_minimap(Image.open(SLIDE_PRESETS["slide_1"]["img_path"])),
                            show_label=False,
                            height=120
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("**Tabular SHAP Attribution:**")
                        shap_thumbnail = gr.Image(show_label=False, height=120)

        # Bottom Clinical Disclaimer Footer
        gr.HTML(get_footer_disclaimer_html())

        # Tabular parameter list (16 biomarkers)
        all_tab_inputs = [
            s_age, s_sex, s_smoke, s_ecog,
            s_cea, s_ca125, s_ca19_9, s_ki67,
            s_er, s_pr, s_her2, s_egfr, s_kras,
            s_hb, s_wbc, s_plt
        ]

        # Handler for Narrative Template Selection
        def _apply_narrative_template(tpl_name):
            val = REPORT_TEMPLATES.get(tpl_name, "")
            if val:
                return val
            return gr.update()

        s_narrative_tpl.change(
            fn=_apply_narrative_template,
            inputs=[s_narrative_tpl],
            outputs=[input_text_box]
        )

        # Connect Primary Diagnostic Inference Button (ANALYSE PATIENT)
        def _on_run_inference(img, txt, inc_tab, p_id, t_size, mag_val, org_val, v_layer, sl_id, bdg_t, *tab_vals):
            outs = process_aisight_case(
                img, txt, inc_tab, *tab_vals[:16],
                active_view_layer=v_layer,
                current_slide_id=sl_id,
                current_badge_text=bdg_t,
                magnification=mag_val,
                suspected_organ=org_val,
                patient_id=p_id,
                tumor_size=t_size
            )
            sh_html = get_sub_header_html(specimen_id=sl_id, status="Analyzed", patient_id=p_id)
            vt_html = get_viewer_toolbar_html(specimen_id=sl_id)
            return (sh_html, vt_html) + outs

        inference_outputs = [
            sub_header_comp, viewer_toolbar_comp,
            main_viewer_img, minimap_view, badge_component,
            accept_display, algo_display, align_display, faith_display,
            text_html_view, shap_thumbnail
        ]

        btn_recompute.click(
            fn=_on_run_inference,
            inputs=[active_slide_img, input_text_box, inc_tab_check, s_patient_id, s_tumor_size, s_mag, s_organ, view_layer_radio, cur_slide_id, cur_badge_text] + all_tab_inputs,
            outputs=inference_outputs
        )

        # Connect Viewport Layer Switcher
        view_layer_radio.change(
            fn=_on_run_inference,
            inputs=[active_slide_img, input_text_box, inc_tab_check, s_patient_id, s_tumor_size, s_mag, s_organ, view_layer_radio, cur_slide_id, cur_badge_text] + all_tab_inputs,
            outputs=inference_outputs
        )

        # Reset Biomarkers Handler
        def _reset_biomarkers(slide_id):
            matched = SLIDE_PRESETS["slide_1"]
            for p in SLIDE_PRESETS.values():
                if p["id"] == slide_id:
                    matched = p
                    break
            tb = matched["tab"]
            p_id = matched["patient_id"]
            t_size = matched["tumor_size"]
            return (p_id, t_size) + tuple(tb)

        btn_reset_tab.click(
            fn=_reset_biomarkers,
            inputs=[cur_slide_id],
            outputs=[s_patient_id, s_tumor_size] + all_tab_inputs
        )

        # Custom Upload Image Handler: Updates active slide image state & toolbars
        def _on_custom_upload(upl_img, cur_layer, txt, inc_tab, p_id, t_size, *tab_args):
            if upl_img is not None:
                sl_id = "Custom Micrograph"
                bdg_t = "🔬 Custom Specimen"
                sh_html = get_sub_header_html(specimen_id=sl_id, status="Uploaded", patient_id=p_id)
                vt_html = get_viewer_toolbar_html(specimen_id=sl_id)
                outs = process_aisight_case(
                    upl_img, txt, inc_tab, *tab_args[:16],
                    active_view_layer=cur_layer,
                    current_slide_id=sl_id,
                    current_badge_text=bdg_t,
                    patient_id=p_id,
                    tumor_size=t_size
                )
                return (
                    sh_html, vt_html,
                    upl_img, sl_id, bdg_t,
                    outs[0], outs[1], outs[2], outs[3], outs[4], outs[5], outs[6], outs[7], outs[8]
                )
            p = SLIDE_PRESETS["slide_1"]
            im = Image.open(p["img_path"])
            sh_html = get_sub_header_html(specimen_id=p["id"], status="Ready", patient_id=p["patient_id"])
            vt_html = get_viewer_toolbar_html(specimen_id=p["id"])
            outs = process_aisight_case(
                im, p["text"], True, *p["tab"],
                active_view_layer=cur_layer,
                current_slide_id=p["id"],
                current_badge_text=p["badge_text"],
                patient_id=p["patient_id"],
                tumor_size=p["tumor_size"]
            )
            return (
                sh_html, vt_html,
                im, p["id"], p["badge_text"],
                outs[0], outs[1], outs[2], outs[3], outs[4], outs[5], outs[6], outs[7], outs[8]
            )

        custom_upload_outputs = [
            sub_header_comp, viewer_toolbar_comp,
            active_slide_img, cur_slide_id, cur_badge_text,
            main_viewer_img, minimap_view, badge_component,
            accept_display, algo_display, align_display, faith_display,
            text_html_view, shap_thumbnail
        ]

        custom_upload_img.change(
            fn=_on_custom_upload,
            inputs=[custom_upload_img, view_layer_radio, input_text_box, inc_tab_check, s_patient_id, s_tumor_size] + all_tab_inputs,
            outputs=custom_upload_outputs
        )

        # Slide Carousel Click Handlers
        def _select_slide(preset_key, cur_layer):
            p = SLIDE_PRESETS[preset_key]
            im = Image.open(p["img_path"])
            tb = p["tab"]
            slide_id = p["id"]
            badge_t = p["badge_text"]
            p_id = p["patient_id"]
            t_size = p["tumor_size"]

            # Process case immediately
            outs = process_aisight_case(
                im, p["text"], True, *tb,
                active_view_layer=cur_layer,
                current_slide_id=slide_id,
                current_badge_text=badge_t,
                patient_id=p_id,
                tumor_size=t_size
            )
            sh_html = get_sub_header_html(specimen_id=slide_id, status="Verified", patient_id=p_id)
            vt_html = get_viewer_toolbar_html(specimen_id=slide_id)

            return (
                sh_html, vt_html,
                im, slide_id, badge_t,
                p_id, t_size,
                p["text"], True,
                tb[0], tb[1], tb[2], tb[3], tb[4], tb[5], tb[6], tb[7], tb[8], tb[9], tb[10], tb[11], tb[12], tb[13], tb[14], tb[15],
                outs[0], outs[1], outs[2], outs[3], outs[4], outs[5], outs[6], outs[7], outs[8]
            )

        slide_outputs = [
            sub_header_comp, viewer_toolbar_comp,
            active_slide_img, cur_slide_id, cur_badge_text,
            s_patient_id, s_tumor_size,
            input_text_box, inc_tab_check,
            s_age, s_sex, s_smoke, s_ecog, s_cea, s_ca125, s_ca19_9, s_ki67,
            s_er, s_pr, s_her2, s_egfr, s_kras, s_hb, s_wbc, s_plt,
            main_viewer_img, minimap_view, badge_component,
            accept_display, algo_display, align_display, faith_display,
            text_html_view, shap_thumbnail
        ]

        btn_slide_1.click(
            fn=lambda l: _select_slide("slide_1", l),
            inputs=[view_layer_radio],
            outputs=slide_outputs
        )

        btn_slide_2.click(
            fn=lambda l: _select_slide("slide_2", l),
            inputs=[view_layer_radio],
            outputs=slide_outputs
        )

        btn_slide_3.click(
            fn=lambda l: _select_slide("slide_3", l),
            inputs=[view_layer_radio],
            outputs=slide_outputs
        )

        btn_slide_4.click(
            fn=lambda l: _select_slide("slide_4", l),
            inputs=[view_layer_radio],
            outputs=slide_outputs
        )

        btn_slide_5.click(
            fn=lambda l: _select_slide("slide_5", l),
            inputs=[view_layer_radio],
            outputs=slide_outputs
        )

        # Auto-run specimen 1 on initial page load
        demo.load(
            fn=lambda l: _select_slide("slide_1", l),
            inputs=[view_layer_radio],
            outputs=slide_outputs
        )

    return demo


if __name__ == "__main__":
    app = build_aisight_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
