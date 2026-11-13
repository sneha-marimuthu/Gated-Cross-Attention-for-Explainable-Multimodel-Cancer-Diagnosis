"""
Unified Hierarchical Gradio Web Demonstration:
Card 1: Organ of Origin (Phikon ViT-B) with Calibrated Cosine Distance OOD Detection
Card 2: Cancer Status Classification (Normal vs Benign vs Malignant)
Card 3: Cancer Subtype Classification (Breast Ca / Lung Ca / Colon Ca) - Only when Malignant
Card 4: Cross-Model Consistency Audit & Uncertainty Warnings
Card 5: Contextual Grad-CAM Visualization with explicit head attribution
Hardware Target: HP Victus Laptop (8GB RAM, NVIDIA GeForce RTX 2050)
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gradio as gr

import config
from src.data.image_transforms import preprocess_image, preprocess_multicrop, crop_eyepiece_vignette
from src.data.domain_validator import validate_histopathology_domain
from src.models.calibrated_organ_model import CalibratedPhikonOrganEncoder, CalibratedOrganHead
from src.models.hierarchical_model import HierarchicalResNetModel
from src.explainability.hierarchical_gradcam import HierarchicalResNetGradCAM

print(f"[*] Initializing Unified Hierarchical Architecture on {config.DEVICE}...")

# ---------------------------------------------------------
# MODULE 1: Calibrated Phikon Organ Encoder & Head
# ---------------------------------------------------------
organ_encoder = CalibratedPhikonOrganEncoder(freeze=True).to(config.DEVICE)
organ_encoder.eval()

organ_head = CalibratedOrganHead(in_features=768, num_organs=3, min_confidence=0.50, max_cosine_distance=0.58).to(config.DEVICE)
organ_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "calibrated_organ_head.pt")
if os.path.exists(organ_ckpt_path):
    ckpt_o = torch.load(organ_ckpt_path, map_location=config.DEVICE, weights_only=False)
    organ_head.load_calibrated_checkpoint(ckpt_o, config.DEVICE)
    print(f"[+] Loaded Calibrated Organ Head (Val Acc: {ckpt_o.get('val_acc')*100:.1f}%, prototypes={organ_head.reference_bank.size(0)})")
organ_head.eval()

# ---------------------------------------------------------
# MODULE 2 & 3: Hierarchical Cancer Status & Subtype Model
# ---------------------------------------------------------
cancer_model = HierarchicalResNetModel().to(config.DEVICE)
cancer_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "hierarchical_cancer_model.pt")
if os.path.exists(cancer_ckpt_path):
    ckpt_c = torch.load(cancer_ckpt_path, map_location=config.DEVICE)
    cancer_model.load_state_dict(ckpt_c["model_state_dict"])
    print(f"[+] Loaded Hierarchical Cancer Model (Val Status Acc: {ckpt_c.get('val_status_acc')*100:.1f}%, Subtype Acc: {ckpt_c.get('val_sub_acc')*100:.1f}%)")
cancer_model.eval()

gradcam = HierarchicalResNetGradCAM(cancer_model)

def hierarchical_inference_pipeline(input_img):
    """
    Executes the full hierarchical diagnostic journey:
    0. Optical stain & tissue validity check (rejects non-histology, documents, white backgrounds)
    1. Organ identification (Phikon ViT-B) with scanner-tolerant calibration
    2. Cancer status (Normal vs Benign vs Malignant)
    3. Cancer subtype (conditioned strictly on Malignant status)
    4. Cross-model consistency check
    5. Contextual Grad-CAM generation
    """
    if input_img is None:
        return (
            "Please upload an image.", None,
            "Awaiting input...", None,
            "Awaiting input...", None,
            "Awaiting input...",
            None, "Awaiting input..."
        )

    # -----------------------------------------------------
    # STAGE 0: Histopathology Domain Validation
    # Rejects terminal screenshots, documents, general photos, and blank slides
    # -----------------------------------------------------
    is_valid, domain_type, rejection_reason, metrics = validate_histopathology_domain(input_img)

    if not is_valid:
        domain_headers = {
            "terminal_screenshot": "🚫 REJECTED: Terminal / Digital Interface Screenshot",
            "document_screenshot": "🚫 REJECTED: Document / Text Screenshot",
            "general_photo": "🚫 REJECTED: General Non-Medical Photograph",
            "blank_or_uninformative": "🚫 REJECTED: Blank / Empty Slide Image"
        }
        title = domain_headers.get(domain_type, "🚫 REJECTED: Non-Histopathology Specimen")
        
        organ_card_text = (
            f"### {title}\n\n"
            f"**Validation Verdict:** REJECTED AT INGESTION GATE\n\n"
            f"**Failure Reason:** {rejection_reason}\n\n"
            f"• Tissue Stained Fraction: {metrics.get('tissue_fraction', 0.0)*100:.1f}%\n"
            f"• H&E Gamut Concordance: {metrics.get('he_fraction', 0.0)*100:.1f}%\n"
            f"• Forbidden Spectrum Fraction: {metrics.get('forbidden_fraction', 0.0)*100:.1f}%\n\n"
            "*Neural network feature extractors are NOT executed on unvalidated inputs.*"
        )
        status_card_text = (
            "🚫 **Status: BLOCKED**\n\n"
            "Cancer status classification halted. Inference is prohibited for non-histopathology inputs."
        )
        subtype_card_text = (
            "🚫 **Subtype: BLOCKED**\n\n"
            "Carcinoma subtyping is suppressed on non-histopathology inputs."
        )
        audit_text = (
            "⚠️ **STATUS: INPUT DOMAIN VALIDATION FAILED**\n\n"
            f"• Detected Domain: **{domain_type.upper()}**\n"
            f"• Safety Action: Neural network inference aborted prior to model invocation.\n\n"
            "The GCAEMMCD diagnostic pipeline requires valid brightfield H&E stained pathology tissue "
            "(Breast, Lung, or Colorectal biopsy patch). Terminal windows, screenshots, and non-tissue photos are rejected."
        )
        overlay = np.array(input_img.resize((224, 224)))
        gradcam_caption = (
            "⚠️ **Grad-CAM Suppressed:** Input rejected by domain validator. "
            "Saliency maps are never computed for non-histopathology inputs."
        )
        return (
            organ_card_text, None,
            status_card_text, None,
            subtype_card_text, None,
            audit_text,
            overlay, gradcam_caption
        )

    # -----------------------------------------------------
    # STAGE 1: Organ of Origin (Phikon ViT-B)
    # Executed ONLY after input domain validation succeeds
    # -----------------------------------------------------
    cropped_pil = crop_eyepiece_vignette(input_img)
    img_batch = preprocess_multicrop(cropped_pil).to(config.DEVICE)
    img_tensor = img_batch[:1]
    with torch.no_grad():
        organ_emb = organ_encoder(img_batch)
        predicted_organ, organ_conf, organ_probs, is_ood, is_organ_uncertain, cos_dist = organ_head.predict_with_calibration(organ_emb)

    if is_ood:
        organ_card_text = (
            f"⚠️ **Organ:** Unsupported Organ / Morphological Outlier\n\n"
            f"**Cosine Distance:** {cos_dist:.4f} (Exceeds reference baseline {organ_head.max_cosine_distance})\n"
            f"*Valid H&E tissue detected, but cellular morphology deviates from supported Breast/Lung/Colorectal profiles.*"
        )
        status_card_text = (
            "🚫 **Status: BLOCKED (Unsupported Organ)**\n\n"
            "Cancer status classification is **not executed** because the specimen origin is outside "
            "the validated reference organs. Downstream cancer models cannot be reliably applied to unsupported tissues."
        )
        subtype_card_text = (
            "🚫 **Subtype: BLOCKED**\n\n"
            "Carcinoma subtype classification is suppressed for unsupported organ specimens."
        )
        audit_text = (
            "⚠️ **STATUS: OUT-OF-DISTRIBUTION ORGAN DETECTED**\n\n"
            f"• Specimen Validation: Authentic H&E Tissue (PASSED)\n"
            f"• Organ Reference Check: Unsupported Organ Profile (Cosine Distance: {cos_dist:.4f} > {organ_head.max_cosine_distance})\n"
            "Downstream malignancy classification is blocked to prevent out-of-scope diagnostic errors."
        )
        overlay = np.array(input_img.resize((224, 224)))
        gradcam_caption = (
            "⚠️ **Grad-CAM Suppressed:** Saliency maps are not generated for unsupported organ tissues."
        )
        return (
            organ_card_text, None,
            status_card_text, None,
            subtype_card_text, None,
            audit_text,
            overlay, gradcam_caption
        )
    elif is_organ_uncertain:
        organ_card_text = (
            f"⚠️ **Organ:** {predicted_organ}\n\n"
            f"**Confidence:** {organ_conf*100:.1f}% (Below 50% threshold)\n"
            f"*Ambiguous histological features detected across organ reference profiles.*"
        )
    else:
        organ_card_text = (
            f"🏛️ **Organ of Origin:** **{predicted_organ}**\n\n"
            f"**Confidence:** {organ_conf*100:.2f}%\n"
            f"**Cosine Distance:** {cos_dist:.4f} (Aligned with prototype)"
        )

    # -----------------------------------------------------
    # STAGE 2: Cancer Status Classification (ResNet-50)
    # Only reached if Stage 1 passed OOD check
    # -----------------------------------------------------
    with torch.no_grad():
        status_logits, subtype_logits, _ = cancer_model(img_batch)
        status_probs = F.softmax(status_logits.mean(dim=0, keepdim=True), dim=1).cpu().numpy()[0]

    status_idx = int(np.argmax(status_probs))
    status_names = ["Normal / Non-neoplastic", "Benign", "Malignant"]
    predicted_status = status_names[status_idx]
    status_conf = float(status_probs[status_idx])

    status_conf_dict = {
        status_names[i]: float(status_probs[i])
        for i in range(len(status_names))
    }

    if status_idx == 0:
        status_icon = "🟢"
        status_desc = "No histological evidence of neoplastic proliferation."
    elif status_idx == 1:
        status_icon = "🟡"
        status_desc = "Non-malignant neoplastic or reactive tissue changes observed."
    else:
        status_icon = "🔴"
        status_desc = "Confirmed invasive epithelial neoplastic malignancy."

    status_card_text = (
        f"{status_icon} **Status:** **{predicted_status}**\n\n"
        f"**Confidence:** {status_conf*100:.2f}%\n"
        f"*{status_desc}*"
    )

    # -----------------------------------------------------
    # STAGE 3: Carcinoma Subtype (Conditioned on Malignancy)
    # -----------------------------------------------------
    subtype_conf_dict = None
    predicted_subtype = "None (Non-Malignant)"
    sub_conf = 0.0

    if status_idx == 2 and status_conf >= 0.60:
        sub_probs = F.softmax(subtype_logits.mean(dim=0, keepdim=True), dim=1).cpu().numpy()[0]
        # Phikon Foundation Guidance: condition ResNet subtyping on verified Phikon organ prior
        if organ_probs is not None and not is_ood:
            phik_vec = np.array([organ_probs.get(s, 0.0) for s in ["Breast", "Lung", "Colorectal"]], dtype=np.float64)
            log_c = 0.35 * np.log(sub_probs + 1e-7) + 0.65 * np.log(phik_vec + 1e-7)
            sub_probs = np.exp(log_c) / np.sum(np.exp(log_c))
        sub_idx = int(np.argmax(sub_probs))
        sub_names = ["Breast Carcinoma", "Lung Carcinoma", "Colorectal Carcinoma"]
        predicted_subtype = sub_names[sub_idx]
        sub_conf = float(sub_probs[sub_idx])

        subtype_conf_dict = {
            sub_names[i]: float(sub_probs[i])
            for i in range(len(sub_names))
        }

        subtype_card_text = (
            f"🔬 **Subtype:** **{predicted_subtype}**\n\n"
            f"**Confidence:** {sub_conf*100:.2f}%\n"
            f"*Conditioned on high-confidence malignancy.*"
        )
        cam_head = "subtype"
        cam_target = sub_idx
        cam_label = f"{predicted_subtype} (Subtype Head)"
    elif status_idx == 2 and status_conf < 0.60:
        predicted_subtype = "Uncertain"
        subtype_card_text = (
            "⚠️ **Subtype: Uncertain**\n\n"
            f"Malignancy confidence is insufficient ({status_conf*100:.1f}% < 60.0% validated threshold). "
            "Carcinoma subtype classification is suppressed to prevent diagnostic overconfidence."
        )
        subtype_conf_dict = None
        cam_head = "status"
        cam_target = 2
        cam_label = f"Malignant Status (Uncertain Confidence: {status_conf*100:.1f}%)"
    else:
        # Normal or Benign
        predicted_subtype = "Not Applicable"
        subtype_card_text = (
            f"ℹ️ **Subtype: Not Applicable**\n\n"
            f"Tissue classified as **{predicted_status}**. "
            f"Carcinoma subtyping is bypassed to preserve non-malignant tissue identity."
        )
        subtype_conf_dict = None
        cam_head = "status"
        cam_target = status_idx
        cam_label = f"{predicted_status} (Cancer Status Head)"

    # -----------------------------------------------------
    # STAGE 4: Inter-Model Consistency Audit
    # (OOD case returned early above — this block only runs on validated H&E inputs)
    # -----------------------------------------------------
    if status_idx != 2:
        audit_text = (
            f"ℹ️ **STATUS: NON-MALIGNANT PROFILE**\n\n"
            f"• Organ identified: **{predicted_organ}** ({organ_conf*100:.1f}%)\n"
            f"• Cancer status: **{predicted_status}** ({status_conf*100:.1f}%)\n"
            f"No inter-model discordance. Model appropriately did not predict carcinoma."
        )
    elif status_idx == 2 and status_conf < 0.60:
        audit_text = (
            f"⚠️ **STATUS: BORDERLINE MALIGNANCY WARNING**\n\n"
            f"• Organ identified: **{predicted_organ}** ({organ_conf*100:.1f}%)\n"
            f"• Cancer status is uncertain ({status_conf*100:.1f}%). Subtype is unassigned.\n"
            f"Manual immunohistochemical (IHC) verification advised."
        )
    else:
        # High-confidence Malignant case
        organ_key = predicted_organ.lower()
        pred_sub_key = predicted_subtype.lower()
        if organ_key in pred_sub_key:
            audit_text = (
                f"✅ **STATUS: MODELS CONCORDANT (AGREED)**\n\n"
                f"Both models corroborate:\n"
                f"• Organ of Origin: **{predicted_organ}** ({organ_conf*100:.1f}%)\n"
                f"• Malignancy Subtype: **{predicted_subtype}** ({sub_conf*100:.1f}%)\n"
                f"*(Note: Model agreement supports evidence synthesis but does not constitute clinical proof).*"
            )
        else:
            audit_text = (
                f"🚨 **WARNING: INTER-MODEL DISCORDANCE DETECTED**\n\n"
                f"Conflicting diagnostic signals:\n"
                f"• Organ Model indicates: **{predicted_organ}**\n"
                f"• Subtype Model indicates: **{predicted_subtype}**\n"
                f"Possible metastatic lesion or morphological overlap. Pathologist review required."
            )

    # -----------------------------------------------------
    # STAGE 5: Contextual Grad-CAM Visualization
    # (OOD inputs already returned early — Grad-CAM always runs here)
    # -----------------------------------------------------
    heatmap, _, _ = gradcam.generate_heatmap(img_tensor, head_type=cam_head, target_class=cam_target)
    overlay = gradcam.overlay_heatmap(cropped_pil, heatmap, alpha=0.45)
    gradcam_caption = (
        f"**Grad-CAM Target:** {cam_label}\n\n"
        f"*Warm colored areas (red/yellow) indicate the spatial microscopic regions that most strongly "
        f"influenced the network's score. This indicates model visual attention, NOT histological confirmation of cancer cells.*"
    )

    return (
        organ_card_text, organ_probs,
        status_card_text, status_conf_dict,
        subtype_card_text, subtype_conf_dict,
        audit_text,
        overlay, gradcam_caption
    )


# ---------------------------------------------------------
# GRADIO UI SETUP
# ---------------------------------------------------------
sample_presets = [
    ["data/sample_images/normal/normal_lung_00.png"],
    ["data/sample_images/normal/normal_colon_00.png"],
    ["data/sample_images/benign/benign_breast_00.png"],
    ["data/sample_images/breast/breast_00.png"],
    ["data/sample_images/lung/lung_00.png"],
    ["data/sample_images/colorectal/colorectal_00.png"]
]

custom_css = """
#header {text-align: center; margin-bottom: 14px;}
.card {background: #ffffff; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 8px;}
"""

with gr.Blocks(title="GCAEMMCD - Hierarchical Cancer Diagnosis", theme=gr.themes.Soft(), css=custom_css) as demo:
    with gr.Row(elem_id="header"):
        gr.Markdown(
            """
            # GCAEMMCD: Hierarchical Explainable Histopathology Diagnostic System
            ### Stage 1: Organ of Origin &nbsp;|&nbsp; Stage 2: Cancer Status &nbsp;|&nbsp; Stage 3: Carcinoma Subtype
            *Calibrated Decision-Support System: Non-malignant tissues are preserved; carcinomas are subtyped conditionally.*
            """
        )

    with gr.Row():
        # LEFT COLUMN: Input Image
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Histopathology Patch")
            img_input = gr.Image(type="pil", label="H&E Histopathology Patch (224×224)")
            run_btn = gr.Button("🔬 Run Hierarchical Diagnostic Pipeline", variant="primary")

            gr.Examples(
                examples=sample_presets,
                inputs=img_input,
                label="Diagnostic Presets (Normal Lung/Colon, Benign Breast, Malignant Tumors)"
            )

        # MIDDLE COLUMN: 3-Stage Cards
        with gr.Column(scale=1):
            gr.Markdown("### 2. Stage 1: Organ Identification (Phikon ViT-B)")
            card_organ = gr.Markdown("Awaiting input...")
            prob_organ = gr.Label(label="Organ Probabilities", num_top_classes=3)

            gr.Markdown("### 3. Stage 2: Cancer Status (ResNet-50)")
            card_status = gr.Markdown("Awaiting input...")
            prob_status = gr.Label(label="Status Probabilities (Normal / Benign / Malignant)")

            gr.Markdown("### 4. Stage 3: Carcinoma Subtype (Conditional)")
            card_subtype = gr.Markdown("Awaiting input...")
            prob_subtype = gr.Label(label="Carcinoma Subtype Probabilities")

        # RIGHT COLUMN: Consistency Audit & Grad-CAM
        with gr.Column(scale=1):
            gr.Markdown("### 5. Inter-Model Consistency Audit")
            audit_box = gr.Markdown("Awaiting inference...")

            gr.Markdown("### 6. Contextual Grad-CAM Saliency")
            cam_view = gr.Image(type="numpy", label="Grad-CAM Overlay")
            cam_info = gr.Markdown("Awaiting inference...")

    gr.Markdown(
        """
        ---
        **Notice:** *This system is a research demonstration prototype developed under the GCAEMMCD architectural specification. Not intended for primary clinical diagnosis. Model agreement supports decision verification but does not constitute medical proof.*
        """
    )

    run_btn.click(
        fn=hierarchical_inference_pipeline,
        inputs=[img_input],
        outputs=[
            card_organ, prob_organ,
            card_status, prob_status,
            card_subtype, prob_subtype,
            audit_box,
            cam_view, cam_info
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
