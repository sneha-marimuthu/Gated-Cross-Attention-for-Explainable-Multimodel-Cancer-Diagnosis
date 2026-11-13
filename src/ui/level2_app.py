"""
GCAEMMCD Level 2: Scientifically Hardened Multimodal Diagnostic & Triage Workbench.
Enforces:
  1. Explicit 4-State Modality Availability Detection (PRESENT_VALID, PRESENT_INVALID, MISSING, PREPROCESSING_FAILED).
  2. Strict Separation of Tissue-Verified Diagnosis (Image present) vs Clinical Risk/Triage Only (Patient data without biopsy).
  3. Dual-Task Prediction (Disease Status: Normal/Benign/Malignant + Conditional Carcinoma Subtype).
  4. Inter-Modal Patient Identity, Age/Sex, Negation & Cross-Organ Concordance Auditing.
  5. Faithful Level 2 Grad-CAM (backpropagated directly through Level 2's active GCA or Image head to ResNet-50 layer4).
  6. Unbiased UI controls supporting 'Unknown / Not Tested' biomarkers and physiological sanity bounds.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gradio as gr

import config
from src.data.image_transforms import preprocess_image, preprocess_multicrop, crop_eyepiece_vignette
from src.data.multimodal_dataset import BIOMARKER_NAMES, CLINICAL_NARRATIVES, generate_biomarker_vector
from src.models.calibrated_organ_model import CalibratedPhikonOrganEncoder, CalibratedOrganHead
from src.models.text_encoder import BioBERTTextEncoder, TextProjectionHead
from src.models.tabular_encoder import TabularBiomarkerMLP
from src.models.multimodal_classifier import UnifiedMultimodalClassifier
from src.models.modality_detector import (
    STATE_MISSING,
    STATE_PRESENT_INVALID,
    STATE_PRESENT_VALID,
    STATE_PREPROC_FAILED,
    detect_image_modality,
    detect_text_modality,
    detect_tabular_modality,
    verify_cross_modal_concordance,
)
from src.training.train_multimodal import load_calibrated_image_extractor
from src.explainability.level2_gradcam import Level2FaithfulGradCAM

print(f"[*] Initializing Hardened GCAEMMCD Level 2 System on {config.DEVICE}...")

# 1. Calibrated Histopathology ResNet-50 Extractor (Level 1 layer4 weights, frozen)
img_extractor = load_calibrated_image_extractor(config.DEVICE)

# 2. Calibrated Phikon ViT-B Organ Gatekeeper
organ_extractor = CalibratedPhikonOrganEncoder(freeze=True).to(config.DEVICE)
organ_extractor.eval()
organ_head = CalibratedOrganHead(in_features=768, num_organs=3, min_confidence=0.50, max_cosine_distance=0.58).to(config.DEVICE)
org_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "calibrated_organ_head.pt")
if os.path.exists(org_ckpt_path):
    org_ckpt = torch.load(org_ckpt_path, map_location=config.DEVICE, weights_only=False)
    organ_head.load_calibrated_checkpoint(org_ckpt, config.DEVICE)
    organ_head.eval()
    print(
        f"[+] Loaded Calibrated Phikon Multi-Prototype Gatekeeper "
        f"(prototypes={organ_head.reference_bank.size(0)}, max_cos={organ_head.max_cosine_distance:.2f})"
    )

# 3. Text & Tabular Encoders + Unified Multimodal Classifier
text_extractor = BioBERTTextEncoder(freeze=True).to(config.DEVICE)
text_extractor.eval()
text_proj = TextProjectionHead(in_features=768, out_features=256).to(config.DEVICE)
tab_mlp = TabularBiomarkerMLP(in_features=32, latent_dim=256).to(config.DEVICE)
classifier = UnifiedMultimodalClassifier(latent_dim=256, num_classes=3).to(config.DEVICE)

ckpt_path = os.path.join(config.CHECKPOINT_DIR, "multimodal_gca_model.pt")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
    text_proj.load_state_dict(ckpt["text_proj_state_dict"])
    tab_mlp.load_state_dict(ckpt["tab_mlp_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])
    print(
        f"[+] Loaded Hardened Level 2 Multimodal Checkpoint "
        f"(Val Status: {ckpt.get('val_status_acc', 0)*100:.1f}%, Val Site: {ckpt.get('val_site_acc', 0)*100:.1f}%)"
    )

text_proj.eval()
tab_mlp.eval()
classifier.eval()

# 4. Faithful Level 2 Grad-CAM bound to exact Level 2 weights
gradcam = Level2FaithfulGradCAM(img_extractor, classifier)

STATUS_NAMES = ["Normal / Non-Neoplastic", "Benign Neoplasm", "Malignant Carcinoma"]
SITE_NAMES = ["Breast", "Lung", "Colorectal"]
SUBTYPE_NAMES = ["Breast Carcinoma", "Lung Carcinoma", "Colorectal Carcinoma"]
MALIGNANCY_SUBTYPE_THRESHOLD = 0.60


def _map_radio_binary(val, pos_token):
    if val is None or "Unknown" in str(val) or "Not Tested" in str(val):
        return "Not Tested / Unknown"
    return 1.0 if str(val) == pos_token else 0.0


def _map_radio_int(val):
    if val is None or "Unknown" in str(val) or "Not Tested" in str(val):
        return "Not Tested / Unknown"
    return float(val)


def _format_hierarchical_label(status_probs, site_probs, is_tissue_verified=True):
    s_idx = int(np.argmax(status_probs))
    s_conf = float(status_probs[s_idx])
    o_idx = int(np.argmax(site_probs))
    o_conf = float(site_probs[o_idx])

    if not is_tissue_verified:
        # Patient-data-only triage mode: NEVER claim definitive carcinoma diagnosis
        return f"Triage Alignment: {STATUS_NAMES[s_idx]} ({s_conf*100:.1f}%) — Site Risk: {SITE_NAMES[o_idx]} ({o_conf*100:.1f}%)"

    if s_idx == 2:  # Malignant
        if s_conf >= MALIGNANCY_SUBTYPE_THRESHOLD:
            return f"Malignant — {SUBTYPE_NAMES[o_idx]} ({o_conf*100:.1f}%)"
        return f"Uncertain Malignancy ({s_conf*100:.1f}% < 60% threshold) — Subtype Suppressed"
    elif s_idx == 1:
        return f"Benign Lesion — {SITE_NAMES[o_idx]} Tissue ({s_conf*100:.1f}%)"
    else:
        return f"Normal Tissue — {SITE_NAMES[o_idx]} ({s_conf*100:.1f}%)"


def _state_icon(state):
    return {
        STATE_PRESENT_VALID: "🟢 PRESENT_VALID",
        STATE_PRESENT_INVALID: "🔴 PRESENT_INVALID",
        STATE_MISSING: "⚪ MISSING",
        STATE_PREPROC_FAILED: "⚠️ PREPROCESSING_FAILED",
    }.get(state, state)


def run_multimodal_pipeline(
    input_img,
    input_text,
    include_tabular,
    age, sex, smoking, ecog, cea, ca125, ca19_9, ki67,
    er, pr, her2, egfr, kras, hb, wbc, plt
):
    raw_dict = {
        "age": age,
        "sex": _map_radio_binary(sex, "Female"),
        "smoking_pack_yrs": smoking,
        "ecog_score": _map_radio_int(ecog),
        "cea_level": cea,
        "ca125_level": ca125,
        "ca19_9_level": ca19_9,
        "ki67_index": ki67,
        "er_status": _map_radio_binary(er, "Positive (+)"),
        "pr_status": _map_radio_binary(pr, "Positive (+)"),
        "her2_status": _map_radio_binary(her2, "Positive (+)"),
        "egfr_mutation": _map_radio_binary(egfr, "Mutant"),
        "kras_mutation": _map_radio_binary(kras, "Mutant"),
        "hemoglobin": hb,
        "wbc_count": wbc,
        "platelet_count": plt,
    }

    # ---------------------------------------------------------
    # STAGE 1: Explicit 4-State Availability & Domain Auditing
    # ---------------------------------------------------------
    img_info = detect_image_modality(input_img, organ_extractor=organ_extractor, organ_head=organ_head, device=config.DEVICE)
    txt_info = detect_text_modality(input_text)
    tab_info = detect_tabular_modality(include_tabular, raw_dict)

    # Format Modality Availability Banner
    phik = img_info.get("phikon")
    phik_note = ""
    if phik:
        phik_note = f" | Phikon: `{phik['predicted_organ']}` ($D_{{cos}}={phik['cosine_distance']:.4f}$)"

    t_meta = txt_info.get("metadata", {})
    txt_extra = ""
    if txt_info["state"] == STATE_PRESENT_VALID:
        cues = []
        if t_meta.get("sex"):
            cues.append(f"Sex={t_meta['sex']}")
        if t_meta.get("age") is not None:
            cues.append(f"Age={t_meta['age']}")
        if t_meta.get("has_negation"):
            cues.append("⚠️ Negation detected")
        if cues:
            txt_extra = " | Parsed: `" + ", ".join(cues) + "`"

    badge_lines = [
        "### 🔍 Modality Availability & Sanity Status",
        f"- **Histopathology Image:** {_state_icon(img_info['state'])}{phik_note} — {img_info['reason']}",
        f"- **Clinical Narrative Text:** {_state_icon(txt_info['state'])}{txt_extra} — {txt_info['reason']}",
        f"- **Structured Biomarkers:** {_state_icon(tab_info['state'])} — {tab_info['reason']}",
    ]
    if tab_info.get("warnings"):
        badge_lines.append("- **Biomarker Clinical Alerts:** " + " ; ".join(tab_info["warnings"]))
    modality_badge_md = "\n".join(badge_lines)

    # Check if ANY provided modality failed validation/preprocessing
    invalid_msgs = []
    for label, info in [("Image", img_info), ("Clinical Text", txt_info), ("Biomarkers", tab_info)]:
        if info["state"] in (STATE_PRESENT_INVALID, STATE_PREPROC_FAILED):
            invalid_msgs.append(f"**{label} ({info['state']}):** {info['reason']}")

    if invalid_msgs:
        diag_card = (
            "### 🚫 EXECUTION HALTED — INVALID INPUT MODALITY DETECTED\n\n"
            "Multimodal synthesis was **not executed** because one or more submitted inputs failed domain or physiological sanity verification:\n\n"
            + "\n".join([f"- {m}" for m in invalid_msgs])
            + "\n\n*Correct or clear the flagged invalid modality before running inference.*"
        )
        ablation_md = "🚫 **Ablation Comparison Halted:** Fix or remove invalid inputs above."
        gate_md = "⚠️ **Attribution Not Computed:** Halted on invalid input."
        cam_out = None
        cam_cap = "⚠️ **Grad-CAM Suppressed:** Inference halted because input validation failed."
        return modality_badge_md, diag_card, {}, {}, ablation_md, gate_md, cam_out, cam_cap

    # Check if ALL 3 are missing
    has_i = img_info["state"] == STATE_PRESENT_VALID
    has_t = txt_info["state"] == STATE_PRESENT_VALID
    has_r = tab_info["state"] == STATE_PRESENT_VALID

    if not (has_i or has_t or has_r):
        diag_card = (
            "### ⚪ Awaiting Clinical Inputs\n\n"
            "All three modalities are currently `MISSING`. Please provide at least one valid clinical input:\n"
            "- Upload an **H&E Histopathology image** for morphological assessment, **or**\n"
            "- Enter **clinical/pathology narrative notes** for text triage, **or**\n"
            "- Check **Include Structured Clinical & Biomarker Data** and enter patient parameters."
        )
        return modality_badge_md, diag_card, {}, {}, "Awaiting valid inputs...", "Awaiting valid inputs...", None, "No image uploaded."

    # ---------------------------------------------------------
    # STAGE 2: Independent Feature Extraction for Valid Streams
    # ---------------------------------------------------------
    img_tensor = None
    cropped_pil = None
    raw_img_feats = None
    f_t = None
    tab_tensor_32 = None
    v_t = None
    v_r = None

    with torch.no_grad():
        if has_i:
            cropped_pil = crop_eyepiece_vignette(img_info["pil_image"])
            img_batch = preprocess_multicrop(cropped_pil).to(config.DEVICE)
            img_tensor = img_batch[:1]
            raw_views, _ = img_extractor(img_batch)
            raw_img_feats = raw_views.mean(dim=0, keepdim=True)  # [1, 2048]
        if has_t:
            f_t = text_extractor(txt_info["metadata"]["cleaned_text"], device=config.DEVICE)
            v_t = text_proj(f_t)  # [1, 256]
        if has_r:
            tab_tensor_32 = torch.tensor(tab_info["vec_32"], dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
            v_r = tab_mlp(tab_tensor_32)  # [1, 256]

        outputs = classifier.forward_available(v_i=raw_img_feats, v_t=v_t, v_r=v_r)

    # Extract active pathway softmax arrays
    probs = {}
    for pkey in ["image", "text", "tabular", "early", "gca"]:
        l_stat = outputs.get(f"logits_{pkey}_status")
        l_site = outputs.get(f"logits_{pkey}_site")
        if l_stat is not None and l_site is not None:
            probs[f"{pkey}_status"] = F.softmax(l_stat, dim=1).cpu().numpy()[0]
            probs[f"{pkey}_site"] = F.softmax(l_site, dim=1).cpu().numpy()[0]

    # Phikon Foundation Guidance Coupling:
    # When Phikon produces a valid in-scope organ profile, couple ResNet image-only site
    # via Bayesian opinion pool with Phikon's ViT-B organ prior to eliminate domain/stain site confusion.
    if has_i and img_info.get("phikon") and img_info["phikon"].get("prob_dict") and not img_info["phikon"].get("is_ood"):
        p_dict = img_info["phikon"]["prob_dict"]
        p_phik = np.array([p_dict.get(s, 0.0) for s in SITE_NAMES], dtype=np.float64)
        if probs.get("image_site") is not None:
            r_p = probs["image_site"]
            log_c = 0.35 * np.log(r_p + 1e-7) + 0.65 * np.log(p_phik + 1e-7)
            probs["image_site"] = np.exp(log_c) / np.sum(np.exp(log_c))

    # ---------------------------------------------------------
    # STAGE 3: Cross-Modal Patient Identity & Clinical Concordance
    # ---------------------------------------------------------
    concordance = verify_cross_modal_concordance(img_info, txt_info, tab_info, probs)

    # ---------------------------------------------------------
    # STAGE 4: Epistemic Mode Routing & Diagnostic Formatting
    # ---------------------------------------------------------
    mode = outputs["mode"]
    dict_status = {}
    dict_site = {}
    cam_overlay = None
    cam_caption = "⚠️ **Grad-CAM Not Applicable:** No valid histopathology image present."

    if mode == "MULTIMODAL_GCA":
        p_stat = probs["gca_status"]
        p_site = probs["gca_site"]
        if has_t and has_r and probs.get("text_site") is not None and probs.get("image_site") is not None:
            t_s = probs["text_site"]
            im_s = probs["image_site"]
            log_cons = 0.40 * np.log(p_site + 1e-7) + 0.35 * np.log(t_s + 1e-7) + 0.25 * np.log(im_s + 1e-7)
            p_site = np.exp(log_cons) / np.sum(np.exp(log_cons))
            probs["gca_site"] = p_site
        s_idx, s_conf = int(np.argmax(p_stat)), float(np.max(p_stat))
        o_idx, o_conf = int(np.argmax(p_site)), float(np.max(p_site))

        if concordance["is_discordant"]:
            all_alerts = concordance["identity_alerts"] + concordance["organ_alerts"] + concordance["status_alerts"]
            diag_card = (
                f"### 🚨 CROSS-MODAL DISCORDANCE — MANUAL SPECIMEN VERIFICATION REQUIRED\n\n"
                f"**Verification Status:** `{concordance['overall_status']}`\n\n"
                "**Conflicting Clinical Evidence Detected Across Modalities:**\n"
                + "\n".join([f"- {a}" for a in all_alerts])
                + "\n\n**Clinical Decision Hold:** Automated GCA carcinoma synthesis is **suppressed** because active patient data streams contradict each other. "
                "Verify patient ID, biopsy site requisition (metastatic vs primary), and pathology report pairing."
            )
            img_s_idx, img_s_conf = int(np.argmax(probs["image_status"])), float(np.max(probs["image_status"]))
            img_o_idx = int(np.argmax(probs["image_site"]))
            dict_status = {STATUS_NAMES[i]: float(probs["image_status"][i]) for i in range(3)}
            if img_s_idx == 2 and img_s_conf >= MALIGNANCY_SUBTYPE_THRESHOLD:
                dict_site = {SUBTYPE_NAMES[i]: float(probs["image_site"][i]) for i in range(3)}
                cam_head, cam_cls, cam_label = "site", img_o_idx, SUBTYPE_NAMES[img_o_idx]
            else:
                dict_site = {f"{SITE_NAMES[i]} Tissue": float(probs["image_site"][i]) for i in range(3)}
                cam_head, cam_cls, cam_label = "status", img_s_idx, STATUS_NAMES[img_s_idx]
            heatmap, _, _ = gradcam.generate_heatmap(img_tensor, pathway="image", head_type=cam_head, target_class=cam_cls)
            cam_overlay = gradcam.overlay_heatmap(cropped_pil, heatmap)
            cam_caption = (
                f"**Grad-CAM Pathway:** Unimodal ResNet Image Head (GCA Suppressed due to Discordance) | "
                f"Target (`head_image_{cam_head}`): `{cam_label}`"
            )
        else:
            dict_status = {STATUS_NAMES[i]: float(p_stat[i]) for i in range(3)}
            if s_idx == 2 and s_conf >= MALIGNANCY_SUBTYPE_THRESHOLD:
                dict_site = {SUBTYPE_NAMES[i]: float(p_site[i]) for i in range(3)}
                headline = f"🏆 **GCA Multimodal Diagnosis: Malignant — {SUBTYPE_NAMES[o_idx]}**"
                sub_line = f"• **Carcinoma Subtype Confidence:** **{o_conf*100:.2f}%** (Concordant with Phikon, BioBERT & Biomarkers)"
                cam_head, cam_cls, cam_label = "site", o_idx, SUBTYPE_NAMES[o_idx]
            elif s_idx == 2:
                dict_site = {SITE_NAMES[i]: float(p_site[i]) for i in range(3)}
                headline = f"⚠️ **Uncertain Disease Status ({s_conf*100:.1f}% < 60% Threshold)**"
                sub_line = "• Carcinoma subtype confirmation suppressed pending higher morphological certainty or specialist review."
                cam_head, cam_cls, cam_label = "status", s_idx, STATUS_NAMES[s_idx]
            else:
                dict_site = {f"{SITE_NAMES[i]} Tissue": float(p_site[i]) for i in range(3)}
                headline = f"✅ **Non-Malignant Finding: {STATUS_NAMES[s_idx]} ({SITE_NAMES[o_idx]} Tissue)**"
                sub_line = "• **Carcinoma Subtyping Bypassed:** Non-malignant specimens are never forced into carcinoma classes."
                cam_head, cam_cls, cam_label = "status", s_idx, STATUS_NAMES[s_idx]

            diag_card = (
                f"### 🔬 Prediction Mode: Tri-Modal Gated Cross-Attention (Concordant)\n\n"
                f"{headline}\n\n"
                f"• **Disease Status Confidence:** **{s_conf*100:.2f}%** (`{STATUS_NAMES[s_idx]}`)\n"
                f"{sub_line}\n"
                f"• **Cross-Modal Integrity:** Patient identity, site, and status verified concordant."
            )

            heatmap, _, _ = gradcam.generate_heatmap(
                img_tensor, pathway="gca", head_type=cam_head, target_class=cam_cls, v_t=v_t, v_r=v_r
            )
            cam_overlay = gradcam.overlay_heatmap(cropped_pil, heatmap)
            cam_caption = (
                f"**Faithful GCA Grad-CAM Target:** `{cam_label}` (`gca_fusion -> head_gca_{cam_head}` backpropagated to ResNet `layer4`)"
            )

    elif mode == "UNIMODAL_IMAGE":
        p_stat, p_site = probs["image_status"], probs["image_site"]
        s_idx, s_conf = int(np.argmax(p_stat)), float(np.max(p_stat))
        o_idx, o_conf = int(np.argmax(p_site)), float(np.max(p_site))
        dict_status = {STATUS_NAMES[i]: float(p_stat[i]) for i in range(3)}

        if s_idx == 2 and s_conf >= MALIGNANCY_SUBTYPE_THRESHOLD:
            dict_site = {SUBTYPE_NAMES[i]: float(p_site[i]) for i in range(3)}
            headline = f"🖼️ **Image-Only Diagnosis: Malignant — {SUBTYPE_NAMES[o_idx]}**"
            cam_head, cam_cls, cam_label = "site", o_idx, SUBTYPE_NAMES[o_idx]
        elif s_idx == 2:
            dict_site = {SITE_NAMES[i]: float(p_site[i]) for i in range(3)}
            headline = f"⚠️ **Image-Only Status Uncertain ({s_conf*100:.1f}% < 60%) — Subtype Suppressed**"
            cam_head, cam_cls, cam_label = "status", s_idx, STATUS_NAMES[s_idx]
        else:
            dict_site = {f"{SITE_NAMES[i]} Tissue": float(p_site[i]) for i in range(3)}
            headline = f"✅ **Image-Only Non-Malignant Finding: {STATUS_NAMES[s_idx]} ({SITE_NAMES[o_idx]})**"
            cam_head, cam_cls, cam_label = "status", s_idx, STATUS_NAMES[s_idx]

        diag_card = (
            f"### 🖼️ Prediction Mode: Unimodal Histopathology Image Only\n\n"
            f"{headline}\n\n"
            f"• **Status Confidence:** **{s_conf*100:.2f}%** | **Site/Subtype Confidence:** **{o_conf*100:.2f}%**\n"
            f"• **Multimodal Notice:** Clinical Text and Tabular Biomarkers are `MISSING`; GCA cross-attention was **not executed**."
        )
        heatmap, _, _ = gradcam.generate_heatmap(img_tensor, pathway="image", head_type=cam_head, target_class=cam_cls)
        cam_overlay = gradcam.overlay_heatmap(cropped_pil, heatmap)
        cam_caption = f"**Faithful Unimodal Grad-CAM Target:** `{cam_label}` (`head_image_{cam_head}` -> ResNet `layer4`)"

    elif mode in ("UNIMODAL_TEXT", "UNIMODAL_TABULAR"):
        pkey = "text" if mode == "UNIMODAL_TEXT" else "tabular"
        p_stat, p_site = probs[f"{pkey}_status"], probs[f"{pkey}_site"]
        s_idx, s_conf = int(np.argmax(p_stat)), float(np.max(p_stat))
        o_idx, o_conf = int(np.argmax(p_site)), float(np.max(p_site))
        dict_status = {f"Risk Profile: {STATUS_NAMES[i]}": float(p_stat[i]) for i in range(3)}
        dict_site = {f"Site Alignment: {SITE_NAMES[i]}": float(p_site[i]) for i in range(3)}

        stream_label = "BioBERT Clinical Narrative" if pkey == "text" else "16-Parameter Clinical Biomarkers"
        neg_warn = ""
        if pkey == "text" and txt_info["metadata"].get("has_negation") and s_idx == 2:
            neg_warn = "\n- ⚠️ **Clinical Negation Detected:** Text contains exclusion/negative phrasing; interpret malignancy risk cautiously."

        diag_card = (
            f"### 📋 Prediction Mode: Clinical Risk & Organ Triage ONLY ({stream_label})\n\n"
            f"**Triage Risk Alignment:** `{STATUS_NAMES[s_idx]}` (**{s_conf*100:.1f}%**) | "
            f"**Associated Organ Profile:** `{SITE_NAMES[o_idx]}` (**{o_conf*100:.1f}%**)\n"
            f"{neg_warn}\n\n"
            "> **Clinical Reality Guardrail:** *No histopathology tissue specimen (`Image = MISSING`). "
            "Demographics, serum tumor markers, CBC labs, or pre-biopsy clinical notes alone **cannot definitively confirm or rule out cancer**. "
            "Treat output strictly as referral triage pending biopsy histology.*"
        )

    else:  # PARTIAL_DUAL_UNSUPPORTED
        all_alerts = concordance["identity_alerts"] + concordance["organ_alerts"] + concordance["status_alerts"]
        alert_block = (
            "\n\n**🚨 Cross-Modal Concordance Alerts:**\n" + "\n".join([f"- {a}" for a in all_alerts])
            if all_alerts
            else "\n\n**Concordance Check:** The two active modalities are mutually consistent."
        )
        active_desc = []
        if has_i:
            active_desc.append(f"Image: `{_format_hierarchical_label(probs['image_status'], probs['image_site'], True)}`")
        if has_t:
            active_desc.append(f"Text Triage: `{_format_hierarchical_label(probs['text_status'], probs['text_site'], False)}`")
        if has_r:
            active_desc.append(f"Tabular Triage: `{_format_hierarchical_label(probs['tabular_status'], probs['tabular_site'], False)}`")

        diag_card = (
            f"### ⚠️ Insufficient Modality Support for 3-Way GCA Fusion\n\n"
            f"**Fusion Verdict:** {outputs.get('fusion_message')}\n\n"
            f"**Independent Active Pathways:**\n"
            + "\n".join([f"- {d}" for d in active_desc])
            + alert_block
        )

        # Populate labels from Image if present, else highest-priority triage stream
        if has_i:
            dict_status = {STATUS_NAMES[i]: float(probs["image_status"][i]) for i in range(3)}
            s_i = int(np.argmax(probs["image_status"]))
            dict_site = {
                (SUBTYPE_NAMES[i] if s_i == 2 else f"{SITE_NAMES[i]} Tissue"): float(probs["image_site"][i])
                for i in range(3)
            }
            heatmap, _, _ = gradcam.generate_heatmap(
                img_tensor, pathway="image", head_type=("site" if s_i == 2 else "status")
            )
            cam_overlay = gradcam.overlay_heatmap(cropped_pil, heatmap)
            cam_caption = "**Faithful Unimodal Grad-CAM:** Generated strictly from ResNet Image Head (3-Way GCA inactive)."
        else:
            p_stat = probs["text_status"]
            p_site = probs["text_site"]
            dict_status = {f"Text Risk: {STATUS_NAMES[i]}": float(p_stat[i]) for i in range(3)}
            dict_site = {f"Text Organ Alignment: {SITE_NAMES[i]}": float(p_site[i]) for i in range(3)}

    # ---------------------------------------------------------
    # STAGE 5: Honest Pathway & Ablation Comparison Table
    # ---------------------------------------------------------
    ab_rows = [
        "### 📊 Executed Modality Pathway Audit",
        "| Pathway | Execution State | Predicted Outcome | Status / Site Confidence |",
        "| :--- | :--- | :--- | :--- |",
    ]
    pathway_defs = [
        ("gca", "🔬 GCA Multimodal Fusion", mode == "MULTIMODAL_GCA", has_i),
        ("early", "🔗 Early Concatenation", mode == "MULTIMODAL_GCA", has_i),
        ("image", "🖼️ Image-Only (ResNet-50)", has_i, True),
        ("text", "📝 Text-Only (BioBERT Triage)", has_t, False),
        ("tabular", "🧪 Tabular-Only (Masked Biomarkers)", has_r, False),
    ]
    for pkey, plabel, is_active, is_tissue in pathway_defs:
        if not is_active:
            reason_off = (
                "Insufficient modality support (requires all 3 valid)"
                if pkey in ("gca", "early") and (int(has_i) + int(has_t) + int(has_r)) == 2
                else "MISSING / Not Executed"
            )
            ab_rows.append(f"| {plabel} | ⚪ `{reason_off}` | — | — |")
        else:
            if pkey == "gca" and concordance["is_discordant"]:
                ab_rows.append(
                    f"| {plabel} | 🟡 `Held (Discordant Inputs)` | Suppressed for manual review | JS Div: {concordance['max_js_organ']:.2f} |"
                )
            else:
                st_p, si_p = probs[f"{pkey}_status"], probs[f"{pkey}_site"]
                desc = _format_hierarchical_label(st_p, si_p, is_tissue_verified=is_tissue)
                ab_rows.append(
                    f"| {plabel} | 🟢 `Executed` | {desc} | Status {np.max(st_p)*100:.1f}% / Site {np.max(si_p)*100:.1f}% |"
                )
    ablation_md = "\n".join(ab_rows)

    # ---------------------------------------------------------
    # STAGE 6: Real Gate Activations & Tabular Feature Attribution
    # ---------------------------------------------------------
    gate_sections = []
    if outputs.get("gate_weights") is not None:
        gw = outputs["gate_weights"]
        g_vals = {
            "Image → Text (g_IT)": float(gw["image_to_text"][0].item()),
            "Text → Image (g_TI)": float(gw["text_to_image"][0].item()),
            "Image → Tabular (g_IR)": float(gw["image_to_tabular"][0].item()),
            "Tabular → Image (g_RI)": float(gw["tabular_to_image"][0].item()),
            "Text → Tabular (g_TR)": float(gw["text_to_tabular"][0].item()),
            "Tabular → Text (g_RT)": float(gw["tabular_to_text"][0].item()),
        }
        gate_sections.append("### 🎛️ Dynamic Sigmoid Attention Gates $g = \\sigma(\\text{GateNet}[X \\parallel Y])$\n")
        for gname, gval in g_vals.items():
            tag = "🟢 Active Consensus" if gval >= 0.50 else "🟡 Attenuated / Suppressed"
            gate_sections.append(f"- **{gname}:** **{gval*100:.1f}%** ({tag})")
    else:
        gate_sections.append(
            "### 🎛️ Inter-Modal Attention Gates\n"
            "*Not computed — dynamic bidirectional cross-attention requires all 3 validated modalities simultaneously.*"
        )

    if has_r and tab_tensor_32 is not None:
        ranked_feats = gradcam.compute_tabular_saliency(tab_mlp, classifier, tab_tensor_32, head_type="site")
        obs_cnt = tab_info["metadata"]["observed_count"]
        miss_list = tab_info["metadata"]["missing_fields"]
        gate_sections.append(f"\n### 🧬 Observed Biomarker Gradient Attribution ({obs_cnt}/16 Observed)")
        for idx, imp in ranked_feats[:5]:
            gate_sections.append(f"- `{BIOMARKER_NAMES[idx]}` (val=`{tab_info['raw_vec'][idx]:.2f}`): **{imp*100:.1f}%** relative influence")
        if miss_list:
            gate_sections.append(f"- *Masked (`Not Tested / Unknown`, excluded from attribution):* `{', '.join(miss_list)}`")

    gate_md = "\n".join(gate_sections)

    return (
        modality_badge_md,
        diag_card,
        dict_status,
        dict_site,
        ablation_md,
        gate_md,
        cam_overlay,
        cam_caption,
    )


# ---------------------------------------------------------
# GRADIO INTERFACE
# ---------------------------------------------------------
custom_css = """
#header {text-align: center; margin-bottom: 12px;}
"""

with gr.Blocks(title="GCAEMMCD Level 2 - Hardened Multimodal Workbench", theme=gr.themes.Soft(), css=custom_css) as demo:
    with gr.Row(elem_id="header"):
        gr.Markdown(
            """
            # GCAEMMCD Level 2: Scientifically Hardened Multimodal Diagnostic & Triage System
            ### Availability-Aware Gated Cross-Attention • Patient Identity Concordance • Faithful Level 2 Grad-CAM
            *Strict epistemic boundaries: Unimodal patient data provides Clinical Triage Risk Only; definitive cancer confirmation requires verified histopathology.*
            """
        )

    with gr.Row():
        # LEFT COLUMN: Modalities
        with gr.Column(scale=1):
            gr.Markdown("### 1. Histopathology Image Modality")
            img_input = gr.Image(type="pil", label="H&E Stained Patch (Leave empty to run Text/Tabular only)")

            gr.Markdown("### 2. Clinical Narrative Text Modality")
            text_input = gr.Textbox(
                label="Clinical / Pathology Notes (Optional — leave blank if absent)",
                placeholder="Enter synchronized clinical vignette or biopsy note (e.g., age, sex, symptoms/findings)...",
                lines=4,
            )

            gr.Markdown("### 3. Structured Biomarkers (16 Parameters)")
            include_tabular = gr.Checkbox(
                value=False,
                label="Include Clinical & Biomarker Data (Check to activate; when unchecked, controls below are ignored)"
            )
            with gr.Accordion("Expand Clinical & Biomarker Panel", open=False):
                with gr.Row():
                    age_input = gr.Number(value=None, label="Age (Years, ref 18-90)")
                    sex_input = gr.Radio(["Unknown / Not Tested", "Female", "Male"], value="Unknown / Not Tested", label="Sex")
                with gr.Row():
                    smoking_input = gr.Number(value=0.0, label="Smoking (Pack-Years)")
                    ecog_input = gr.Radio(["Unknown / Not Tested", "0", "1", "2", "3"], value="Unknown / Not Tested", label="ECOG Status")
                with gr.Row():
                    cea_input = gr.Number(value=None, label="CEA (ng/mL, blank = Not Tested)")
                    ca125_input = gr.Number(value=None, label="CA-125 (U/mL, blank = Not Tested)")
                    ca19_9_input = gr.Number(value=None, label="CA19-9 (U/mL, blank = Not Tested)")
                with gr.Row():
                    ki67_input = gr.Number(value=None, label="Ki-67 % (blank = Not Tested)")
                    er_input = gr.Radio(["Unknown / Not Tested", "Negative (-)", "Positive (+)"], value="Unknown / Not Tested", label="ER")
                    pr_input = gr.Radio(["Unknown / Not Tested", "Negative (-)", "Positive (+)"], value="Unknown / Not Tested", label="PR")
                with gr.Row():
                    her2_input = gr.Radio(["Unknown / Not Tested", "Negative (-)", "Positive (+)"], value="Unknown / Not Tested", label="HER2")
                    egfr_input = gr.Radio(["Unknown / Not Tested", "Wild-type", "Mutant"], value="Unknown / Not Tested", label="EGFR")
                    kras_input = gr.Radio(["Unknown / Not Tested", "Wild-type", "Mutant"], value="Unknown / Not Tested", label="KRAS")
                with gr.Row():
                    hb_input = gr.Number(value=None, label="Hemoglobin (g/dL, ref 11-16)")
                    wbc_input = gr.Number(value=None, label="WBC (x10^3/uL, ref 4-11)")
                    plt_input = gr.Number(value=None, label="Platelets (x10^3/uL, ref 150-400)")

            with gr.Row():
                run_btn = gr.Button("🚀 Run Verified Multimodal Pipeline", variant="primary")
                clear_btn = gr.Button("🧹 Clear All Inputs")

        # MIDDLE COLUMN: Modality Status, Diagnostic Card, Ablations
        with gr.Column(scale=1):
            modality_badge_box = gr.Markdown("Awaiting inference...")
            diag_card = gr.Markdown("Awaiting inference...")
            prob_status = gr.Label(label="Disease Status Probabilities (Normal / Benign / Malignant)", num_top_classes=3)
            prob_site = gr.Label(label="Anatomical Site / Carcinoma Probabilities (Breast / Lung / Colorectal)", num_top_classes=3)
            ablation_box = gr.Markdown("Awaiting inference...")

        # RIGHT COLUMN: Gates, Tabular Attribution & Faithful Grad-CAM
        with gr.Column(scale=1):
            gate_box = gr.Markdown("Awaiting inference...")
            gr.Markdown("### 7. Faithful Level 2 Grad-CAM")
            cam_view = gr.Image(type="numpy", label="Grad-CAM Overlay")
            cam_info = gr.Markdown("Awaiting inference...")

    all_biomarkers = [
        age_input, sex_input, smoking_input, ecog_input,
        cea_input, ca125_input, ca19_9_input, ki67_input,
        er_input, pr_input, her2_input, egfr_input, kras_input,
        hb_input, wbc_input, plt_input,
    ]

    def load_preset(cat_key):
        vec = generate_biomarker_vector(cat_key, seed=42)
        if cat_key == "breast_malignant":
            img = os.path.join(config.IMAGE_DIR, "breast", "breast_00.png")
            txt = CLINICAL_NARRATIVES["breast_malignant"]
            vec[0], vec[1], vec[2], vec[3] = 58.0, 1.0, 0.0, 1.0
        elif cat_key == "lung_malignant":
            img = os.path.join(config.IMAGE_DIR, "lung", "lung_00.png")
            txt = CLINICAL_NARRATIVES["lung_malignant"]
            vec[0], vec[1], vec[2], vec[3] = 68.0, 0.0, 35.0, 1.0
        elif cat_key == "colorectal_malignant":
            img = os.path.join(config.IMAGE_DIR, "colorectal", "colorectal_00.png")
            txt = CLINICAL_NARRATIVES["colorectal_malignant"]
            vec[0], vec[1], vec[2], vec[3] = 64.0, 0.0, 10.0, 1.0
        elif cat_key == "breast_benign":
            img = os.path.join(config.IMAGE_DIR, "benign", "benign_breast_00.png")
            txt = CLINICAL_NARRATIVES["breast_benign"]
            vec[0], vec[1], vec[2], vec[3] = 34.0, 1.0, 0.0, 0.0
        elif cat_key == "lung_normal":
            img = os.path.join(config.IMAGE_DIR, "normal", "normal_lung_00.png")
            txt = CLINICAL_NARRATIVES["lung_normal"]
            vec[0], vec[1], vec[2], vec[3] = 46.0, 0.0, 0.0, 0.0
        elif cat_key == "mismatch_demo":
            # Deliberate specimen swap & identity mismatch test!
            img = os.path.join(config.IMAGE_DIR, "colorectal", "colorectal_00.png")
            txt = CLINICAL_NARRATIVES["breast_malignant"]  # 58yo female breast carcinoma
            vec = generate_biomarker_vector("lung_malignant", seed=99)
            vec[0], vec[1], vec[2], vec[3] = 72.0, 0.0, 42.0, 2.0  # 72yo male smoker lung profile
        else:
            return clear_all()

        return [
            img,
            txt,
            True,
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

    def clear_all():
        return [
            None,
            "",
            False,
            None,
            "Unknown / Not Tested",
            0.0,
            "Unknown / Not Tested",
            None,
            None,
            None,
            None,
            "Unknown / Not Tested",
            "Unknown / Not Tested",
            "Unknown / Not Tested",
            "Unknown / Not Tested",
            "Unknown / Not Tested",
            None,
            None,
            None,
        ]

    with gr.Row():
        gr.Markdown("### 🌟 Clinical Test Cases & Verification Presets")
    with gr.Row():
        btn_br = gr.Button("📋 Breast Malignant")
        btn_lu = gr.Button("📋 Lung Malignant")
        btn_co = gr.Button("📋 Colorectal Malignant")
        btn_ben = gr.Button("📋 Benign Breast Fibroadenoma")
        btn_norm = gr.Button("📋 Normal Lung Parenchyma")
        btn_mis = gr.Button("⚠️ Patient/Organ Mismatch Test")

    preset_outputs = [img_input, text_input, include_tabular] + all_biomarkers
    btn_br.click(fn=lambda: load_preset("breast_malignant"), outputs=preset_outputs)
    btn_lu.click(fn=lambda: load_preset("lung_malignant"), outputs=preset_outputs)
    btn_co.click(fn=lambda: load_preset("colorectal_malignant"), outputs=preset_outputs)
    btn_ben.click(fn=lambda: load_preset("breast_benign"), outputs=preset_outputs)
    btn_norm.click(fn=lambda: load_preset("lung_normal"), outputs=preset_outputs)
    btn_mis.click(fn=lambda: load_preset("mismatch_demo"), outputs=preset_outputs)
    clear_btn.click(fn=clear_all, outputs=preset_outputs)

    run_btn.click(
        fn=run_multimodal_pipeline,
        inputs=[img_input, text_input, include_tabular] + all_biomarkers,
        outputs=[
            modality_badge_box,
            diag_card,
            prob_status,
            prob_site,
            ablation_box,
            gate_box,
            cam_view,
            cam_info,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
