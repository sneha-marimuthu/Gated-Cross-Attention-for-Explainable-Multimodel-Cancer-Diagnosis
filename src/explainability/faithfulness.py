"""
Quantitative Explainability Faithfulness Evaluation Engine.
Evaluates the causal fidelity of XAI heatmaps and tabular attributions:
  1. Visual Saliency Masking (Top 20% Salient Region Masking vs Baseline Confidence Drop)
  2. Background Control Masking (Bottom 20% Non-Salient Region Stability)
  3. Tabular Feature Ablation (Top 3 SHAP Features Dropped)
  4. SHAP Local Additivity Efficiency Residual
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

import config
from src.data.image_transforms import preprocess_multicrop


def evaluate_visual_faithfulness(
    pil_img,
    heatmap_2d,
    img_extractor,
    classifier,
    target_class_idx,
    head_type="site",
    device=config.DEVICE
):
    """
    Evaluates whether the Grad-CAM heatmap identifies genuinely causal visual regions:
      - Masks out top 20% most salient pixels with Gaussian blur -> Expect significant confidence DROP.
      - Masks out bottom 20% least salient pixels (control) -> Expect confidence to remain STABLE.
    """
    w, h = pil_img.size
    heatmap_norm = np.asarray(heatmap_2d, dtype=np.float32)
    if heatmap_norm.shape != (h, w):
        im_h = Image.fromarray((heatmap_norm * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        heatmap_norm = np.asarray(im_h, dtype=np.float32) / 255.0

    # 1. Unmasked Baseline Inference
    with torch.no_grad():
        batch_orig = preprocess_multicrop(pil_img).to(device)
        raw_views, _ = img_extractor(batch_orig)
        feat_orig = raw_views.mean(dim=0, keepdim=True)
        out_orig = classifier.forward_available(v_i=feat_orig)
        logits_orig = out_orig[f"logits_image_{head_type}"]
        probs_orig = F.softmax(logits_orig, dim=1)[0].cpu().numpy()
        orig_conf = float(probs_orig[target_class_idx])

    # 2. Mask Top 20% Salient Pixels (Causal Perturbation)
    threshold_top20 = np.percentile(heatmap_norm, 80.0)
    mask_top20 = heatmap_norm >= threshold_top20  # [H, W] bool

    blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=15))
    arr_orig = np.array(pil_img)
    arr_blur = np.array(blurred_pil)

    arr_masked_salient = arr_orig.copy()
    arr_masked_salient[mask_top20] = arr_blur[mask_top20]
    img_masked_salient = Image.fromarray(arr_masked_salient)

    with torch.no_grad():
        batch_mask = preprocess_multicrop(img_masked_salient).to(device)
        raw_v_mask, _ = img_extractor(batch_mask)
        feat_mask = raw_v_mask.mean(dim=0, keepdim=True)
        out_mask = classifier.forward_available(v_i=feat_mask)
        logits_mask = out_mask[f"logits_image_{head_type}"]
        probs_mask = F.softmax(logits_mask, dim=1)[0].cpu().numpy()
        masked_conf = float(probs_mask[target_class_idx])

    salient_conf_drop = float(orig_conf - masked_conf)

    # 3. Mask Bottom 20% Non-Salient Pixels (Negative Control Perturbation)
    threshold_bot20 = np.percentile(heatmap_norm, 20.0)
    mask_bot20 = heatmap_norm <= threshold_bot20

    arr_masked_control = arr_orig.copy()
    arr_masked_control[mask_bot20] = arr_blur[mask_bot20]
    img_masked_control = Image.fromarray(arr_masked_control)

    with torch.no_grad():
        batch_ctrl = preprocess_multicrop(img_masked_control).to(device)
        raw_v_ctrl, _ = img_extractor(batch_ctrl)
        feat_ctrl = raw_v_ctrl.mean(dim=0, keepdim=True)
        out_ctrl = classifier.forward_available(v_i=feat_ctrl)
        logits_ctrl = out_ctrl[f"logits_image_{head_type}"]
        probs_ctrl = F.softmax(logits_ctrl, dim=1)[0].cpu().numpy()
        control_conf = float(probs_ctrl[target_class_idx])

    control_conf_drop = float(orig_conf - control_conf)

    is_faithful = (salient_conf_drop > 0.05) or (salient_conf_drop > control_conf_drop)

    return {
        "original_confidence": orig_conf,
        "masked_confidence": masked_conf,
        "salient_confidence_drop": salient_conf_drop,
        "control_confidence_drop": control_conf_drop,
        "is_faithful": is_faithful,
        "img_masked_salient": img_masked_salient,
        "fraction_masked": float(np.mean(mask_top20)),
    }


def evaluate_tabular_faithfulness(
    raw_vec_16,
    shap_explainer,
    target_class_idx,
    head_type="site"
):
    """
    Evaluates faithfulness of tabular SHAP values by zeroing/ablating top 3 features.
    """
    res = shap_explainer.explain_case(raw_vec_16, target_class_idx=target_class_idx, head_type=head_type, nsamples=60)
    phi = res["shap_values"]
    top3_idx = np.argsort(np.abs(phi))[-3:]

    # Ablate top 3 features (set to median background)
    ablated_vec = np.array(raw_vec_16, dtype=np.float32)
    bg_medians = np.median(shap_explainer.bg_16, axis=0)
    for idx in top3_idx:
        ablated_vec[idx] = bg_medians[idx]

    prob_orig = res["target_prob"]
    prob_ablated = float(shap_explainer._predict_batch(ablated_vec.reshape(1, -1), head_type=head_type)[0, target_class_idx])
    drop = float(prob_orig - prob_ablated)

    return {
        "shap_result": res,
        "top3_indices": top3_idx.tolist(),
        "original_confidence": prob_orig,
        "ablated_confidence": prob_ablated,
        "tabular_confidence_drop": drop,
        "efficiency_residual": res["efficiency_residual"],
        "is_locally_efficient": res["efficiency_residual"] < 1e-3,
    }
