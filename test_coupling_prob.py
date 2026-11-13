import json, os, torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import config
from src.data.image_transforms import preprocess_multicrop, crop_eyepiece_vignette
from src.ui.level2_app import organ_extractor, organ_head, img_extractor, classifier

# Calibrate energy threshold to -1.10
organ_head.max_energy = -1.10

manifest_path = "data/external_validation/provenance_manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

organs = ["Breast", "Lung", "Colorectal"]

for lam in [0.5, 0.6, 0.7, 0.8]:
    print(f"\n=================== PROBABILITY COUPLING WITH LAMBDA = {lam} (max_energy=-1.10) ===================")
    print(f"{'FILE':<28} | {'EXPECTED':<12} | {'RESNET ALONE':<14} | {'PHIKON':<14} | {'COUPLED':<14} | {'CONF'}")
    print("-" * 95)
    for item in manifest:
        fname = item["filename"]
        img_path = os.path.join("data/external_validation", fname)
        if not os.path.exists(img_path):
            continue
        pil = Image.open(img_path).convert("RGB")
        cropped = crop_eyepiece_vignette(pil)
        batch = preprocess_multicrop(cropped).to(config.DEVICE)
        
        with torch.no_grad():
            p_feats = organ_extractor(batch)
            pred_org, p_conf, p_prob_dict, is_ood, is_unc, d_cos = organ_head.predict_with_calibration(p_feats)
            
            raw_views, _ = img_extractor(batch)
            raw_feat = raw_views.mean(dim=0, keepdim=True)
            out = classifier.forward_available(v_i=raw_feat)
            r_probs = F.softmax(out["logits_image_site"], dim=1)[0].cpu().numpy()
            
            r_pred = organs[np.argmax(r_probs)]
            
            if p_prob_dict is not None and not is_ood:
                p_vec = np.array([p_prob_dict["Breast"], p_prob_dict["Lung"], p_prob_dict["Colorectal"]])
                log_c = (1.0 - lam) * np.log(r_probs + 1e-7) + lam * np.log(p_vec + 1e-7)
                c_probs = np.exp(log_c) / np.sum(np.exp(log_c))
                c_pred = organs[np.argmax(c_probs)]
                c_conf = np.max(c_probs)
            else:
                c_pred = f"OOD ({pred_org[:7]})" if is_ood else r_pred
                c_conf = np.max(r_probs)
                
            print(f"{fname:<28} | {str(item['expected_organ'])[:12]:<12} | {r_pred:<14} | {pred_org[:14]:<14} | {c_pred:<14} | {c_conf*100:5.1f}%")
