import json, os, torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import config
from src.data.image_transforms import preprocess_multicrop, crop_eyepiece_vignette
from src.ui.level2_app import organ_extractor, organ_head, img_extractor, classifier

manifest_path = "data/external_validation/provenance_manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

organs = ["Breast", "Lung", "Colorectal"]

for beta in [0.5, 1.0, 1.5, 2.0]:
    print(f"\n--- LOGIT COUPLING WITH BETA = {beta} ---")
    print(f"{'FILE':<28} | {'EXPECTED':<12} | {'PREDICTED':<12} | {'CONF':<6} | {'PROBS (B/L/C)'}")
    print("-" * 75)
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
            p_logits = organ_head(p_feats).mean(dim=0, keepdim=True)
            
            raw_views, _ = img_extractor(batch)
            raw_feat = raw_views.mean(dim=0, keepdim=True)
            out = classifier.forward_available(v_i=raw_feat)
            r_logits = out["logits_image_site"]
            
            coupled_logits = r_logits + beta * p_logits
            coupled_probs = F.softmax(coupled_logits, dim=1)[0].cpu().numpy()
            pred_idx = np.argmax(coupled_probs)
            pred_name = organs[pred_idx]
            conf = coupled_probs[pred_idx]
            prob_str = f"{coupled_probs[0]:.2f} / {coupled_probs[1]:.2f} / {coupled_probs[2]:.2f}"
            
        print(f"{fname:<28} | {str(item['expected_organ'])[:12]:<12} | {pred_name:<12} | {conf*100:5.1f}% | {prob_str}")
