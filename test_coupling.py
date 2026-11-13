import json, os, torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import config
from src.data.image_transforms import preprocess_multicrop, crop_eyepiece_vignette
from src.ui.level1_app import cancer_model
from src.ui.level2_app import organ_extractor, organ_head, img_extractor, classifier

manifest_path = "data/external_validation/provenance_manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

organs = ["Breast", "Lung", "Colorectal"]

print("=" * 110)
print(f"{'FILE':<28} | {'EXPECTED':<12} | {'PHIKON PROBS (B/L/C)':<22} | {'RESNET PROBS':<20} | {'COUPLED 50/50'}")
print("=" * 110)

for item in manifest:
    fname = item["filename"]
    img_path = os.path.join("data/external_validation", fname)
    if not os.path.exists(img_path):
        continue
    pil = Image.open(img_path).convert("RGB")
    cropped = crop_eyepiece_vignette(pil)
    batch = preprocess_multicrop(cropped).to(config.DEVICE)
    
    with torch.no_grad():
        # Phikon
        p_feats = organ_extractor(batch)
        p_logits = organ_head(p_feats).mean(dim=0, keepdim=True)
        p_probs = F.softmax(p_logits, dim=1)[0].cpu().numpy()
        
        # ResNet Level 2 image-only site
        raw_views, _ = img_extractor(batch)
        raw_feat = raw_views.mean(dim=0, keepdim=True)
        out = classifier.forward_available(v_i=raw_feat)
        l_site = out["logits_image_site"]
        r_probs = F.softmax(l_site, dim=1)[0].cpu().numpy()
        
        # Coupled 50/50 geometric mean / log-prob fusion
        # log_coupled = 0.5 * log(r_probs) + 0.5 * log(p_probs)
        log_coupled = 0.5 * np.log(r_probs + 1e-7) + 0.5 * np.log(p_probs + 1e-7)
        c_probs = np.exp(log_coupled) / np.sum(np.exp(log_coupled))
        
        p_str = f"{p_probs[0]:.2f}/{p_probs[1]:.2f}/{p_probs[2]:.2f}"
        r_str = f"{r_probs[0]:.2f}/{r_probs[1]:.2f}/{r_probs[2]:.2f} -> {organs[np.argmax(r_probs)]}"
        c_str = f"{c_probs[0]:.2f}/{c_probs[1]:.2f}/{c_probs[2]:.2f} -> {organs[np.argmax(c_probs)]}"
        
        print(f"{fname:<28} | {str(item['expected_organ'])[:12]:<12} | {p_str:<22} | {r_str:<24} | {c_str}")
