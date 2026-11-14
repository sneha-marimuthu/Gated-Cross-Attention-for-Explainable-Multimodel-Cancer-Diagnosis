import json, os, sys, torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import config
from src.data.image_transforms import preprocess_multicrop
from src.ui.level2_app import organ_extractor, organ_head

ckpt = torch.load("checkpoints/calibrated_organ_head.pt", map_location=config.DEVICE, weights_only=False)
ref = organ_head.reference_bank  # [12, 768]
# Prototypes were generated with K=4 for Breast (0..3), Lung (4..7), Colorectal (8..11)
manifest_path = "data/external_validation/provenance_manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

organs = ["Breast", "Lung", "Colorectal"]

print("PROTOTYPE SIMILARITY BREAKDOWN:")
print("{:<30} | {:<10} | {:<7} | {:<7} | {:<7} | {:<10}".format("FILE", "EXPECTED", "BREAST", "LUNG", "COLON", "NEAREST"))
print("-" * 80)
for item in manifest:
    fname = item["filename"]
    img_path = os.path.join("data/external_validation", fname)
    if not os.path.exists(img_path):
        continue
    pil = Image.open(img_path).convert("RGB")
    t = preprocess_multicrop(pil).to(config.DEVICE)
    with torch.no_grad():
        feats = organ_extractor(t) # [2, 768]
        norm_emb = F.normalize(feats.mean(dim=0, keepdim=True), p=2, dim=1) # [1, 768]
        sims = torch.mm(norm_emb, ref.t())[0] # [12]
        
        sim_breast = sims[0:4].max().item()
        sim_lung = sims[4:8].max().item()
        sim_colon = sims[8:12].max().item()
        
        sim_map = {"Breast": sim_breast, "Lung": sim_lung, "Colorectal": sim_colon}
        best_org = max(sim_map, key=sim_map.get)
        
    print("{:<30} | {:<10} | {:0.4f}  | {:0.4f} | {:0.4f} | {:<10}".format(
        fname, str(item["expected_organ"])[:10], sim_breast, sim_lung, sim_colon, best_org
    ))
