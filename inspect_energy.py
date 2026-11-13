import json, os, torch
import torch.nn.functional as F
from PIL import Image
import config
from src.data.image_transforms import preprocess_multicrop
from src.ui.level2_app import organ_extractor, organ_head

manifest_path = "data/external_validation/provenance_manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

header = "{:<30} | {:<7} | {:<8} | {:<12} | {:<6}".format("FILE", "d_cos", "energy", "top_organ", "prob")
print(header)
print("-" * 75)
for item in manifest:
    fname = item["filename"]
    img_path = os.path.join("data/external_validation", fname)
    if not os.path.exists(img_path):
        continue
    pil = Image.open(img_path).convert("RGB")
    t = preprocess_multicrop(pil).to(config.DEVICE)
    with torch.no_grad():
        feats = organ_extractor(t)
        logits = organ_head(feats)
        mean_logits = logits.mean(dim=0, keepdim=True)
        energy = float(-torch.logsumexp(mean_logits, dim=1).item())
        probs = torch.softmax(mean_logits, dim=1)
        prob_val, pred_idx = torch.max(probs, dim=1)
        norm_emb = F.normalize(feats, p=2, dim=1)
        sims = torch.mm(norm_emb, organ_head.reference_bank.t())
        d_cos = float(max(0.0, 1.0 - torch.max(sims).item()))
        organs = ["Breast", "Lung", "Colorectal"]
        pred_organ = organs[pred_idx.item()]
    row = "{:<30} | {:0.4f}  | {:0.4f}   | {:<12} | {:0.4f}".format(fname, d_cos, energy, pred_organ, prob_val.item())
    print(row)
