import os, sys, torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "calibrated_organ_head.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

old_th = ckpt.get("calibrated_thresholds", {})
print("Old thresholds:", old_th)

new_th = {
    "soft_cosine_distance": 0.38,
    "max_cosine_distance": 0.535,
    "max_energy": -1.10,
}
ckpt["calibrated_thresholds"] = new_th
torch.save(ckpt, ckpt_path)
print("Updated calibrated thresholds in checkpoint:", new_th)
