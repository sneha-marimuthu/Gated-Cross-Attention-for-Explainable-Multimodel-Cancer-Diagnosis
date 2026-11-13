import json, os, torch
from PIL import Image
import config
from src.ui.level2_app import run_multimodal_pipeline, organ_head
from scripts.run_external_validation import EXTERNAL_CLINICAL_PAIRS, make_tab_args
from src.data.multimodal_dataset import generate_biomarker_vector

# Test relaxed energy threshold
organ_head.max_energy = -1.10

manifest_path = "data/external_validation/provenance_manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

print("=" * 105)
print(f"{'FILE':<28} | {'EXPECTED':<18} | {'TRIMODAL STATUS':<22} | {'TRIMODAL SITE':<20} | {'CAM'}")
print("=" * 105)

for idx, item in enumerate(manifest):
    fname = item["filename"]
    if fname not in EXTERNAL_CLINICAL_PAIRS:
        continue
    img_path = os.path.join("data/external_validation", fname)
    pil = Image.open(img_path).convert("RGB")
    txt, cat_key, (age, sex, smoke, ecog) = EXTERNAL_CLINICAL_PAIRS[fname]
    vec = generate_biomarker_vector(cat_key, seed=100 + idx)
    vec[0], vec[1], vec[2], vec[3] = age, sex, smoke, ecog
    tab_args = make_tab_args(vec)
    
    b_md, d_md, p_stat, p_site, abl_md, gate_md, cam_out, cam_cap = run_multimodal_pipeline(
        pil, txt, True, *tab_args
    )
        
    s_top = max(p_stat, key=p_stat.get) if p_stat else "HALTED"
    o_top = max(p_site, key=p_site.get) if p_site else "HALTED"
    has_cam = cam_out is not None
    print(f"{fname:<28} | {cat_key:<18} | {s_top:<22} | {o_top:<20} | {has_cam}")
