import json, os, sys, torch, numpy as np
from PIL import Image
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import config
from src.data.image_transforms import preprocess_multicrop, crop_eyepiece_vignette
from src.ui.level2_app import img_extractor, text_extractor, text_proj, tab_mlp, classifier, organ_extractor, organ_head
from scripts.run_external_validation import EXTERNAL_CLINICAL_PAIRS, make_tab_args
from src.data.multimodal_dataset import generate_biomarker_vector
from src.models.modality_detector import validate_tabular_biomarkers

SITE_NAMES = ["Breast", "Lung", "Colorectal"]

def test_consensus():
    print("=" * 115)
    print(f"{'FILE':<28} | {'EXPECTED':<18} | {'IMAGE_SITE':<16} | {'TEXT_SITE':<14} | {'GCA_SITE':<16} | {'CONSENSUS_SITE'}")
    print("=" * 115)

    for idx, fname in enumerate(EXTERNAL_CLINICAL_PAIRS.keys()):
        img_path = os.path.join("data/external_validation", fname)
        pil = Image.open(img_path).convert("RGB")
        txt, cat_key, (age, sex, smoke, ecog) = EXTERNAL_CLINICAL_PAIRS[fname]
        vec = generate_biomarker_vector(cat_key, seed=100 + idx)
        vec[0], vec[1], vec[2], vec[3] = age, sex, smoke, ecog
        raw_dict = {
            "age": age, "sex": sex, "smoking_pack_yrs": smoke, "ecog_score": ecog,
            "cea_level": vec[4], "ca125_level": vec[5], "ca19_9_level": vec[6], "ki67_index": vec[7],
            "er_status": vec[8], "pr_status": vec[9], "her2_status": vec[10],
            "egfr_mutation": vec[11], "kras_mutation": vec[12],
            "hemoglobin": vec[13], "wbc_count": vec[14], "platelet_count": vec[15]
        }
        _, _, _, vec_32, _, _ = validate_tabular_biomarkers(raw_dict)

        with torch.no_grad():
            t_img = preprocess_multicrop(crop_eyepiece_vignette(pil)).to(config.DEVICE)
            
            # Phikon
            p_feats = organ_extractor(t_img)
            pred_org, p_conf, p_prob_dict, is_ood, is_unc, d_cos = organ_head.predict_with_calibration(p_feats)
            p_phik = np.array([p_prob_dict.get(s, 0.0) for s in SITE_NAMES], dtype=np.float64) if p_prob_dict else None
            
            # Extractor & MLP
            raw_v, _ = img_extractor(t_img)
            v_i = raw_v.mean(dim=0, keepdim=True)
            f_t = text_extractor(txt, device=config.DEVICE)
            v_t = text_proj(f_t)
            tab_t = torch.tensor(vec_32, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
            v_r = tab_mlp(tab_t)
            
            out = classifier.forward_available(v_i=v_i, v_t=v_t, v_r=v_r)
            
            r_im_site = F.softmax(out["logits_image_site"], dim=1)[0].cpu().numpy()
            # Couple with Phikon
            if p_phik is not None:
                log_c = 0.35 * np.log(r_im_site + 1e-7) + 0.65 * np.log(p_phik + 1e-7)
                c_im_site = np.exp(log_c) / np.sum(np.exp(log_c))
            else:
                c_im_site = r_im_site
                
            t_site = F.softmax(out["logits_text_site"], dim=1)[0].cpu().numpy()
            gca_site = F.softmax(out["logits_gca_site"], dim=1)[0].cpu().numpy()
            
            # Multimodal consensus pool
            log_cons = 0.40 * np.log(gca_site + 1e-7) + 0.35 * np.log(t_site + 1e-7) + 0.25 * np.log(c_im_site + 1e-7)
            cons_site = np.exp(log_cons) / np.sum(np.exp(log_cons))
            
            im_top = f"{SITE_NAMES[np.argmax(c_im_site)]} ({np.max(c_im_site)*100:.0f}%)"
            tx_top = f"{SITE_NAMES[np.argmax(t_site)]} ({np.max(t_site)*100:.0f}%)"
            gca_top = f"{SITE_NAMES[np.argmax(gca_site)]} ({np.max(gca_site)*100:.0f}%)"
            cons_top = f"{SITE_NAMES[np.argmax(cons_site)]} ({np.max(cons_site)*100:.0f}%)"
            
            print(f"{fname:<28} | {cat_key:<18} | {im_top:<16} | {tx_top:<14} | {gca_top:<16} | {cons_top}")


if __name__ == "__main__":
    test_consensus()
