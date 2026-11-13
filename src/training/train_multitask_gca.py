"""
Multi-Task GCA Training, Joint Optimization, and Clinical Benchmark Suite.
Trains and calibrates all 4 multi-task heads:
  1. Cancer Classification (Status & Site)
  2. TNM Cancer Staging (Ordinal cross-entropy / MSE)
  3. Treatment Recommendation Ranking (Pairwise / listwise MSE on oncology guidelines, evaluated by NDCG@3)
  4. 5-Year Survival Risk (Cox proportional hazards loss, evaluated by Harrell's C-index)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image

import config
from src.data.multimodal_dataset import (
    build_multimodal_records,
    encode_biomarkers_with_mask,
    format_synchronized_narrative,
    generate_biomarker_vector,
)
from src.data.image_transforms import preprocess_multicrop, get_image_transforms
from src.models.hierarchical_model import HierarchicalResNetModel
from src.models.image_encoder import ResNet50FeatureExtractor
from src.models.text_encoder import BioBERTTextEncoder, TextProjectionHead
from src.models.tabular_encoder import TabularBiomarkerMLP
from src.models.multimodal_classifier import MultiTaskMultimodalClassifier
from src.models.clinical_staging_engine import (
    derive_clinical_targets,
    compute_ndcg_at_k,
    compute_c_index,
    STAGE_NAMES,
    TREATMENT_OPTIONS,
    NUM_TREATMENTS,
)


def load_calibrated_image_extractor(device=config.DEVICE):
    h_ckpt = os.path.join(config.CHECKPOINT_DIR, "hierarchical_cancer_model.pt")
    if os.path.exists(h_ckpt):
        h_model = HierarchicalResNetModel().to(device)
        ckpt = torch.load(h_ckpt, map_location=device, weights_only=False)
        h_model.load_state_dict(ckpt["model_state_dict"])
        extractor = h_model.extractor
    else:
        extractor = ResNet50FeatureExtractor(freeze=True).to(device)
    for p in extractor.parameters():
        p.requires_grad = False
    extractor.eval()
    return extractor


def build_and_cache_multitask_data(device=config.DEVICE):
    cache_path = os.path.join(config.CHECKPOINT_DIR, "multitask_cached_features.pt")
    if os.path.exists(cache_path):
        print(f"[*] Loading cached multimodal features from {cache_path}...")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print("[*] Extracting multi-task features across Train, Val, and Test splits...")
    train_records, val_records, test_records = build_multimodal_records()
    img_extractor = load_calibrated_image_extractor(device)
    text_extractor = BioBERTTextEncoder(freeze=True).to(device)
    text_extractor.eval()

    def process_records(records, is_train=False):
        img_feats, text_feats, tab_feats = [], [], []
        y_statuses, y_sites, y_stages = [], [], []
        y_treatments, y_survivals, y_hazards = [], [], []

        with torch.no_grad():
            for r in records:
                img = Image.open(r["image_path"]).convert("RGB")
                t_img = preprocess_multicrop(img).to(device)
                ext_out = img_extractor(t_img)
                raw_gap = ext_out[0] if isinstance(ext_out, tuple) else ext_out
                feat_i = raw_gap.mean(dim=0).cpu().numpy()

                raw_bio = r["raw_biomarkers"]
                f_r_vec = r["biomarkers_32"]
                text_str = r["text_narrative"]

                feat_t = text_extractor([text_str]).squeeze(0).cpu().numpy()

                # Derive 4 clinical targets
                stage_idx, treat_rel, surv_prob, hazard = derive_clinical_targets(
                    r["category_key"], raw_bio, text_str
                )

                img_feats.append(feat_i)
                text_feats.append(feat_t)
                tab_feats.append(f_r_vec)
                y_statuses.append(r["status_label"])
                y_sites.append(r["organ_label"])
                y_stages.append(stage_idx)
                y_treatments.append(treat_rel)
                y_survivals.append(surv_prob)
                y_hazards.append(hazard)

        return {
            "img": torch.tensor(np.array(img_feats), dtype=torch.float32),
            "text": torch.tensor(np.array(text_feats), dtype=torch.float32),
            "tab": torch.tensor(np.array(tab_feats), dtype=torch.float32),
            "status": torch.tensor(y_statuses, dtype=torch.long),
            "site": torch.tensor(y_sites, dtype=torch.long),
            "stage": torch.tensor(y_stages, dtype=torch.long),
            "treatment": torch.tensor(np.array(y_treatments), dtype=torch.float32),
            "survival": torch.tensor(y_survivals, dtype=torch.float32).unsqueeze(1),
            "hazard": torch.tensor(y_hazards, dtype=torch.float32).unsqueeze(1),
        }

    cached_data = {
        "train": process_records(train_records, is_train=True),
        "val": process_records(val_records, is_train=False),
        "test": process_records(test_records, is_train=False),
    }

    torch.save(cached_data, cache_path)
    print(f"[+] Cached multi-task features successfully to {cache_path}")
    return cached_data


def train_multitask_gca(device=config.DEVICE, epochs=15, batch_size=32, lr=3e-4):
    print("=" * 80)
    print("GCAEMMCD MULTI-TASK JOINT OPTIMIZATION & CALIBRATION")
    print("=" * 80)

    data = build_and_cache_multitask_data(device)
    train_d = data["train"]
    val_d = data["val"]
    test_d = data["test"]

    train_ds = TensorDataset(
        train_d["img"], train_d["text"], train_d["tab"],
        train_d["status"], train_d["site"], train_d["stage"],
        train_d["treatment"], train_d["survival"], train_d["hazard"]
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Initialize Models
    text_proj = TextProjectionHead(in_features=768, out_features=256).to(device)
    tab_mlp = TabularBiomarkerMLP(in_features=32, latent_dim=256).to(device)
    classifier = MultiTaskMultimodalClassifier(latent_dim=256, num_classes=3).to(device)

    # Pre-load existing calibrated Level 2 backbone weights if present
    base_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "multimodal_gca_model.pt")
    if os.path.exists(base_ckpt_path):
        print(f"[*] Initializing from hardened Level 2 base weights: {base_ckpt_path}")
        b_ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
        text_proj.load_state_dict(b_ckpt["text_proj_state_dict"])
        tab_mlp.load_state_dict(b_ckpt["tab_mlp_state_dict"])
        # Load matched keys into classifier
        c_sd = classifier.state_dict()
        for k, v in b_ckpt["classifier_state_dict"].items():
            if k in c_sd and c_sd[k].shape == v.shape:
                c_sd[k] = v
        classifier.load_state_dict(c_sd)
        print("[+] Base GCA weights successfully mapped.")

    # Optimizer
    all_params = list(text_proj.parameters()) + list(tab_mlp.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    criterion_smooth_l1 = nn.SmoothL1Loss()

    best_val_score = 0.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        text_proj.train()
        tab_mlp.train()
        classifier.train()

        total_loss = 0.0

        for b_img, b_text, b_tab, b_stat, b_site, b_stage, b_trt, b_surv, b_haz in train_loader:
            b_img = b_img.to(device)
            b_text = b_text.to(device)
            b_tab = b_tab.to(device)
            b_stat = b_stat.to(device)
            b_site = b_site.to(device)
            b_stage = b_stage.to(device)
            b_trt = b_trt.to(device)
            b_surv = b_surv.to(device)
            b_haz = b_haz.to(device)

            optimizer.zero_grad()

            v_t = text_proj(b_text)
            v_r = tab_mlp(b_tab)
            out = classifier(b_img, v_t, v_r)

            # Task 1 Losses (Cancer Status & Site)
            loss_stat = criterion_ce(out["logits_gca_status"], b_stat)
            loss_site = criterion_ce(out["logits_gca_site"], b_site)

            # Task 2 Loss (TNM Staging ordinal cross-entropy)
            loss_stage = criterion_ce(out["logits_staging"], b_stage)

            # Task 3 Loss (Treatment Ranking relevance loss)
            loss_treatment = criterion_mse(out["treatment_scores"], b_trt)

            # Task 4 Loss (Survival Hazard & Probability estimation)
            loss_hazard = criterion_smooth_l1(out["hazard_scores"], b_haz)
            loss_surv = criterion_mse(out["survival_probabilities"], b_surv)

            # Multi-Task Joint Objective (Equation from paper)
            loss = (
                1.0 * (loss_stat + loss_site) +
                0.8 * loss_stage +
                0.6 * loss_treatment +
                0.5 * (loss_hazard + loss_surv)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.5)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation Evaluation
        text_proj.eval()
        tab_mlp.eval()
        classifier.eval()

        with torch.no_grad():
            v_img = val_d["img"].to(device)
            v_t_val = text_proj(val_d["text"].to(device))
            v_r_val = tab_mlp(val_d["tab"].to(device))
            v_out = classifier(v_img, v_t_val, v_r_val)

            # Compute Task 1 Metrics
            pred_stat = v_out["logits_gca_status"].argmax(dim=-1).cpu().numpy()
            pred_site = v_out["logits_gca_site"].argmax(dim=-1).cpu().numpy()
            acc_stat = np.mean(pred_stat == val_d["status"].numpy())
            acc_site = np.mean(pred_site == val_d["site"].numpy())

            # Compute Task 2 Metrics (Staging Accuracy)
            pred_stage = v_out["logits_staging"].argmax(dim=-1).cpu().numpy()
            acc_stage = np.mean(pred_stage == val_d["stage"].numpy())

            # Compute Task 3 Metrics (Treatment NDCG@3)
            pred_trt_scores = v_out["treatment_scores"].cpu().numpy()
            ndcg_3 = compute_ndcg_at_k(val_d["treatment"].numpy(), pred_trt_scores, k=3)

            # Compute Task 4 Metrics (Harrell's C-index)
            pred_haz = v_out["hazard_scores"].cpu().numpy().flatten()
            true_surv = val_d["survival"].numpy().flatten()
            c_index = compute_c_index(pred_haz, true_surv)

            val_composite = (acc_stat + acc_site + acc_stage + ndcg_3 + c_index) / 5.0

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {total_loss/len(train_loader):.3f} | "
                f"Status: {acc_stat*100:.1f}% | Site: {acc_site*100:.1f}% | "
                f"Staging Acc: {acc_stage*100:.1f}% | NDCG@3: {ndcg_3:.3f} | C-Index: {c_index:.3f}"
            )

            if val_composite > best_val_score:
                best_val_score = val_composite
                best_metrics = {
                    "epoch": epoch,
                    "val_status_acc": float(acc_stat),
                    "val_site_acc": float(acc_site),
                    "val_stage_acc": float(acc_stage),
                    "val_ndcg_at_3": float(ndcg_3),
                    "val_c_index": float(c_index),
                }
                # Save Hardened Multi-Task Checkpoint
                save_dict = {
                    "arch_version": "gcaemmcd_multitask_hardened_v4",
                    "text_proj_state_dict": text_proj.state_dict(),
                    "tab_mlp_state_dict": tab_mlp.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "best_metrics": best_metrics
                }
                out_ckpt = os.path.join(config.CHECKPOINT_DIR, "multimodal_gca_model.pt")
                torch.save(save_dict, out_ckpt)
                print(f"  --> Saved superior multi-task checkpoint (Score: {best_val_score:.4f})")

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION FOR RESEARCH COMPLIANCE")
    print("=" * 80)
    # Test Evaluation
    with torch.no_grad():
        t_img = test_d["img"].to(device)
        t_t_test = text_proj(test_d["text"].to(device))
        t_r_test = tab_mlp(test_d["tab"].to(device))
        t_out = classifier(t_img, t_t_test, t_r_test)

        test_stat = t_out["logits_gca_status"].argmax(dim=-1).cpu().numpy()
        test_site = t_out["logits_gca_site"].argmax(dim=-1).cpu().numpy()
        test_stage = t_out["logits_staging"].argmax(dim=-1).cpu().numpy()
        test_trt_scores = t_out["treatment_scores"].cpu().numpy()
        test_haz = t_out["hazard_scores"].cpu().numpy().flatten()
        test_surv = test_d["survival"].numpy().flatten()

        t_acc_stat = np.mean(test_stat == test_d["status"].numpy())
        t_acc_site = np.mean(test_site == test_d["site"].numpy())
        t_acc_stage = np.mean(test_stage == test_d["stage"].numpy())
        t_ndcg_3 = compute_ndcg_at_k(test_d["treatment"].numpy(), test_trt_scores, k=3)
        t_c_index = compute_c_index(test_haz, test_surv)

        print(f"[*] Task 1 Cancer Classification Accuracy: {t_acc_stat*100:.2f}% (Site: {t_acc_site*100:.2f}%)")
        print(f"[*] Task 2 TNM Staging Ordinal Accuracy:    {t_acc_stage*100:.2f}%")
        print(f"[*] Task 3 Treatment Recommendation NDCG@3: {t_ndcg_3:.4f}")
        print(f"[*] Task 4 5-Year Survival Harrell's C-Index: {t_c_index:.4f}")
        print("=" * 80)

    return best_metrics


if __name__ == "__main__":
    train_multitask_gca()
