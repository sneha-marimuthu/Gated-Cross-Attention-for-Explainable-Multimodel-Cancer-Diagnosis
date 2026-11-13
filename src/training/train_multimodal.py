"""
Scientifically Hardened Multimodal Training & Verification Pipeline for GCAEMMCD Level 2.
Key improvements over prior version:
  1. Uses fine-tuned histopathology ResNet-50 features from Level 1 (`hierarchical_cancer_model.pt`)
     for visual-feature consistency with Grad-CAM.
  2. Dual-Task supervision across ALL 5 pathways (`Status`: Normal/Benign/Malignant AND
     `Site`: Breast/Lung/Colorectal), eliminating the flaw where Normal/Benign cases were
     labeled and predicted as Carcinoma.
  3. Multi-variant clinical narratives in training (pathology + symptom vignettes + negated text).
  4. Dynamic availability masking in tabular batches (pre-biopsy panels without IHC).
  5. Controlled 1-modality discordance augmentation so GCA gates learn to attenuate noisy inputs.
  6. Comprehensive evaluation across Clean Test Set, Partial-Biomarker Stress Test, and
     Single-Modality Perturbation Gate Audit.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
from PIL import Image

import config
from src.data.image_transforms import preprocess_image, preprocess_multicrop, get_image_transforms
from src.data.multimodal_dataset import (
    build_multimodal_records,
    format_synchronized_narrative,
    generate_biomarker_vector,
    encode_biomarkers_with_mask,
)
from src.models.hierarchical_model import HierarchicalResNetModel
from src.models.image_encoder import ResNet50FeatureExtractor
from src.models.text_encoder import BioBERTTextEncoder, TextProjectionHead
from src.models.tabular_encoder import TabularBiomarkerMLP
from src.models.multimodal_classifier import UnifiedMultimodalClassifier


def load_calibrated_image_extractor(device=config.DEVICE):
    """Loads Level 1 calibrated ResNet-50 extractor (layer4 fine-tuned on histology) and freezes it."""
    h_ckpt = os.path.join(config.CHECKPOINT_DIR, "hierarchical_cancer_model.pt")
    if os.path.exists(h_ckpt):
        print(f"[*] Loading calibrated histopathology ResNet-50 extractor from {h_ckpt}...")
        h_model = HierarchicalResNetModel().to(device)
        ckpt = torch.load(h_ckpt, map_location=device, weights_only=False)
        h_model.load_state_dict(ckpt["model_state_dict"])
        extractor = h_model.extractor
    else:
        print("[!] hierarchical_cancer_model.pt not found, falling back to ImageNet ResNet-50.")
        extractor = ResNet50FeatureExtractor(freeze=True).to(device)

    for p in extractor.parameters():
        p.requires_grad = False
    extractor.eval()
    return extractor


def extract_and_cache_multimodal_features(train_records, val_records, test_records, device=config.DEVICE):
    """
    Caches frozen ResNet-50 GAP [2048], BioBERT [CLS] [768], Masked Tabular [32],
    y_status, and y_site.
    For training records ONLY:
      - Applies stochastic histopathology image augmentations (flips, rotation, color jitter)
      - Expands across 5 disjoint training narrative templates
      - Samples realistic overlapping biomarker boundary draws
    For Val / Test records:
      - Uses strictly unaugmented images and strictly disjoint 'eval' narratives.
    """
    img_extractor = load_calibrated_image_extractor(device)
    text_extractor = BioBERTTextEncoder(freeze=True).to(device)
    text_extractor.eval()
    train_tf = get_image_transforms(is_train=True)

    def process_split(records, split_name, expand_variants=False):
        print(f"[*] Caching multimodal features for {split_name} split ({len(records)} base samples)...")
        img_feats, text_feats, tab_feats = [], [], []
        y_statuses, y_sites = [], []

        with torch.no_grad():
            for idx, r in enumerate(records):
                img = Image.open(r["image_path"]).convert("RGB")
                variants = [0, 1, 2, 3, 4] if expand_variants else [0]
                for v_idx in variants:
                    if expand_variants:
                        t_img = train_tf(img).unsqueeze(0).to(device)
                        raw_bio = (
                            r["raw_biomarkers"]
                            if v_idx == 0
                            else generate_biomarker_vector(r["category_key"], seed=abs(hash((r["image_path"], v_idx))) % (2**31))
                        )
                        f_r_vec = encode_biomarkers_with_mask(raw_bio, np.ones(16, dtype=np.float32))
                        text_str = format_synchronized_narrative(
                            r["category_key"], raw_bio, variant_idx=v_idx, split_role="train"
                        )
                    else:
                        t_img = preprocess_multicrop(img).to(device)
                        f_r_vec = r["biomarkers_32"]
                        text_str = r["text_narrative"]

                    raw_views, _ = img_extractor(t_img)  # [N, 2048]
                    raw_img = raw_views.mean(dim=0, keepdim=True)  # [1, 2048]
                    f_t = text_extractor(text_str, device=device)  # [1, 768]
                    f_r = torch.tensor(f_r_vec, dtype=torch.float32).unsqueeze(0)  # [1, 32]

                    img_feats.append(raw_img.cpu())
                    text_feats.append(f_t.cpu())
                    tab_feats.append(f_r)
                    y_statuses.append(r["status_label"])
                    y_sites.append(r["organ_label"])

        return TensorDataset(
            torch.cat(img_feats, dim=0),
            torch.cat(text_feats, dim=0),
            torch.cat(tab_feats, dim=0),
            torch.tensor(y_statuses, dtype=torch.long),
            torch.tensor(y_sites, dtype=torch.long),
        )

    train_ds = process_split(train_records, "train", expand_variants=True)
    val_ds = process_split(val_records, "val", expand_variants=False)
    test_ds = process_split(test_records, "test", expand_variants=False)
    return train_ds, val_ds, test_ds


def apply_tabular_mask_dropout(f_r_batch, rng):
    """
    Randomly masks optional molecular/serum markers on training mini-batches
    so TabularBiomarkerMLP handles 'Unknown / Not Tested' entries safely.
    Indices [0:16] are normalized values; [16:32] are availability mask bits.
    """
    out = f_r_batch.clone()
    bsz = out.size(0)
    for b in range(bsz):
        if rng.rand() < 0.40:
            # Drop post-biopsy IHC / molecular markers (Ki67=7, ER=8, PR=9, HER2=10, EGFR=11, KRAS=12)
            for col in [7, 8, 9, 10, 11, 12]:
                out[b, col] = 0.0
                out[b, 16 + col] = 0.0
        if rng.rand() < 0.25:
            # Randomly drop 1-2 serum tumor markers (CEA=4, CA125=5, CA19_9=6)
            for col in rng.choice([4, 5, 6], size=rng.randint(1, 3), replace=False):
                out[b, col] = 0.0
                out[b, 16 + col] = 0.0
    return out


def compute_dual_pathway_loss(preds, y_status, y_site, criterion, gca_only=False):
    l_gca = criterion(preds["logits_gca_status"], y_status) + criterion(preds["logits_gca_site"], y_site)
    if gca_only:
        return l_gca

    l_early = criterion(preds["logits_early_status"], y_status) + criterion(preds["logits_early_site"], y_site)
    l_img = criterion(preds["logits_image_status"], y_status) + criterion(preds["logits_image_site"], y_site)
    l_txt = criterion(preds["logits_text_status"], y_status) + criterion(preds["logits_text_site"], y_site)
    l_tab = criterion(preds["logits_tabular_status"], y_status) + criterion(preds["logits_tabular_site"], y_site)
    return l_gca + 0.5 * l_early + 0.35 * (l_img + l_txt + l_tab)


def _evaluate_dataset_split(text_proj, tab_mlp, classifier, ds, device):
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    y_st_true, y_si_true = [], []
    st_preds = {k: [] for k in ["image", "text", "tabular", "early", "gca"]}
    si_preds = {k: [] for k in ["image", "text", "tabular", "early", "gca"]}
    with torch.no_grad():
        for f_i, f_t, f_r, y_st, y_si in loader:
            f_i, f_t, f_r = f_i.to(device), f_t.to(device), f_r.to(device)
            preds = classifier(f_i, text_proj(f_t), tab_mlp(f_r))
            y_st_true.extend(y_st.numpy())
            y_si_true.extend(y_si.numpy())
            for k in st_preds:
                st_preds[k].extend(torch.argmax(preds[f"logits_{k}_status"], dim=1).cpu().numpy())
                si_preds[k].extend(torch.argmax(preds[f"logits_{k}_site"], dim=1).cpu().numpy())
    metrics = {}
    for k in st_preds:
        metrics[k] = (
            accuracy_score(y_st_true, st_preds[k]),
            accuracy_score(y_si_true, si_preds[k]),
        )
    return metrics, y_st_true, y_si_true, st_preds, si_preds


def run_5fold_unseen_cv(all_records, device=config.DEVICE):
    """
    Runs 5-Fold Stratified Cross-Validation across all 100 paired samples using:
      - Uncontaminated frozen ImageNet ResNet-50 features (zero Level-1 fold leakage!)
      - Disjoint 'train' vs 'eval' text narrative templates
      - Overlapping clinical biomarker distributions
    Reports honest out-of-fold Mean +/- Std across all 5 pathways.
    """
    print("\n" + "=" * 72)
    print("HONEST 5-FOLD STRATIFIED CROSS-VALIDATION (ZERO TRAIN/TEST LEAKAGE)")
    print("=" * 72)

    imagenet_ext = ResNet50FeatureExtractor(freeze=True).to(device).eval()
    txt_enc = BioBERTTextEncoder(freeze=True).to(device).eval()
    train_tf = get_image_transforms(is_train=True)

    # Pre-cache uncontaminated features cleanly per sample
    labels_strat = [r["category_key"] for r in all_records]
    groups = [r.get("group_id", r["image_path"]) for r in all_records]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    # Cache eval views (unaugmented multi-crop image + unseen eval narrative) and train augmentations per sample
    eval_cache = []
    train_cache = []
    with torch.no_grad():
        for idx, r in enumerate(all_records):
            im = Image.open(r["image_path"]).convert("RGB")
            fi_views, _ = imagenet_ext(preprocess_multicrop(im).to(device))
            fi_eval = fi_views.mean(dim=0, keepdim=True)
            eval_txt = format_synchronized_narrative(r["category_key"], r["raw_biomarkers"], variant_idx=idx, split_role="eval")
            ft_eval = txt_enc(eval_txt, device=device)
            fr_eval = torch.tensor(r["biomarkers_32"], dtype=torch.float32).unsqueeze(0)
            eval_cache.append((fi_eval.cpu(), ft_eval.cpu(), fr_eval, r["status_label"], r["organ_label"]))

            aug_views = []
            for v in range(3):
                fi_tr, _ = imagenet_ext(train_tf(im).unsqueeze(0).to(device))
                raw_b = r["raw_biomarkers"] if v == 0 else generate_biomarker_vector(
                    r["category_key"], seed=abs(hash((r["image_path"], "cv", v))) % (2**31)
                )
                fr_tr = torch.tensor(encode_biomarkers_with_mask(raw_b, np.ones(16, dtype=np.float32)), dtype=torch.float32).unsqueeze(0)
                txt_tr = format_synchronized_narrative(r["category_key"], raw_b, variant_idx=v, split_role="train")
                ft_tr = txt_enc(txt_tr, device=device)
                aug_views.append((fi_tr.cpu(), ft_tr.cpu(), fr_tr, r["status_label"], r["organ_label"]))
            train_cache.append(aug_views)

    fold_scores = {k: {"status": [], "site": []} for k in ["image", "text", "tabular", "early", "gca"]}
    rng = np.random.RandomState(config.SEED)

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(all_records, labels_strat, groups=groups), start=1):
        tr_items = [item for i in tr_idx for item in train_cache[i]]
        te_items = [eval_cache[i] for i in te_idx]

        tr_ds = TensorDataset(
            torch.cat([x[0] for x in tr_items], 0),
            torch.cat([x[1] for x in tr_items], 0),
            torch.cat([x[2] for x in tr_items], 0),
            torch.tensor([x[3] for x in tr_items], dtype=torch.long),
            torch.tensor([x[4] for x in tr_items], dtype=torch.long),
        )
        te_ds = TensorDataset(
            torch.cat([x[0] for x in te_items], 0),
            torch.cat([x[1] for x in te_items], 0),
            torch.cat([x[2] for x in te_items], 0),
            torch.tensor([x[3] for x in te_items], dtype=torch.long),
            torch.tensor([x[4] for x in te_items], dtype=torch.long),
        )

        tr_loader = DataLoader(tr_ds, batch_size=24, shuffle=True, drop_last=True)
        t_proj = TextProjectionHead(768, 256, dropout=0.35).to(device)
        tb_mlp = TabularBiomarkerMLP(32, 256, dropout=0.35).to(device)
        clf = UnifiedMultimodalClassifier(256, 3, dropout=0.35).to(device)

        params = list(t_proj.parameters()) + list(tb_mlp.parameters()) + list(clf.parameters())
        opt = torch.optim.AdamW(params, lr=6e-4, weight_decay=2e-3)
        crit = nn.CrossEntropyLoss(label_smoothing=0.10)

        t_proj.train()
        tb_mlp.train()
        clf.train()
        for _ in range(20):
            for fi, ft, fr, yst, ysi in tr_loader:
                fi, ft = fi.to(device), ft.to(device)
                fr = apply_tabular_mask_dropout(fr, rng).to(device)
                yst, ysi = yst.to(device), ysi.to(device)
                opt.zero_grad()
                preds = clf(fi, t_proj(ft), tb_mlp(fr))
                loss = compute_dual_pathway_loss(preds, yst, ysi, crit, gca_only=False)
                loss.backward()
                opt.step()

        t_proj.eval()
        tb_mlp.eval()
        clf.eval()
        m, _, _, _, _ = _evaluate_dataset_split(t_proj, tb_mlp, clf, te_ds, device)
        for k in fold_scores:
            fold_scores[k]["status"].append(m[k][0] * 100.0)
            fold_scores[k]["site"].append(m[k][1] * 100.0)

    print(f"{'Pathway (Zero-Leakage 5-Fold CV)':<34} | {'Status (Mean±Std)':<19} | {'Site (Mean±Std)':<19}")
    print("-" * 76)
    names = {
        "image": "1. Image-Only (Frozen ResNet-50)",
        "text": "2. Text-Only (BioBERT Disjoint)",
        "tabular": "3. Tabular-Only (Overlapping)",
        "early": "4. Early Concatenation",
        "gca": "5. GCA Multimodal Fusion",
    }
    for k in ["image", "text", "tabular", "early", "gca"]:
        s_m, s_sd = np.mean(fold_scores[k]["status"]), np.std(fold_scores[k]["status"])
        o_m, o_sd = np.mean(fold_scores[k]["site"]), np.std(fold_scores[k]["site"])
        print(f"{names[k]:<34} | {s_m:6.2f}% ± {s_sd:5.2f}%   | {o_m:6.2f}% ± {o_sd:5.2f}%")
    print("=" * 76)


def train_multimodal_model():
    print("=" * 72)
    print("GCAEMMCD LEVEL 2 REGULARIZED TRAINING, CV & VERIFICATION")
    print("=" * 72)

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    rng = np.random.RandomState(config.SEED)

    train_recs, val_recs, test_recs = build_multimodal_records()
    device = config.DEVICE

    # 1. Run honest 5-Fold Stratified Cross-Validation with zero Level-1 fold leakage first
    run_5fold_unseen_cv(train_recs + val_recs + test_recs, device=device)

    # 2. Train deployed Level 2 checkpoint with augmented train features & disjoint Val/Test eval
    train_ds, val_ds, test_ds = extract_and_cache_multimodal_features(train_recs, val_recs, test_recs, device)

    train_loader = DataLoader(train_ds, batch_size=24, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=14, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    text_proj = TextProjectionHead(in_features=768, out_features=256, dropout=0.35).to(device)
    tab_mlp = TabularBiomarkerMLP(in_features=32, latent_dim=256, dropout=0.35).to(device)
    classifier = UnifiedMultimodalClassifier(latent_dim=256, num_classes=3, dropout=0.35).to(device)

    all_params = list(text_proj.parameters()) + list(tab_mlp.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=6e-4, weight_decay=2e-3)
    num_epochs = 40
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)

    best_val_loss = float("inf")
    patience = 10
    stale_epochs = 0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "multimodal_gca_model.pt")

    for epoch in range(1, num_epochs + 1):
        text_proj.train()
        tab_mlp.train()
        classifier.train()

        total_loss = 0.0
        total_samples = 0

        for f_i, f_t, f_r, y_stat, y_site in train_loader:
            f_i = f_i.to(device)
            f_t = f_t.to(device)
            f_r_masked = apply_tabular_mask_dropout(f_r, rng).to(device)
            y_stat = y_stat.to(device)
            y_site = y_site.to(device)

            optimizer.zero_grad()

            # 1. Clean tri-modal forward step
            v_t = text_proj(f_t)
            v_r = tab_mlp(f_r_masked)
            preds = classifier(f_i, v_t, v_r)
            loss_clean = compute_dual_pathway_loss(preds, y_stat, y_site, criterion, gca_only=False)

            gw_clean = preds["gate_weights"]
            target_high = torch.full_like(gw_clean["image_to_text"], 0.82)
            loss_gate_clean = sum(F.mse_loss(g, target_high) for g in gw_clean.values()) / 6.0

            # 2. Controlled 1-modality discordance training step (if batch >= 2)
            loss_robust = torch.tensor(0.0, device=device)
            loss_gate_disc = torch.tensor(0.0, device=device)
            if f_i.size(0) >= 2:
                perm = torch.randperm(f_i.size(0), device=device)
                is_diff = ((y_stat[perm] != y_stat) | (y_site[perm] != y_site)).float()
                if epoch % 2 == 0:
                    preds_corr = classifier(f_i, v_t[perm].detach(), v_r.detach())
                    disc_keys = ["image_to_text", "text_to_image", "text_to_tabular", "tabular_to_text"]
                    conc_keys = ["image_to_tabular", "tabular_to_image"]
                else:
                    preds_corr = classifier(f_i, v_t.detach(), v_r[perm].detach())
                    disc_keys = ["image_to_tabular", "tabular_to_image", "text_to_tabular", "tabular_to_text"]
                    conc_keys = ["image_to_text", "text_to_image"]

                loss_robust = compute_dual_pathway_loss(preds_corr, y_stat, y_site, criterion, gca_only=True)

                gw_corr = preds_corr["gate_weights"]
                target_disc = 0.82 * (1.0 - is_diff) + 0.18 * is_diff
                target_conc = torch.full_like(target_disc, 0.82)
                loss_gate_disc = (
                    sum(F.mse_loss(gw_corr[k], target_disc) for k in disc_keys)
                    + sum(F.mse_loss(gw_corr[k], target_conc) for k in conc_keys)
                ) / 6.0

            loss = loss_clean + 0.30 * loss_robust + 1.5 * (loss_gate_clean + loss_gate_disc)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y_stat.size(0)
            total_samples += y_stat.size(0)

        scheduler.step()

        # Validation on unmodified Val set (monitor LOSS + accuracy for early stopping)
        text_proj.eval()
        tab_mlp.eval()
        classifier.eval()

        v_stat_ok, v_site_ok, v_tot, v_loss_sum = 0, 0, 0, 0.0
        v_clean_g, v_disc_g = [], []
        v_gate_loss_sum = 0.0
        with torch.no_grad():
            for f_i, f_t, f_r, y_stat, y_site in val_loader:
                f_i, f_t, f_r = f_i.to(device), f_t.to(device), f_r.to(device)
                y_stat, y_site = y_stat.to(device), y_site.to(device)

                v_t = text_proj(f_t)
                v_r = tab_mlp(f_r)
                preds = classifier(f_i, v_t, v_r)
                v_loss = compute_dual_pathway_loss(preds, y_stat, y_site, criterion, gca_only=False)
                v_loss_sum += v_loss.item() * y_stat.size(0)

                gw_vc = preds["gate_weights"]
                t_high = torch.full_like(gw_vc["image_to_text"], 0.82)
                g_loss_c = sum(F.mse_loss(g, t_high) for g in gw_vc.values()) / 6.0
                v_clean_g.extend(gw_vc["text_to_image"].cpu().numpy())

                g_loss_d = torch.tensor(0.0, device=device)
                if f_i.size(0) >= 2:
                    v_t_flip = torch.flip(v_t, dims=[0])
                    is_diff_v = ((torch.flip(y_stat, dims=[0]) != y_stat) | (torch.flip(y_site, dims=[0]) != y_site)).float()
                    preds_vd = classifier(f_i, v_t_flip, v_r)
                    gw_vd = preds_vd["gate_weights"]
                    d_keys = ["image_to_text", "text_to_image", "text_to_tabular", "tabular_to_text"]
                    c_keys = ["image_to_tabular", "tabular_to_image"]
                    t_d = 0.82 * (1.0 - is_diff_v) + 0.18 * is_diff_v
                    g_loss_d = (
                        sum(F.mse_loss(gw_vd[k], t_d) for k in d_keys)
                        + sum(F.mse_loss(gw_vd[k], t_high) for k in c_keys)
                    ) / 6.0
                    diff_np = is_diff_v.cpu().numpy() > 0.5
                    if diff_np.any():
                        v_disc_g.extend(gw_vd["text_to_image"].cpu().numpy()[diff_np])

                v_gate_loss_sum += float(g_loss_c.item() + g_loss_d.item()) * y_stat.size(0)

                p_stat = torch.argmax(preds["logits_gca_status"], dim=1)
                p_site = torch.argmax(preds["logits_gca_site"], dim=1)
                v_stat_ok += (p_stat == y_stat).sum().item()
                v_site_ok += (p_site == y_site).sum().item()
                v_tot += y_stat.size(0)

        acc_stat = v_stat_ok / v_tot
        acc_site = v_site_ok / v_tot
        val_cls_loss = v_loss_sum / v_tot
        val_gate_loss = v_gate_loss_sum / v_tot
        val_loss = val_cls_loss + 1.5 * val_gate_loss
        g_sep = (np.mean(v_clean_g) - np.mean(v_disc_g)) if (v_clean_g and v_disc_g) else 0.0

        if epoch % 5 == 0 or epoch == num_epochs or val_loss < best_val_loss:
            print(
                f"Epoch {epoch:02d}/{num_epochs:02d} | TrLoss: {total_loss/total_samples:.4f} | "
                f"ValObj: {val_loss:.4f} (Cls {val_cls_loss:.3f}, GateSep {g_sep:+.3f}) | "
                f"Val Status: {acc_stat*100:.1f}% | Val Site: {acc_site*100:.1f}%"
            )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            stale_epochs = 0
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                "arch_version": "level2_regularized_dual_task_v3",
                "tabular_in_features": 32,
                "text_proj_state_dict": text_proj.state_dict(),
                "tab_mlp_state_dict": tab_mlp.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "val_loss": val_loss,
                "val_status_acc": acc_stat,
                "val_site_acc": acc_site,
                "epoch": epoch,
            }, ckpt_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience and epoch >= 22:
                print(f"[*] Early stopping triggered at epoch {epoch} (best ValObj: {best_val_loss:.4f}).")
                break

    print(f"\n[+] Saved lowest-val-loss checkpoint to {ckpt_path} (Best ValLoss: {best_val_loss:.4f})")

    # ---------------------------------------------------------
    # Honest Train vs Val vs Held-Out Test Comparison
    # ---------------------------------------------------------
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    text_proj.load_state_dict(checkpoint["text_proj_state_dict"])
    tab_mlp.load_state_dict(checkpoint["tab_mlp_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    text_proj.eval()
    tab_mlp.eval()
    classifier.eval()

    tr_metrics, _, _, _, _ = _evaluate_dataset_split(text_proj, tab_mlp, classifier, train_ds, device)
    va_metrics, _, _, _, _ = _evaluate_dataset_split(text_proj, tab_mlp, classifier, val_ds, device)
    te_metrics, y_stat_true, y_site_true, stat_preds, site_preds = _evaluate_dataset_split(
        text_proj, tab_mlp, classifier, test_ds, device
    )

    tab_partial_site_preds, tab_partial_stat_preds = [], []
    clean_gates_ti, corr_gates_ti = [], []
    with torch.no_grad():
        for f_i, f_t, f_r, y_stat, y_site in test_loader:
            f_i, f_t, f_r = f_i.to(device), f_t.to(device), f_r.to(device)
            v_t = text_proj(f_t)
            v_r = tab_mlp(f_r)
            preds = classifier(f_i, v_t, v_r)
            clean_gates_ti.extend(preds["gate_weights"]["text_to_image"].cpu().numpy())

            f_r_pre = f_r.clone()
            for col in [7, 8, 9, 10, 11, 12]:
                f_r_pre[:, col] = 0.0
                f_r_pre[:, 16 + col] = 0.0
            v_r_pre = tab_mlp(f_r_pre)
            tab_partial_stat_preds.extend(torch.argmax(classifier.head_tabular_status(v_r_pre), dim=1).cpu().numpy())
            tab_partial_site_preds.extend(torch.argmax(classifier.head_tabular_site(v_r_pre), dim=1).cpu().numpy())

            if f_i.size(0) >= 2:
                v_t_corr = torch.flip(v_t, dims=[0])
                y_st_corr = torch.flip(y_stat, dims=[0])
                y_si_corr = torch.flip(y_site, dims=[0])
                diff_mask = (y_st_corr != y_stat) | (y_si_corr != y_site)
                if diff_mask.sum() > 0:
                    preds_corr = classifier(f_i, v_t_corr, v_r)
                    g_ti_c = preds_corr["gate_weights"]["text_to_image"].cpu().numpy()
                    corr_gates_ti.extend(g_ti_c[diff_mask.numpy()])

    print("\n" + "=" * 88)
    print(f"TRAIN (N={len(train_ds)}) vs VAL (N={len(val_ds)}) vs HELD-OUT TEST (N={len(y_stat_true)}) GENERALIZATION AUDIT")
    print("=" * 88)
    print(f"{'Pathway':<28} | {'Train (St/Si)':<15} | {'Val (St/Si)':<15} | {'Test (St/Si)':<15}")
    print("-" * 88)
    label_map = {
        "image": "1. Image-Only (ResNet-50)",
        "text": "2. Text-Only (BioBERT)",
        "tabular": "3. Tabular-Only (Masked)",
        "early": "4. Early Concatenation",
        "gca": "5. GCA Multimodal Fusion",
    }
    for k in ["image", "text", "tabular", "early", "gca"]:
        trs, tro = tr_metrics[k]
        vas, vao = va_metrics[k]
        tes, teo = te_metrics[k]
        print(
            f"{label_map[k]:<28} | {trs*100:5.1f}% / {tro*100:5.1f}% | "
            f"{vas*100:5.1f}% / {vao*100:5.1f}% | {tes*100:5.1f}% / {teo*100:5.1f}%"
        )

    print("-" * 88)
    pre_s_acc = accuracy_score(y_stat_true, tab_partial_stat_preds) * 100
    pre_o_acc = accuracy_score(y_site_true, tab_partial_site_preds) * 100
    print(f"{'Tabular Stress (Pre-Biopsy IHC Masked)':<38} | Test Status: {pre_s_acc:>6.2f}% | Test Site: {pre_o_acc:>6.2f}%")
    if corr_gates_ti:
        print(
            f"GCA Text->Image Gate Mean | Clean Concordant: {np.mean(clean_gates_ti):.4f} "
            f"| Discordant Corrupted: {np.mean(corr_gates_ti):.4f}"
        )
    print("=" * 88)

    print("\nGCA Disease Status Classification Report (Normal / Benign / Malignant):")
    print(classification_report(y_stat_true, stat_preds["gca"], target_names=["Normal", "Benign", "Malignant"], digits=4))
    print("\nGCA Anatomical Site Classification Report (Breast / Lung / Colorectal):")
    print(classification_report(y_site_true, site_preds["gca"], target_names=["Breast", "Lung", "Colorectal"], digits=4))

    return ckpt_path


if __name__ == "__main__":
    train_multimodal_model()
