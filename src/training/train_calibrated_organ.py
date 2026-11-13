"""
Retrain and Calibrate the Phikon Organ Head with Multi-Prototype Clustering
and Logit-Energy OOD Threshold Calibration across Multi-Cohort & Multi-Scale Views.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
from PIL import Image

import config
from src.data.hierarchical_dataset import (
    build_hierarchical_splits,
    HierarchicalHistopathologyDataset,
    get_ood_calibration_paths,
)
from src.data.image_transforms import preprocess_multicrop, get_image_transforms
from src.models.calibrated_organ_model import CalibratedPhikonOrganEncoder, CalibratedOrganHead


def train_calibrated_organ_head(k_per_organ=4):
    print(f"[*] Training Multi-Prototype Phikon Organ Head on {config.DEVICE}...")
    train_records, val_records, test_records = build_hierarchical_splits()

    encoder = CalibratedPhikonOrganEncoder(freeze=True).to(config.DEVICE)
    encoder.eval()
    train_tf = get_image_transforms(is_train=True)

    # Pre-cache stain-normalized + multi-scale augmented training embeddings (3 views per image)
    print("[*] Extracting stain-normalized multi-scale Phikon embeddings...")
    tr_embs, tr_labels = [], []
    with torch.no_grad():
        for r in train_records:
            im = Image.open(r["image_path"]).convert("RGB")
            # View 0: deterministic multicrop (full + center)
            mc = preprocess_multicrop(im).to(config.DEVICE)
            f_mc = encoder(mc).cpu()
            for v in range(f_mc.size(0)):
                tr_embs.append(f_mc[v : v + 1])
                tr_labels.append(r["organ_label"])
            # Views 1..2: stochastic multi-scale RandomResizedCrop + rotation + stain-norm
            for _ in range(2):
                t_aug = train_tf(im).unsqueeze(0).to(config.DEVICE)
                tr_embs.append(encoder(t_aug).cpu())
                tr_labels.append(r["organ_label"])

    tr_X = torch.cat(tr_embs, dim=0)
    tr_y = torch.tensor(tr_labels, dtype=torch.long)

    # Build K=4 K-Means sub-prototypes per organ on normalized training embeddings (12 total)
    print(f"[*] Clustering K={k_per_organ} multi-scale sub-prototypes per organ...")
    proto_list = []
    tr_X_norm = F.normalize(tr_X, p=2, dim=1).numpy()
    for org_idx in range(3):
        org_vecs = tr_X_norm[tr_y.numpy() == org_idx]
        n_cl = min(k_per_organ, len(org_vecs))
        km = KMeans(n_clusters=n_cl, random_state=config.SEED, n_init=10).fit(org_vecs)
        proto_list.append(torch.tensor(km.cluster_centers_, dtype=torch.float32))
    prototypes = torch.cat(proto_list, dim=0).to(config.DEVICE)

    val_ds = HierarchicalHistopathologyDataset(val_records, is_train=False)
    test_ds = HierarchicalHistopathologyDataset(test_records, is_train=False)
    tr_ds = torch.utils.data.TensorDataset(tr_X, tr_y)
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    organ_head = CalibratedOrganHead(in_features=768, num_organs=3, min_confidence=0.50).to(config.DEVICE)
    organ_head.set_reference_prototypes(prototypes)

    optimizer = torch.optim.AdamW(organ_head.parameters(), lr=8e-4, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)

    best_val_loss = float("inf")
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "calibrated_organ_head.pt")

    for epoch in range(1, 26):
        organ_head.train()
        tot_loss = 0.0
        for embs, labels in tr_loader:
            embs, labels = embs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            logits = organ_head(embs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * embs.size(0)
        scheduler.step()

        organ_head.eval()
        v_ok, v_tot, v_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = batch["image"].to(config.DEVICE), batch["organ_label"].to(config.DEVICE)
                logits = organ_head(encoder(imgs))
                v_loss_sum += criterion(logits, labels).item() * imgs.size(0)
                v_ok += (torch.argmax(logits, dim=1) == labels).sum().item()
                v_tot += imgs.size(0)
        val_acc = v_ok / v_tot
        val_loss = v_loss_sum / v_tot
        if epoch % 5 == 0 or val_loss < best_val_loss:
            print(f"Epoch {epoch:02d}/25 - TrLoss: {tot_loss/len(tr_ds):.4f} | ValLoss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                "organ_head_state_dict": organ_head.state_dict(),
                "reference_prototypes": prototypes.cpu(),
                "val_acc": val_acc,
                "epoch": epoch,
            }, ckpt_path)

    # Calibrate OOD cosine + energy thresholds on Validation (in-scope) vs OOD Calibration Tissues
    checkpoint = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    organ_head.load_calibrated_checkpoint(checkpoint, config.DEVICE)
    organ_head.eval()

    def score_records(paths):
        dists, energies = [], []
        ref = organ_head.reference_bank
        with torch.no_grad():
            for p in paths:
                im = Image.open(p).convert("RGB")
                mc = preprocess_multicrop(im).to(config.DEVICE)
                embs = encoder(mc)
                logits = organ_head(embs).mean(dim=0, keepdim=True)
                energy = float(-torch.logsumexp(logits, dim=1).item())
                sims = torch.mm(F.normalize(embs, p=2, dim=1), ref.t())
                dcos = max(0.0, 1.0 - float(torch.max(sims).item()))
                dists.append(dcos)
                energies.append(energy)
        return dists, energies

    in_dists, in_energies = score_records([r["image_path"] for r in val_records])
    ood_paths = get_ood_calibration_paths()
    ood_dists, ood_energies = score_records(ood_paths)

    if ood_dists:
        y_ood = [0] * len(in_dists) + [1] * len(ood_dists)
        auroc = roc_auc_score(y_ood, in_dists + ood_dists)
        soft_cos = float(np.percentile(in_dists, 90))
        hard_cos = float(max(np.percentile(in_dists, 98) + 0.04, 0.54))
        max_energy = float(np.percentile(in_energies, 90) + 0.35)
        print(
            f"[*] OOD Calibration (Val N={len(in_dists)} vs Distinct OOD Organs N={len(ood_dists)}): "
            f"Dist AUROC={auroc:.4f} | InDist 95th={np.percentile(in_dists,95):.3f} vs OOD Min={min(ood_dists):.3f} | "
            f"Thresholds -> soft={soft_cos:.3f}, hard={hard_cos:.3f}, max_E={max_energy:.2f}"
        )
    else:
        soft_cos, hard_cos, max_energy = 0.48, 0.58, -1.8

    checkpoint["calibrated_thresholds"] = {
        "soft_cosine_distance": round(soft_cos, 4),
        "max_cosine_distance": round(hard_cos, 4),
        "max_energy": round(max_energy, 4),
    }
    torch.save(checkpoint, ckpt_path)
    print(f"[+] Saved multi-prototype Organ Head ({prototypes.size(0)} prototypes) -> {ckpt_path}")

    # Test evaluation
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            imgs, labels = batch["image"].to(config.DEVICE), batch["organ_label"].to(config.DEVICE)
            preds = torch.argmax(organ_head(encoder(imgs)), dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n" + "=" * 50)
    print("STAGE 1 TEST EVALUATION (MULTI-COHORT ORGAN ID)")
    print("=" * 50)
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["Breast", "Lung", "Colorectal"], digits=4))
    return ckpt_path


if __name__ == "__main__":
    train_calibrated_organ_head()
