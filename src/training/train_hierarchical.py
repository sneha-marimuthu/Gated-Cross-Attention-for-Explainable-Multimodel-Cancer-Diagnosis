"""
Trainer module for the Hierarchical ResNet-50 Architecture on Multi-Cohort Data:
Trains Cancer Status Head (Normal/Benign/Malignant) and Carcinoma Subtype Head (Breast/Lung/Colon)
with Reinhard stain normalization, RandomResizedCrop multi-scale augmentation,
CosineAnnealingLR, label smoothing, and validation-loss checkpoint selection.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

import config
from src.data.hierarchical_dataset import build_hierarchical_splits, HierarchicalHistopathologyDataset
from src.models.hierarchical_model import HierarchicalResNetModel


def train_hierarchical_model():
    print(f"[*] Initializing Multi-Cohort Hierarchical ResNet-50 Training on {config.DEVICE}...")
    train_records, val_records, test_records = build_hierarchical_splits()

    train_loader = DataLoader(HierarchicalHistopathologyDataset(train_records, is_train=True), batch_size=16, shuffle=True)
    val_loader = DataLoader(HierarchicalHistopathologyDataset(val_records, is_train=False), batch_size=16, shuffle=False)
    test_loader = DataLoader(HierarchicalHistopathologyDataset(test_records, is_train=False), batch_size=16, shuffle=False)

    model = HierarchicalResNetModel().to(config.DEVICE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=1e-3)
    num_epochs = 22
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    status_criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    subtype_criterion = nn.CrossEntropyLoss(label_smoothing=0.08)

    best_val_loss = float("inf")
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "hierarchical_cancer_model.pt")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, total_status = 0.0, 0

        for batch in train_loader:
            imgs = batch["image"].to(config.DEVICE)
            status_labels = batch["status_label"].to(config.DEVICE)
            subtype_labels = batch["subtype_label"].to(config.DEVICE)

            optimizer.zero_grad()
            status_logits, subtype_logits, _ = model(imgs)
            loss_status = status_criterion(status_logits, status_labels)

            mal_mask = subtype_labels >= 0
            if mal_mask.sum() > 0:
                loss_sub = subtype_criterion(subtype_logits[mal_mask], subtype_labels[mal_mask])
                loss = loss_status + loss_sub
            else:
                loss = loss_status

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            total_status += imgs.size(0)

        scheduler.step()

        model.eval()
        val_loss_sum, val_status_ok, val_status_tot = 0.0, 0, 0
        val_sub_ok, val_sub_tot = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(config.DEVICE)
                status_labels = batch["status_label"].to(config.DEVICE)
                subtype_labels = batch["subtype_label"].to(config.DEVICE)

                status_logits, subtype_logits, _ = model(imgs)
                v_loss = status_criterion(status_logits, status_labels)
                mal_mask = subtype_labels >= 0
                if mal_mask.sum() > 0:
                    v_loss = v_loss + subtype_criterion(subtype_logits[mal_mask], subtype_labels[mal_mask])
                    val_sub_ok += (torch.argmax(subtype_logits[mal_mask], dim=1) == subtype_labels[mal_mask]).sum().item()
                    val_sub_tot += mal_mask.sum().item()

                val_loss_sum += v_loss.item() * imgs.size(0)
                val_status_ok += (torch.argmax(status_logits, dim=1) == status_labels).sum().item()
                val_status_tot += imgs.size(0)

        val_loss = val_loss_sum / val_status_tot
        val_status_acc = val_status_ok / val_status_tot
        val_sub_acc = (val_sub_ok / val_sub_tot) if val_sub_tot > 0 else 1.0

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | TrLoss: {total_loss/total_status:.4f} | "
            f"ValLoss: {val_loss:.4f} | Val Status: {val_status_acc*100:.1f}% | Val Subtype: {val_sub_acc*100:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_status_acc": val_status_acc,
                "val_sub_acc": val_sub_acc,
            }, ckpt_path)

    print(f"\n[+] Multi-Cohort Hierarchical training complete! Best ValLoss={best_val_loss:.4f} -> {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_status_true, y_status_pred = [], []
    y_sub_true, y_sub_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(config.DEVICE)
            status_labels = batch["status_label"].to(config.DEVICE)
            subtype_labels = batch["subtype_label"].to(config.DEVICE)
            status_logits, subtype_logits, _ = model(imgs)
            y_status_true.extend(status_labels.cpu().numpy())
            y_status_pred.extend(torch.argmax(status_logits, dim=1).cpu().numpy())

            mal_mask = subtype_labels >= 0
            if mal_mask.sum() > 0:
                y_sub_true.extend(subtype_labels[mal_mask].cpu().numpy())
                y_sub_pred.extend(torch.argmax(subtype_logits[mal_mask], dim=1).cpu().numpy())

    print("\n" + "=" * 50)
    print("STAGE 2 TEST EVALUATION: CANCER STATUS (MULTI-COHORT)")
    print("=" * 50)
    print(confusion_matrix(y_status_true, y_status_pred))
    print(classification_report(y_status_true, y_status_pred, target_names=["Normal", "Benign", "Malignant"], digits=4))

    print("\n" + "=" * 50)
    print("STAGE 3 TEST EVALUATION: CARCINOMA SUBTYPE (MULTI-COHORT)")
    print("=" * 50)
    print(confusion_matrix(y_sub_true, y_sub_pred))
    print(classification_report(y_sub_true, y_sub_pred, target_names=["Breast Ca", "Lung Ca", "Colon Ca"], digits=4))
    return ckpt_path


if __name__ == "__main__":
    train_hierarchical_model()
