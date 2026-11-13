"""
Training and evaluation module for the Organ Identification Head.
Extracts frozen Phikon ViT-B embeddings from the existing labelled dataset,
computes reference organ prototypes for OOD rejection, and trains the lightweight classification head.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import config
from src.data.dataset import HistopathologyDataset, build_splits
from src.models.organ_encoder import PhikonOrganEncoder, OrganClassifierHead

def train_organ_head():
    print(f"[*] Training Organ Identification Head with frozen Phikon ViT-B on {config.DEVICE}...")
    train_records, val_records, test_records = build_splits()
    
    train_loader = DataLoader(HistopathologyDataset(train_records, is_train=True), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(HistopathologyDataset(val_records, is_train=False), batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(HistopathologyDataset(test_records, is_train=False), batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 1. Instantiate frozen foundation encoder
    encoder = PhikonOrganEncoder(freeze=True).to(config.DEVICE)
    encoder.eval()
    
    # 2. Compute organ reference prototypes for OOD detection
    print("[*] Computing reference organ prototypes from training split...")
    prototypes = torch.zeros(3, 768).to(config.DEVICE)
    class_counts = torch.zeros(3).to(config.DEVICE)
    
    with torch.no_grad():
        for r in train_records:
            from PIL import Image
            from src.data.image_transforms import preprocess_image
            img = Image.open(r["image_path"]).convert("RGB")
            t = preprocess_image(img).to(config.DEVICE)
            emb = encoder(t)
            lbl = r["label"]
            prototypes[lbl] += emb[0]
            class_counts[lbl] += 1
            
    for i in range(3):
        prototypes[i] = prototypes[i] / class_counts[i]
        
    # 3. Instantiate lightweight head
    organ_head = OrganClassifierHead(in_features=768, num_organs=3, min_confidence=0.60, min_similarity=0.45).to(config.DEVICE)
    organ_head.set_reference_prototypes(prototypes)
    
    optimizer = torch.optim.AdamW(organ_head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 20
    best_val_acc = 0.0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "organ_classifier_head.pt")
    
    print(f"[*] Training organ classification head across {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        organ_head.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for batch in train_loader:
            imgs = batch["image"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)
            
            optimizer.zero_grad()
            with torch.no_grad():
                embeddings = encoder(imgs)
                
            logits = organ_head(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            
        train_acc = correct / total
        
        # Validation
        organ_head.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(config.DEVICE)
                labels = batch["label"].to(config.DEVICE)
                embeddings = encoder(imgs)
                logits = organ_head(embeddings)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch:02d}/{num_epochs:02d} - Loss: {total_loss/total:.4f} - Train Acc: {train_acc*100:.1f}% - Val Acc: {val_acc*100:.1f}%")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                "organ_head_state_dict": organ_head.state_dict(),
                "reference_prototypes": prototypes.cpu(),
                "val_acc": val_acc,
                "epoch": epoch
            }, ckpt_path)
            
    print(f"[+] Organ Head training complete! Best Val Acc: {best_val_acc*100:.1f}%. Saved to {ckpt_path}")
    
    # 4. Final Evaluation on Held-out Test Split
    print("\n[*] Evaluating on held-out test split (9 samples: 3 breast, 3 lung, 3 colorectal)...")
    checkpoint = torch.load(ckpt_path, map_location=config.DEVICE)
    organ_head.load_state_dict(checkpoint["organ_head_state_dict"])
    organ_head.set_reference_prototypes(checkpoint["reference_prototypes"].to(config.DEVICE))
    organ_head.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)
            embeddings = encoder(imgs)
            logits = organ_head(embeddings)
            preds = torch.argmax(logits, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    target_names = [c.capitalize() for c in config.CATEGORIES]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    
    return ckpt_path

if __name__ == "__main__":
    train_organ_head()
