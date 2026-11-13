"""
Trainer module for Level 1 Image Classification baseline.
Preserves frozen layers 1-3 of ResNet-50 while fine-tuning layer4.2 and projection
with weight decay and cosine annealing on NVIDIA RTX 2050 GPU.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import config
from src.data.dataset import HistopathologyDataset, build_splits
from src.models.image_encoder import ResNet50FeatureExtractor, ImageCancerClassifier, FullImageModel

def train_image_classifier():
    print(f"[*] Initializing Calibrated Image Model on {config.DEVICE}...")
    train_records, val_records, test_records = build_splits()
    
    train_loader = DataLoader(HistopathologyDataset(train_records, is_train=True), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(HistopathologyDataset(val_records, is_train=False), batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(HistopathologyDataset(test_records, is_train=False), batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 1. Instantiate extractor with layer4 unfreezing
    extractor = ResNet50FeatureExtractor(freeze=False).to(config.DEVICE)
    for name, p in extractor.named_parameters():
        if "layer4" in name or "projection" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
            
    classifier = ImageCancerClassifier(latent_dim=config.LATENT_DIM, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    trainable_params = [p for p in extractor.parameters() if p.requires_grad] + list(classifier.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 20
    best_val_acc = 0.0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "level1_image_model.pt")
    
    print(f"[*] Training on {len(train_records)} images, validating on {len(val_records)} images across {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        extractor.train()
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for batch in train_loader:
            imgs = batch["image"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)
            
            optimizer.zero_grad()
            _, v_I = extractor(imgs)
            logits = classifier(v_I)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            
        train_acc = correct / total
        
        # Validation
        extractor.eval()
        classifier.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(config.DEVICE)
                labels = batch["label"].to(config.DEVICE)
                _, v_I = extractor(imgs)
                logits = classifier(v_I)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch:02d}/{num_epochs:02d} - Loss: {total_loss/total:.4f} - Train Acc: {train_acc*100:.1f}% - Val Acc: {val_acc*100:.1f}%")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                "extractor_state_dict": extractor.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc
            }, ckpt_path)
            
    print(f"[+] Training complete! Best validation accuracy: {best_val_acc*100:.1f}%. Saved to {ckpt_path}")
    
    # 2. Evaluation on Held-Out Test Set
    print("\n[*] Evaluating on held-out test split (9 samples: 3 breast, 3 lung, 3 colorectal)...")
    checkpoint = torch.load(ckpt_path, map_location=config.DEVICE)
    extractor.load_state_dict(checkpoint["extractor_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    extractor.eval()
    classifier.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)
            _, v_I = extractor(imgs)
            logits = classifier(v_I)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    target_names = [f"{c.capitalize()} Carcinoma" for c in config.CATEGORIES]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    
    return ckpt_path

if __name__ == "__main__":
    train_image_classifier()
