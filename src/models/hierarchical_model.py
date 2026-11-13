"""
Hierarchical Multi-Task Vision Architecture:
- Backbone: Pretrained ResNet-50 (with calibrated layer4 fine-tuning)
- Head A: Cancer Status Classifier (Normal vs Benign vs Malignant)
- Head B: Carcinoma Subtype Classifier (Breast Ca vs Lung Ca vs Colorectal Ca)
"""
import torch
import torch.nn as nn
import config
from src.models.image_encoder import ResNet50FeatureExtractor

class HierarchicalResNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Extractor maps [B, 3, 224, 224] -> v_I [B, 256]
        self.extractor = ResNet50FeatureExtractor(freeze=False)
        for name, p in self.extractor.named_parameters():
            if "layer4" in name or "projection" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # Head 1: Cancer Status (Normal: 0, Benign: 1, Malignant: 2)
        self.status_head = nn.Sequential(
            nn.Linear(config.LATENT_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, config.NUM_STATUSES)
        )

        # Head 2: Carcinoma Subtype (Breast: 0, Lung: 1, Colorectal: 2)
        self.subtype_head = nn.Sequential(
            nn.Linear(config.LATENT_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, config.NUM_SUBTYPES)
        )

    def forward(self, x):
        _, v_I = self.extractor(x)
        status_logits = self.status_head(v_I)
        subtype_logits = self.subtype_head(v_I)
        return status_logits, subtype_logits, v_I
