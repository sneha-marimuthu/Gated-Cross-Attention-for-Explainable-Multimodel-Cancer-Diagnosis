"""
Pretrained ResNet-50 visual feature extractor and task-specific classification head.
Corresponds to Section III-C of the GCAEMMCD paper:
- Backbone: ResNet-50 up to layer4.2 (2048 x 7 x 7)
- Global Average Pooling (GAP) -> 2048-dim embedding
- Linear projection -> v_I in R^256
- Task-specific classification head -> 3 cancer classes
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import config

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)
        
        # We preserve layers up to layer4 for Grad-CAM access
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        
        # Projection layer to latent space d = 256
        self.projection = nn.Sequential(
            nn.Linear(config.IMAGE_RAW_DIM, config.LATENT_DIM),
            nn.LayerNorm(config.LATENT_DIM),
            nn.ReLU()
        )
        
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            # Projection parameters remain trainable
            for p in self.projection.parameters():
                p.requires_grad = True

    def forward_features(self, x):
        """Returns 2048x7x7 feature activation maps before GAP."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        """
        Input: [B, 3, 224, 224]
        Output:
          raw_features: [B, 2048]
          v_I: [B, 256]
        """
        feats = self.forward_features(x)
        pooled = self.avgpool(feats)
        raw_features = torch.flatten(pooled, 1)  # [B, 2048]
        v_I = self.projection(raw_features)       # [B, 256]
        return raw_features, v_I

class ImageCancerClassifier(nn.Module):
    def __init__(self, latent_dim=config.LATENT_DIM, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, v_I):
        """
        Input: v_I [B, 256]
        Output: logits [B, 3]
        """
        return self.classifier(v_I)

class FullImageModel(nn.Module):
    def __init__(self, extractor, classifier):
        super().__init__()
        self.extractor = extractor
        self.classifier = classifier

    def forward(self, x):
        _, v_I = self.extractor(x)
        logits = self.classifier(v_I)
        return logits
