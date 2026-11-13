"""
Pretrained Histopathology Foundation Model Encoder for Organ Identification.
Uses Owkin/Phikon (ViT-B/16 self-supervised on 40M+ histology tiles from TCGA).
Kept completely frozen. Feeds an independent Organ Classification Head with an
Uncertainty / Unknown reject option for low confidence and out-of-distribution inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import config

class PhikonOrganEncoder(nn.Module):
    def __init__(self, model_name="owkin/phikon", freeze=True):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.vit = AutoModel.from_pretrained(model_name)
        
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
                
        self.embedding_dim = self.vit.config.hidden_size # 768

    def forward(self, pixel_values):
        """
        Input: pixel_values [B, 3, 224, 224]
        Output: cls_embedding [B, 768]
        """
        outputs = self.vit(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

class OrganClassifierHead(nn.Module):
    def __init__(self, in_features=768, num_organs=3, min_confidence=0.60, min_similarity=0.45):
        """
        Lightweight MLP head for Organ Classification:
          - Breast
          - Lung
          - Colorectal
        Includes dual-criterion rejection:
          1. Softmax confidence threshold (< 0.60 -> Uncertain)
          2. Prototype cosine similarity threshold (< 0.45 -> Out-of-Distribution / Unknown)
        """
        super().__init__()
        self.min_confidence = min_confidence
        self.min_similarity = min_similarity
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_organs)
        )
        
        # Register reference exemplar vectors for OOD rejection
        self.register_buffer("reference_bank", torch.zeros(3, in_features))
        self.has_references = False

    def set_reference_prototypes(self, prototypes):
        """prototypes: [3, 768] tensor of mean embeddings per organ."""
        self.reference_bank = F.normalize(prototypes, p=2, dim=1)
        self.has_references = True

    def forward(self, x):
        """
        Input: x [B, 768]
        Output: logits [B, 3]
        """
        return self.head(x)

    def predict_with_uncertainty(self, x):
        """
        Evaluates input embedding x [1, 768]:
        Returns predicted organ class, probability, confidence dict, and uncertainty flag.
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        max_prob, pred_idx = torch.max(probs, dim=1)
        
        max_prob_val = max_prob.item()
        pred_idx_val = pred_idx.item()
        
        organs = config.CATEGORIES # ['breast', 'lung', 'colorectal']
        
        # Check cosine similarity to known histology distribution
        is_ood = False
        if self.has_references:
            norm_x = F.normalize(x, p=2, dim=1)
            sims = torch.mm(norm_x, self.reference_bank.t())
            max_sim = torch.max(sims).item()
            if max_sim < self.min_similarity:
                is_ood = True
                
        is_low_conf = max_prob_val < self.min_confidence
        
        if is_ood or is_low_conf:
            is_uncertain = True
            if is_ood:
                predicted_organ = "Unknown / Out-of-Distribution"
            else:
                predicted_organ = "Uncertain / Ambiguous Organ"
        else:
            is_uncertain = False
            predicted_organ = organs[pred_idx_val].capitalize()
            
        prob_dict = {
            organs[i].capitalize(): float(probs[0, i].item())
            for i in range(len(organs))
        }
        
        return predicted_organ, max_prob_val, prob_dict, is_uncertain
