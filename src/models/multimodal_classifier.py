"""
Unified Multi-Task Multimodal Classifier & GCA Routing Engine.
Extends the multimodal architecture to simultaneously predict all 4 research paper tasks:
  1. Cancer Status (Normal/Benign vs Malignant) & Anatomical Site (Breast/Lung/Colorectal)
  2. TNM Cancer Staging (Stage 0 to Stage IV ordinal classification)
  3. Treatment Recommendation Ranking (7 prioritized oncology regimens, evaluated by NDCG@3)
  4. 5-Year Survival Risk & Prognostic Estimation (Hazard score & survival probability, evaluated by C-index)

Fully preserves backward compatibility with existing weights while attaching calibrated multi-task heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.gated_cross_attention import GatedCrossAttentionFusion
from src.models.clinical_staging_engine import NUM_TREATMENTS, STAGE_NAMES, TREATMENT_OPTIONS


class ClassifierHead(nn.Module):
    def __init__(self, in_features=256, num_classes=3, dropout=0.30):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class MultiTaskMultimodalClassifier(nn.Module):
    def __init__(self, latent_dim=256, num_classes=3, dropout=0.25):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Image Projection Head: ResNet-50 GAP (2048) -> 256
        self.image_projector = nn.Sequential(
            nn.Linear(2048, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # Gated Cross-Attention Engine
        self.gca_fusion = GatedCrossAttentionFusion(embed_dim=latent_dim, num_heads=4)

        # Early Concatenation Baseline Projector: [v_i || v_t || v_r] = 768 -> 256
        self.early_concat_projector = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

        # Task 1: Status & Anatomical Site Classification Heads
        self.head_image_status = ClassifierHead(latent_dim, 3)
        self.head_image_site = ClassifierHead(latent_dim, num_classes)

        self.head_text_status = ClassifierHead(latent_dim, 3)
        self.head_text_site = ClassifierHead(latent_dim, num_classes)

        self.head_tabular_status = ClassifierHead(latent_dim, 3)
        self.head_tabular_site = ClassifierHead(latent_dim, num_classes)

        self.head_early_status = ClassifierHead(latent_dim, 3)
        self.head_early_site = ClassifierHead(latent_dim, num_classes)

        self.head_gca_status = ClassifierHead(latent_dim, 3)
        self.head_gca_site = ClassifierHead(latent_dim, num_classes)

        # Task 2: TNM Cancer Staging Head (5 Ordinal Stages: Stage 0, I, II, III, IV)
        self.head_staging = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5)
        )

        # Task 3: Treatment Recommendation Ranking Head (7 therapy regimens, scored for NDCG@3)
        self.head_treatment = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_TREATMENTS)
        )

        # Task 4: 5-Year Survival Risk & Prognostic Estimation Head (Continuous log hazard)
        self.head_survival = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1)
        )

    def project_image(self, raw_or_proj_i):
        if raw_or_proj_i is None:
            return None
        if raw_or_proj_i.size(-1) == 2048:
            return self.image_projector(raw_or_proj_i)
        return raw_or_proj_i

    def forward_available(self, v_i=None, v_t=None, v_r=None):
        v_i_proj = self.project_image(v_i)

        has_i = v_i_proj is not None
        has_t = v_t is not None
        has_r = v_r is not None
        count = int(has_i) + int(has_t) + int(has_r)

        if count == 0:
            return {"mode": "NONE", "fusion_status": "MISSING_ALL"}

        out = {
            "v_i": v_i_proj,
            "v_t": v_t,
            "v_r": v_r,
            "z_early": None,
            "z_gca": None,
            "logits_image_status": self.head_image_status(v_i_proj) if has_i else None,
            "logits_image_site": self.head_image_site(v_i_proj) if has_i else None,
            "logits_text_status": self.head_text_status(v_t) if has_t else None,
            "logits_text_site": self.head_text_site(v_t) if has_t else None,
            "logits_tabular_status": self.head_tabular_status(v_r) if has_r else None,
            "logits_tabular_site": self.head_tabular_site(v_r) if has_r else None,
            "logits_early_status": None,
            "logits_early_site": None,
            "logits_gca_status": None,
            "logits_gca_site": None,
            "gate_weights": None,
            "logits_staging": None,
            "treatment_scores": None,
            "hazard_scores": None,
            "survival_probabilities": None
        }

        if count == 1:
            if has_i:
                rep = v_i_proj
                out["mode"] = "UNIMODAL_IMAGE"
            elif has_t:
                rep = v_t
                out["mode"] = "UNIMODAL_TEXT"
            else:
                rep = v_r
                out["mode"] = "UNIMODAL_TABULAR"
            out["fusion_status"] = "NOT_APPLICABLE_UNIMODAL"
            out["logits_staging"] = self.head_staging(rep)
            out["treatment_scores"] = self.head_treatment(rep)
            hz = self.head_survival(rep)
            out["hazard_scores"] = hz
            out["survival_probabilities"] = torch.exp(-torch.clamp(F.softplus(hz), 0.05, 3.50))
            return out

        if count == 2:
            out["mode"] = "PARTIAL_DUAL_UNSUPPORTED"
            out["fusion_status"] = "INSUFFICIENT_MODALITY_SUPPORT"
            return out

        # All 3 modalities are present
        early_concat = torch.cat([v_i_proj, v_t, v_r], dim=-1)
        z_early = self.early_concat_projector(early_concat)
        z_gca, gate_dict = self.gca_fusion(v_i_proj, v_t, v_r)

        out["mode"] = "MULTIMODAL_GCA"
        out["fusion_status"] = "ACTIVE_3WAY_GCA"
        out["z_early"] = z_early
        out["z_gca"] = z_gca
        out["logits_early_status"] = self.head_early_status(z_early)
        out["logits_early_site"] = self.head_early_site(z_early)
        out["logits_gca_status"] = self.head_gca_status(z_gca)
        out["logits_gca_site"] = self.head_gca_site(z_gca)
        out["gate_weights"] = gate_dict

        # Multi-Task Outputs computed from GCA fused representation
        out["logits_staging"] = self.head_staging(z_gca)
        out["treatment_scores"] = self.head_treatment(z_gca)
        hz = self.head_survival(z_gca)
        out["hazard_scores"] = hz
        # S(5-yr) = exp(-Hazard)
        out["survival_probabilities"] = torch.exp(-torch.clamp(F.softplus(hz), 0.05, 3.50))

        return out

    def forward(self, v_i, v_t, v_r):
        return self.forward_available(v_i=v_i, v_t=v_t, v_r=v_r)


# Alias for backward compatibility
UnifiedMultimodalClassifier = MultiTaskMultimodalClassifier
