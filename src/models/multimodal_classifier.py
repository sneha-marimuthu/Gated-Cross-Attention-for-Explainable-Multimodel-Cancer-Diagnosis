"""
Unified Dual-Task Multimodal Classifier & Availability-Aware Routing Engine.
Provides dual heads (Disease Status: Normal/Benign/Malignant AND Anatomical Site/Subtype: Breast/Lung/Colorectal)
across 5 pathways:
  1. Image-only baseline (v_I)
  2. Text-only baseline (v_T)
  3. Tabular-only baseline (v_R)
  4. Early Concatenation fusion [v_I || v_T || v_R] (3-way only)
  5. Gated Cross-Attention (GCA) Fusion (z_fused, 3-way only)

Prevents silent dummy injection: missing modalities remain None and fusion/ablations only execute
for strictly valid modality subsets.
"""

import torch
import torch.nn as nn
from src.models.gated_cross_attention import GatedCrossAttentionFusion


class ClassifierHead(nn.Module):
    def __init__(self, in_features=256, num_classes=3, dropout=0.35):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class UnifiedMultimodalClassifier(nn.Module):
    def __init__(self, latent_dim=256, num_classes=3, dropout=0.30):
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

        # Unimodal + Fusion Dual Heads (Status: Normal/Benign/Malignant, Site: Breast/Lung/Colorectal)
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

    def project_image(self, raw_or_proj_i):
        if raw_or_proj_i is None:
            return None
        if raw_or_proj_i.size(-1) == 2048:
            return self.image_projector(raw_or_proj_i)
        return raw_or_proj_i

    def forward_available(self, v_i=None, v_t=None, v_r=None):
        """
        Executes ONLY the valid modality pathways present in the inputs.
        Never synthesizes dummy inputs for missing streams.
        """
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
        }

        if count == 1:
            if has_i:
                out["mode"] = "UNIMODAL_IMAGE"
            elif has_t:
                out["mode"] = "UNIMODAL_TEXT"
            else:
                out["mode"] = "UNIMODAL_TABULAR"
            out["fusion_status"] = "NOT_APPLICABLE_UNIMODAL"
            return out

        if count == 2:
            out["mode"] = "PARTIAL_DUAL_UNSUPPORTED"
            out["fusion_status"] = "INSUFFICIENT_MODALITY_SUPPORT"
            out["fusion_message"] = (
                "Insufficient modality support for this prediction: "
                "3-way GCA fusion requires all three validated streams (Image, Text, and Tabular)."
            )
            return out

        # All 3 modalities are present and validated
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
        return out

    def forward(self, v_i, v_t, v_r):
        return self.forward_available(v_i=v_i, v_t=v_t, v_r=v_r)
