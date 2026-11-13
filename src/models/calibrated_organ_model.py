"""
Calibrated Organ Encoder and Multi-Prototype + Energy OOD Classifier Head (Owkin/Phikon ViT-B).
Supports:
  - Breast
  - Lung
  - Colorectal
  - Unknown / Out-of-Distribution (via multi-prototype cosine distance + logit energy)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


class CalibratedPhikonOrganEncoder(nn.Module):
    def __init__(self, model_name="owkin/phikon", freeze=True):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
        self.vit = AutoModel.from_pretrained(model_name, local_files_only=True)

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.embedding_dim = self.vit.config.hidden_size  # 768

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]


class CalibratedOrganHead(nn.Module):
    def __init__(
        self,
        in_features=768,
        num_organs=3,
        min_confidence=0.50,
        max_cosine_distance=0.535,
        soft_cosine_distance=0.38,
        max_energy=-1.10,
    ):
        """
        Multi-prototype + logit-energy calibrated organ gatekeeper:
          - min_dist <= soft_cosine_distance: confident in-distribution morphology
          - soft_cosine_distance < min_dist <= max_cosine_distance: accepted only when
            in-distribution energy <= max_energy (strong organ activation despite scale shift)
          - min_dist > max_cosine_distance: rejected as Out-of-Distribution
        """
        super().__init__()
        self.min_confidence = min_confidence
        self.max_cosine_distance = max_cosine_distance
        self.soft_cosine_distance = soft_cosine_distance
        self.max_energy = max_energy

        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_organs),
        )
        self.register_buffer("reference_bank", torch.zeros(num_organs, in_features))
        self.has_references = False

    def set_reference_prototypes(self, prototypes):
        normed = F.normalize(prototypes, p=2, dim=1)
        # Re-register buffer cleanly to support multi-prototype [M, 768] or single [3, 768]
        self.register_buffer("reference_bank", normed)
        self.has_references = True

    def load_calibrated_checkpoint(self, ckpt_dict, device):
        sd = {k: v for k, v in ckpt_dict["organ_head_state_dict"].items() if k != "reference_bank"}
        self.load_state_dict(sd, strict=False)
        if "reference_prototypes" in ckpt_dict:
            self.set_reference_prototypes(ckpt_dict["reference_prototypes"].to(device))
        if "calibrated_thresholds" in ckpt_dict:
            th = ckpt_dict["calibrated_thresholds"]
            self.max_cosine_distance = float(th.get("max_cosine_distance", self.max_cosine_distance))
            self.soft_cosine_distance = float(th.get("soft_cosine_distance", self.soft_cosine_distance))
            self.max_energy = float(th.get("max_energy", self.max_energy))

    def forward(self, x):
        return self.head(x)

    def predict_with_calibration(self, x):
        """
        Input: x [N, 768] (single crop N=1 or multi-scale views N>=2 of one specimen)
        Returns:
          predicted_organ (str), confidence (float), prob_dict (dict|None),
          ood_flag (bool), uncertainty_flag (bool), min_dist (float)
        """
        logits = self.forward(x)
        mean_logits = logits.mean(dim=0, keepdim=True)
        probs = torch.softmax(mean_logits, dim=1)
        max_prob, pred_idx = torch.max(probs, dim=1)

        max_prob_val = max_prob.item()
        pred_idx_val = pred_idx.item()
        energy = float(-torch.logsumexp(mean_logits, dim=1).item())
        organs = ["Breast", "Lung", "Colorectal"]

        is_ood = False
        min_dist = 0.0
        if self.has_references:
            norm_x = F.normalize(x, p=2, dim=1)
            sims = torch.mm(norm_x, self.reference_bank.t())  # [N, M]
            max_sim = torch.max(sims).item()
            min_dist = max(0.0, 1.0 - max_sim)
            if min_dist > self.max_cosine_distance:
                is_ood = True
            elif min_dist > self.soft_cosine_distance and self.max_energy is not None and energy > self.max_energy:
                is_ood = True

        is_uncertain = max_prob_val < self.min_confidence

        if is_ood:
            predicted_organ = "Unknown / Out-of-Distribution"
            prob_dict = None
        elif is_uncertain:
            predicted_organ = f"Uncertain ({organs[pred_idx_val]}?)"
            prob_dict = {organs[i]: float(probs[0, i].item()) for i in range(len(organs))}
        else:
            predicted_organ = organs[pred_idx_val]
            prob_dict = {organs[i]: float(probs[0, i].item()) for i in range(len(organs))}

        return predicted_organ, max_prob_val, prob_dict, is_ood, is_uncertain, min_dist
