"""
Faithful Level 2 Multimodal Explainability Module.
Fixes prior Level 2 weight-mismatch loophole by computing Grad-CAM directly through:
  - The exact ResNet-50 layer4 feature maps feeding Level 2 (`img_extractor`)
  - The trained Level 2 `UnifiedMultimodalClassifier` projection & prediction heads
  - Supports BOTH Unimodal (`head_image_*`) and True Multimodal GCA (`gca_fusion -> head_gca_*`) backprop!

Also provides gradient-based biomarker importance attribution across the 16 tabular inputs.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class Level2FaithfulGradCAM:
    def __init__(self, img_extractor, classifier):
        self.extractor = img_extractor
        self.classifier = classifier
        self.extractor.eval()
        self.classifier.eval()

    def generate_heatmap(
        self,
        input_tensor,
        pathway="image",
        head_type="site",
        target_class=None,
        v_t=None,
        v_r=None,
    ):
        """
        pathway: 'image' (unimodal ResNet head) or 'gca' (true multimodal GCA fused head back to layer4)
        head_type: 'status' (Normal/Benign/Malignant) or 'site' (Breast/Lung/Colorectal)
        target_class: int or None (uses argmax if None)
        """
        self.extractor.zero_grad()
        self.classifier.zero_grad()

        x = input_tensor.clone().detach().requires_grad_(True)
        feats = self.extractor.forward_features(x)  # [1, 2048, 7, 7]
        feats.retain_grad()

        pooled = self.extractor.avgpool(feats)
        raw_gap = torch.flatten(pooled, 1)  # [1, 2048]
        v_i_proj = self.classifier.image_projector(raw_gap)  # [1, 256]

        if pathway == "gca" and v_t is not None and v_r is not None:
            z_gca, _ = self.classifier.gca_fusion(v_i_proj, v_t.detach(), v_r.detach())
            if head_type == "status":
                logits = self.classifier.head_gca_status(z_gca)
            else:
                logits = self.classifier.head_gca_site(z_gca)
        else:
            if head_type == "status":
                logits = self.classifier.head_image_status(v_i_proj)
            else:
                logits = self.classifier.head_image_site(v_i_proj)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward()

        gradients = feats.grad
        activations = feats.detach()

        if gradients is None:
            return np.zeros((224, 224), dtype=np.float32), logits.detach().cpu().numpy()[0], target_class

        alpha = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(alpha * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cmin, cmax = float(np.min(cam)), float(np.max(cam))
        if cmax - cmin > 1e-8:
            cam = (cam - cmin) / (cmax - cmin)
        else:
            cam = np.zeros_like(cam)

        return cam, logits.detach().cpu().numpy()[0], target_class

    @staticmethod
    def overlay_heatmap(original_image, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
        if hasattr(original_image, "resize"):
            orig = np.array(original_image.convert("RGB").resize((224, 224)))
        else:
            orig = cv2.resize(original_image, (224, 224))

        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
        colored_cam = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)

        overlay = np.float32(colored_cam) * alpha + np.float32(orig) * (1.0 - alpha)
        return np.uint8(np.clip(overlay, 0, 255))

    @staticmethod
    def compute_tabular_saliency(tab_mlp, classifier, tab_tensor_32, head_type="site", target_class=None):
        """
        Computes input-gradient sensitivity (|grad * x|) across observed tabular features.
        Returns list of (feature_index, importance_score) sorted descending.
        """
        tab_mlp.eval()
        classifier.eval()
        tab_mlp.zero_grad()
        classifier.zero_grad()

        x_tab = tab_tensor_32.clone().detach().requires_grad_(True)
        v_r = tab_mlp(x_tab)
        logits = (
            classifier.head_tabular_status(v_r)
            if head_type == "status"
            else classifier.head_tabular_site(v_r)
        )
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())
        score = logits[0, target_class]
        score.backward()

        grad = x_tab.grad.detach().cpu().numpy()[0]
        inp = x_tab.detach().cpu().numpy()[0]
        # Sensitivity over the first 16 value dimensions, masked by availability bits [16:32]
        mask = inp[16:32]
        scores = np.abs(grad[:16]) * (np.abs(inp[:16]) + 0.1) * mask
        total = float(np.sum(scores))
        if total > 1e-8:
            scores = scores / total

        ranked = sorted([(i, float(scores[i])) for i in range(16) if mask[i] > 0.5], key=lambda x: x[1], reverse=True)
        return ranked
