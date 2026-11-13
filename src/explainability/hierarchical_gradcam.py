"""
Hierarchical Grad-CAM implementation.
Generates gradient-weighted visual saliency specifically for:
1. Cancer Status (Normal / Benign / Malignant)
2. Cancer Subtype (Breast Ca / Lung Ca / Colorectal Ca)
Clearly documents which head and target class the heatmap represents.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class HierarchicalResNetGradCAM:
    def __init__(self, model):
        """
        model: HierarchicalResNetModel
        """
        self.model = model
        self.model.eval()

    def generate_heatmap(self, input_tensor, head_type="status", target_class=None):
        """
        head_type: 'status' (0: Normal, 1: Benign, 2: Malignant)
                   or 'subtype' (0: Breast, 1: Lung, 2: Colorectal)
        target_class: int (if None, argmax of chosen head)
        Returns:
          heatmap: numpy float array [224, 224] in [0, 1]
          logits: numpy array
          target_class: int
        """
        self.model.zero_grad()
        x = input_tensor.clone().requires_grad_(True)

        # Forward through extractor to layer4
        feats = self.model.extractor.forward_features(x)
        feats.retain_grad()

        # Complete forward
        pooled = self.model.extractor.avgpool(feats)
        raw_gap = torch.flatten(pooled, 1)
        v_I = self.model.extractor.projection(raw_gap)

        if head_type == "status":
            logits = self.model.status_head(v_I)
        else:
            logits = self.model.subtype_head(v_I)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = feats.grad # [1, 2048, 7, 7]
        activations = feats.detach() # [1, 2048, 7, 7]

        # Global average pooling of gradients: alpha_k^c
        alpha = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(alpha * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        if np.max(cam) - np.min(cam) > 1e-8:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        else:
            cam = np.zeros_like(cam)

        return cam, logits.detach().cpu().numpy()[0], target_class

    def overlay_heatmap(self, original_image, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
        if hasattr(original_image, 'resize'):
            orig = np.array(original_image.resize((224, 224)))
        else:
            orig = cv2.resize(original_image, (224, 224))

        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_cam = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)

        overlay = np.float32(colored_cam) * alpha + np.float32(orig) * (1 - alpha)
        overlay = np.uint8(np.clip(overlay, 0, 255))
        return overlay
