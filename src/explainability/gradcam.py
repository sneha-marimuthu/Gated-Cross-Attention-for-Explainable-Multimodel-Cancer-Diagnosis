"""
Hook-based Grad-CAM implementation on ResNet-50 layer4.2.
Corresponds to Section III-F of the GCAEMMCD paper:
  L_Grad-CAM^c = ReLU( sum_k alpha_k^c * A_k )
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class ResNetGradCAM:
    def __init__(self, model):
        """
        model: FullImageModel containing extractor and classifier
        """
        self.model = model
        self.model.eval()

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Direct gradient computation on layer4 activations without persistent hooks.
        input_tensor: [1, 3, 224, 224] on model device
        target_class: int (if None, uses argmax of model logits)
        Returns:
          heatmap: numpy float array [224, 224] in range [0, 1]
          logits: numpy array [3]
          target_class: int
        """
        self.model.zero_grad()
        
        # Forward pass up to layer4 with gradient tracking enabled
        x = input_tensor.clone().requires_grad_(True)
        feats = self.model.extractor.forward_features(x)
        feats.retain_grad()
        
        # Complete forward pass
        pooled = self.model.extractor.avgpool(feats)
        raw_gap = torch.flatten(pooled, 1)
        v_I = self.model.extractor.projection(raw_gap)
        logits = self.model.classifier(v_I)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = feats.grad # [1, 2048, 7, 7]
        activations = feats.detach() # [1, 2048, 7, 7]

        # Global average pooling of gradients: alpha_k^c (Paper Eq. 6)
        alpha = torch.mean(gradients, dim=(2, 3), keepdim=True) # [1, 2048, 1, 1]
        
        # Weighted combination of forward activation maps
        cam = torch.sum(alpha * activations, dim=1, keepdim=True) # [1, 1, 7, 7]
        cam = F.relu(cam) # Paper Eq. 6
        
        # Bilinear interpolation to 224x224
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize between [0, 1]
        if np.max(cam) - np.min(cam) > 1e-8:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        else:
            cam = np.zeros_like(cam)

        return cam, logits.detach().cpu().numpy()[0], target_class

    def overlay_heatmap(self, original_image, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
        """
        original_image: PIL Image or RGB numpy array [224, 224, 3] in range [0, 255]
        heatmap: float numpy array [224, 224] in [0, 1]
        Returns: overlay RGB image [224, 224, 3] as uint8
        """
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
