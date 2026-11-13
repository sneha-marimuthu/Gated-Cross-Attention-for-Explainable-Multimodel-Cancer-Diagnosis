"""
Tabular Biomarker MLP Encoder (Masked Missingness-Aware).
Processes 16 standardized clinical biomarkers + 16 binary availability mask bits (32-D total)
into latent space v_R in R^256.

Distinguishes observed negative/zero values [0.0, 1.0] from 'Unknown / Not Tested' [0.0, 0.0].
"""

import torch
import torch.nn as nn


class TabularBiomarkerMLP(nn.Module):
    def __init__(self, in_features=32, latent_dim=256, dropout=0.3):
        super().__init__()
        self.in_features = in_features
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        """
        x: [B, 32] masked biomarker float tensor ([norm_16, mask_16])
        returns: [B, 256] projected biomarker vector v_R
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.encoder(x)
