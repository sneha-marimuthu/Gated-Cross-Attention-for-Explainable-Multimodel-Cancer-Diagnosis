"""
Gated Cross-Attention (GCA) Module.
Exact mathematical implementation from the GCAEMMCD research paper:
  Section III-B: Bidirectional Cross-Modal Cross-Attention with Sigmoid Gating.

Equations:
  1. Attention(X, Y) = softmax( (X W_Q)(Y W_K)^T / sqrt(d_k) ) (Y W_V)
  2. g = sigma( W_g [X || Y] + b_g )  in (0, 1)^d
  3. X_tilde = X + g (elementwise_mul) Attention(X, Y)
  4. GCA(X, Y) = LayerNorm( X_tilde + FFN(X_tilde) )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PairwiseGatedCrossAttention(nn.Module):
    """
    Directional Cross-Attention from Query Modality X to Context Modality Y,
    modulated by a parameterized sigmoid gate g = sigma(W_g [X || Y] + b_g).
    """
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Parameterized Sigmoid Gate over concatenated modalities [X || Y]
        # Uses hidden interaction layer so gate can learn cross-modal agreement vs discordance
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Sigmoid()
        )

        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        """
        x: Query Modality [B, d]
        y: Context Modality [B, d]
        returns:
          out: Gated updated representation [B, d]
          gate: Mean gate activation value [B] in (0, 1)
        """
        batch_size = x.size(0)

        # Project Q, K, V
        # Reshape to [B, num_heads, 1, head_dim]
        q = self.w_q(x).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(y).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(y).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention: [B, num_heads, 1, 1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # Reshape back to [B, d]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, self.embed_dim)
        attn_out = self.out_proj(attn_out)

        # Compute Sigmoid Gate: g in (0, 1)^d
        xy_concat = torch.cat([x, y], dim=-1)
        g = self.gate_net(xy_concat)  # [B, d]

        # Gated residual integration: X_tilde = X + g * Attn(X, Y)
        x_tilde = x + g * attn_out
        x_norm1 = self.ln1(x_tilde)

        # FFN residual block
        ffn_out = self.ffn(x_norm1)
        out = self.ln2(x_norm1 + self.dropout(ffn_out))

        # Mean gate activation across feature dimensions for interpretability
        mean_gate = torch.mean(g, dim=-1)

        return out, mean_gate

class GatedCrossAttentionFusion(nn.Module):
    """
    Synthesizes all 3 modalities (Image I, Text T, Tabular R) via 6 bidirectional GCA blocks:
    (I->T, T->I, I->R, R->I, T->R, R->T).
    Projects fused concatenation back to unified diagnostic embedding z_fused in R^256.
    """
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim

        # 6 Pairwise Bidirectional GCA Blocks
        self.gca_it = PairwiseGatedCrossAttention(embed_dim, num_heads)
        self.gca_ti = PairwiseGatedCrossAttention(embed_dim, num_heads)
        self.gca_ir = PairwiseGatedCrossAttention(embed_dim, num_heads)
        self.gca_ri = PairwiseGatedCrossAttention(embed_dim, num_heads)
        self.gca_tr = PairwiseGatedCrossAttention(embed_dim, num_heads)
        self.gca_rt = PairwiseGatedCrossAttention(embed_dim, num_heads)

        # Fusion bottleneck: Linear(1536 -> 256)
        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(embed_dim * 6, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, v_i, v_t, v_r):
        """
        v_i: Image embedding [B, 256]
        v_t: Text embedding [B, 256]
        v_r: Tabular embedding [B, 256]
        returns:
          z_fused: Unified diagnostic embedding [B, 256]
          gate_dict: Dictionary of pairwise gate scalar activations
        """
        out_it, g_it = self.gca_it(v_i, v_t)
        out_ti, g_ti = self.gca_ti(v_t, v_i)

        out_ir, g_ir = self.gca_ir(v_i, v_r)
        out_ri, g_ri = self.gca_ri(v_r, v_i)

        out_tr, g_tr = self.gca_tr(v_t, v_r)
        out_rt, g_rt = self.gca_rt(v_r, v_t)

        # Concatenate all 6 directional representations [B, 6 * 256 = 1536]
        concat_rep = torch.cat([out_it, out_ti, out_ir, out_ri, out_tr, out_rt], dim=-1)
        z_fused = self.fusion_bottleneck(concat_rep)

        gate_dict = {
            "image_to_text": g_it,
            "text_to_image": g_ti,
            "image_to_tabular": g_ir,
            "tabular_to_image": g_ri,
            "text_to_tabular": g_tr,
            "tabular_to_text": g_rt
        }

        return z_fused, gate_dict
