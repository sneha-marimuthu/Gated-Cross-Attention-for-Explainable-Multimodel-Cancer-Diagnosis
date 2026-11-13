"""
BioBERT Clinical Narrative Text Encoder.
Extracts contextual clinical semantic representations from free-text pathology notes.
Model: dmis-lab/biobert-v1.1 (PubMed + PMC pretrained, frozen).
Outputs projected text vector v_T in R^256.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import config

class BioBERTTextEncoder(nn.Module):
    def __init__(self, model_name="dmis-lab/biobert-v1.1", freeze=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
                
        self.hidden_dim = self.bert.config.hidden_size # 768

    def forward(self, text_list, device=None):
        """
        Tokenizes and extracts the [CLS] embedding from BioBERT.
        text_list: list of strings or single string
        """
        if isinstance(text_list, str):
            text_list = [text_list]

        if device is None:
            device = next(self.parameters()).device

        tokens = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        outputs = self.bert(**tokens)
        # [CLS] token representation: shape [B, 768]
        cls_rep = outputs.last_hidden_state[:, 0, :]
        return cls_rep

class TextProjectionHead(nn.Module):
    def __init__(self, in_features=768, out_features=256, dropout=0.30):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.projector(x)
