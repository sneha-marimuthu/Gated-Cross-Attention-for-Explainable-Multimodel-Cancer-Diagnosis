"""
BioBERT Word & Token Saliency Engine for GCAEMMCD.
Computes Input x Gradient word-level attributions through BioBERT v1.1 embeddings,
reassembles WordPiece subwords into natural language clinical terms,
and generates rich HTML highlighted visualizations.
"""

import html
import re
import numpy as np
import torch
import torch.nn.functional as F

import config

SITE_NAMES = ["Breast Carcinoma", "Lung Carcinoma", "Colorectal Carcinoma"]
STATUS_NAMES = ["Normal / Non-Neoplastic", "Benign Neoplasm", "Malignant Carcinoma"]


class BioBERTTextSaliencyExplainer:
    def __init__(self, text_extractor, text_proj, classifier, device=config.DEVICE):
        self.tokenizer = text_extractor.tokenizer
        self.bert = text_extractor.bert
        self.text_proj = text_proj
        self.classifier = classifier
        self.device = device

    def explain_text(self, text, target_class_idx=None, head_type="site"):
        """
        Computes token-level and word-level attributions for the input text.
        head_type: 'site' or 'status'
        returns dict with:
          - html_view: rich HTML with highlighted spans
          - word_attributions: list of (word, score)
          - top_supporting_words: list of (word, score)
          - top_suppressing_words: list of (word, score)
          - target_class_name: str
          - predicted_prob: float
        """
        if not text or not text.strip():
            return {
                "html_view": "<em>No clinical text available for attribution.</em>",
                "word_attributions": [],
                "top_supporting_words": [],
                "top_suppressing_words": [],
                "target_class_name": "None",
                "predicted_prob": 0.0,
            }

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(self.device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Forward with grad tracking on word embeddings
        word_embs = self.bert.embeddings.word_embeddings(input_ids).detach().requires_grad_(True)
        position_ids = self.bert.embeddings.position_ids[:, :input_ids.size(1)]
        token_type_ids = torch.zeros_like(input_ids)
        embeddings = word_embs + self.bert.embeddings.position_embeddings(position_ids) + self.bert.embeddings.token_type_embeddings(token_type_ids)
        embeddings = self.bert.embeddings.LayerNorm(embeddings)
        embeddings = self.bert.embeddings.dropout(embeddings)

        ext_mask = self.bert.get_extended_attention_mask(attention_mask, input_ids.shape)
        encoder_outputs = self.bert.encoder(embeddings, attention_mask=ext_mask)
        cls_rep = encoder_outputs[0][:, 0, :]

        v_t = self.text_proj(cls_rep)
        if head_type == "site":
            logits = self.classifier.head_text_site(v_t)
            names = SITE_NAMES
        else:
            logits = self.classifier.head_text_status(v_t)
            names = STATUS_NAMES

        probs = F.softmax(logits, dim=1)[0]
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(probs).item())

        target_prob = float(probs[target_class_idx].item())
        target_name = names[target_class_idx]

        # Backward pass on target logit
        target_logit = logits[0, target_class_idx]
        target_logit.backward()

        grad = word_embs.grad[0]  # [seq_len, 768]
        token_scores = (word_embs[0] * grad).sum(dim=-1).detach().cpu().numpy()

        raw_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Reassemble subwords (WordPiece ##) into whole words with combined attribution
        words_assembled = []
        scores_assembled = []
        cur_word = ""
        cur_score = 0.0

        for tok, sc in zip(raw_tokens, token_scores):
            if tok in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            if tok.startswith("##"):
                cur_word += tok[2:]
                cur_score += sc
            else:
                if cur_word:
                    words_assembled.append(cur_word)
                    scores_assembled.append(cur_score)
                cur_word = tok
                cur_score = sc
        if cur_word:
            words_assembled.append(cur_word)
            scores_assembled.append(cur_score)

        if not scores_assembled:
            return {
                "html_view": html.escape(text),
                "word_attributions": [],
                "top_supporting_words": [],
                "top_suppressing_words": [],
                "target_class_name": target_name,
                "predicted_prob": target_prob,
            }

        scores_arr = np.array(scores_assembled)
        max_abs = float(np.max(np.abs(scores_arr))) if len(scores_arr) > 0 else 1.0
        if max_abs < 1e-6:
            max_abs = 1.0

        # Build styled HTML
        html_spans = []
        word_attr_tuples = []
        for w, sc in zip(words_assembled, scores_assembled):
            norm_sc = sc / max_abs  # in [-1.0, 1.0]
            word_attr_tuples.append((w, float(sc)))
            safe_w = html.escape(w)

            is_negation = bool(re.match(r"^(no|not|without|negative|unremarkable|denies)$", w, re.IGNORECASE))
            neg_badge = " ⚠️" if is_negation else ""

            if norm_sc > 0.08:
                alpha = min(0.85, max(0.18, abs(norm_sc)))
                style = (
                    f"background-color: rgba(34, 197, 94, {alpha:.2f}); color: #064e3b; "
                    "font-weight: 600; padding: 2px 5px; border-radius: 4px; margin: 1px 2px; display: inline-block;"
                )
            elif norm_sc < -0.08:
                alpha = min(0.85, max(0.18, abs(norm_sc)))
                style = (
                    f"background-color: rgba(239, 68, 68, {alpha:.2f}); color: #7f1d1d; "
                    "font-weight: 600; padding: 2px 5px; border-radius: 4px; margin: 1px 2px; display: inline-block;"
                )
            else:
                style = "color: #334155; padding: 2px 1px; display: inline-block;"

            span = f'<span style="{style}" title="Attribution: {sc:+.4f} (towards {target_name})">{safe_w}{neg_badge}</span>'
            html_spans.append(span)

        # Container styling
        html_view = (
            f'<div style="font-family: ui-sans-serif, system-ui, sans-serif; line-height: 1.8; '
            f'padding: 16px; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; color: #1e293b;">'
            f'<div style="font-size: 0.85rem; font-weight: 700; color: #475569; margin-bottom: 8px;">'
            f'Target Prediction: <span style="color: #0f172a;">{target_name}</span> ({target_prob*100:.1f}%) | '
            f'<span style="color: #16a34a;">■ Supporting</span> / <span style="color: #dc2626;">■ Suppressing</span>'
            f'</div>'
            f'{" ".join(html_spans)}'
            f'</div>'
        )

        sorted_by_sc = sorted(word_attr_tuples, key=lambda x: x[1], reverse=True)
        top_supporting = [x for x in sorted_by_sc if x[1] > 0][:5]
        top_suppressing = [x for x in sorted_by_sc[::-1] if x[1] < 0][:5]

        return {
            "html_view": html_view,
            "word_attributions": word_attr_tuples,
            "top_supporting_words": top_supporting,
            "top_suppressing_words": top_suppressing,
            "target_class_name": target_name,
            "predicted_prob": target_prob,
        }
