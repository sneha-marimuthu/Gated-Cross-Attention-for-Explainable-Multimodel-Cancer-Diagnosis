"""
Clinical Staging, Treatment Guideline, and Survival Prognostication Module.
Implements ground truth synthesis and rule-based oncology knowledge aligned with NCCN / ESMO clinical standards:
  Task 1: Cancer Classification (Breast, Lung, Colorectal, Normal/Benign)
  Task 2: TNM Cancer Staging (Stage 0, Stage I, Stage II, Stage III, Stage IV)
  Task 3: Treatment Recommendation Ranking (Surgery, Adjuvant Chemotherapy, Targeted Therapy, Immunotherapy, Radiation, Endocrine, Palliative)
  Task 4: 5-Year Survival Risk & Prognostic Estimation (Continuous hazard index & 5-year survival probability)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. CANONICAL CLINICAL DEFINITIONS
# -----------------------------------------------------------------------------
STAGE_NAMES = [
    "Stage 0 (Non-Neoplastic / In Situ)",
    "Stage I (Early Localized)",
    "Stage II (Locally Advanced)",
    "Stage III (Regional Nodal Involvement)",
    "Stage IV (Distant Metastasis / Disseminated)"
]

TREATMENT_OPTIONS = [
    "Surgical Resection / Lumpectomy / Colectomy",   # 0
    "Adjuvant Systemic Chemotherapy (FOLFOX/Carbo)", # 1
    "Targeted Kinase Therapy (EGFR/HER2 Inhibitors)", # 2
    "Immune Checkpoint Blockade (Anti-PD-L1)",       # 3
    "Endocrine / Hormone Therapy (Tamoxifen/AI)",    # 4
    "Radiation Oncology (External Beam / SBRT)",     # 5
    "Supportive / Palliative Symptom Care"           # 6
]
NUM_TREATMENTS = len(TREATMENT_OPTIONS)


# -----------------------------------------------------------------------------
# 2. DETERMINISTIC CLINICAL GROUND TRUTH DERIVATION
# -----------------------------------------------------------------------------
def derive_clinical_targets(category_key, raw_biomarkers, text_narrative=None):
    """
    Derives deterministic, clinically realistic multi-task labels based on
    biological tumor pathology, 16 biomarkers, and clinical observations.
    
    raw_biomarkers indices:
      0: age, 1: sex, 2: smoking, 3: ecog, 4: cea, 5: ca125, 6: ca19_9, 7: ki67,
      8: er, 9: pr, 10: her2, 11: egfr, 12: kras, 13: hb, 14: wbc, 15: plt
    """
    # Default non-malignant
    if "normal" in category_key or "benign" in category_key:
        stage_idx = 0
        # Treatment ranking for normal/benign: Observation/Surgery only if benign, zero chemo/targeted
        if "benign" in category_key:
            treat_rel = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32) # Excision optional
        else:
            treat_rel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32) # No oncological therapy
        survival_prob = 0.98
        hazard_score = 0.05
        return stage_idx, treat_rel, survival_prob, hazard_score

    # Malignant neoplasm: evaluate disease burden
    age = raw_biomarkers[0]
    ecog = raw_biomarkers[3]
    cea = raw_biomarkers[4]
    ca125 = raw_biomarkers[5]
    ca19_9 = raw_biomarkers[6]
    ki67 = raw_biomarkers[7]
    er = raw_biomarkers[8]
    pr = raw_biomarkers[9]
    her2 = raw_biomarkers[10]
    egfr = raw_biomarkers[11]
    kras = raw_biomarkers[12]

    # Staging rules:
    # Stage IV if metastatic keywords in narrative OR extreme tumor markers/mutations with high ECOG
    has_met = False
    if text_narrative:
        text_lower = text_narrative.lower()
        if any(w in text_lower for w in ["metast", "pleural infiltration", "peritoneal", "disseminat", "distant"]):
            has_met = True

    if has_met or (cea > 30.0 and "colorectal" in category_key) or (egfr == 1.0 and ecog >= 2.0 and "lung" in category_key):
        stage_idx = 4 # Stage IV
    elif ki67 > 45.0 or cea > 15.0 or ca19_9 > 35.0 or (her2 == 1.0 and ki67 > 35.0):
        stage_idx = 3 # Stage III (Nodal involvement)
    elif ki67 > 20.0 or cea > 5.0 or ca125 > 25.0:
        stage_idx = 2 # Stage II (Locally advanced)
    else:
        stage_idx = 1 # Stage I (Early localized)

    # Treatment relevance scores (0 to 3 scale for NDCG evaluation):
    treat_rel = np.zeros(NUM_TREATMENTS, dtype=np.float32)

    if "colorectal" in category_key:
        if stage_idx in [1, 2]:
            treat_rel[0] = 3.0 # Colectomy primary
            treat_rel[1] = 2.0 if stage_idx == 2 else 0.0 # Adjuvant FOLFOX
            treat_rel[5] = 1.0 # Radiation if rectal
        elif stage_idx == 3:
            treat_rel[0] = 3.0 # Surgery
            treat_rel[1] = 3.0 # FOLFOX/CAPOX chemo
            treat_rel[3] = 2.0 if kras == 0.0 else 0.0 # Immunotherapy
        else: # Stage IV
            treat_rel[1] = 3.0 # Systemic Chemo
            treat_rel[2] = 3.0 if kras == 0.0 else 1.0 # Targeted (anti-EGFR if KRAS wild-type)
            treat_rel[0] = 1.0 # Palliative resection
            treat_rel[6] = 2.0 # Supportive care

    elif "lung" in category_key:
        if stage_idx == 1:
            treat_rel[0] = 3.0 # Lobectomy / VATS
            treat_rel[5] = 2.0 # SBRT
        elif stage_idx in [2, 3]:
            treat_rel[0] = 2.0 # Resection if resectable
            treat_rel[1] = 3.0 # Platinum doublet chemo
            treat_rel[5] = 3.0 # Radiotherapy
            treat_rel[2] = 3.0 if egfr == 1.0 else 0.0 # EGFR TKI
        else: # Stage IV
            treat_rel[2] = 3.0 if egfr == 1.0 else 1.0 # Osimertinib TKI
            treat_rel[3] = 3.0 if egfr == 0.0 else 1.0 # Pembrolizumab (anti-PD-L1)
            treat_rel[1] = 2.0 # Chemotherapy
            treat_rel[6] = 3.0 # Supportive palliative care

    elif "breast" in category_key:
        if stage_idx in [1, 2]:
            treat_rel[0] = 3.0 # Lumpectomy / Mastectomy
            treat_rel[4] = 3.0 if (er == 1.0 or pr == 1.0) else 0.0 # Tamoxifen / AI
            treat_rel[2] = 3.0 if her2 == 1.0 else 0.0 # Trastuzumab (anti-HER2)
            treat_rel[1] = 2.0 if ki67 > 30.0 else 1.0 # Chemo
            treat_rel[5] = 2.0 # Post-lumpectomy radiation
        elif stage_idx == 3:
            treat_rel[0] = 3.0 # Surgery
            treat_rel[1] = 3.0 # AC-T Chemotherapy
            treat_rel[2] = 3.0 if her2 == 1.0 else 0.0 # Trastuzumab + Pertuzumab
            treat_rel[4] = 3.0 if (er == 1.0 or pr == 1.0) else 0.0 # Endocrine
            treat_rel[5] = 3.0 # Radiation
        else: # Stage IV
            treat_rel[2] = 3.0 if her2 == 1.0 else 0.0 # Anti-HER2
            treat_rel[4] = 3.0 if (er == 1.0 or pr == 1.0) else 0.0 # Endocrine CDK4/6
            treat_rel[1] = 2.0 # Chemotherapy
            treat_rel[6] = 3.0 # Palliative symptom care

    # 5-Year Survival Probability & Cox Hazard Estimation
    # Baseline survival rate decreases with stage, age, high ecog, and high ki67
    base_surv_by_stage = [0.98, 0.91, 0.74, 0.48, 0.16]
    stage_surv = base_surv_by_stage[stage_idx]
    
    # Biomarker risk modifier
    risk_mod = (age - 50.0) * 0.003 + float(ecog) * 0.08 + (ki67 / 100.0) * 0.10
    survival_prob = float(np.clip(stage_surv - risk_mod, 0.05, 0.99))
    hazard_score = float(np.clip(-np.log(survival_prob), 0.01, 3.50))

    return stage_idx, treat_rel, survival_prob, hazard_score


# -----------------------------------------------------------------------------
# 3. NDCG@3 & C-INDEX EVALUATION METRICS
# -----------------------------------------------------------------------------
def compute_ndcg_at_k(y_true_relevance, y_pred_scores, k=3):
    """
    Computes Normalized Discounted Cumulative Gain at rank k (NDCG@k).
    y_true_relevance: [N, num_items]
    y_pred_scores: [N, num_items]
    """
    batch_size = y_true_relevance.shape[0]
    ndcg_list = []

    for i in range(batch_size):
        true_rel = y_true_relevance[i]
        pred_sc = y_pred_scores[i]

        # Top-k predicted indices
        pred_rank = np.argsort(-pred_sc)[:k]
        dcg = np.sum((2.0 ** true_rel[pred_rank] - 1.0) / np.log2(np.arange(2, k + 2)))

        # Ideal top-k indices
        ideal_rank = np.argsort(-true_rel)[:k]
        idcg = np.sum((2.0 ** true_rel[ideal_rank] - 1.0) / np.log2(np.arange(2, k + 2)))

        if idcg > 0:
            ndcg_list.append(dcg / idcg)
        else:
            ndcg_list.append(1.0)

    return float(np.mean(ndcg_list))


def compute_c_index(hazard_scores, survival_probs):
    """
    Computes Harrell's Concordance Index (C-index).
    Higher hazard must correspond to lower survival probability.
    """
    n = len(hazard_scores)
    concordant = 0.0
    permissible = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            # If survival probabilities are distinct
            if survival_probs[i] != survival_probs[j]:
                permissible += 1.0
                # True higher risk = lower survival probability
                true_higher_risk = i if survival_probs[i] < survival_probs[j] else j
                pred_higher_risk = i if hazard_scores[i] > hazard_scores[j] else j
                if true_higher_risk == pred_higher_risk:
                    concordant += 1.0
                elif hazard_scores[i] == hazard_scores[j]:
                    concordant += 0.5

    return float(concordant / permissible) if permissible > 0 else 0.5
