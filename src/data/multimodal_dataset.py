"""
Multimodal Dataset Module for GCAEMMCD Level 2 (Scientifically Hardened).
Constructs paired tri-modal samples:
1. Histopathology Image (H&E patch)
2. Clinical Narrative Text (Synchronized patient vignettes, biopsy reports & negated non-malignant variants)
3. Structured Biomarkers (16 clinical parameters + 16 availability mask bits -> 32-D vector)

Fixes prior loopholes:
- Preserves hierarchical labels (organ_label, status_label, subtype_label) instead of collapsing Normal/Benign into Carcinoma.
- Synchronizes age, sex, and smoking metadata between clinical narratives and tabular vectors.
- Distinguishes explicit '0 / Negative' from 'Unknown / Not Tested' via binary availability masking.
- Enforces physiological bounds and detects lab-unit errors instead of silent clipping.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import config
from src.data.image_transforms import preprocess_image
from src.data.hierarchical_dataset import build_hierarchical_splits

BIOMARKER_NAMES = [
    "age",              # 0: Years [15 - 105]
    "sex",              # 1: 0=Male, 1=Female
    "smoking_pack_yrs", # 2: Pack-years [0 - 120]
    "ecog_score",       # 3: ECOG [0, 1, 2, 3]
    "cea_level",        # 4: ng/mL [0 - 200]
    "ca125_level",      # 5: U/mL [0 - 300]
    "ca19_9_level",     # 6: U/mL [0 - 400]
    "ki67_index",       # 7: % [0 - 100]
    "er_status",        # 8: 0=Negative, 1=Positive
    "pr_status",        # 9: 0=Negative, 1=Positive
    "her2_status",      # 10: 0=Negative, 1=Positive
    "egfr_mutation",    # 11: 0=Wild-type, 1=Mutant
    "kras_mutation",    # 12: 0=Wild-type, 1=Mutant
    "hemoglobin",       # 13: g/dL [4.0 - 22.0]
    "wbc_count",        # 14: x10^3/uL [0.5 - 100.0]
    "platelet_count"    # 15: x10^3/uL [10 - 1500]
]

# Normalization bounds (aligned with typical oncology cohort ranges)
NORM_MINS = np.array([20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 3.0, 100.0], dtype=np.float32)
NORM_MAXS = np.array([90.0, 1.0, 60.0, 3.0, 100.0, 150.0, 200.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 18.0, 20.0, 500.0], dtype=np.float32)

# Hard physiological / unit-sanity bounds (reject outside these rather than silently clipping)
PHYS_BOUNDS = {
    "age": (15.0, 105.0, "years"),
    "sex": (0.0, 1.0, "binary"),
    "smoking_pack_yrs": (0.0, 120.0, "pack-years"),
    "ecog_score": (0.0, 4.0, "score"),
    "cea_level": (0.0, 500.0, "ng/mL"),
    "ca125_level": (0.0, 1000.0, "U/mL"),
    "ca19_9_level": (0.0, 2000.0, "U/mL"),
    "ki67_index": (0.0, 100.0, "%"),
    "er_status": (0.0, 1.0, "binary"),
    "pr_status": (0.0, 1.0, "binary"),
    "her2_status": (0.0, 1.0, "binary"),
    "egfr_mutation": (0.0, 1.0, "binary"),
    "kras_mutation": (0.0, 1.0, "binary"),
    "hemoglobin": (4.0, 22.0, "g/dL"),
    "wbc_count": (0.5, 100.0, "x10^3/uL"),
    "platelet_count": (10.0, 1500.0, "x10^3/uL"),
}

# Preserve backward-compatible summary dictionary for UI preset buttons
CLINICAL_NARRATIVES = {
    "breast_malignant": (
        "58-year-old female, lifelong non-smoker, presenting with firm spiculated left breast mass on mammography. "
        "Core needle biopsy reveals high-grade invasive ductal carcinoma with nuclear pleomorphism, "
        "mitotic figures, stromal desmoplasia, ER+/PR+ profile and elevated Ki-67 proliferation."
    ),
    "breast_benign": (
        "34-year-old female with mobile painless right breast nodule. Ultrasound shows well-circumscribed oval lesion. "
        "Excisional biopsy confirms benign fibroadenoma with compressed glandular spaces and uniform epithelium; "
        "negative for malignancy, no evidence of breast carcinoma or atypia."
    ),
    "lung_malignant": (
        "68-year-old male with 35 pack-year smoking history presenting with persistent cough, weight loss, and right upper lobe mass. "
        "CT-guided biopsy confirms poorly differentiated lung adenocarcinoma with mucin production, pleomorphic nucleoli, "
        "and desmoplastic pleural infiltration."
    ),
    "lung_normal": (
        "46-year-old non-smoking male evaluated after incidental chest shadow. "
        "Bronchoscopic transbronchial biopsy reveals normal lung parenchyma with intact alveolar septa, thin pneumocytes, "
        "and clear airspaces; negative for carcinoma or neoplastic proliferation."
    ),
    "colorectal_malignant": (
        "64-year-old male with iron-deficiency anemia, altered bowel habits, and elevated CEA. "
        "Colonoscopy biopsy of ulcerated sigmoid mass demonstrates infiltrating cribriform glands with dirty necrosis, "
        "hyperchromatic nuclei, and muscularis propria invasion confirming colorectal adenocarcinoma."
    ),
    "colorectal_normal": (
        "52-year-old female undergoing routine surveillance colonoscopy; asymptomatic with normal blood counts. "
        "Mucosal biopsy shows preserved tubular crypt architecture, abundant goblet cells, and healthy lamina propria; "
        "ruled out colorectal adenocarcinoma, negative for dysplasia."
    ),
}


def format_synchronized_narrative(category_key, raw_vec, variant_idx=0, split_role="train"):
    """
    Constructs medically realistic clinical narratives synchronized with patient age, sex, and smoking.
    Uses STRICTLY DISJOINT template pools for 'train' vs 'eval' (val/test) so BioBERT cannot memorize
    exact evaluation phrasings or trivial class-label strings.
    Also includes negative-prior-history clauses and pre-biopsy descriptions.
    """
    age = int(round(float(raw_vec[0])))
    sex_str = "female" if raw_vec[1] >= 0.5 else "male"
    smoking = float(raw_vec[2])
    if smoking < 1.0:
        smoke_phrase = "lifelong non-smoker"
    else:
        smoke_phrase = f"with a {int(round(smoking))} pack-year smoking history"

    cea = float(raw_vec[4])
    ca125 = float(raw_vec[5])
    ca19_9 = float(raw_vec[6])
    hb = float(raw_vec[13])

    train_templates = {
        "breast_malignant": [
            f"{age}-year-old {sex_str}, {smoke_phrase}. Core needle biopsy of palpable mammary mass reveals invasive ductal architecture with marked nuclear pleomorphism, high mitotic rate, and desmoplastic stromal invasion.",
            f"Clinical workup in {age}-year-old {sex_str} ({smoke_phrase}) with irregular BI-RADS 5 left upper outer lesion and axillary fullness. Tissue histology confirms malignant epithelial nests invading fibrous stroma.",
            f"Surgical pathology consultation for {age}-year-old {sex_str}: atypical cohesive glandular sheets, focal microcalcifications, and elevated proliferative activity consistent with mammary adenocarcinoma.",
            f"{age}-year-old {sex_str}, no prior history of colon or pulmonary disease. Core specimen of spiculated pectoral nodule shows high-grade infiltrating carcinoma with stromal desmoplasia.",
            f"Oncology review ({age}yo {sex_str}, CA-125 {ca125:.0f} U/mL): core mammary sampling demonstrates hyperchromatic pleomorphic nuclei infiltrating adipose tissue, diagnostic of carcinoma.",
        ],
        "breast_benign": [
            f"{age}-year-old {sex_str}, {smoke_phrase}, evaluated for well-circumscribed breast nodule. Biopsy confirms benign fibroadenoma with bilayered epithelium; negative for carcinoma and no evidence of malignancy.",
            f"Screening follow-up of {age}-year-old {sex_str} ({smoke_phrase}) with smooth mobile mammary lump. Microscopy shows intact myoepithelial layer and intracanalicular stroma; ruled out invasive cancer.",
            f"Asymptomatic {age}-year-old {sex_str} with hemoglobin {hb:.1f} g/dL. Histopathology shows benign proliferative mammary lesion without cytological atypia or carcinoma.",
            f"{age}-year-old {sex_str} presenting with tender cyclical breast lump on ultrasound. Excisional tissue shows benign sclerosing adenosis and uniform ducts, negative for malignancy.",
            f"Routine breast clinic note ({age}yo {sex_str}): smooth rubbery painless mass, stromal hyalinization with preserved lobular architecture and absence of neoplastic atypia.",
        ],
        "lung_malignant": [
            f"{age}-year-old {sex_str}, {smoke_phrase}, presenting with progressive dyspnea and spiculated pulmonary nodule. Transthoracic biopsy demonstrates invasive lung adenocarcinoma with atypical nucleoli and stromal desmoplasia.",
            f"Oncology referral for {age}-year-old {sex_str} ({smoke_phrase}) with weight loss, cough, and CEA {cea:.1f} ng/mL. Bronchial tissue sections reveal malignant glandular architecture infiltrating alveoli.",
            f"{age}-year-old {sex_str}, {smoke_phrase}. CT-guided core biopsy of right hilar mass confirms non-small cell pulmonary malignancy with mucinous acinar proliferation and pleural invasion.",
            f"{age}-year-old {sex_str} ({smoke_phrase}), negative for prior breast malignancy. Pleural-based pulmonary mass shows pleomorphic acinar carcinoma invading respiratory parenchyma.",
            f"Thoracic pathology ({age}yo {sex_str}): irregular lower lobe parenchymal lesion with desmoplastic stroma and hyperchromatic mitotic cells diagnostic of bronchogenic carcinoma.",
        ],
        "lung_normal": [
            f"{age}-year-old {sex_str}, {smoke_phrase}, undergoing evaluation for transient pulmonary opacity. Biopsy shows normal alveolar septa, thin pneumocytes, and clear airspaces; negative for lung carcinoma.",
            f"Routine thoracic clearance in {age}-year-old {sex_str} ({smoke_phrase}) with hemoglobin {hb:.1f} g/dL. Tissue histology demonstrates benign unremarkable pulmonary parenchyma; malignancy ruled out.",
            f"{age}-year-old {sex_str}, {smoke_phrase}. Transbronchial sample contains intact ciliated respiratory mucosa and normal alveoli without neoplastic changes or adenocarcinoma.",
            f"{age}-year-old {sex_str} ({smoke_phrase}) with chronic bronchitis and mild COPD symptoms. Endobronchial biopsy shows reactive inflammation only, negative for pulmonary carcinoma.",
            f"Surveillance thoracic washings and parenchymal biopsy in {age}-year-old {sex_str} ({smoke_phrase}): preserved alveolar spaces, no atypia, no evidence of malignancy.",
        ],
        "colorectal_malignant": [
            f"{age}-year-old {sex_str}, {smoke_phrase}, with hematochezia, anemia (Hb {hb:.1f} g/dL), and CEA {cea:.1f} ng/mL. Endoscopic biopsy confirms colorectal adenocarcinoma with cribriform glands and dirty necrosis.",
            f"GI surgical consultation for {age}-year-old {sex_str} ({smoke_phrase}) with circumferential sigmoid mass. Histopathology shows infiltrating stratified hyperchromatic epithelial glands invading muscularis propria.",
            f"{age}-year-old {sex_str}, {smoke_phrase}. Colonoscopic tissue specimen demonstrates invasive intestinal-type adenocarcinoma with high mitotic index and desmoplastic stroma.",
            f"{age}-year-old {sex_str}, no prior history of breast cancer, colonoscopy demonstrates ulcerated rectosigmoid mass with malignant cribriform proliferation and CA19-9 {ca19_9:.0f} U/mL.",
            f"Lower GI pathology ({age}yo {sex_str}): cecal biopsy confirms invasive glandular carcinoma penetrating submucosa with necrotic intraluminal debris.",
        ],
        "colorectal_normal": [
            f"{age}-year-old {sex_str}, {smoke_phrase}, presenting for preventive surveillance colonoscopy. Biopsy reveals normal colonic mucosa with regular parallel crypts and goblet cells; negative for colorectal carcinoma.",
            f"Asymptomatic {age}-year-old {sex_str} ({smoke_phrase}) with CEA {cea:.1f} ng/mL. Pathology confirms unremarkable lamina propria and intact crypt architecture; adenocarcinoma ruled out.",
            f"{age}-year-old {sex_str}, {smoke_phrase}. Endoscopic rectal mucosal biopsy demonstrates preserved glandular epithelium without dysplasia or evidence of malignancy.",
            f"{age}-year-old {sex_str} evaluated for mild IBS-like bowel changes. Random mucosal samples show healthy tubular crypts and abundant mucin, negative for neoplasia.",
            f"Screening lower endoscopy in {age}-year-old {sex_str} ({smoke_phrase}, Hb {hb:.1f} g/dL): normal non-neoplastic mucosal biopsy with ruled out carcinoma.",
        ],
    }

    # Strictly disjoint unseen paraphrases reserved exclusively for Val / Test evaluation
    eval_templates = {
        "breast_malignant": [
            f"{age}yo {sex_str} ({smoke_phrase}) referred for irregular BI-RADS 4C pectoral mass; needle sampling reveals malignant mammary ducts with pleomorphism.",
            f"Pathology evaluation in {age}-year-old {sex_str} shows invasive carcinoma arising in breast lobular stroma with desmoplastic reaction.",
        ],
        "breast_benign": [
            f"{age}yo {sex_str} ({smoke_phrase}) with painless encapsulated breast lesion; excisional microscopy confirms fibroadenoma free of atypia or malignancy.",
            f"Follow-up in {age}-year-old {sex_str} shows normal blood counts (Hb {hb:.1f} g/dL) and benign proliferative mammary glands without cancer.",
        ],
        "lung_malignant": [
            f"{age}yo {sex_str} ({smoke_phrase}) with persistent hemoptysis and hilar opacity; core biopsy establishes primary bronchogenic adenocarcinoma.",
            f"Thoracic oncology note for {age}-year-old {sex_str} (CEA {cea:.1f} ng/mL): pulmonary tissue confirms invasive malignant acinar glands.",
        ],
        "lung_normal": [
            f"{age}yo {sex_str} ({smoke_phrase}) evaluated for airway cough; bronchoscopic sampling reveals intact respiratory alveoli negative for tumor.",
            f"Clear thoracic parenchymal biopsy in {age}-year-old {sex_str} ({smoke_phrase}): thin alveolar walls with zero evidence of carcinoma.",
        ],
        "colorectal_malignant": [
            f"{age}yo {sex_str} ({smoke_phrase}) with iron-deficiency anemia (Hb {hb:.1f}) and positive FOBT; sigmoid biopsy confirms invasive intestinal adenocarcinoma.",
            f"Surgical GI workup in {age}-year-old {sex_str} (CEA {cea:.1f}): descending colonic lesion demonstrates malignant glands with dirty necrosis.",
        ],
        "colorectal_normal": [
            f"{age}yo {sex_str} ({smoke_phrase}) after routine polyp surveillance; ascending mucosal histology confirms healthy crypts without dysplasia.",
            f"Surveillance sigmoid sample in {age}-year-old {sex_str}: preserved goblet cell lining and lamina propria, free of neoplastic growth.",
        ],
    }

    pool = eval_templates[category_key] if split_role == "eval" else train_templates[category_key]
    return pool[variant_idx % len(pool)]


def generate_biomarker_vector(category_key, seed=None):
    """
    Generates a medically coherent 16-parameter biomarker vector with realistic
    clinical boundary overlap across Normal, Benign, and Malignant cohorts
    (e.g., COPD smokers with borderline CEA, low-Ki67 Luminal A breast Ca,
    and CEA-normal colorectal Ca) so the MLP cannot exploit trivial hard thresholds.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    if category_key == "breast_malignant":
        age = rng.uniform(38.0, 80.0)
        sex = 1.0 if rng.rand() < 0.97 else 0.0
        smoking = rng.uniform(0.0, 18.0)
        ecog = rng.choice([0.0, 1.0, 2.0], p=[0.45, 0.42, 0.13])
        cea = rng.uniform(0.8, 12.0)
        # Include ~25% indolent Luminal A / early breast cases with normal CA125 & lower Ki67
        if rng.rand() < 0.25:
            ca125 = rng.uniform(10.0, 28.0)
            ki67 = rng.uniform(8.0, 22.0)
        else:
            ca125 = rng.uniform(22.0, 130.0)
            ki67 = rng.uniform(20.0, 78.0)
        ca19_9 = rng.uniform(4.0, 34.0)
        er = rng.choice([0.0, 1.0], p=[0.25, 0.75])
        pr = rng.choice([0.0, 1.0], p=[0.35, 0.65])
        her2 = rng.choice([0.0, 1.0], p=[0.75, 0.25])
        egfr = 0.0
        kras = 0.0
        hb = rng.uniform(9.8, 14.2)
        wbc = rng.uniform(4.8, 11.8)
        plt = rng.uniform(165.0, 400.0)

    elif category_key == "breast_benign":
        age = rng.uniform(20.0, 58.0)
        sex = 1.0
        smoking = rng.uniform(0.0, 10.0)
        ecog = rng.choice([0.0, 1.0], p=[0.85, 0.15])
        cea = rng.uniform(0.4, 3.8)
        ca125 = rng.uniform(7.0, 36.0)
        ca19_9 = rng.uniform(3.0, 25.0)
        ki67 = rng.uniform(2.0, 16.0)
        er = rng.choice([0.0, 1.0], p=[0.20, 0.80])
        pr = rng.choice([0.0, 1.0], p=[0.25, 0.75])
        her2 = 0.0
        egfr = 0.0
        kras = 0.0
        hb = rng.uniform(11.4, 15.0)
        wbc = rng.uniform(4.2, 9.8)
        plt = rng.uniform(175.0, 355.0)

    elif category_key == "lung_malignant":
        age = rng.uniform(46.0, 84.0)
        # Include ~25% non-smoker EGFR-mutant adenocarcinoma subgroup
        is_nonsmoke_egfr = rng.rand() < 0.25
        if is_nonsmoke_egfr:
            sex = rng.choice([0.0, 1.0], p=[0.35, 0.65])
            smoking = rng.uniform(0.0, 5.0)
            egfr = 1.0
            kras = 0.0
        else:
            sex = rng.choice([0.0, 1.0], p=[0.62, 0.38])
            smoking = rng.uniform(12.0, 60.0)
            egfr = rng.choice([0.0, 1.0], p=[0.70, 0.30])
            kras = rng.choice([0.0, 1.0], p=[0.68, 0.32])
        ecog = rng.choice([0.0, 1.0, 2.0, 3.0], p=[0.28, 0.44, 0.20, 0.08])
        cea = rng.uniform(1.8, 60.0)
        ca125 = rng.uniform(8.0, 42.0)
        ca19_9 = rng.uniform(5.0, 36.0)
        ki67 = rng.uniform(18.0, 82.0)
        er = 0.0
        pr = 0.0
        her2 = 0.0
        hb = rng.uniform(9.2, 14.0)
        wbc = rng.uniform(5.5, 14.8)
        plt = rng.uniform(195.0, 440.0)

    elif category_key == "lung_normal":
        age = rng.uniform(32.0, 74.0)
        sex = rng.choice([0.0, 1.0], p=[0.52, 0.48])
        # Realistic smoker / COPD controls overlapping smoking & mildly elevated reactive CEA
        smoking = rng.uniform(0.0, 40.0)
        ecog = rng.choice([0.0, 1.0], p=[0.75, 0.25])
        cea = rng.uniform(0.5, 5.2 if smoking > 15.0 else 3.2)
        ca125 = rng.uniform(5.0, 26.0)
        ca19_9 = rng.uniform(3.0, 24.0)
        ki67 = rng.uniform(1.0, 10.0)
        er = 0.0
        pr = 0.0
        her2 = 0.0
        egfr = 0.0
        kras = 0.0
        hb = rng.uniform(12.0, 16.0)
        wbc = rng.uniform(4.2, 10.8)
        plt = rng.uniform(165.0, 340.0)

    elif category_key == "colorectal_malignant":
        age = rng.uniform(45.0, 85.0)
        sex = rng.choice([0.0, 1.0], p=[0.55, 0.45])
        smoking = rng.uniform(0.0, 35.0)
        ecog = rng.choice([0.0, 1.0, 2.0], p=[0.38, 0.44, 0.18])
        # ~25% of colon cancers present with normal-borderline CEA (<5) & CA19-9
        if rng.rand() < 0.25:
            cea = rng.uniform(1.4, 5.0)
            ca19_9 = rng.uniform(8.0, 32.0)
        else:
            cea = rng.uniform(6.0, 95.0)
            ca19_9 = rng.uniform(24.0, 175.0)
        ca125 = rng.uniform(6.0, 34.0)
        ki67 = rng.uniform(22.0, 86.0)
        er = 0.0
        pr = 0.0
        her2 = 0.0
        egfr = 0.0
        kras = rng.choice([0.0, 1.0], p=[0.50, 0.50])
        hb = rng.uniform(7.8, 13.4)
        wbc = rng.uniform(5.5, 13.8)
        plt = rng.uniform(195.0, 450.0)

    else:  # colorectal_normal
        age = rng.uniform(35.0, 76.0)
        sex = rng.choice([0.0, 1.0], p=[0.5, 0.5])
        smoking = rng.uniform(0.0, 22.0)
        ecog = rng.choice([0.0, 1.0], p=[0.85, 0.15])
        cea = rng.uniform(0.5, 4.2)
        ca125 = rng.uniform(5.0, 25.0)
        ca19_9 = rng.uniform(4.0, 30.0)
        ki67 = rng.uniform(2.0, 12.0)
        er = 0.0
        pr = 0.0
        her2 = 0.0
        egfr = 0.0
        kras = 0.0
        hb = rng.uniform(11.2, 15.6)
        wbc = rng.uniform(4.3, 9.8)
        plt = rng.uniform(165.0, 350.0)

    return np.array([
        age, sex, smoking, ecog, cea, ca125, ca19_9, ki67,
        er, pr, her2, egfr, kras, hb, wbc, plt
    ], dtype=np.float32)


def normalize_biomarkers(raw_vec):
    """Normalized 16-D vector in [0, 1]."""
    norm_vec = (raw_vec - NORM_MINS) / (NORM_MAXS - NORM_MINS + 1e-6)
    return np.clip(norm_vec, 0.0, 1.0)


def encode_biomarkers_with_mask(raw_vec, obs_mask=None):
    """
    Constructs a 32-D tensor representation: [norm_values (16) || obs_mask (16)].
    When obs_mask[i] == 0.0 ('Unknown / Not Tested'), norm_values[i] is explicitly zeroed
    so 'Not Tested' ([0.0, 0.0]) is mathematically distinct from 'Negative / Zero' ([0.0, 1.0]).
    """
    raw_arr = np.asarray(raw_vec, dtype=np.float32)
    if obs_mask is None:
        mask_arr = np.ones(16, dtype=np.float32)
    else:
        mask_arr = np.asarray(obs_mask, dtype=np.float32)

    norm_vec = normalize_biomarkers(raw_arr)
    norm_vec = norm_vec * mask_arr
    return np.concatenate([norm_vec, mask_arr], axis=0).astype(np.float32)


def validate_tabular_biomarkers(raw_dict):
    """
    Rigorous validation & physiological unit auditing for 16 biomarkers.
    Supports None / 'Not Tested / Unknown' for optional tumor & molecular markers.

    raw_dict: dict mapping BIOMARKER_NAMES -> float | str('Not Tested / Unknown') | None
    Returns:
      is_valid (bool), reason (str), warnings (list[str]),
      vec_32 (np.ndarray or None), raw_vec (np.ndarray), obs_mask (np.ndarray)
    """
    errors = []
    warnings = []
    raw_vec = np.zeros(16, dtype=np.float32)
    obs_mask = np.zeros(16, dtype=np.float32)

    required_baseline = {"age", "sex", "hemoglobin", "wbc_count", "platelet_count"}

    for idx, name in enumerate(BIOMARKER_NAMES):
        val = raw_dict.get(name, None)

        if val is None or (isinstance(val, str) and ("unknown" in val.lower() or "not tested" in val.lower() or val.strip() == "")):
            if name in required_baseline:
                errors.append(f"Mandatory baseline clinical field '{name}' is missing or marked unknown.")
            raw_vec[idx] = 0.0
            obs_mask[idx] = 0.0
            continue

        try:
            fval = float(val)
        except (TypeError, ValueError):
            errors.append(f"Biomarker '{name}' has non-numeric invalid value: {val}")
            continue

        if np.isnan(fval) or np.isinf(fval):
            errors.append(f"Biomarker '{name}' contains NaN or Inf.")
            continue

        lo, hi, unit = PHYS_BOUNDS[name]
        # Catch common all-zero biological impossibility on vital lab parameters
        if name in {"age", "hemoglobin", "wbc_count", "platelet_count"} and fval <= 0.0:
            errors.append(f"Biological zero violation: '{name}' cannot be {fval} {unit} in a live patient.")
            continue

        if fval < lo or fval > hi:
            unit_hint = ""
            if name == "hemoglobin" and fval > 50:
                unit_hint = " (possible g/L vs g/dL unit confusion)"
            elif name == "platelet_count" and fval > 5000:
                unit_hint = " (possible /mm^3 vs x10^3/uL unit confusion)"
            errors.append(
                f"Physiological/unit range error for '{name}': {fval} {unit} is outside valid [{lo}, {hi}] {unit}{unit_hint}."
            )
            continue

        raw_vec[idx] = fval
        obs_mask[idx] = 1.0

    # Cross-field tabular physiological checks
    if obs_mask[0] == 1.0 and obs_mask[2] == 1.0:
        age_v = raw_vec[0]
        smoke_v = raw_vec[2]
        max_plausible_py = max(0.0, (age_v - 10.0) * 3.0)
        if smoke_v > max_plausible_py:
            errors.append(
                f"Physiologically impossible smoking history: {smoke_v:.0f} pack-years is incompatible with Age {age_v:.0f} years "
                f"(exceeds max ~{max_plausible_py:.0f} pack-years assuming 3 packs/day starting age 10)."
            )

    # Biological plausibility warning for unusual receptor/gender profiles (warn, do not hard-fail rare cases)
    if obs_mask[1] == 1.0 and raw_vec[1] == 0.0:  # Male
        if (obs_mask[8] == 1.0 and raw_vec[8] == 1.0) or (obs_mask[9] == 1.0 and raw_vec[9] == 1.0):
            warnings.append("Rare biological profile: ER/PR positivity reported in a Male patient — verify pathology requisition.")
        if obs_mask[5] == 1.0 and raw_vec[5] > 45.0:
            warnings.append("Unusual elevation of CA-125 in a Male patient — monitor non-gynecologic effusions/serosal inflammation.")

    # Ensure enough clinical markers were entered (at least baseline + 2 additional clinical markers)
    if np.sum(obs_mask) < 6:
        errors.append("Insufficient clinical parameters observed (fewer than 6 fields available).")

    if errors:
        return False, " | ".join(errors), warnings, None, raw_vec, obs_mask

    vec_32 = encode_biomarkers_with_mask(raw_vec, obs_mask)
    return True, "Valid structured biomarker profile.", warnings, vec_32, raw_vec, obs_mask


def build_multimodal_records():
    """
    Constructs train, val, and test splits preserving hierarchical status_label, organ_label, and subtype_label.
    Generates synchronized clinical narratives and 32-D masked biomarker features.
    """
    train_splits, val_splits, test_splits = build_hierarchical_splits()

    def enrich_records(record_list, split_idx_offset=0, split_role="train"):
        enriched = []
        for idx, r in enumerate(record_list):
            organ_lbl = r["organ_label"]        # 0: Breast, 1: Lung, 2: Colorectal
            status_lbl = r["status_label"]      # 0: Normal, 1: Benign, 2: Malignant
            sub_lbl = r.get("subtype_label", -1)

            if status_lbl == 0:
                key = "lung_normal" if organ_lbl == 1 else "colorectal_normal"
            elif status_lbl == 1:
                key = "breast_benign"
            else:
                if organ_lbl == 0:
                    key = "breast_malignant"
                elif organ_lbl == 1:
                    key = "lung_malignant"
                else:
                    key = "colorectal_malignant"

            seed = abs(hash(os.path.basename(r["image_path"]))) % (2**31)
            raw_biomarkers = generate_biomarker_vector(key, seed=seed)
            obs_mask = np.ones(16, dtype=np.float32)
            masked_vec_32 = encode_biomarkers_with_mask(raw_biomarkers, obs_mask)
            narrative = format_synchronized_narrative(key, raw_biomarkers, variant_idx=idx + split_idx_offset, split_role=split_role)

            enriched.append({
                "image_path": r["image_path"],
                "group_id": r.get("group_id", r["image_path"]),
                "organ_label": organ_lbl,
                "status_label": status_lbl,
                "subtype_label": sub_lbl,
                "category_key": key,
                "text_narrative": narrative,
                "raw_biomarkers": raw_biomarkers,
                "obs_mask": obs_mask,
                "norm_biomarkers": normalize_biomarkers(raw_biomarkers),
                "biomarkers_32": masked_vec_32,
            })
        return enriched

    return (
        enrich_records(train_splits, 0, split_role="train"),
        enrich_records(val_splits, 1, split_role="eval"),
        enrich_records(test_splits, 2, split_role="eval"),
    )


class MultimodalCancerDataset(Dataset):
    """PyTorch Dataset yielding synchronized (Image, Text, Masked Tabular [32], Multi-Task Labels)."""
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        img_tensor = preprocess_image(img)

        return {
            "image": img_tensor,
            "text": item["text_narrative"],
            "tabular": torch.tensor(item["biomarkers_32"], dtype=torch.float32),
            "organ_label": torch.tensor(item["organ_label"], dtype=torch.long),
            "status_label": torch.tensor(item["status_label"], dtype=torch.long),
            "subtype_label": torch.tensor(item["subtype_label"], dtype=torch.long),
            "image_path": item["image_path"],
        }
