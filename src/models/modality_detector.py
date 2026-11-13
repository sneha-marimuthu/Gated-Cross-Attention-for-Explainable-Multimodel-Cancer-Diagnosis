"""
Modality Availability, Sanity Verification & Inter-Modal Patient Concordance Engine.
Implements explicit 4-state detection:
  - PRESENT_VALID
  - PRESENT_INVALID
  - MISSING
  - PREPROCESSING_FAILED

Eliminates silent defaults, validates clinical/physiological integrity, parses patient identity
metadata across streams, and flags specimen swaps or cross-modal clinical discordance.
"""

import re
import numpy as np
import torch
from PIL import Image

import config
from src.data.domain_validator import validate_histopathology_domain
from src.data.image_transforms import preprocess_image, preprocess_multicrop
from src.data.multimodal_dataset import validate_tabular_biomarkers, BIOMARKER_NAMES

STATE_PRESENT_VALID = "PRESENT_VALID"
STATE_PRESENT_INVALID = "PRESENT_INVALID"
STATE_MISSING = "MISSING"
STATE_PREPROC_FAILED = "PREPROCESSING_FAILED"

CLINICAL_KEYWORDS = {
    "patient", "biopsy", "pathology", "histopathology", "carcinoma", "adenocarcinoma",
    "tumor", "tumour", "mass", "lesion", "nodule", "lump", "breast", "mammary",
    "ductal", "fibroadenoma", "lung", "pulmonary", "alveolar", "bronchial", "colon",
    "colorectal", "sigmoid", "rectal", "colonic", "mucosa", "crypt", "gland", "glands",
    "malignant", "malignancy", "benign", "normal", "unremarkable", "neoplasm", "neoplastic",
    "cancer", "dysplasia", "metastatic", "metastasis", "stroma", "stromal", "nuclei",
    "mitotic", "epithelium", "epithelial", "necrosis", "cea", "ca125", "ca-125", "ca19-9",
    "ki67", "ki-67", "ki67", "her2", "egfr", "kras", "er+", "pr+", "smoker", "smoking",
    "cough", "dyspnea", "anemia", "hematochezia", "screening", "colonoscopy", "mammography",
    "male", "female", "year-old", "yo", "symptom", "symptoms", "clinical", "diagnosis",
    "negative", "positive", "rule", "ruled", "evidence", "hemoglobin", "biopsied", "resection"
}

NEGATION_PATTERNS = [
    r"\bno evidence of (?:breast |lung |colorectal |invasive |)?(?:malignancy|carcinoma|adenocarcinoma|cancer|neoplasm)\b",
    r"\bnegative for (?:breast |lung |colorectal |invasive |)?(?:malignancy|carcinoma|adenocarcinoma|cancer|dysplasia)\b",
    r"\bruled out (?:breast |lung |colorectal |invasive |)?(?:malignancy|carcinoma|adenocarcinoma|cancer)\b",
    r"\babsence of (?:neoplastic|malignant|carcinoma|dysplasia)\b",
    r"\bwithout (?:evidence of |)(?:malignancy|carcinoma|atypia|dysplasia|invasive growth)\b",
]


def _js_divergence(p, q):
    """Symmetric Jensen-Shannon divergence (base 2, in [0, 1]) between two probability vectors."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def detect_image_modality(input_img, organ_extractor=None, organ_head=None, device=config.DEVICE):
    """
    Validates image availability, optical brightfield H&E physics, and optional Phikon organ OOD.
    Returns dict with state, reason, metrics, and optional Phikon calibration details.
    """
    if input_img is None:
        return {
            "state": STATE_MISSING,
            "reason": "No histopathology image uploaded.",
            "metrics": {},
            "phikon": None,
            "pil_image": None,
        }

    try:
        pil_img = input_img.convert("RGB") if hasattr(input_img, "convert") else Image.fromarray(np.asarray(input_img)).convert("RGB")
        w, h = pil_img.size
        if w < 32 or h < 32:
            return {
                "state": STATE_PRESENT_INVALID,
                "reason": f"Image resolution ({w}x{h}) is too small for histopathology feature extraction (minimum 32x32).",
                "metrics": {"width": w, "height": h},
                "phikon": None,
                "pil_image": pil_img,
            }
    except Exception as exc:
        return {
            "state": STATE_PREPROC_FAILED,
            "reason": f"Failed to decode uploaded image: {exc}",
            "metrics": {},
            "phikon": None,
            "pil_image": None,
        }

    is_valid_he, domain_type, optical_reason, metrics = validate_histopathology_domain(pil_img)
    if not is_valid_he:
        return {
            "state": STATE_PRESENT_INVALID,
            "reason": f"[{domain_type.upper()}] {optical_reason}",
            "domain_type": domain_type,
            "metrics": metrics,
            "phikon": None,
            "pil_image": pil_img,
        }

    phikon_info = None
    if organ_extractor is not None and organ_head is not None:
        try:
            with torch.no_grad():
                pixel_values = preprocess_multicrop(pil_img).to(device)
                phikon_feats = organ_extractor(pixel_values)
                pred_org, conf, prob_dict, is_ood, is_uncertain, min_dist = organ_head.predict_with_calibration(phikon_feats)

            phikon_info = {
                "predicted_organ": pred_org,
                "confidence": float(conf),
                "prob_dict": prob_dict,
                "is_ood": bool(is_ood),
                "is_uncertain": bool(is_uncertain),
                "cosine_distance": float(min_dist),
            }

            if is_ood:
                return {
                    "state": STATE_PRESENT_INVALID,
                    "reason": (
                        f"Unsupported Tissue / Out-of-Distribution Histology Rejected by Phikon Gatekeeper: "
                        f"Cosine distance {min_dist:.4f} exceeds calibrated threshold ({organ_head.max_cosine_distance:.2f}). "
                        "Supported H&E tissues: Breast, Lung, and Colorectal specimens."
                    ),
                    "domain_type": "unsupported_tissue_ood",
                    "metrics": metrics,
                    "phikon": phikon_info,
                    "pil_image": pil_img,
                }
        except Exception as exc:
            return {
                "state": STATE_PREPROC_FAILED,
                "reason": f"Phikon organ preprocessing failed: {exc}",
                "metrics": metrics,
                "phikon": None,
                "pil_image": pil_img,
            }

    return {
        "state": STATE_PRESENT_VALID,
        "reason": "Valid brightfield H&E histopathology patch verified.",
        "domain_type": "histopathology",
        "metrics": metrics,
        "phikon": phikon_info,
        "pil_image": pil_img,
    }


def detect_text_modality(input_text):
    """
    Validates clinical text presence, detects gibberish/off-topic text, and extracts
    patient metadata + clinical negation cues for cross-modal verification.
    """
    if input_text is None:
        return {"state": STATE_MISSING, "reason": "Clinical text box is empty.", "metadata": {}}

    cleaned = input_text.strip()
    if len(cleaned) == 0:
        return {"state": STATE_MISSING, "reason": "No clinical narrative entered.", "metadata": {}}

    alpha_words = re.findall(r"[A-Za-z]+", cleaned)
    if len(cleaned) < 10 or len(alpha_words) < 3:
        return {
            "state": STATE_PRESENT_INVALID,
            "reason": (
                f"Text input too short or lacks diagnostic context ({len(alpha_words)} words, {len(cleaned)} chars). "
                "Provide a meaningful clinical vignette or pathology summary (or clear field to omit text)."
            ),
            "metadata": {},
        }

    # Check gibberish / random keystrokes (very low vowel ratio or extreme repeated chars)
    lower = cleaned.lower()
    if re.search(r"(.)\1{5,}", lower):
        return {
            "state": STATE_PRESENT_INVALID,
            "reason": "Repetitive non-clinical character sequence detected in text input.",
            "metadata": {},
        }

    tokens = set(re.findall(r"[a-z0-9\-/+]+", lower))
    matched_clinical = tokens.intersection(CLINICAL_KEYWORDS)
    if len(matched_clinical) == 0:
        return {
            "state": STATE_PRESENT_INVALID,
            "reason": (
                "Off-topic or non-clinical text rejected: no recognized medical, anatomical, or symptom vocabulary found. "
                "Cannot run BioBERT clinical reasoning on unrelated prose."
            ),
            "metadata": {"tokens": len(alpha_words)},
        }

    # Extract clinical metadata for cross-modal consistency auditing
    age_match = re.search(r"\b(\d{1,3})\s*[-\s]?(?:year[s]?[-\s]*old|y/?o|yrs?\b)", lower)
    parsed_age = int(age_match.group(1)) if age_match else None

    has_female = bool(re.search(r"\b(female|woman|women|lady|she|her|hers|mrs\.?|ms\.?)\b", lower))
    has_male = bool(re.search(r"\b(male|man|men|gentleman|he|his|mr\.?)\b", lower))
    parsed_sex = None
    if has_female and not has_male:
        parsed_sex = "female"
    elif has_male and not has_female:
        parsed_sex = "male"

    nonsmoker = bool(re.search(r"\b(non[-\s]?smoker|never smoked|no smoking history|0 pack[-\s]?years?)\b", lower))
    smoke_match = re.search(r"\b(\d+)\s*pack[-\s]?years?\b", lower)
    parsed_pack_yrs = int(smoke_match.group(1)) if smoke_match else None

    negation_hits = [p for p in NEGATION_PATTERNS if re.search(p, lower)]
    has_negation = len(negation_hits) > 0

    # Strip negated prior-history or exclusion clauses so they do not hijack primary organ/malignancy cues
    # e.g. "no prior history of breast cancer, colonoscopy shows invasive sigmoid colon mass" -> primary cue: Colorectal / Malignant
    affirmative_lower = re.sub(
        r"\b(?:no|without|negative\s+for|ruled\s+out|absence\s+of|denies)\s+"
        r"(?:prior\s+|past\s+)?(?:history\s+of\s+|evidence\s+of\s+)?[^.;,\n]+",
        " ",
        lower,
    ).strip()
    aff_tokens = set(re.findall(r"[a-z0-9\-/+]+", affirmative_lower))

    has_affirmative_mal = bool(
        aff_tokens.intersection({"carcinoma", "adenocarcinoma", "malignant", "malignancy", "invasive", "metastatic"})
    )
    has_affirmative_benign = bool(aff_tokens.intersection({"benign", "fibroadenoma"}))
    has_histologic_normal = bool(
        re.search(r"\b(?:normal|unremarkable)\s+(?:lung|pulmonary|alveolar|colonic|colon|mucosa|parenchyma|epithelium|tissue|biopsy|crypts?)\b", affirmative_lower)
        or "unremarkable" in aff_tokens
    )

    status_cue = None
    if has_affirmative_mal:
        status_cue = "Malignant"
    elif has_affirmative_benign:
        status_cue = "Benign"
    elif has_negation or has_histologic_normal:
        status_cue = "Non-Malignant / Negative"

    cue_source = affirmative_lower if any(
        k in affirmative_lower for k in [
            "breast", "mammary", "ductal", "bi-rads", "fibroadenoma",
            "lung", "pulmonary", "alveolar", "bronchial", "hilar", "luad",
            "colon", "colorectal", "sigmoid", "rectal", "colonic", "colonoscopy", "coad",
        ]
    ) else lower

    organ_cues = []
    if any(k in cue_source for k in ["breast", "mammary", "ductal", "bi-rads", "fibroadenoma"]):
        organ_cues.append("Breast")
    if any(k in cue_source for k in ["lung", "pulmonary", "alveolar", "bronchial", "hilar", "luad"]):
        organ_cues.append("Lung")
    if any(k in cue_source for k in ["colon", "colorectal", "sigmoid", "rectal", "colonic", "colonoscopy", "coad"]):
        organ_cues.append("Colorectal")

    metadata = {
        "word_count": len(alpha_words),
        "matched_clinical_terms": sorted(list(matched_clinical))[:8],
        "age": parsed_age,
        "sex": parsed_sex,
        "nonsmoker": nonsmoker,
        "pack_years": parsed_pack_yrs,
        "has_negation": has_negation,
        "status_cue": status_cue,
        "organ_cues": organ_cues,
        "cleaned_text": cleaned,
    }

    return {
        "state": STATE_PRESENT_VALID,
        "reason": f"Valid clinical text ({len(alpha_words)} words, {len(matched_clinical)} medical terms detected).",
        "metadata": metadata,
    }


def detect_tabular_modality(include_tabular, raw_dict):
    """
    Validates structured biomarker availability strictly via explicit user opt-in and
    physiological/unit auditing. Never reads untouched UI defaults silently.
    """
    if not include_tabular:
        return {
            "state": STATE_MISSING,
            "reason": "Structured biomarkers excluded ('Include Clinical & Biomarker Data' is unchecked; default slider values ignored).",
            "warnings": [],
            "vec_32": None,
            "raw_vec": None,
            "obs_mask": None,
            "metadata": {},
        }

    try:
        is_valid, reason, warnings, vec_32, raw_vec, obs_mask = validate_tabular_biomarkers(raw_dict)
    except Exception as exc:
        return {
            "state": STATE_PREPROC_FAILED,
            "reason": f"Tabular preprocessing error: {exc}",
            "warnings": [],
            "vec_32": None,
            "raw_vec": None,
            "obs_mask": None,
            "metadata": {},
        }

    if not is_valid:
        return {
            "state": STATE_PRESENT_INVALID,
            "reason": reason,
            "warnings": warnings,
            "vec_32": None,
            "raw_vec": raw_vec,
            "obs_mask": obs_mask,
            "metadata": {},
        }

    metadata = {
        "age": float(raw_vec[0]) if obs_mask[0] > 0.5 else None,
        "sex": ("female" if raw_vec[1] >= 0.5 else "male") if obs_mask[1] > 0.5 else None,
        "smoking_pack_yrs": float(raw_vec[2]) if obs_mask[2] > 0.5 else None,
        "observed_count": int(np.sum(obs_mask)),
        "missing_fields": [
            BIOMARKER_NAMES[i]
            for i in range(16) if obs_mask[i] < 0.5
        ],
    }

    return {
        "state": STATE_PRESENT_VALID,
        "reason": f"Valid biomarker vector ({int(np.sum(obs_mask))}/16 fields observed).",
        "warnings": warnings,
        "vec_32": vec_32,
        "raw_vec": raw_vec,
        "obs_mask": obs_mask,
        "metadata": metadata,
    }


def verify_cross_modal_concordance(img_info, txt_info, tab_info, pathway_probs):
    """
    Audits consistency across active modalities for:
      1. Patient Identity & Demographic Consistency (Text vs Tabular Sex, Age, Smoking)
      2. Disease Status Concordance (Normal / Benign / Malignant + Negation flags)
      3. Anatomical Organ / Site Concordance (JS divergence & high-confidence argmax conflicts)

    pathway_probs: dict with optional keys:
      'image_status', 'image_site', 'text_status', 'text_site', 'tabular_status', 'tabular_site'
    Returns rich audit report dict.
    """
    STATUS_NAMES = ["Normal", "Benign", "Malignant"]
    SITE_NAMES = ["Breast", "Lung", "Colorectal"]

    identity_alerts = []
    status_alerts = []
    organ_alerts = []

    has_i = img_info["state"] == STATE_PRESENT_VALID
    has_t = txt_info["state"] == STATE_PRESENT_VALID
    has_r = tab_info["state"] == STATE_PRESENT_VALID

    # 1. Patient Identity & Metadata Audit (Text vs Tabular)
    if has_t and has_r:
        t_meta = txt_info["metadata"]
        r_meta = tab_info["metadata"]

        if t_meta.get("sex") and r_meta.get("sex") and t_meta["sex"] != r_meta["sex"]:
            identity_alerts.append(
                f"PATIENT SEX MISMATCH: Clinical text states '{t_meta['sex'].upper()}', "
                f"whereas Tabular profile states '{r_meta['sex'].upper()}'."
            )

        if t_meta.get("age") is not None and r_meta.get("age") is not None:
            age_diff = abs(t_meta["age"] - r_meta["age"])
            if age_diff > 5:
                identity_alerts.append(
                    f"PATIENT AGE MISMATCH: Clinical text states age {t_meta['age']}, "
                    f"while Tabular Age is {r_meta['age']:.0f} (difference: {age_diff:.0f} years)."
                )

        if t_meta.get("nonsmoker") and r_meta.get("smoking_pack_yrs") is not None and r_meta["smoking_pack_yrs"] > 5.0:
            identity_alerts.append(
                f"CLINICAL HISTORY CONFLICT: Narrative states 'non-smoker', while Tabular records {r_meta['smoking_pack_yrs']:.0f} pack-years."
            )
        elif t_meta.get("pack_years") is not None and r_meta.get("smoking_pack_yrs") is not None:
            if abs(t_meta["pack_years"] - r_meta["smoking_pack_yrs"]) > 10:
                identity_alerts.append(
                    f"SMOKING HISTORY CONFLICT: Narrative reports {t_meta['pack_years']} pack-years vs Tabular {r_meta['smoking_pack_yrs']:.0f} pack-years."
                )

    # 2. Compute Disease Status First (Normal vs Benign vs Malignant)
    active_statuses = {}
    if has_i and pathway_probs.get("image_status") is not None:
        active_statuses["Image (ResNet)"] = pathway_probs["image_status"]
    if has_t and pathway_probs.get("text_status") is not None:
        active_statuses["Text (BioBERT)"] = pathway_probs["text_status"]
    if has_r and pathway_probs.get("tabular_status") is not None:
        active_statuses["Tabular (Biomarkers)"] = pathway_probs["tabular_status"]

    status_preds = {}
    for name, vec in active_statuses.items():
        idx = int(np.argmax(vec))
        conf = float(vec[idx])
        status_preds[name] = (STATUS_NAMES[idx], conf)

    # Normal peripheral labs/markers lack organ specificity (normal colon vs normal lung bloods overlap)
    tab_is_normal = status_preds.get("Tabular (Biomarkers)", ("", 0.0))[0] == "Normal"

    # 3. Organ / Anatomical Site Concordance
    active_sites = {}
    if has_i and img_info.get("phikon") and img_info["phikon"].get("prob_dict"):
        p_phik = [img_info["phikon"]["prob_dict"].get(s, 0.0) for s in SITE_NAMES]
        active_sites["Phikon Organ"] = np.array(p_phik, dtype=np.float64)
    if has_i and pathway_probs.get("image_site") is not None:
        active_sites["Image (ResNet)"] = pathway_probs["image_site"]
    if has_t and pathway_probs.get("text_site") is not None:
        active_sites["Text (BioBERT)"] = pathway_probs["text_site"]
    if has_r and pathway_probs.get("tabular_site") is not None and not tab_is_normal:
        active_sites["Tabular (Biomarkers)"] = pathway_probs["tabular_site"]

    site_predictions = {}
    for name, vec in active_sites.items():
        idx = int(np.argmax(vec))
        conf = float(vec[idx])
        site_predictions[name] = (SITE_NAMES[idx], conf)

    # Check high-confidence cross-modality disagreements (>= 0.68)
    confident_sites = {k: v for k, v in site_predictions.items() if v[1] >= 0.68}
    unique_organs = set(v[0] for v in confident_sites.values())

    # Cross-check explicit text organ keywords vs Image/Tabular predictions
    if has_t and len(txt_info["metadata"].get("organ_cues", [])) == 1:
        text_explicit_org = txt_info["metadata"]["organ_cues"][0]
        for mod_name, (org_pred, org_conf) in site_predictions.items():
            if "Text" not in mod_name and org_conf >= 0.68 and org_pred != text_explicit_org:
                organ_alerts.append(
                    f"TEXT-KEYWORD ORGAN DISCORDANCE: Text explicitly mentions '{text_explicit_org}', "
                    f"but {mod_name} predicts '{org_pred}' ({org_conf*100:.1f}%)."
                )

    max_js_organ = 0.0
    keys = list(active_sites.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            js = _js_divergence(active_sites[keys[i]], active_sites[keys[j]])
            if js > max_js_organ:
                max_js_organ = js

    if len(unique_organs) > 1 or max_js_organ > 0.48:
        detail_str = ", ".join([f"{k}: {v[0]} ({v[1]*100:.0f}%)" for k, v in site_predictions.items()])
        organ_alerts.append(
            f"INTER-MODAL ORGAN DISCORDANCE (Max JS Div: {max_js_organ:.2f}): Conflicting site signals across [{detail_str}]. "
            "Could indicate (a) specimen/patient record mix-up OR (b) metastatic tumor at a secondary organ biopsy site."
        )

    # Check text negation vs predicted Malignancy (only when overall text verdict is NOT affirmatively malignant)
    if has_t and txt_info["metadata"].get("has_negation") and txt_info["metadata"].get("status_cue") != "Malignant":
        for mod_name, (st_pred, st_conf) in status_preds.items():
            if st_pred == "Malignant" and st_conf >= 0.60:
                status_alerts.append(
                    f"NEGATION VS MALIGNANCY CONFLICT: Clinical narrative contains explicit exclusion/negation phrasing "
                    f"('no evidence of malignancy / negative for carcinoma'), while {mod_name} predicts Malignant ({st_conf*100:.1f}%)."
                )

    confident_statuses = {k: v for k, v in status_preds.items() if v[1] >= 0.55}
    # Check Malignant vs Non-Malignant (Normal/Benign) disagreement
    has_mal = any(v[0] == "Malignant" for v in confident_statuses.values())
    has_nonmal = any(v[0] in {"Normal", "Benign"} for v in confident_statuses.values())
    if has_mal and has_nonmal:
        st_str = ", ".join([f"{k}: {v[0]} ({v[1]*100:.0f}%)" for k, v in status_preds.items()])
        status_alerts.append(
            f"DISEASE STATUS DISCORDANCE: Modalities disagree on Malignancy vs Non-Malignant state [{st_str}]."
        )

    is_discordant = bool(identity_alerts or organ_alerts or status_alerts)
    if identity_alerts:
        overall = "PATIENT_IDENTITY_MISMATCH"
    elif organ_alerts:
        overall = "INTER_MODAL_ORGAN_DISCORDANCE"
    elif status_alerts:
        overall = "DISEASE_STATUS_DISCORDANCE"
    elif (int(has_i) + int(has_t) + int(has_r)) >= 2:
        overall = "CONCORDANT"
    else:
        overall = "SINGLE_OR_EMPTY"

    return {
        "overall_status": overall,
        "is_discordant": is_discordant,
        "identity_alerts": identity_alerts,
        "organ_alerts": organ_alerts,
        "status_alerts": status_alerts,
        "site_predictions": site_predictions,
        "status_predictions": status_preds,
        "max_js_organ": max_js_organ,
    }
