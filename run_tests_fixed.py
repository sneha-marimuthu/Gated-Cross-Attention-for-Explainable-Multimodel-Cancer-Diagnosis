import os, json, sys, numpy as np, torch

# Add project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.ui.level2_app import (
    run_multimodal_pipeline,
    STATE_PRESENT_VALID,
    STATE_PRESENT_INVALID,
    STATE_MISSING,
    STATE_PREPROC_FAILED,
)

from PIL import Image

def load_image(path):
    try:
        img = Image.open(path).convert('RGB')
        return img
    except Exception as exc:
        print(f'[WARN] Could not open image {path!r}: {exc}')
        return None

# Helper to extract a concise summary from the pipeline output list
def summarise_output(output_list):
    # output_list may contain strings, dicts, ndarrays, etc.
    summary = {
        "modality_badge": None,
        "diagnostic_label": None,
        "gradcam_generated": False,
        "raw": []
    }
    for item in output_list:
        # Capture plain strings for later reference
        if isinstance(item, str):
            summary["raw"].append(item)
            # The first string usually is the modality badge
            if summary["modality_badge"] is None:
                summary["modality_badge"] = item
            # Look for a line that contains a disease status/subtype
            if any(tag in item.lower() for tag in ["malignant", "benign", "normal", "triage"]):
                summary["diagnostic_label"] = item
        # Detect Grad‑CAM image (numpy array or torch Tensor)
        if isinstance(item, (np.ndarray, torch.Tensor)):
            summary["gradcam_generated"] = True
    return summary

# Define test cases
test_cases = [
    {
        "name": "Valid Breast Histopathology",
        "img_path": "/home/frappe/.gemini/antigravity/brain/0ed6296f-b1fc-4214-a2f2-24348269df2b/.user_uploaded/media_1794297363465_8ad144c5.jpg",
        "text": "Patient female, 55 years, invasive ductal carcinoma, ER positive, HER2 negative.",
        "tabular": True,
        "tab_vals": {
            "age": 55, "sex": "Female", "smoking": 0, "ecog": 1,
            "cea": 2.0, "ca125": 5.0, "ca19_9": 3.0, "ki67": 20,
            "er": "Positive (+)", "pr": "Positive (+)", "her2": "Negative (-)",
            "egfr": "Not Tested", "kras": "Not Tested", "hb": 13.5,
            "wbc": 6.0, "plt": 250,
        },
    },
    {
        "name": "Valid Lung Histopathology",
        "img_path": "/home/frappe/.gemini/antigravity/brain/0ed6296f-b1fc-4214-a2f2-24348269df2b/.user_uploaded/media_1794299412232_0bc21083.jpg",
        "text": "Male, 68, non‑small cell lung carcinoma, EGFR mutant.",
        "tabular": True,
        "tab_vals": {
            "age": 68, "sex": "Male", "smoking": 30, "ecog": 2,
            "cea": 7.0, "ca125": 1.0, "ca19_9": 2.0, "ki67": 30,
            "er": "Not Tested", "pr": "Not Tested", "her2": "Not Tested",
            "egfr": "Mutant", "kras": "Not Tested", "hb": 12.0,
            "wbc": 7.5, "plt": 300,
        },
    },
    {
        "name": "Terminal Screenshot (invalid)",
        "img_path": "/home/frappe/.gemini/antigravity/brain/0ed6296f-b1fc-4214-a2f2-24348269df2b/.user_uploaded/media_1794301479562_df9b4da5.jpg",
        "text": "",
        "tabular": False,
    },
    {
        "name": "General Photograph (invalid)",
        "img_path": "/home/frappe/.gemini/antigravity/brain/0ed6296f-b1fc-4214-a2f2-24348269df2b/.user_uploaded/media_1794301542394_f1d24202.jpg",
        "text": "",
        "tabular": False,
    },
]

results = []
for case in test_cases:
    img = load_image(case['img_path']) if os.path.exists(case['img_path']) else None
    txt = case.get('text', '')
    include_tab = case.get('tabular', False)
    tb = case.get('tab_vals', {}) if include_tab else {}
    age = tb.get('age')
    sex = tb.get('sex')
    smoking = tb.get('smoking')
    ecog = tb.get('ecog')
    cea = tb.get('cea')
    ca125 = tb.get('ca125')
    ca19_9 = tb.get('ca19_9')
    ki67 = tb.get('ki67')
    er = tb.get('er')
    pr = tb.get('pr')
    her2 = tb.get('her2')
    egfr = tb.get('egfr')
    kras = tb.get('kras')
    hb = tb.get('hb')
    wbc = tb.get('wbc')
    plt = tb.get('plt')

    output = run_multimodal_pipeline(
        input_img=img,
        input_text=txt,
        include_tabular=include_tab,
        age=age, sex=sex, smoking=smoking, ecog=ecog, cea=cea, ca125=ca125,
        ca19_9=ca19_9, ki67=ki67, er=er, pr=pr, her2=her2, egfr=egfr,
        kras=kras, hb=hb, wbc=wbc, plt=plt,
    )
    summary = summarise_output(output)
    results.append({"name": case['name'], "summary": summary})

out_path = os.path.join(os.path.dirname(__file__), 'test_results_summary.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'✅ Summary saved to {out_path}')
