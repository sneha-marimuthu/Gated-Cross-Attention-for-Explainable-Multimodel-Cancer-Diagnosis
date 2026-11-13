import os, sys, json
import torch
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import config
from src.ui.level2_app import run_multimodal_pipeline, STATE_PRESENT_VALID, STATE_PRESENT_INVALID, STATE_MISSING, STATE_PREPROC_FAILED

# Define test cases
test_cases = [
    {
        "name": "Valid Breast Histopathology",
        "img_path": "/home/frappe/.gemini/antigravity/brain/0ed6296f-b1fc-4214-a2f2-24348269df2b/.user_uploaded/media_1794297363465_8ad144c5.jpg",
        "text": "Patient female, 55 years, invasive ductal carcinoma, ER positive, HER2 negative.",
        "tabular": True,
        "tab_vals": {
            "age": 55, "sex": "Female", "smoking": 0, "ecog": 1, "cea": 2.0, "ca125": 5.0, "ca19_9": 3.0, "ki67": 20, "er": "Positive (+)", "pr": "Positive (+)", "her2": "Negative (-)", "egfr": "Not Tested", "kras": "Not Tested", "hb": 13.5, "wbc": 6.0, "plt": 250
        }
    },
    {
        "name": "Valid Lung Histopathology",
        "img_path": "/home/frappe/.gemini/antigravity/brain/0ed6296f-b1fc-4214-a2f2-24348269df2b/.user_uploaded/media_1794299412232_0bc21083.jpg",
        "text": "Male, 68, non-small cell lung carcinoma, EGFR mutant.",
        "tabular": True,
        "tab_vals": {
            "age": 68, "sex": "Male", "smoking": 30, "ecog": 2, "cea": 7.0, "ca125": 1.0, "ca19_9": 2.0, "ki67": 30, "er": "Not Tested", "pr": "Not Tested", "her2": "Not Tested", "egfr": "Mutant", "kras": "Not Tested", "hb": 12.0, "wbc": 7.5, "plt": 300
        }
    },
    {
        "name": "Terminal Screenshot (invalid)",
        "img_path": "/home/frappe/.gemini/antigravity/brain/0ed6296f-b1fc-4214-a2f2-24348269df2b/.user_uploaded/media_1794301479562_df9b4da5.jpg",
        "text": "", "tabular": False
    },
    {
        "name": "General Photograph (invalid)",
        "img_path": "/home/frappe/.gemini/antigravity/brain/0ed6296f-b1fc-4214-a2f2-24348269df2b/.user_uploaded/media_1794301542394_f1d24202.jpg",
        "text": "", "tabular": False
    },
]

results = []
for case in test_cases:
    img = case.get('img_path')
    if img and os.path.exists(img):
        img_file = open(img, 'rb')
    else:
        img_file = None
    txt = case.get('text', '')
    include_tab = case.get('tabular', False)
    # unpack tabular values or use placeholders
    age = case['tab_vals'].get('age') if include_tab else None
    sex = case['tab_vals'].get('sex') if include_tab else None
    smoking = case['tab_vals'].get('smoking') if include_tab else None
    ecog = case['tab_vals'].get('ecog') if include_tab else None
    cea = case['tab_vals'].get('cea') if include_tab else None
    ca125 = case['tab_vals'].get('ca125') if include_tab else None
    ca19_9 = case['tab_vals'].get('ca19_9') if include_tab else None
    ki67 = case['tab_vals'].get('ki67') if include_tab else None
    er = case['tab_vals'].get('er') if include_tab else None
    pr = case['tab_vals'].get('pr') if include_tab else None
    her2 = case['tab_vals'].get('her2') if include_tab else None
    egfr = case['tab_vals'].get('egfr') if include_tab else None
    kras = case['tab_vals'].get('kras') if include_tab else None
    hb = case['tab_vals'].get('hb') if include_tab else None
    wbc = case['tab_vals'].get('wbc') if include_tab else None
    plt = case['tab_vals'].get('plt') if include_tab else None
    # Run pipeline
    output = run_multimodal_pipeline(
        input_img=img_file,
        input_text=txt,
        include_tabular=include_tab,
        age=age, sex=sex, smoking=smoking, ecog=ecog, cea=cea, ca125=ca125, ca19_9=ca19_9, ki67=ki67,
        er=er, pr=pr, her2=her2, egfr=egfr, kras=kras, hb=hb, wbc=wbc, plt=plt
    )
    results.append({"name": case['name'], "output": output})
    if img_file:
        img_file.close()

print(json.dumps(results, indent=2))
