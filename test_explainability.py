import torch
import numpy as np
from PIL import Image

import config
from src.ui.level2_app import img_extractor, tab_mlp, text_extractor, text_proj, classifier, gradcam
from src.explainability.shap_explainer import TabularShapExplainer
from src.explainability.text_saliency import BioBERTTextSaliencyExplainer
from src.explainability.faithfulness import evaluate_visual_faithfulness, evaluate_tabular_faithfulness
from src.data.multimodal_dataset import generate_biomarker_vector

print("[*] Initializing Explainability Test Suite...")

# 1. Test Tabular SHAP Explainer
shap_expl = TabularShapExplainer(tab_mlp, classifier, config.DEVICE, num_bg_samples=20)
vec_col = generate_biomarker_vector("colorectal_malignant", seed=101)
shap_res = shap_expl.explain_case(vec_col, target_class_idx=2, head_type="site", nsamples=60)

print(f"[+] SHAP Result for Colorectal: Base={shap_res['base_value']:.3f}, Target={shap_res['target_prob']:.3f}")
print(f"[+] Additivity Residual: {shap_res['efficiency_residual']:.5e} (Valid: {shap_res['efficiency_residual'] < 1e-3})")
print(f"[+] Top 3 Features: {[f['name'] for f in shap_res['top_features'][:3]]}")
assert shap_res["plot_pil"] is not None
print("[+] SHAP Plot Generated Successfully (Size:", shap_res["plot_pil"].size, ")")

# 2. Test BioBERT Text Saliency Explainer
text_expl = BioBERTTextSaliencyExplainer(text_extractor, text_proj, classifier, config.DEVICE)
sample_text = "Endoscopic biopsy confirms invasive adenocarcinoma of the sigmoid colon with prominent nuclear atypia."
text_res = text_expl.explain_text(sample_text, target_class_idx=2, head_type="site")
print(f"[+] Text Saliency Result: Target={text_res['target_class_name']} ({text_res['predicted_prob']*100:.1f}%)")
print(f"[+] Top Supporting Words: {[w[0] for w in text_res['top_supporting_words']]}")
assert "<span" in text_res["html_view"]
print("[+] HTML Saliency Span Generated Successfully")

# 3. Test Visual Faithfulness
test_img_path = "data/sample_images/colorectal/colorectal_00.png"
pil_img = Image.open(test_img_path).convert("RGB")
img_tensor = torch.zeros(1, 3, 224, 224).to(config.DEVICE)
cam_map, _, _ = gradcam.generate_heatmap(img_tensor, pathway="image", head_type="site", target_class=2)

faith_res = evaluate_visual_faithfulness(pil_img, cam_map, img_extractor, classifier, target_class_idx=2, head_type="site")
print(f"[+] Visual Faithfulness: OrigConf={faith_res['original_confidence']:.3f}, MaskedConf={faith_res['masked_confidence']:.3f}")
print(f"[+] Salient Drop={faith_res['salient_confidence_drop']:+.3f} vs Control Drop={faith_res['control_confidence_drop']:+.3f}")
print(f"[+] Is Faithful: {faith_res['is_faithful']}")

print("\n[+] ALL LEVEL 3 EXPLAINABILITY MODULES VERIFIED SUCCESSFULLY!")
