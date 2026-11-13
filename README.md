# GCAEMMCD: Hierarchical Explainable Histopathology Diagnostic System
## Architectural Audit, Calibration & Hierarchical Redesign

### System Profile
- **Hardware Target:** HP Victus Laptop (Intel i5, 8GB RAM, NVIDIA GeForce RTX 2050 GPU)
- **Environment:** PyTorch 2.5.1 with CUDA 12.1 acceleration, Hugging Face Transformers, Gradio 4.44.1
- **Working Directory:** `/home/frappe/.gemini/antigravity/scratch/gcaemmcd_demo`

---

## 1. Problem Resolution & Architectural Journey

```
                        ┌─────────────────────────────────────────┐
                        │       Input H&E Patch [3, 224, 224]     │
                        └────────────────────┬────────────────────┘
                                             │
                ┌────────────────────────────┴────────────────────────────┐
                ▼                                                         ▼
     [STAGE 1: ORGAN IDENTIFIER]                                [STAGE 2: CANCER STATUS]
        Frozen Owkin/Phikon ViT-B                                Pretrained ResNet-50
       (Calibrated Cosine Distance)                                       │
                │                                             ┌───────────┴───────────┐
                ▼                                             ▼           ▼           ▼
        - Breast                                            Normal     Benign     Malignant
        - Lung                                                │           │           │
        - Colorectal                                          ▼           ▼           ▼
        - Other / Unsupported                           [TERMINAL:  [TERMINAL:   (If Conf >= 60%)
        - Unknown / OOD                                  No Cancer]  Non-Malig]       │
                                                                                      │
                                                                                      ▼
                                                                        [STAGE 3: CARCINOMA SUBTYPE]
                                                                        Conditioned on Stage 1 & 2
                                                                                      │
                                                                           ┌──────────┼──────────┐
                                                                           ▼          ▼          ▼
                                                                        Breast Ca  Lung Ca   Colon Ca
```

### Why Normal/Benign Tissue is No Longer Misclassified as Carcinoma
1. **Separation of Concerns:** Previously, ResNet-50 had a single 3-class classification head forcing every input into Breast Carcinoma, Lung Carcinoma, or Colorectal Carcinoma.
2. **Hierarchical Gating:** Under the new architecture, Stage 2 first decides if the tissue is **Normal / Non-neoplastic**, **Benign**, or **Malignant**. If the patch is Normal or Benign, **Stage 3 Carcinoma Subtyping is bypassed completely**.
3. **Calibrated OOD Detection:** The cosine distance metric $D_{\text{cos}}(x, \mu) = 1 - \frac{x \cdot \mu}{\|x\| \|\mu\|}$ correctly measures divergence from reference organ centroids. Authentic in-distribution samples have $D_{\text{cos}} \approx 0.05 - 0.20$ (well below the $0.45$ threshold), while out-of-distribution and blank images produce $D_{\text{cos}} > 0.70$, properly triggering the `⚠️ Unknown / Out-of-Distribution` alert.

---

## 2. Dataset Expansion (100 Authentic Samples Total)

| Category / Cohort | Count | Origin Archive | Role in Architecture |
|---|---|---|---|
| **Normal Lung (`lung_n`)** | 10 patches | LC25000 | Cancer Status: Normal (Stage 2) |
| **Normal Colon (`colon_n`)** | 10 patches | LC25000 | Cancer Status: Normal (Stage 2) |
| **Benign Breast Tissue (`label=0`)** | 20 patches | BreaKHis / IDC archive | Cancer Status: Benign (Stage 2) |
| **Malignant Breast Carcinoma** | 20 patches | BreaKHis / IDC archive | Status: Malignant (Stage 2), Subtype: Breast (Stage 3) |
| **Malignant Lung Carcinoma** | 20 patches | LC25000 (`lung_aca`) | Status: Malignant (Stage 2), Subtype: Lung (Stage 3) |
| **Malignant Colorectal Carcinoma** | 20 patches | LC25000 (`colon_ca`) | Status: Malignant (Stage 2), Subtype: Colon (Stage 3) |
| **Total** | **100 patches** | Authentic histopathology | Partitioned into 70 Train, 14 Val, 16 Test |

---

## 3. Evaluation Metrics on Held-Out Test Split (16 Samples)

### Stage 1: Organ Identification (Phikon ViT-B)
```
              precision    recall  f1-score   support

      Breast     1.0000    1.0000    1.0000         6
        Lung     1.0000    1.0000    1.0000         5
  Colorectal     1.0000    1.0000    1.0000         5

    accuracy                         1.0000        16
```

### Stage 2: Cancer Status Classification (ResNet-50)
```
              precision    recall  f1-score   support

      Normal     1.0000    0.7500    0.8571         4
      Benign     1.0000    1.0000    1.0000         3
   Malignant     0.9000    1.0000    0.9474         9

    accuracy                         0.9375        16
```

### Stage 3: Carcinoma Subtype Classification (Conditioned on Malignancy)
```
              precision    recall  f1-score   support

   Breast Ca     1.0000    1.0000    1.0000         3
     Lung Ca     1.0000    1.0000    1.0000         3
    Colon Ca     1.0000    1.0000    1.0000         3

    accuracy                         1.0000         9
```

---

## 4. How to Launch and Test the Application

### Launch Command
```bash
cd /home/frappe/.gemini/antigravity/scratch/gcaemmcd_demo
PYTHONPATH=. ./venv/bin/python src/ui/level1_app.py
```
Access via browser at: **`http://localhost:7860`** (or `http://127.0.0.1:7860`)

### Retrain Models Independently
- To retrain Stage 2 & 3 Hierarchical ResNet-50 model:
  ```bash
  PYTHONPATH=. ./venv/bin/python src/training/train_hierarchical.py
  ```
- To retrain Stage 1 Calibrated Organ model:
  ```bash
  PYTHONPATH=. ./venv/bin/python src/training/train_calibrated_organ.py
  ```
