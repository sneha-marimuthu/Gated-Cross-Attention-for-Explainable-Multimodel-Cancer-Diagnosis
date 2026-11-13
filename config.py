"""
Configuration parameters for GCAEMMCD Demonstration Project.
Hardware Target: HP Victus (Intel i5, 8GB RAM, NVIDIA RTX 2050 Laptop GPU)
"""
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "sample_images")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic_clinical")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# 1. Organ Categories (Module 1)
ORGANS = ["breast", "lung", "colorectal", "other"]
ORGAN_TO_IDX = {org: idx for idx, org in enumerate(ORGANS)}
IDX_TO_ORGAN = {idx: org for org, idx in ORGAN_TO_IDX.items()}
NUM_ORGANS = len(ORGANS)

# 2. Cancer Status Categories (Module 2)
CANCER_STATUSES = ["normal", "benign", "malignant"]
STATUS_TO_IDX = {st: idx for idx, st in enumerate(CANCER_STATUSES)}
IDX_TO_STATUS = {idx: st for st, idx in STATUS_TO_IDX.items()}
NUM_STATUSES = len(CANCER_STATUSES)

# 3. Cancer Subtype Categories (Module 3 - Activated only when Malignant)
CANCER_SUBTYPES = ["breast", "lung", "colorectal"]
SUBTYPE_TO_IDX = {sub: idx for idx, sub in enumerate(CANCER_SUBTYPES)}
IDX_TO_SUBTYPE = {idx: sub for sub, idx in SUBTYPE_TO_IDX.items()}
NUM_SUBTYPES = len(CANCER_SUBTYPES)

# Legacy alias for backward compatibility
CATEGORIES = CANCER_SUBTYPES
CATEGORY_TO_IDX = SUBTYPE_TO_IDX
IDX_TO_CATEGORY = IDX_TO_SUBTYPE
NUM_CLASSES = NUM_SUBTYPES

# Model Dimensions
LATENT_DIM = 256
IMAGE_RAW_DIM = 2048   # ResNet-50 GAP
TEXT_RAW_DIM = 768     # BioBERT [CLS]
TABULAR_RAW_DIM = 16   # Biomarkers

# Hardware & Training Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20

# Image Preprocessing Standards (ImageNet)
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
