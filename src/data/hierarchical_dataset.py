"""
Multi-Task Histopathology Dataset and Group-Stratified Split Management.
Combines the baseline 99 valid specimens with the expanded multi-cohort archive
(Breast IDC, NCT-CRC-HE Colorectal, LC25000 Lung ACA+SCC+Normal, and Wikimedia micrographs)
using strict group-level splitting so crops from the same slide never cross splits.
"""
import glob
import json
import os
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from PIL import Image

import config
from src.data.image_transforms import get_image_transforms


class HierarchicalHistopathologyDataset(Dataset):
    def __init__(self, records, is_train=False):
        self.records = records
        self.transform = get_image_transforms(is_train=is_train)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        tensor = self.transform(image)
        return {
            "image": tensor,
            "organ_label": torch.tensor(item["organ_label"], dtype=torch.long),
            "status_label": torch.tensor(item["status_label"], dtype=torch.long),
            "subtype_label": torch.tensor(item["subtype_label"], dtype=torch.long),
            "image_path": item["image_path"],
        }


CATEGORY_META = {
    "lung_normal": (1, 0, -1),
    "colorectal_normal": (2, 0, -1),
    "breast_benign": (0, 1, -1),
    "breast_malignant": (0, 2, 0),
    "lung_malignant": (1, 2, 1),
    "colorectal_malignant": (2, 2, 2),
}


def get_ood_calibration_paths():
    manifest_path = os.path.join(config.DATA_DIR, "expanded_cohort", "cohort_manifest.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path) as f:
        return [x["path"] for x in json.load(f) if x.get("category") == "ood_calibration" and os.path.exists(x["path"])]


def build_hierarchical_splits(seed=config.SEED, include_expanded=True):
    """
    Constructs balanced, group-stratified 70/15/15 splits across all supported classes.
    Excludes non-tissue slide artifacts (e.g., empty coverslip edge `benign_breast_13.png`).
    """
    rng = random.Random(seed)
    grouped_entries = defaultdict(lambda: defaultdict(list))  # category -> group_id -> list[path]

    def add_sample(cat, gid, path):
        if not os.path.exists(path) or "benign_breast_13.png" in path:
            return
        grouped_entries[cat][gid].append(path)

    # 1. Baseline 99 authentic sample_images
    for f in sorted(glob.glob(os.path.join(config.IMAGE_DIR, "normal", "normal_lung_*.png"))):
        add_sample("lung_normal", f"base_{os.path.basename(f)}", f)
    for f in sorted(glob.glob(os.path.join(config.IMAGE_DIR, "normal", "normal_colon_*.png"))):
        add_sample("colorectal_normal", f"base_{os.path.basename(f)}", f)
    for f in sorted(glob.glob(os.path.join(config.IMAGE_DIR, "benign", "*.png"))):
        add_sample("breast_benign", f"base_{os.path.basename(f)}", f)
    for f in sorted(glob.glob(os.path.join(config.IMAGE_DIR, "breast", "*.png"))):
        add_sample("breast_malignant", f"base_{os.path.basename(f)}", f)
    for f in sorted(glob.glob(os.path.join(config.IMAGE_DIR, "lung", "*.png"))):
        add_sample("lung_malignant", f"base_{os.path.basename(f)}", f)
    for f in sorted(glob.glob(os.path.join(config.IMAGE_DIR, "colorectal", "*.png"))):
        add_sample("colorectal_malignant", f"base_{os.path.basename(f)}", f)

    # 2. Expanded multi-cohort dataset (excluding ood_calibration & external_validation)
    manifest_path = os.path.join(config.DATA_DIR, "expanded_cohort", "cohort_manifest.json")
    if include_expanded and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for item in json.load(f):
                cat = item.get("category")
                if cat in CATEGORY_META:
                    add_sample(cat, item["group_id"], item["path"])

    train_records, val_records, test_records = [], [], []

    for cat, (organ_lbl, status_lbl, subtype_lbl) in CATEGORY_META.items():
        gids = sorted(grouped_entries[cat].keys())
        rng.shuffle(gids)
        n = len(gids)
        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.15))
        tr_g = gids[:n_train]
        va_g = gids[n_train : n_train + n_val]
        te_g = gids[n_train + n_val :]

        for g_list, bucket in [(tr_g, train_records), (va_g, val_records), (te_g, test_records)]:
            for gid in g_list:
                for p in grouped_entries[cat][gid]:
                    bucket.append({
                        "image_path": p,
                        "organ_label": organ_lbl,
                        "status_label": status_lbl,
                        "subtype_label": subtype_lbl,
                        "group_id": gid,
                    })

    print(f"Group-Stratified Multi-Cohort Splits: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")
    return train_records, val_records, test_records
