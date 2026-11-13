"""
Dataset and split management for GCAEMMCD.
Enforces stratified train/val/test splits (70% train, 15% val, 15% test) without leakage.
"""
import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import config
from src.data.image_transforms import get_image_transforms

class HistopathologyDataset(Dataset):
    def __init__(self, sample_records, is_train=False):
        """
        sample_records: list of dicts with keys {'image_path', 'category', 'label'}
        """
        self.records = sample_records
        self.transform = get_image_transforms(is_train=is_train)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        tensor = self.transform(image)
        label = item["label"]
        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "category": item["category"],
            "image_path": item["image_path"]
        }

def build_splits(seed=config.SEED):
    """
    Partitions the 60 patches (20 per category) into:
    Train: 14 per category = 42 total (70%)
    Val: 3 per category = 9 total (15%)
    Test: 3 per category = 9 total (15%)
    """
    random.seed(seed)
    train_records, val_records, test_records = [], [], []

    for cat in config.CATEGORIES:
        cat_dir = os.path.join(config.IMAGE_DIR, cat)
        files = sorted(glob.glob(os.path.join(cat_dir, "*.png")))
        random.shuffle(files)
        
        label = config.CATEGORY_TO_IDX[cat]
        # 14 train, 3 val, 3 test
        train_files = files[:14]
        val_files = files[14:17]
        test_files = files[17:20]

        for f in train_files:
            train_records.append({"image_path": f, "category": cat, "label": label})
        for f in val_files:
            val_records.append({"image_path": f, "category": cat, "label": label})
        for f in test_files:
            test_records.append({"image_path": f, "category": cat, "label": label})

    # Save split IDs
    os.makedirs(config.SPLITS_DIR, exist_ok=True)
    with open(os.path.join(config.SPLITS_DIR, "train_ids.txt"), "w") as f:
        for r in train_records:
            f.write(f"{r['image_path']},{r['category']},{r['label']}\n")
    with open(os.path.join(config.SPLITS_DIR, "val_ids.txt"), "w") as f:
        for r in val_records:
            f.write(f"{r['image_path']},{r['category']},{r['label']}\n")
    with open(os.path.join(config.SPLITS_DIR, "test_ids.txt"), "w") as f:
        for r in test_records:
            f.write(f"{r['image_path']},{r['category']},{r['label']}\n")

    return train_records, val_records, test_records
