"""
save_gradcam_images.py
----------------------
Utility to convert Grad-CAM overlay arrays (uint8 [0..255] or float [0..1])
to PNG files, either directly via `save_gradcam_array()` or from a JSON report
containing serialized image arrays or saved image paths.
"""

import json
import os
import sys
import numpy as np
from PIL import Image


def _to_numpy(item):
    """Convert nested list, NumPy array, or torch.Tensor to a NumPy array."""
    if isinstance(item, np.ndarray):
        return item
    if isinstance(item, list):
        arr = np.asarray(item)
        if arr.ndim in (2, 3) and arr.shape[0] >= 16 and arr.dtype.kind in ("u", "i", "f"):
            return arr
        raise TypeError("List is not an image array.")
    try:
        import torch
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().numpy()
    except Exception:
        pass
    raise TypeError(f"Unsupported Grad-CAM type: {type(item)}")


def save_gradcam_array(cam_array, case_name: str, out_dir: str) -> str:
    """Saves a Grad-CAM NumPy array or Tensor cleanly as an RGB PNG file."""
    os.makedirs(out_dir, exist_ok=True)
    arr = _to_numpy(cam_array)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.dtype == np.uint8:
        uint8_arr = arr
    elif np.issubdtype(arr.dtype, np.floating) and float(np.max(arr)) <= 1.0 + 1e-6:
        uint8_arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        uint8_arr = np.clip(arr, 0, 255).astype(np.uint8)

    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in case_name).strip("_")
    out_path = os.path.join(out_dir, f"gradcam_{safe_name}.png")
    Image.fromarray(uint8_arr).save(out_path)
    return out_path


def main(summary_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(summary_path, "r") as f:
        records = json.load(f)

    saved_count = 0
    for rec in records:
        case_name = rec.get("name", "unnamed_case")
        existing_path = rec.get("gradcam_path") or rec.get("summary", {}).get("gradcam_path")
        if existing_path and os.path.exists(existing_path):
            print(f"[✓] Verified Grad-CAM PNG for {case_name} -> {existing_path}")
            saved_count += 1
            continue

        raw_items = rec.get("raw") or rec.get("summary", {}).get("raw") or []
        for idx, item in enumerate(raw_items):
            if isinstance(item, (str, dict)):
                continue
            try:
                out_path = save_gradcam_array(item, f"{case_name}_{idx}", out_dir)
                print(f"[✓] Saved Grad-CAM for {case_name} -> {out_path}")
                saved_count += 1
                break
            except Exception:
                continue

    print(f"[*] Total Grad-CAM PNGs verified/saved: {saved_count}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python save_gradcam_images.py <summary.json> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
