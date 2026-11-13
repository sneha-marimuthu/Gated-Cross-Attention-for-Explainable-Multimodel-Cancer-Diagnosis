"""
Disk-Safe Multi-Cohort Histopathology Expansion Builder.
Builds / resumes `data/expanded_cohort/` across:
  1. dbzadnen/breast-histopathology-images (Breast IDC Benign/Normal vs Malignant)
  2. 1aurent/NCT-CRC-HE (Kather NCT-CRC-HE: label 6=NORM vs label 8=TUM)
  3. 1aurent/LC25000 (Lung Benign/Normal, Lung Adenocarcinoma, Lung Squamous Carcinoma)
  4. Wikimedia Commons CC0/CC-BY clinical micrographs (Breast, Lung, Colon + OOD Calibration Organs)
Strictly excludes every file in `data/external_validation/provenance_manifest.json`.
"""

import glob
import io
import json
import os
import sys
import time
import requests
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.domain_validator import validate_histopathology_domain


def download_file(url, dest_path):
    headers = {"User-Agent": "GCAEMMCD-Research-Trainer/1.0"}
    with requests.get(url, headers=headers, stream=True, timeout=90) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def decode_parquet_image(img_cell):
    if isinstance(img_cell, dict):
        if img_cell.get("bytes"):
            return Image.open(io.BytesIO(img_cell["bytes"])).convert("RGB")
        if img_cell.get("path") and os.path.exists(img_cell["path"]):
            return Image.open(img_cell["path"]).convert("RGB")
    raise ValueError("Unsupported parquet image cell")


def save_valid_patch(pil_img, out_path, max_dim=384):
    ok, dom, _, _ = validate_histopathology_domain(pil_img)
    if not ok:
        return False
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        pil_img = pil_img.resize((max(64, int(w * scale)), max(64, int(h * scale))), Image.BICUBIC)
    pil_img.save(out_path, format="PNG")
    return True


def main():
    out_root = os.path.join(PROJECT_ROOT, "data", "expanded_cohort")
    categories = [
        "breast_malignant", "breast_benign",
        "lung_malignant", "lung_normal",
        "colorectal_malignant", "colorectal_normal",
        "ood_calibration",
    ]
    for c in categories:
        os.makedirs(os.path.join(out_root, c), exist_ok=True)

    ext_manifest_path = os.path.join(PROJECT_ROOT, "data", "external_validation", "provenance_manifest.json")
    blacklist_titles = set()
    if os.path.exists(ext_manifest_path):
        with open(ext_manifest_path) as f:
            for item in json.load(f):
                blacklist_titles.add(item["source_title"].lower())

    tmp_pq = "/tmp/gca_tmp_shard.parquet"

    # 1. Breast IDC (skip if already populated)
    existing_bb = glob.glob(os.path.join(out_root, "breast_benign", "idc_*.png"))
    existing_bm = glob.glob(os.path.join(out_root, "breast_malignant", "idc_*.png"))
    if len(existing_bb) < 60 or len(existing_bm) < 60:
        print("[1/4] Streaming Breast IDC cohort (83.9 MB)...")
        try:
            download_file(
                "https://huggingface.co/datasets/dbzadnen/breast-histopathology-images/resolve/main/data/validation-00000-of-00001.parquet",
                tmp_pq,
            )
            tbl = pq.read_table(tmp_pq)
            labels = tbl["label"].to_pylist()
            imgs = tbl["image"].to_pylist()
            rng = np.random.RandomState(42)
            for target_lbl, cat in [(0, "breast_benign"), (1, "breast_malignant")]:
                idxs = [i for i, l in enumerate(labels) if l == target_lbl]
                rng.shuffle(idxs)
                kept = 0
                for i in idxs:
                    if kept >= 60:
                        break
                    im = decode_parquet_image(imgs[i])
                    out_p = os.path.join(out_root, cat, f"idc_{cat}_{kept:03d}.png")
                    if save_valid_patch(im, out_p, max_dim=224):
                        kept += 1
                print(f"  Saved {kept} patches -> {cat}")
        finally:
            if os.path.exists(tmp_pq):
                os.remove(tmp_pq)
    else:
        print(f"[1/4] Breast IDC already present ({len(existing_bb)} benign, {len(existing_bm)} malignant).")

    # 2. NCT-CRC-HE Colorectal (ClassLabel: 6=NORM, 8=TUM)
    existing_cm = glob.glob(os.path.join(out_root, "colorectal_malignant", "nct_*.png"))
    existing_cn = glob.glob(os.path.join(out_root, "colorectal_normal", "nct_*.png"))
    if len(existing_cm) < 60 or len(existing_cn) < 60:
        print("[2/4] Streaming NCT-CRC-HE Colorectal shards (NORM=6, TUM=8)...")
        nct_shards = [
            "https://huggingface.co/datasets/1aurent/NCT-CRC-HE/resolve/main/data/CRC_VAL_HE_7K-00002-of-00003-2009e87e0c48fccf.parquet",
            "https://huggingface.co/datasets/1aurent/NCT-CRC-HE/resolve/main/data/CRC_VAL_HE_7K-00001-of-00003-fbb5de8b1164f4ea.parquet",
        ]
        cnt_cm, cnt_cn = len(existing_cm), len(existing_cn)
        rng = np.random.RandomState(43)
        for s_url in nct_shards:
            if cnt_cm >= 60 and cnt_cn >= 60:
                break
            try:
                download_file(s_url, tmp_pq)
                tbl = pq.read_table(tmp_pq)
                labels = tbl["label"].to_pylist()
                imgs = tbl["image"].to_pylist()
                for target_lbl, cat, current_cnt in [(8, "colorectal_malignant", cnt_cm), (6, "colorectal_normal", cnt_cn)]:
                    if current_cnt >= 60:
                        continue
                    idxs = [i for i, l in enumerate(labels) if l == target_lbl]
                    rng.shuffle(idxs)
                    for i in idxs:
                        if cat == "colorectal_malignant" and cnt_cm >= 60:
                            break
                        if cat == "colorectal_normal" and cnt_cn >= 60:
                            break
                        im = decode_parquet_image(imgs[i])
                        idx_num = cnt_cm if cat == "colorectal_malignant" else cnt_cn
                        out_p = os.path.join(out_root, cat, f"nct_{cat}_{idx_num:03d}.png")
                        if save_valid_patch(im, out_p, max_dim=224):
                            if cat == "colorectal_malignant":
                                cnt_cm += 1
                            else:
                                cnt_cn += 1
            finally:
                if os.path.exists(tmp_pq):
                    os.remove(tmp_pq)
        print(f"  NCT-CRC-HE extracted: colorectal_malignant={cnt_cm}, colorectal_normal={cnt_cn}")
    else:
        print(f"[2/4] NCT-CRC-HE already present ({len(existing_cm)} mal, {len(existing_cn)} norm).")

    # 3. LC25000 Lung (extract remaining lung_aca if needed; lung_n & lung_scc already done)
    existing_ln = glob.glob(os.path.join(out_root, "lung_normal", "lc25k_lung_normal_*.png"))
    existing_lscc = glob.glob(os.path.join(out_root, "lung_malignant", "lc25k_lung_scc_*.png"))
    existing_laca = glob.glob(os.path.join(out_root, "lung_malignant", "lc25k_lung_aca_*.png"))
    if len(existing_ln) < 50 or len(existing_lscc) < 35 or len(existing_laca) < 35:
        print("[3/4] Streaming LC25000 shard for missing Lung subclasses...")
        lc_shards = [
            "https://huggingface.co/datasets/1aurent/LC25000/resolve/main/data/train-00000-of-00004-40f80292828d4b6c.parquet",
            "https://huggingface.co/datasets/1aurent/LC25000/resolve/main/data/train-00002-of-00004-8b746b230324f3f3.parquet",
        ]
        cnts = {"lung_normal": len(existing_ln), "lung_aca": len(existing_laca), "lung_scc": len(existing_lscc)}
        lims = {"lung_normal": 50, "lung_aca": 35, "lung_scc": 35}
        rng = np.random.RandomState(44)
        for s_url in lc_shards:
            if all(cnts[k] >= lims[k] for k in cnts):
                break
            try:
                download_file(s_url, tmp_pq)
                tbl = pq.read_table(tmp_pq)
                organs = tbl["organ"].to_pylist()
                labels = tbl["label"].to_pylist()
                imgs = tbl["image"].to_pylist()
                for (org_id, lbl_id, key, cat) in [
                    (0, 0, "lung_normal", "lung_normal"),
                    (0, 1, "lung_aca", "lung_malignant"),
                    (0, 2, "lung_scc", "lung_malignant"),
                ]:
                    if cnts[key] >= lims[key]:
                        continue
                    idxs = [i for i in range(len(organs)) if organs[i] == org_id and labels[i] == lbl_id]
                    rng.shuffle(idxs)
                    for i in idxs:
                        if cnts[key] >= lims[key]:
                            break
                        im = decode_parquet_image(imgs[i])
                        out_p = os.path.join(out_root, cat, f"lc25k_{key}_{cnts[key]:03d}.png")
                        if save_valid_patch(im, out_p, max_dim=384):
                            cnts[key] += 1
            finally:
                if os.path.exists(tmp_pq):
                    os.remove(tmp_pq)
        print("  LC25000 totals:", cnts)
    else:
        print(f"[3/4] LC25000 Lung already present (norm={len(existing_ln)}, aca={len(existing_laca)}, scc={len(existing_lscc)}).")

    # 4. Wikimedia Commons Clinical Multi-Mag Micrographs + OOD Calibration Organs
    print("[4/4] Fetching Wikimedia clinical micrographs & OOD calibration tissues...")
    wiki_titles = {
        "File:Histopathology of invasive ductal carcinoma, intermediate magnification.jpg": ("breast_malignant", "wiki_br_m1"),
        "File:Histopathology of invasive ductal carcinoma, low magnification.jpg": ("breast_malignant", "wiki_br_m2"),
        "File:Histopathology of invasive lobular carcinoma of the breast.jpg": ("breast_malignant", "wiki_br_m3"),
        "File:Histopathology of a lymph node with metastatic invasive ductal carcinoma from the breast, Original.jpg": ("breast_malignant", "wiki_br_m4"),
        "File:Histopathology of fibroadenoma with small cell clusters.jpg": ("breast_benign", "wiki_br_b1"),
        "File:Fibroadenoma - high mag.jpg": ("breast_benign", "wiki_br_b2"),
        "File:Fibroadenoma - low mag.jpg": ("breast_benign", "wiki_br_b3"),
        "File:Breast fibroadenoma (1).jpg": ("breast_benign", "wiki_br_b4"),
        "File:Histopathology of lepidic predominant adenocarcinoma.jpg": ("lung_malignant", "wiki_lu_m1"),
        "File:Histopathology of lung adenocarcinoma with solid pattern.jpg": ("lung_malignant", "wiki_lu_m2"),
        "File:Lung squamous carcinoma -- high mag.jpg": ("lung_malignant", "wiki_lu_m3"),
        "File:Lung adenocarcinoma - high mag.jpg": ("lung_malignant", "wiki_lu_m4"),
        "File:Normal lung - high mag.jpg": ("lung_normal", "wiki_lu_n1"),
        "File:Normal alveoli.jpg": ("lung_normal", "wiki_lu_n2"),
        "File:Colon adenocarcinoma - high mag.jpg": ("colorectal_malignant", "wiki_co_m1"),
        "File:Colorectal adenocarcinoma - very high mag.jpg": ("colorectal_malignant", "wiki_co_m2"),
        "File:Colon adenocarcinoma -- high mag.jpg": ("colorectal_malignant", "wiki_co_m3"),
        "File:Colonic mucosa -- high mag.jpg": ("colorectal_normal", "wiki_co_n1"),
        "File:Normal colon - high mag.jpg": ("colorectal_normal", "wiki_co_n2"),
        "File:Papillary thyroid carcinoma - high mag.jpg": ("ood_calibration", "wiki_ood_thyroid"),
        "File:Gastric adenocarcinoma - high mag.jpg": ("ood_calibration", "wiki_ood_gastric"),
        "File:Urothelial carcinoma - high mag.jpg": ("ood_calibration", "wiki_ood_uro"),
        "File:Normal pancreas - high mag.jpg": ("ood_calibration", "wiki_ood_pancreas"),
        "File:Melanoma - high mag.jpg": ("ood_calibration", "wiki_ood_skin"),
        "File:Glioblastoma - high mag.jpg": ("ood_calibration", "wiki_ood_brain"),
    }

    S = requests.Session()
    S.headers.update({"User-Agent": "GCAEMMCD-Research-Trainer/1.0 (https://commons.wikimedia.org)"})
    r = S.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": "|".join(wiki_titles.keys()),
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": 800,
        },
        timeout=25,
    )
    if r.status_code == 200:
        pages = r.json().get("query", {}).get("pages", {})
        for p in pages.values():
            title = p.get("title", "")
            if title.lower() in blacklist_titles or title not in wiki_titles:
                continue
            ii = p.get("imageinfo", [{}])[0]
            url = ii.get("thumburl") or ii.get("url")
            if not url:
                continue
            cat, gid = wiki_titles[title]
            out_p = os.path.join(out_root, cat, f"{gid}_full.png")
            if os.path.exists(out_p):
                continue
            time.sleep(0.6)
            resp = S.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            im = Image.open(io.BytesIO(resp.content)).convert("RGB")
            save_valid_patch(im, out_p, max_dim=384)
            w, h = im.size
            if min(w, h) >= 500 and cat != "ood_calibration":
                cw, ch = w // 2, h // 2
                for q_idx, box in enumerate([(w//4, h//4, 3*w//4, 3*h//4), (0, 0, cw, ch), (cw, ch, w, h)]):
                    c_path = os.path.join(out_root, cat, f"{gid}_crop{q_idx}.png")
                    save_valid_patch(im.crop(box), c_path, max_dim=384)

    # Rebuild full manifest cleanly from disk so group_id prevents any cross-split leakage
    manifest = []
    for cat in categories:
        for fpath in sorted(glob.glob(os.path.join(out_root, cat, "*.png"))):
            bn = os.path.basename(fpath)
            if bn.startswith("wiki_"):
                gid = "_".join(bn.split("_")[:3])  # keep *_full and *_crop0..2 in same group!
                src = "Wikimedia"
            elif bn.startswith("idc_"):
                gid = os.path.splitext(bn)[0]
                src = "Breast_IDC"
            elif bn.startswith("nct_"):
                gid = os.path.splitext(bn)[0]
                src = "NCT_CRC_HE"
            else:
                gid = os.path.splitext(bn)[0]
                src = "LC25000"
            manifest.append({"path": fpath, "category": cat, "source": src, "group_id": gid})

    with open(os.path.join(out_root, "cohort_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    from collections import Counter
    print("\n[+] Final Expanded Cohort Summary:", dict(Counter(m["category"] for m in manifest)))


if __name__ == "__main__":
    main()
