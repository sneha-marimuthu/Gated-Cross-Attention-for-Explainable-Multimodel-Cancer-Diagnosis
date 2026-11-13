"""
Image transformation, vignette cropping, Reinhard H&E stain normalization,
and multi-scale preprocessing module for GCAEMMCD.
"""
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import config

# Measured global reference brightfield H&E OpenCV-LAB statistics across valid training specimens
REF_LAB_MEAN = np.array([172.52, 151.69, 102.06], dtype=np.float32)
REF_LAB_STD = np.array([30.63, 10.16, 12.03], dtype=np.float32)


def crop_eyepiece_vignette(pil_img: Image.Image) -> Image.Image:
    """
    Detects circular microscope eyepiece vignetting (dark corner borders) and crops
    to the illuminated brightfield specimen region without clipping normal histology.
    """
    rgb = np.array(pil_img.convert("RGB"))
    h, w, _ = rgb.shape
    if min(h, w) < 64:
        return pil_img.convert("RGB")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    # Trigger only when image perimeter is predominantly black eyepiece framing
    if float(np.mean(border < 28)) < 0.35:
        return pil_img.convert("RGB")

    ys, xs = np.where(gray >= 30)
    if len(ys) < 0.15 * h * w:
        return pil_img.convert("RGB")

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    # Inset slightly (6%) to remove residual curved dark arc corners inside bounding box
    dy = int(0.06 * (y1 - y0))
    dx = int(0.06 * (x1 - x0))
    y0, y1 = min(y1 - 32, y0 + dy), max(y0 + 32, y1 - dy)
    x0, x1 = min(x1 - 32, x0 + dx), max(x0 + 32, x1 - dx)
    if (y1 - y0) >= 64 and (x1 - x0) >= 64:
        return Image.fromarray(rgb[y0:y1, x0:x1])
    return pil_img.convert("RGB")


def reinhard_stain_normalize(pil_img: Image.Image, blend: float = 0.85) -> Image.Image:
    """
    Lightweight Reinhard LAB stain normalization calibrated to brightfield H&E reference
    statistics so cross-scanner stain intensity differences do not act as class shortcuts.
    """
    rgb = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_mean = lab.mean(axis=(0, 1))
    src_std = np.maximum(lab.std(axis=(0, 1)), np.array([6.0, 3.0, 3.0], dtype=np.float32))

    norm_lab = ((lab - src_mean) / src_std) * REF_LAB_STD + REF_LAB_MEAN
    out_lab = blend * norm_lab + (1.0 - blend) * lab
    out_lab = np.clip(out_lab, 0.0, 255.0).astype(np.uint8)
    out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(out_rgb)


def _prepare_pil(image_input, apply_stain_norm=True) -> Image.Image:
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.fromarray(image_input).convert("RGB")

    image = crop_eyepiece_vignette(image)
    if apply_stain_norm:
        image = reinhard_stain_normalize(image)
    return image


def get_image_transforms(is_train=False, stain_norm=True):
    """
    Returns torchvision transforms pipeline with multi-scale cropping and
    optional Reinhard H&E stain normalization.
    """
    pre_steps = [
        transforms.Lambda(lambda im: crop_eyepiece_vignette(im)),
    ]
    if stain_norm:
        pre_steps.append(transforms.Lambda(lambda im: reinhard_stain_normalize(im)))

    if is_train:
        return transforms.Compose(pre_steps + [
            transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.40, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
        ])
    else:
        return transforms.Compose(pre_steps + [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
        ])


def preprocess_image(image_input, stain_norm=True):
    """
    Takes PIL Image, numpy array, or file path and returns [1, 3, 224, 224] tensor.
    """
    image = _prepare_pil(image_input, apply_stain_norm=False)
    transform = get_image_transforms(is_train=False, stain_norm=stain_norm)
    return transform(image).unsqueeze(0)


def preprocess_multicrop(image_input, stain_norm=True):
    """
    Returns multi-scale tensor batch [N, 3, 224, 224] (full field + center 50% crop
    for high-resolution micrographs >= 360px) so magnification shifts stay robust.
    """
    clean_pil = _prepare_pil(image_input, apply_stain_norm=stain_norm)
    tf = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
    ])
    tensors = [tf(clean_pil)]
    w, h = clean_pil.size
    if min(w, h) >= 360:
        center_crop = clean_pil.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
        tensors.append(tf(center_crop))
    return torch.stack(tensors, dim=0)
