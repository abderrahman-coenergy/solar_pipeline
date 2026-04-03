"""Sky masking utilities using MiDaS for monocular depth estimation.

This module provides compact, focused functions that compose a depth-based
sky masking pipeline suitable for fisheye images.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import torch


# ─── Singleton MiDaS ──────────────────────────────────────────────────────
# Le modèle est chargé une seule fois au premier appel dans le processus
# worker, puis réutilisé pour toutes les tâches suivantes.
_MIDAS_MODEL = None
_MIDAS_TRANSFORMS = None


def _get_midas(device: str):
    global _MIDAS_MODEL, _MIDAS_TRANSFORMS

    if _MIDAS_MODEL is None:
        _MIDAS_MODEL = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small",
            trust_repo=True
        )
        _MIDAS_MODEL.to(device).eval()

        _MIDAS_TRANSFORMS = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True
        )

    return _MIDAS_MODEL, _MIDAS_TRANSFORMS


def _select_device(device: Optional[str]) -> str:
    """Return the device string to use for computation."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _largest_nonblack_component(img_rgb: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray], Optional[int]]:
    """Detect the largest non-black connected component and return its box."""
    non_black = np.any(img_rgb != 0, axis=2).astype(np.uint8) * 255
    _, comp_labels, comp_stats, _ = cv2.connectedComponentsWithStats(non_black, connectivity=8)
    if comp_labels is None or comp_labels.max() < 1:
        return None, comp_labels, None
    max_label = 1
    max_area = 0
    for lab in range(1, int(comp_labels.max()) + 1):
        area = int(comp_stats[lab, cv2.CC_STAT_AREA])
        if area > max_area:
            max_area = area
            max_label = lab
    left = int(comp_stats[max_label, cv2.CC_STAT_LEFT])
    top = int(comp_stats[max_label, cv2.CC_STAT_TOP])
    width = int(comp_stats[max_label, cv2.CC_STAT_WIDTH])
    height = int(comp_stats[max_label, cv2.CC_STAT_HEIGHT])
    return (left, top, left + width, top + height), comp_labels, max_label


def _run_midas_on_roi(roi: np.ndarray, device: str) -> np.ndarray:
    """Compute a depth crop for the ROI using the shared MiDaS singleton."""
    midas, transforms = _get_midas(device)
    transform = transforms.small_transform
    inp = transform(roi)
    if isinstance(inp, np.ndarray):
        inp = torch.from_numpy(inp)
    if inp.ndim == 3:
        inp = inp.unsqueeze(0)
    inp = inp.to(device)
    with torch.no_grad():
        prediction = midas(inp)
    if prediction.ndim == 4 and prediction.shape[1] == 1:
        pred = prediction[:, 0, :, :]
    elif prediction.ndim == 3:
        pred = prediction
    else:
        pred = prediction.squeeze()
    roi_h, roi_w = roi.shape[:2]
    pred_up = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(roi_h, roi_w), mode="bicubic", align_corners=False
    ).squeeze(1)
    return pred_up[0].cpu().numpy() if pred_up.ndim == 3 else pred_up.cpu().numpy()


def _normalize_and_build_full(depth_crop, H, W, crop_box, comp_labels, max_label):
    depth_norm_full = np.zeros((H, W), dtype=float)
    dmin, dmax = float(depth_crop.min()), float(depth_crop.max())
    if dmax - dmin < 1e-8:
        raise RuntimeError("Depth map is flat; cannot produce depth-based mask.")
    depth_crop_norm = (depth_crop - dmin) / (dmax - dmin)
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        depth_norm_full[y0:y1, x0:x1] = depth_crop_norm
        valid_mask = (comp_labels == max_label).astype(np.uint8) if comp_labels is not None else None
    else:
        depth_norm_full = depth_crop_norm
        valid_mask = None
    return depth_norm_full, depth_crop_norm, valid_mask


def _depth_to_sky_mask(depth_norm_full: np.ndarray, depth_threshold: float) -> np.ndarray:
    return (depth_norm_full <= float(depth_threshold)).astype(np.uint8) * 255


def _filter_components(sky_mask, min_area):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(sky_mask, connectivity=8)
    out_mask = np.zeros_like(sky_mask)
    if labels is None:
        return out_mask, labels, stats
    for lab in range(1, int(labels.max()) + 1):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area:
            out_mask[labels == lab] = 255
    return out_mask, labels, stats


def _ensure_nonempty(out_mask, labels, stats):
    if out_mask.sum() != 0:
        return out_mask
    if labels is None or labels.max() < 1:
        return out_mask
    areas = [int(stats[l, cv2.CC_STAT_AREA]) for l in range(1, int(labels.max()) + 1)]
    if len(areas) == 0:
        return out_mask
    largest = 1 + int(np.argmax(np.array(areas)))
    out_mask[labels == largest] = 255
    return out_mask


def _show_debug(img_rgb, roi, depth_crop_norm, out_mask):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1); plt.title('Input (RGB)'); plt.imshow(img_rgb); plt.axis('off')
    if roi is not None:
        plt.subplot(1, 4, 2); plt.title('Cropped ROI'); plt.imshow(roi); plt.axis('off')
    plt.subplot(1, 4, 3); plt.title('Depth (crop norm)'); plt.imshow(depth_crop_norm, cmap='plasma'); plt.axis('off')
    plt.subplot(1, 4, 4); plt.title('Depth Sky Mask'); plt.imshow(out_mask, cmap='gray'); plt.axis('off')
    plt.tight_layout(); plt.show()


def mask_depth_anything(image: np.ndarray, device: Optional[str] = None,
                        depth_threshold: float = 0.3, min_area: int = 200,
                        debug: bool = False) -> Optional[np.ndarray]:
    """Compute a depth-based sky mask for the provided BGR image."""
    device = _select_device(device)
    img_rgb = _bgr_to_rgb(image)
    H, W = img_rgb.shape[:2]
    crop_box, comp_labels, max_label = _largest_nonblack_component(img_rgb)
    roi = img_rgb if crop_box is None else img_rgb[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    try:
        depth_crop = _run_midas_on_roi(roi, device)
    except Exception as e:
        print(f"Depth model load/compute failed: {e}")
        return None
    try:
        depth_norm_full, depth_crop_norm, valid_mask = _normalize_and_build_full(
            depth_crop, H, W, crop_box, comp_labels, max_label
        )
    except RuntimeError as e:
        print(str(e))
        return None
    sky_mask = _depth_to_sky_mask(depth_norm_full, depth_threshold)
    if valid_mask is not None:
        sky_mask = cv2.bitwise_and(sky_mask, (valid_mask * 255).astype(np.uint8))
    out_mask, labels, stats = _filter_components(sky_mask, min_area)
    out_mask = _ensure_nonempty(out_mask, labels, stats)
    if debug:
        _show_debug(img_rgb, None if crop_box is None else roi, depth_crop_norm, out_mask)
    return out_mask


def mask_sky(image: np.ndarray, device: Optional[str] = None,
             debug: bool = False) -> Optional[np.ndarray]:
    """High-level wrapper to produce a sky mask using depth-based method."""
    device = _select_device(device)
    final_mask = mask_depth_anything(image, device=device, debug=debug)
    if final_mask is None:
        print("Depth-based masking failed or model not available.")
        return None
    return final_mask


if __name__ == "__main__":
    img_path = "/Users/eliotjanvier/Documents/freelance/coe/cosol/frame_26s.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Test image at {img_path} could not be loaded.")
    img = img[:, img.shape[1] // 2 :]
    mask_sam_res = mask_sky(img, debug=True)
    if mask_sam_res is not None:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1); plt.title("Input Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis("off")
        plt.subplot(1, 2, 2); plt.title("Sky Mask")
        plt.imshow(mask_sam_res, cmap="gray"); plt.axis("off")
        plt.tight_layout(); plt.show()
    else:
        print("Sky mask could not be generated.")
