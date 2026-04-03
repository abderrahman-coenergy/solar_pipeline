import numpy as np
from typing import Optional
import cv2
import coe_sol.masking as masking
from coe_sol.fisheye import dual_fisheye_to_equirectangular, Equirectangular
import logging

log = logging.getLogger(__name__)


def _ensure_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise RuntimeError("mask_sky returned None")

    if mask.dtype == bool:
        mask = (mask.astype(np.uint8) * 255)
    elif mask.dtype != np.uint8:
        try:
            mask = mask.astype(np.uint8)
        except Exception:
            mask = (mask > 0).astype(np.uint8) * 255

    unique = np.unique(mask)
    if set(unique.tolist()).issubset({0, 1}):
        mask = mask * 255

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return mask


def _split_dual_fisheye(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    w_half = w // 2
    left = image[:, :w_half]
    right = image[:, w_half:]
    return left, right


def _combine_dual_masks(mask_left: np.ndarray, mask_right: np.ndarray) -> np.ndarray:
    return np.concatenate([mask_left, mask_right], axis=1)


SINGLE_HALF_LEFT = 'left'
SINGLE_HALF_RIGHT = 'right'
DUAL = None


def compute_horizon_from_image(
    image_path: str,
    fov_deg: int,
    single_half=DUAL,
    azimuth_deg=0.0,
    inclination_deg=90.0,
    preview=False
) -> np.ndarray:

    azimuth = float(azimuth_deg)
    inclination = float(inclination_deg)

    image: Optional[np.ndarray] = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    w_half = w // 2

    # ─────────────────────────────────────────────
    # CAS LEFT
    # ─────────────────────────────────────────────
    if single_half == SINGLE_HALF_LEFT:
        left, _ = _split_dual_fisheye(image)

        mask_left = masking.mask_sky(left)
        if mask_left is None:
            log.warning("mask_left is None → fallback")
            mask_left = np.zeros((h, w_half), dtype=np.uint8)

        black_right = np.zeros((h, w_half), dtype=np.uint8)
        image_mask = _combine_dual_masks(mask_left, black_right)

    # ─────────────────────────────────────────────
    # CAS RIGHT
    # ─────────────────────────────────────────────
    elif single_half == SINGLE_HALF_RIGHT:
        _, right = _split_dual_fisheye(image)

        mask_right = masking.mask_sky(right)
        if mask_right is None:
            log.warning("mask_right is None → fallback")
            mask_right = np.zeros((h, w_half), dtype=np.uint8)

        black_left = np.zeros((h, w_half), dtype=np.uint8)
        image_mask = _combine_dual_masks(black_left, mask_right)

    # ─────────────────────────────────────────────
    # CAS DUAL
    # ─────────────────────────────────────────────
    else:
        left, right = _split_dual_fisheye(image)

        mask_left = masking.mask_sky(left)
        mask_right = masking.mask_sky(right)

        if mask_left is None:
            log.warning("mask_left is None → fallback")
            mask_left = np.zeros((h, w_half), dtype=np.uint8)

        if mask_right is None:
            log.warning("mask_right is None → fallback")
            mask_right = np.zeros((h, w_half), dtype=np.uint8)

        image_mask = _combine_dual_masks(mask_left, mask_right)

    # ─────────────────────────────────────────────
    # NORMALISATION
    # ─────────────────────────────────────────────
    image_mask = _ensure_uint8_mask(image_mask)

    # ─────────────────────────────────────────────
    # CALCUL HORIZON
    # ─────────────────────────────────────────────
    horizon = sample_horizon_from_mask(
        image_mask=image_mask,
        fov_deg=fov_deg,
        azimuth_deg=azimuth,
        inclination_deg=inclination,
        single_half=single_half,
        preview=preview,
        original_image_for_preview=image if preview else None
    )

    # ─────────────────────────────────────────────
    # FALLBACK FINAL
    # ─────────────────────────────────────────────
    if horizon is None:
        log.warning("Horizon computation failed → fallback flat horizon")
        horizon = np.zeros(360)

    return horizon


def get_horizon_from_sphere(eq: Equirectangular, preview=False) -> np.ndarray:

    if eq.array.ndim == 3:
        lum = eq.luminance()
        eq_mask = (np.clip(lum, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        eq_mask = eq.array.astype(np.uint8)

    horizon = []
    h, w = eq_mask.shape[:2]

    for col in range(w):
        col_data = eq_mask[:, col]
        non_black = np.where(col_data == 0)[0]

        if non_black.size == 0:
            horizon.append(h - 1)
        else:
            horizon.append(int(non_black[0]))

    if preview:
        eq_preview = eq
        arr = eq_preview.array

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            arr = arr.copy()

        for col in range(arr.shape[1]):
            row = int(np.clip(horizon[col], 0, arr.shape[0] - 1))
            arr[row, col] = np.array([255, 0, 0], dtype=arr.dtype)

        eq_preview.array = arr
        eq_preview.preview(show_2d=True, show_3d=True)

    return np.array(horizon)


def sample_horizon_from_mask(
    image_mask: np.ndarray,
    fov_deg: int,
    azimuth_deg: float,
    inclination_deg: float,
    single_half=DUAL,
    preview=False,
    original_image_for_preview: Optional[np.ndarray] = None
) -> np.ndarray:

    eq = dual_fisheye_to_equirectangular(
        image_mask,
        out_h=180,
        out_w=360,
        fov_deg=fov_deg,
        single_half=single_half
    )

    eq.rotate(
        delta_azimuth=np.deg2rad(float(azimuth_deg)),
        delta_inclination=np.deg2rad(float(inclination_deg))
    )

    return get_horizon_from_sphere(eq, preview=preview)


def hello_horizon():
    print("Hello from horizon module!")