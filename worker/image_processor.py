"""
image_processor.py
──────────────────
Pipeline de traitement d'image ciel avec OpenCV.

Étapes
──────
  1. Décodage de l'image (base64 → numpy array BGR)
  2. Conversion en espace HSV (Hue-Saturation-Value)
  3. Masque de détection des nuages (pixels très clairs / faible saturation)
  4. Calcul de la couverture nuageuse (cloud_cover ∈ [0, 1])
  5. Calcul de la luminosité moyenne (brightness ∈ [0, 255])
  6. Correction de l'irradiance brute

Concepts OpenCV utilisés
────────────────────────
  cv2.imdecode         → charger image depuis bytes
  cv2.cvtColor         → conversion d'espace colorimétrique
  cv2.inRange          → seuillage (thresholding)
  cv2.GaussianBlur     → lissage avant seuillage (réduit le bruit)
  numpy.mean           → statistiques sur l'image
"""

import base64
import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ─── Résultat du traitement ──────────────────────────────────────────────────

@dataclass
class ImageAnalysis:
    cloud_cover:       float   # fraction [0, 1]
    brightness_mean:   float   # luminance moyenne [0, 255]
    cloud_factor:      float   # facteur de correction [0, 1]
    irradiance_pred:   float   # irradiance corrigée (W/m²)


# ─── Pipeline de traitement ──────────────────────────────────────────────────

def analyze_sky_image(image_b64: str, irradiance_raw: float) -> ImageAnalysis:
    """
    Traite une image ciel encodée en base64 et prédit l'irradiance corrigée.

    Parameters
    ----------
    image_b64      : str    Image JPEG encodée en base64
    irradiance_raw : float  Mesure brute du pyranomètre (W/m²)

    Returns
    -------
    ImageAnalysis  : résultats détaillés du traitement
    """
    # ── Étape 1 : Décodage ──────────────────────────────────────────────────
    raw_bytes  = base64.b64decode(image_b64)
    buf        = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr    = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Impossible de décoder l'image")

    log.debug(f"Image décodée : {img_bgr.shape}")

    # ── Étape 2 : Lissage (réduit le bruit avant seuillage) ─────────────────
    img_blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)

    # ── Étape 3 : Conversion BGR → HSV ──────────────────────────────────────
    # HSV est plus pratique pour isoler les nuages (blancs = faible saturation)
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    # ── Étape 4 : Détection des nuages via seuillage HSV ────────────────────
    # Nuages blancs/gris : saturation faible (< 40) ET valeur élevée (> 180)
    lower_cloud = np.array([0,   0, 180], dtype=np.uint8)
    upper_cloud = np.array([180, 40, 255], dtype=np.uint8)
    cloud_mask  = cv2.inRange(img_hsv, lower_cloud, upper_cloud)

    # Couverture nuageuse = fraction de pixels détectés comme nuages
    total_pixels = cloud_mask.size
    cloud_pixels = int(np.sum(cloud_mask > 0))
    cloud_cover  = cloud_pixels / total_pixels

    # ── Étape 5 : Luminosité moyenne (canal Value du HSV) ───────────────────
    brightness_mean = float(np.mean(img_hsv[:, :, 2]))

    # ── Étape 6 : Correction de l'irradiance ────────────────────────────────
    # Modèle simple : les nuages réduisent l'irradiance de max 70%
    # En production → remplacer par un modèle ML entraîné sur données réelles
    CLOUD_ATTENUATION = 0.70   # coefficient d'atténuation maximal
    cloud_factor      = 1.0 - (cloud_cover * CLOUD_ATTENUATION)
    irradiance_pred   = max(0.0, round(irradiance_raw * cloud_factor, 1))

    log.info(
        f"Analyse image → cloud_cover={cloud_cover:.1%} | "
        f"brightness={brightness_mean:.1f} | "
        f"cloud_factor={cloud_factor:.3f} | "
        f"irr_pred={irradiance_pred} W/m²"
    )

    return ImageAnalysis(
        cloud_cover=round(cloud_cover, 4),
        brightness_mean=round(brightness_mean, 2),
        cloud_factor=round(cloud_factor, 4),
        irradiance_pred=irradiance_pred,
    )
