"""
sky_generator.py
────────────────
Génère des images ciel synthétiques simulant différentes conditions
météorologiques (ciel dégagé, partiellement nuageux, couvert).

En production, ces images viendraient d'une caméra fisheye.
"""

import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_sky_image(width: int = 128, height: int = 128) -> np.ndarray:
    """
    Crée une image ciel synthétique en numpy array (H, W, 3) uint8.

    Returns
    -------
    img_array : np.ndarray
        Image BGR (compatible OpenCV) avec nuages simulés.
    cloud_cover_true : float
        Fraction réelle de nuages dans l'image [0, 1].
    """
    # 1. Fond de ciel
    sky_color = (135, 206, 235)   # bleu ciel (RGB)
    img = Image.new("RGB", (width, height), sky_color)
    draw = ImageDraw.Draw(img)

    # 2. Soleil
    sun_x = random.randint(width // 4, 3 * width // 4)
    sun_y = random.randint(height // 6, height // 3)
    sun_r = random.randint(8, 14)
    draw.ellipse(
        [sun_x - sun_r, sun_y - sun_r, sun_x + sun_r, sun_y + sun_r],
        fill=(255, 255, 180),
    )

    # 3. Nuages (ellipses blanches semi-transparentes)
    num_clouds = random.randint(0, 8)
    cloud_pixels = 0
    total_pixels = width * height

    cloud_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(cloud_layer)

    for _ in range(num_clouds):
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        rw = random.randint(15, 50)
        rh = random.randint(8, 25)
        alpha = random.randint(160, 240)
        cdraw.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=(255, 255, 255, alpha))

    # Estimation rapide de la couverture nuageuse
    arr_cloud = np.array(cloud_layer)[:, :, 3]   # canal alpha
    cloud_pixels = np.sum(arr_cloud > 50)
    cloud_cover_true = min(cloud_pixels / total_pixels, 1.0)

    # Composite
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, cloud_layer)
    img = img.convert("RGB")

    # Légère texture via flou
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    # → BGR pour OpenCV
    img_array = np.array(img)[:, :, ::-1].copy()

    return img_array, cloud_cover_true


if __name__ == "__main__":
    # Test rapide
    import cv2
    arr, cc = generate_sky_image()
    print(f"Image générée : {arr.shape}, cloud_cover={cc:.2%}")
    cv2.imwrite("/tmp/sky_test.jpg", arr)
