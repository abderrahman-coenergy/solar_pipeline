"""
sky_generator.py
────────────────
Génère des images ciel synthétiques au format dual fisheye, simulant
la sortie réelle d'une caméra fisheye 360° (ex: Insta360, Ricoh Theta).

Format de sortie
────────────────
Image H x 2H pixels (ex: 256 x 512), fond noir.
  - Moitié DROITE : hémisphère supérieur (ciel, zénith au centre du disque)
  - Moitié GAUCHE : hémisphère inférieur (sol, nadir au centre du disque)

Projection utilisée : équidistante  →  r = f * theta
  avec f = R / (fov_rad / 2), R = rayon du disque fisheye en pixels.

Ce format est directement compatible avec dual_fisheye_to_equirectangular()
de coe_sol/fisheye.py (pipeline vision du worker).

En production, ces images viendraient d'une vraie caméra fisheye.
"""

import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ─── Helpers de projection ────────────────────────────────────────────────

def _equidistant_fisheye_disk(size: int, scene_fn, fov_deg: float = 180.0) -> np.ndarray:
    """
    Génère un disque fisheye carré (size x size) en appliquant scene_fn
    à chaque pixel valide du disque.

    Args:
        size      : côté du carré de sortie (= diamètre du disque fisheye)
        scene_fn  : callable (theta_rad, phi_rad) -> (R, G, B) uint8
                    theta = angle depuis l'axe optique  [0, fov/2]
                    phi   = azimut dans le plan image   [-pi, pi]
        fov_deg   : champ de vue total de la lentille (défaut 180°)

    Returns:
        np.ndarray uint8 (size, size, 3), fond noir hors disque.
    """
    fov_rad = math.radians(fov_deg)
    R = size / 2.0          # rayon du disque en pixels
    f = R / (fov_rad / 2.0) # paramètre focal équidistant

    cx = cy = size / 2.0

    # Grille de coordonnées pixel
    ys, xs = np.mgrid[0:size, 0:size]
    dx = xs - cx
    dy = ys - cy
    r_px = np.sqrt(dx**2 + dy**2)

    # Masque circulaire (dans le disque fisheye)
    mask = r_px <= R

    # Angles sphériques
    theta = np.where(mask, r_px / f, 0.0)   # angle depuis l'axe optique
    phi   = np.arctan2(dy, dx)               # azimut

    # Appel vectorisé de la fonction de scène
    img = np.zeros((size, size, 3), dtype=np.uint8)
    if np.any(mask):
        rgb = scene_fn(theta[mask], phi[mask])  # (N, 3)
        img[mask] = rgb

    return img


# ─── Fonctions de scène ───────────────────────────────────────────────────

def _sky_scene(theta: np.ndarray, phi: np.ndarray,
               num_clouds: int, cloud_params: list,
               sun_theta: float, sun_phi: float, sun_r: float) -> np.ndarray:
    """
    Calcule la couleur ciel pour chaque rayon (theta, phi).
    theta=0 → zénith (axe optique du fisheye ciel).
    """
    N = len(theta)
    rgb = np.zeros((N, 3), dtype=np.float32)

    # --- 1. Fond de ciel (dégradé zénith → horizon) ---
    t = np.clip(theta / (math.pi / 2.0), 0.0, 1.0)  # 0=zénith, 1=horizon
    sky_zenith  = np.array([100., 160., 230.], dtype=np.float32)  # bleu foncé
    sky_horizon = np.array([180., 220., 255.], dtype=np.float32)  # bleu clair
    for c in range(3):
        rgb[:, c] = sky_zenith[c] * (1 - t) + sky_horizon[c] * t

    # --- 2. Soleil ---
    cos_sun = (np.cos(theta) * np.cos(sun_theta) +
               np.sin(theta) * np.sin(sun_theta) * np.cos(phi - sun_phi))
    cos_sun = np.clip(cos_sun, -1.0, 1.0)
    ang_sun = np.arccos(cos_sun)
    sun_mask = ang_sun < sun_r
    rgb[sun_mask] = [255., 255., 180.]

    # Halo autour du soleil
    halo_mask = (ang_sun >= sun_r) & (ang_sun < sun_r * 3.0)
    halo_t = ((ang_sun[halo_mask] - sun_r) / (sun_r * 2.0))
    rgb[halo_mask, 0] = np.minimum(255., rgb[halo_mask, 0] + 60. * (1 - halo_t))
    rgb[halo_mask, 1] = np.minimum(255., rgb[halo_mask, 1] + 40. * (1 - halo_t))

    # --- 3. Nuages ---
    for (c_theta, c_phi, c_r, c_alpha) in cloud_params:
        cos_c = (np.cos(theta) * np.cos(c_theta) +
                 np.sin(theta) * np.sin(c_theta) * np.cos(phi - c_phi))
        cos_c = np.clip(cos_c, -1.0, 1.0)
        ang_c = np.arccos(cos_c)
        cloud_mask = ang_c < c_r
        blend = c_alpha
        rgb[cloud_mask] = (rgb[cloud_mask] * (1. - blend) +
                           np.array([255., 255., 255.]) * blend)

    # --- 4. NOUVEAU : Horizon Urbain (Bâtiments) ---
    # On simule 3 gros bâtiments à différents azimuts
    buildings =[
        {"azimuth": 0.0, "width": 0.5, "height_angle": math.pi / 4},    # Bâtiment Nord, monte jusqu'à 45°
        {"azimuth": 1.5, "width": 0.3, "height_angle": math.pi / 6},    # Bâtiment Est, monte à 30°
        {"azimuth": -2.0, "width": 0.8, "height_angle": math.pi / 3},   # Gros bâtiment Ouest, monte à 60°
    ]
    
    for b in buildings:
        # phi est entre -pi et pi. On regarde la distance à l'azimut du bâtiment.
        # On utilise une astuce avec cos() pour gérer le rebouclage à pi/-pi.
        ang_dist = np.arccos(np.clip(np.cos(phi - b["azimuth"]), -1.0, 1.0))
        
        # Le masque du bâtiment : il faut être dans la bonne direction (ang_dist < width)
        # ET assez bas sur l'horizon (theta > hauteur du bâtiment depuis le zénith)
        zenith_distance = (math.pi / 2.0) - b["height_angle"]
        building_mask = (ang_dist < b["width"]) & (theta > zenith_distance)
        
        # On dessine les bâtiments en gris très foncé (presque noir)
        rgb[building_mask] =[30., 30., 35.]

    return np.clip(rgb, 0., 255.).astype(np.uint8)

def _ground_scene(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Calcule la couleur sol pour chaque rayon (theta, phi).
    theta=0 → nadir (axe optique du fisheye sol).
    """
    N = len(theta)
    # Dégradé nadir (terre sombre) → horizon (plus clair)
    t = np.clip(theta / (math.pi / 2.0), 0.0, 1.0)
    ground_nadir   = np.array([60.,  45.,  30.], dtype=np.float32)   # brun foncé
    ground_horizon = np.array([120., 100., 70.], dtype=np.float32)   # brun clair

    rgb = np.zeros((N, 3), dtype=np.float32)
    for c in range(3):
        rgb[:, c] = ground_nadir[c] * (1 - t) + ground_horizon[c] * t

    # Légère variation aléatoire texture sol
    noise = np.random.uniform(-15., 15., (N, 3)).astype(np.float32)
    rgb += noise

    return np.clip(rgb, 0., 255.).astype(np.uint8)


# ─── Générateur principal ─────────────────────────────────────────────────

def generate_sky_image(size: int = 256) -> tuple:
    """
    Génère une image dual fisheye synthétique (size x 2*size pixels).

    Args:
        size : hauteur de l'image = diamètre d'un disque fisheye.
               La largeur sera 2*size (deux disques côte à côte).

    Returns:
        img_array       : np.ndarray BGR uint8 (size, 2*size, 3)
                          compatible OpenCV, prêt pour cv2.imwrite()
        cloud_cover_true: float [0, 1] estimation de la couverture nuageuse
    """
    # --- Paramètres aléatoires de la scène ---
    # Soleil (position dans l'hémisphère supérieur)
    sun_theta = random.uniform(0.05, math.pi / 2.5)   # pas trop bas sur l'horizon
    sun_phi   = random.uniform(-math.pi, math.pi)
    sun_r     = math.radians(random.uniform(5., 12.))  # rayon angulaire du soleil

    # Nuages
    num_clouds = random.randint(0, 10)
    cloud_params = []
    total_cloud_solid_angle = 0.0
    for _ in range(num_clouds):
        c_theta = random.uniform(0., math.pi / 2.0)
        c_phi   = random.uniform(-math.pi, math.pi)
        c_r     = math.radians(random.uniform(8., 35.))  # rayon angulaire du nuage
        c_alpha = random.uniform(0.5, 0.95)              # opacité
        cloud_params.append((c_theta, c_phi, c_r, c_alpha))
        # Estimation solid angle ≈ pi * r^2 / (2*pi) pour hémisphère
        total_cloud_solid_angle += c_r ** 2

    # Couverture nuageuse estimée (fraction de l'hémisphère)
    cloud_cover_true = min(total_cloud_solid_angle / (math.pi / 2.) ** 2, 1.0)

    # --- Disque fisheye DROIT : hémisphère ciel (zénith au centre) ---
    def sky_fn(theta, phi):
        return _sky_scene(theta, phi, num_clouds, cloud_params,
                          sun_theta, sun_phi, sun_r)

    disk_sky = _equidistant_fisheye_disk(size, sky_fn, fov_deg=180.0)

    # --- Disque fisheye GAUCHE : hémisphère sol (nadir au centre) ---
    disk_ground = _equidistant_fisheye_disk(size, _ground_scene, fov_deg=180.0)

    # --- Assemblage : [gauche | droite] = [sol | ciel] ---
    # dual_fisheye_to_equirectangular attend : gauche=hémisphère inférieur, droite=supérieur
    dual = np.zeros((size, 2 * size, 3), dtype=np.uint8)
    dual[:, :size]       = disk_ground   # moitié gauche = sol
    dual[:, size:]       = disk_sky      # moitié droite = ciel

    # Légère texture via flou gaussien (PIL)
    pil_img = Image.fromarray(dual, mode='RGB')
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.6))
    dual_rgb = np.array(pil_img, dtype=np.uint8)

    # Conversion RGB → BGR pour OpenCV
    img_bgr = dual_rgb[:, :, ::-1].copy()

    return img_bgr, cloud_cover_true


if __name__ == "__main__":
    import cv2
    arr, cc = generate_sky_image(size=256)
    print(f"Image générée : {arr.shape}  |  cloud_cover={cc:.2%}")
    cv2.imwrite("/tmp/sky_test_fisheye.jpg", arr)
    print("Sauvegardée dans /tmp/sky_test_fisheye.jpg")
