"""
sensor.py
─────────
Simule le protocole expérimental d'un pyranomètre rotatif (Pan-Tilt).

Au lieu d'envoyer une seule mesure, ce script simule un "Scan 360" complet :
  1. Position 0 (Origin) : Mesure à plat (GHI)
  2. Position 1 (Fit 1)  : Mesure inclinée vers le Sud
  3. Position 2 (Fit 2)  : Mesure inclinée vers l'Est
  4. Position 3 (Fit 3)  : Mesure inclinée vers l'Ouest

Il génère également une image Fisheye synthétique (avec bâtiments) et
envoie l'ensemble de ces contraintes géométriques et radiométriques 
au Celery Worker via RabbitMQ.
"""

import os
import time
import math
import random
import base64
import logging
import cv2
from datetime import datetime, timezone

from celery import Celery
from celery.exceptions import OperationalError
from sky_generator import generate_sky_image

RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
# L'intervalle par défaut est de 60s pour simuler la lenteur mécanique
# du vrai capteur Hukseflux SR05 (18s de stabilisation * 4 positions)
INTERVAL      = float(os.environ.get("INTERVAL_SECONDS", "60"))
QUEUE_NAME    = "solar_tasks"

logging.basicConfig(format="%(asctime)s [SENSOR] %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

BROKER_URL = f"pyamqp://guest:guest@{RABBITMQ_HOST}:5672//"
celery_app = Celery("solar_pipeline", broker=BROKER_URL)


def simulate_irradiance(hour_fraction: float) -> float:
    """Modèle simplifié de la courbe solaire journalière."""
    peak_irradiance = 1000.0
    solar_angle     = math.sin(math.pi * hour_fraction)
    raw             = peak_irradiance * solar_angle
    noise           = random.gauss(0, 30)
    return max(0.0, round(raw + noise, 1))


def wait_for_rabbitmq(app: Celery, retries: int = 15, delay: int = 3):
    """
    Tente d'établir une connexion avec le broker avant de démarrer.
    Évite les crashs au démarrage si RabbitMQ n'est pas encore prêt.
    """
    for attempt in range(1, retries + 1):
        try:
            with app.connection_for_write() as conn:
                conn.ensure_connection(max_retries=1)
            log.info("Connexion à RabbitMQ réussie ✓")
            return
        except OperationalError as e:
            log.warning(f"RabbitMQ pas encore prêt (tentative {attempt}/{retries}). Attente {delay}s...")
            time.sleep(delay)
    raise RuntimeError("Impossible de se connecter à RabbitMQ après plusieurs tentatives.")


def run():
    measure_id = 0
    while True:
        measure_id += 1
        
        # 1. Génération de l'environnement visuel (Image Fisheye avec Bâtiments)
        sky_img, cloud_cover_true = generate_sky_image(size=256)
        img_path = f"/app/shared/sky_{measure_id:05d}.jpg"
        cv2.imwrite(img_path, sky_img)

        # 2. Timestamp formaté (Exigé par ModelKd.py : YYYY-MM-DD HH:MM:SS.000)
        now_utc       = datetime.now(timezone.utc)
        timestamp_str = now_utc.strftime("%Y-%m-%d %H:%M:%S.000")

        # 3. Simulation de l'acquisition matérielle (Le Rotor)
        hour_fraction = (time.time() % 86400) / 86400
        base_irr      = simulate_irradiance(hour_fraction)
        
        # A. Position 0 : Zénith (Origin - GHI)
        mesure_origin = base_irr
        
        # B. Position 1 : Sud, incliné à 45° (Simule plus de lumière au Sud)
        mesure_fit_1  = base_irr * 1.15
        
        # C. Position 2 : Est, incliné à 60°
        mesure_fit_2  = base_irr * 0.85
        
        # D. Position 3 : Ouest, incliné à 60°
        mesure_fit_3  = base_irr * 0.60

        # 4. Création du Payload JSON pour le Worker Celery
        message = {
            "id":             measure_id,
            "timestamp":      timestamp_str,
            "image_path":     img_path,
            "cloud_cover_gt": round(cloud_cover_true, 4),
            
            # --- Les données Radiométriques pour le Fitting de Perez ---
            "origin": {
                "irradiance": mesure_origin
            },
            "fits":[
                {"azimuth": 180.0, "inclination": 45.0, "irradiance": mesure_fit_1},
                {"azimuth": 90.0,  "inclination": 60.0, "irradiance": mesure_fit_2},
                {"azimuth": 270.0, "inclination": 60.0, "irradiance": mesure_fit_3}
            ]
        }

        try:
            # Envoi Asynchrone : Le capteur "libère" la tâche et prépare son scan suivant
            celery_app.send_task("solar.process_measurement", args=[message], queue=QUEUE_NAME)
            log.info(f"[#{measure_id:05d}] Scan 360° envoyé (Origin + 3 Fits) | GHI={mesure_origin:.1f} W/m² | t={timestamp_str}")
        except OperationalError as e:
            log.error(f"[#{measure_id:05d}] RECONNEXION - Impossible d'envoyer la tâche : {e}")
            wait_for_rabbitmq(celery_app)

        # Simulation de la lenteur mécanique du rotor (Wait 18s * 4)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    log.info(f"Démarrage du capteur rotatif simulé... Envoi via Celery vers la queue '{QUEUE_NAME}'")
    wait_for_rabbitmq(celery_app)
    run()