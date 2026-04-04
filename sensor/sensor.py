import os
import time
import math
import random
import base64
import logging
import cv2
from datetime import datetime, timezone    # ← FIX bug #1 : pour formater le timestamp

from celery import Celery
from celery.exceptions import OperationalError
from sky_generator import generate_sky_image

RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
INTERVAL      = float(os.environ.get("INTERVAL_SECONDS", "3"))
QUEUE_NAME    = "solar_tasks"

logging.basicConfig(format="%(asctime)s [SENSOR] %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

BROKER_URL = f"pyamqp://guest:guest@{RABBITMQ_HOST}:5672//"
celery_app = Celery("solar_pipeline", broker=BROKER_URL)


def simulate_irradiance(hour_fraction: float) -> float:
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
            log.warning(f"RabbitMQ pas encore prêt (tentative {attempt}/{retries}). Attente {delay}s... Erreur: {e}")
            time.sleep(delay)
    raise RuntimeError("Impossible de se connecter à RabbitMQ après plusieurs tentatives.")


def run():
    measure_id = 0
    while True:
        measure_id += 1
        
        # 1. On génère l'image (qui représente l'environnement physique)
        sky_img, cloud_cover_true = generate_sky_image(size=256)
        img_path = f"/app/shared/sky_{measure_id:05d}.jpg"
        cv2.imwrite(img_path, sky_img)
        
        # 2. Le Timestamp commun pour le "scan"
        now_utc       = datetime.now(timezone.utc)
        timestamp_str = now_utc.strftime("%Y-%m-%d %H:%M:%S.000")
        
        # 3. Simulation du balayage (Les 4 positions du pyranomètre)
        hour_fraction = (time.time() % 86400) / 86400
        base_irr      = simulate_irradiance(hour_fraction)
        
        # Position 0 : Zénith (Origin - GHI)
        mesure_origin = base_irr
        
        # Position 1 : Sud, incliné à 45°
        mesure_fit_1 = base_irr * 1.1 # Simule plus de lumière au Sud
        
        # Position 2 : Est, incliné à 60°
        mesure_fit_2 = base_irr * 0.8 # Simule moins de lumière
        
        # Position 3 : Ouest, incliné à 60°
        mesure_fit_3 = base_irr * 0.6 # Simule encore moins
        
        # 4. Construction du message JSON structuré
        message = {
            "id": measure_id,
            "timestamp": timestamp_str,
            "image_path": img_path,
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
            celery_app.send_task("solar.process_measurement", args=[message], queue=QUEUE_NAME)
            log.info(f"[#{measure_id:05d}] Scan complet envoyé (Origin + 3 Fits) | t={timestamp_str}")
        except OperationalError as e:
            log.error(f"[#{measure_id:05d}] RECONNEXION - Impossible d'envoyer la tâche : {e}")
            wait_for_rabbitmq(celery_app)

        time.sleep(INTERVAL)


if __name__ == "__main__":
    log.info(f"Démarrage du capteur... Envoi via Celery vers la queue '{QUEUE_NAME}'")
    wait_for_rabbitmq(celery_app)
    run()
