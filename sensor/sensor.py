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
        hour_fraction  = (time.time() % 86400) / 86400
        irradiance_raw = simulate_irradiance(hour_fraction)
        sky_img, cloud_cover_true = generate_sky_image(size=256)

        img_path = f"/app/shared/sky_{measure_id:05d}.jpg"
        cv2.imwrite(img_path, sky_img)

        _, buf  = cv2.imencode(".jpg", sky_img)
        img_b64 = base64.b64encode(buf).decode()

        # ── FIX bug #1 : envoyer le timestamp comme string "YYYY-MM-DD HH:MM:SS.000"
        # ModelKd.format_hour_no_24() attend ce format exact (parse("{}-{}-{} {}:{}:{}")).
        # time.time() retourne un float qui crasherait à la désérialisation.
        now_utc       = datetime.now(timezone.utc)
        timestamp_str = now_utc.strftime("%Y-%m-%d %H:%M:%S.000")

        message = {
            "id":             measure_id,
            "timestamp":      timestamp_str,          # ← string, plus float
            "irradiance_raw": irradiance_raw,
            "cloud_cover_gt": round(cloud_cover_true, 4),
            "image_path":     img_path,
            "image_b64":      img_b64,
        }

        try:
            celery_app.send_task("solar.process_measurement", args=[message], queue=QUEUE_NAME)
            log.info(f"[#{measure_id:05d}] Tâche envoyée → irr={irradiance_raw} W/m² | cloud={cloud_cover_true:.1%} | t={timestamp_str}")
        except OperationalError as e:
            log.error(f"[#{measure_id:05d}] RECONNEXION - Impossible d'envoyer la tâche : {e}")
            wait_for_rabbitmq(celery_app)

        time.sleep(INTERVAL)


if __name__ == "__main__":
    log.info(f"Démarrage du capteur... Envoi via Celery vers la queue '{QUEUE_NAME}'")
    wait_for_rabbitmq(celery_app)
    run()
