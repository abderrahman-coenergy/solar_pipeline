"""
tasks.py
────────
Définition des tâches Celery pour le pipeline solaire.

Architecture
────────────
  RabbitMQ (broker)
       │
       ▼
  Celery Worker  ←── ce fichier
       │
       ├── process_solar_measurement()   tâche principale
       └── analyze_image_only()          tâche utilitaire

Lancement du worker
───────────────────
  celery -A tasks worker --loglevel=info --concurrency=2

Concepts Celery illustrés
─────────────────────────
  @app.task(bind=True)    → accès à self (retry, request.id…)
  self.retry()            → relance automatique en cas d'erreur
  task.apply_async()      → appel asynchrone depuis un autre processus
  task.delay()            → raccourci de apply_async sans kwargs
  AsyncResult             → suivre le résultat d'une tâche
"""

import os
import json
import logging
import time
from celery import Celery
from celery.utils.log import get_task_logger

from image_processor import analyze_sky_image

# ─── Configuration ──────────────────────────────────────────────────────────
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
BROKER_URL    = f"pyamqp://guest:guest@{RABBITMQ_HOST}//"

# Celery utilise aussi RabbitMQ comme backend de résultats
# (en prod on préfère Redis pour les résultats, RabbitMQ pour les tâches)
BACKEND_URL   = f"rpc://"

app = Celery(
    "solar_pipeline",
    broker=BROKER_URL,
    backend=BACKEND_URL,
)

# Configuration Celery
app.conf.update(
    task_serializer          = "json",
    result_serializer        = "json",
    accept_content           = ["json"],
    task_acks_late           = True,    # ACK seulement après succès (fiabilité)
    worker_prefetch_multiplier = 1,     # 1 tâche à la fois par worker (fair dispatch)
    task_track_started       = True,
)

log = get_task_logger(__name__)


# ─── Tâche principale ────────────────────────────────────────────────────────

@app.task(
    bind=True,
    name="solar.process_measurement",
    queue="solar_tasks",
    max_retries=3,
    default_retry_delay=5,
)
def process_solar_measurement(self, message: dict) -> dict:
    """
    Tâche Celery principale.

    Reçoit un message du pyranomètre, traite l'image ciel et retourne
    l'irradiance solaire prédite.

    Parameters
    ----------
    message : dict  Message JSON publié par sensor.py :
        {
          "id":             int,
          "timestamp":      float,
          "irradiance_raw": float,
          "cloud_cover_gt": float,   # vérité terrain (évaluation)
          "image_b64":      str,     # image JPEG encodée en base64
        }

    Returns
    -------
    dict  Résultat complet avec prédiction et métriques.
    """
    task_id  = self.request.id
    msg_id   = message.get("id", "?")
    irr_raw  = message.get("irradiance_raw", 0.0)
    img_b64  = message.get("image_b64", "")

    log.info(f"[Task {task_id}] Démarrage — message #{msg_id} | irr_raw={irr_raw} W/m²")
    t_start = time.monotonic()

    try:
        # ── Traitement image ─────────────────────────────────────────────────
        analysis = analyze_sky_image(img_b64, irr_raw)

        # ── Calcul de l'erreur (si vérité terrain disponible) ────────────────
        cloud_gt  = message.get("cloud_cover_gt")
        cloud_err = None
        if cloud_gt is not None:
            cloud_err = round(abs(analysis.cloud_cover - cloud_gt), 4)

        elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)

        result = {
            "measure_id":        msg_id,
            "task_id":           task_id,
            "irradiance_raw":    irr_raw,
            "irradiance_pred":   analysis.irradiance_pred,
            "cloud_cover":       analysis.cloud_cover,
            "brightness_mean":   analysis.brightness_mean,
            "cloud_factor":      analysis.cloud_factor,
            "cloud_cover_error": cloud_err,
            "processing_ms":     elapsed_ms,
            "timestamp":         message.get("timestamp"),
        }

        log.info(
            f"[Task {task_id}] ✓ #{msg_id} — "
            f"irr: {irr_raw} → {analysis.irradiance_pred} W/m² | "
            f"cloud={analysis.cloud_cover:.1%} | "
            f"temps={elapsed_ms}ms"
        )

        return result

    except Exception as exc:
        log.error(f"[Task {task_id}] Erreur: {exc}")
        # Retry automatique avec backoff exponentiel
        raise self.retry(exc=exc, countdown=5 * (self.request.retries + 1))


# ─── Tâche utilitaire ────────────────────────────────────────────────────────

@app.task(name="solar.analyze_image_only")
def analyze_image_only(image_b64: str, irradiance_raw: float) -> dict:
    """
    Version allégée — traite une seule image sans pipeline complet.
    Utile pour tester le traitement image indépendamment.
    """
    analysis = analyze_sky_image(image_b64, irradiance_raw)
    return {
        "cloud_cover":     analysis.cloud_cover,
        "brightness_mean": analysis.brightness_mean,
        "cloud_factor":    analysis.cloud_factor,
        "irradiance_pred": analysis.irradiance_pred,
    }


# ─── Signal de démarrage ─────────────────────────────────────────────────────

@app.on_after_configure.connect
def on_start(sender, **kwargs):
    log.info(f"Worker connecté → broker: {BROKER_URL}")
    log.info("Tâches enregistrées:")
    log.info("  • solar.process_measurement")
    log.info("  • solar.analyze_image_only")
