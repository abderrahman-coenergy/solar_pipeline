import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from celery import Celery
from celery.utils.log import get_task_logger

# ─── IMPORTATION DU PACKAGE INTERNE (coe_sol) ──────────────────────────────
import coe_sol.SolarModel as sm
import coe_sol.horizon

# ─── Configuration Celery ──────────────────────────────────────────────────
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
app = Celery("solar_pipeline", broker=f"pyamqp://guest:guest@{RABBITMQ_HOST}:5672//")
log = get_task_logger(__name__)


def _ensure_timestamp_str(ts) -> str:
    """
    Garantit que le timestamp est une string "YYYY-MM-DD HH:MM:SS.000"
    attendu par ModelKd.format_hour_no_24().
    - sensor.py corrigé envoie déjà une string  → retourne tel quel
    - ancien message avec float Unix             → convertit en string
    """
    if isinstance(ts, str):
        return ts
    dt = datetime.utcfromtimestamp(float(ts))
    return dt.strftime("%Y-%m-%d %H:%M:%S.000")


@app.task(bind=True, name="solar.process_measurement", queue="solar_tasks", max_retries=3)
def process_solar_measurement(self, message: dict) -> dict:
    task_id   = self.request.id
    msg_id    = message.get("id", "?")
    irr_raw   = message.get("irradiance_raw", 0.0)
    img_path  = message.get("image_path")
    timestamp = message.get("timestamp", time.time())

    # Normalise le timestamp : string si sensor corrigé, conversion si float legacy
    timestamp_str = _ensure_timestamp_str(timestamp)

    log.info(f"[Task {task_id}] Démarrage du pipeline scientifique pour #{msg_id}")
    t_start = time.monotonic()

    try:
        # =====================================================================
        # ÉTAPE 1 : VISION (horizon.py + masking.py via MiDaS)
        # =====================================================================
        log.info("Étape 1 : Analyse d'image et calcul de l'horizon...")
        horizon_profile = coe_sol.horizon.compute_horizon_from_image(
            image_path=img_path,
            fov_deg=180,
            single_half=coe_sol.horizon.SINGLE_HALF_RIGHT,
            azimuth_deg=0.0,
            inclination_deg=90.0    # 90° = caméra pointée vers le zénith
        )

        # =====================================================================
        # ÉTAPE 2 : MODÉLISATION PHYSIQUE (SolarModel.py + ModelKd.py)
        # =====================================================================
        log.info("Étape 2 : Configuration du modèle physique...")
        options = sm.SolarModelOptions(latitude=48.85, longitude=2.35, elevation_meter=35, use_riso=True)
        model = sm.SolarModel(options=options)

        # ── A. Le Pyranomètre d'Origine (Zénith) ──────────
        irr_origin = message["origin"]["irradiance"]
        
        info_origin = sm.PyranoInfo(azimuth_deg=0.0, inclination_deg=0.0, horizon=horizon_profile.tolist())
        meas_origin = sm.PyranoMeasure(timestamps=np.array([timestamp_str]), values=np.array([irr_origin]))
        
        model.set_origin(sm.RealPyrano(info=info_origin, measures=meas_origin))

        # ── B. Les Pyranomètres de Fit (Le balayage) ──────
        fits_data = message.get("fits",[])
        
        for fit in fits_data:
            fit_info = sm.PyranoInfo(
                azimuth_deg=fit["azimuth"],
                inclination_deg=fit["inclination"],
                horizon=horizon_profile.tolist() # On utilise le même horizon pour tout le toit
            )
            fit_meas = sm.PyranoMeasure(
                timestamps=np.array([timestamp_str]),
                values=np.array([fit["irradiance"]])
            )
            model.add_fit(sm.RealPyrano(info=fit_info, measures=fit_meas))
            log.info(f"  + Ajout Fit: {fit['irradiance']:.1f}W/m² (Az:{fit['azimuth']}°, Inc:{fit['inclination']}°)")

        # ── C. La Façade Cible Virtuelle (Mur Sud) ────────
        target_info = sm.PyranoInfo(azimuth_deg=180.0, inclination_deg=90.0, horizon=np.zeros(360).tolist())
        model.add_target(sm.VirtualPyrano(target_info))

        # =====================================================================
        # ÉTAPE 3 : CALCULS ET PROJECTION
        # =====================================================================
        log.info("Étape 3 : Fitting du paramètre kd et Projection...")
        model.fit_parameters()
        df_result = model.project()

        projection_data = df_result.to_dict(orient="records")
        elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
        log.info(f"✓ Task {task_id} terminée en {elapsed_ms}ms")

        return {
            "measure_id":         msg_id,
            "input_ghi":          irr_raw,
            "timestamp_utc":      timestamp_str,
            "results":            projection_data,
            "processing_time_ms": elapsed_ms,
            "status":             "SUCCESS"
        }

    except Exception as exc:
        log.error(f"ERREUR CRITIQUE sur Task {task_id} : {exc}")
        raise self.retry(exc=exc, countdown=15)
