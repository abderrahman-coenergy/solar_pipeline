"""
consumer.py  (optionnel — pour comprendre le flux complet)
───────────────────────────────────────────────────────────
Consommateur RabbitMQ bas niveau (sans Celery).
Montre exactement ce que fait Celery en interne.

En pratique : Celery gère lui-même la consommation.
Ce fichier sert à COMPRENDRE le mécanisme.

Usage
─────
  python consumer.py   # dans le container worker
"""

import os
import json
import time
import logging
import pika

from image_processor import analyze_sky_image

logging.basicConfig(
    format="%(asctime)s [CONSUMER] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
QUEUE_NAME    = "solar_tasks"


def on_message(channel, method, properties, body):
    """
    Callback appelé à chaque message reçu.

    Parameters (fournis par pika automatiquement)
    ──────────────────────────────────────────────
    channel    → le canal AMQP
    method     → métadonnées de livraison (delivery_tag, routing_key…)
    properties → propriétés du message (content_type, reply_to…)
    body       → payload brut (bytes)
    """
    t_start = time.monotonic()

    try:
        message      = json.loads(body)
        msg_id       = message["id"]
        irr_raw      = message["irradiance_raw"]
        img_b64      = message["image_b64"]

        log.info(f"Reçu message #{msg_id} | irr_raw={irr_raw} W/m²")

        # Traitement image
        analysis = analyze_sky_image(img_b64, irr_raw)

        elapsed = round((time.monotonic() - t_start) * 1000, 1)
        log.info(
            f"#{msg_id} → irr_pred={analysis.irradiance_pred} W/m² | "
            f"cloud={analysis.cloud_cover:.1%} | {elapsed}ms"
        )

        # ACK = confirme que le message a bien été traité
        # Sans ça, RabbitMQ le redelivre si le process crash
        channel.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as exc:
        log.error(f"Erreur traitement: {exc}")
        # NACK = refuse le message → RabbitMQ peut le remettre en queue
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


def run():
    params     = pika.ConnectionParameters(host=RABBITMQ_HOST, heartbeat=600)
    connection = pika.BlockingConnection(params)
    channel    = connection.channel()

    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    # Ne prendre qu'un message à la fois (évite la surcharge)
    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=QUEUE_NAME,
        on_message_callback=on_message,
    )

    log.info(f"En attente de messages sur '{QUEUE_NAME}'… (Ctrl+C pour arrêter)")
    channel.start_consuming()


if __name__ == "__main__":
    run()
