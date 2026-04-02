# Pipeline Prédiction Flux Solaire
## Architecture complète : Pyranomètre → RabbitMQ → Celery → OpenCV

```
┌──────────────┐     basic_publish()     ┌─────────────┐    basic_consume()    ┌──────────────────┐
│  sensor.py   │ ───────────────────────▶│  RabbitMQ   │ ────────────────────▶ │  Celery Worker   │
│              │                         │  (broker)   │                        │                  │
│ • Mesure brut│                         │ queue:      │                        │ • analyze_sky()  │
│ • Génère img │                         │ solar_tasks │                        │ • OpenCV clouds  │
│ • JSON msg   │                         │             │                        │ • Corrige irr.   │
└──────────────┘                         └─────────────┘                        └──────────────────┘
```

---

## Lancement rapide

```bash
# 1. Cloner / se placer dans le dossier
cd solar_pipeline

# 2. Créer le dossier partagé (images ciel)
mkdir -p shared

# 3. Build + lancement de tout le stack
docker compose up --build

# 4. Ouvrir un autre terminal pour voir les logs du worker
docker compose logs -f worker

# 5. Dashboard RabbitMQ (optionnel mais très utile)
# → http://localhost:15672  (guest / guest)
```

---

## Structure du projet

```
solar_pipeline/
├── docker-compose.yml        # Orchestration des 3 services
│
├── sensor/
│   ├── Dockerfile
│   ├── sensor.py             # Producteur RabbitMQ (pyranomètre simulé)
│   └── sky_generator.py      # Générateur d'images ciel synthétiques
│
├── worker/
│   ├── Dockerfile
│   ├── tasks.py              # Tâches Celery (@app.task)
│   ├── image_processor.py    # Pipeline OpenCV (détection nuages)
│   └── consumer.py           # Consommateur bas-niveau (pédagogique)
│
└── shared/                   # Volume partagé sensor ↔ worker (images .jpg)
```

---

## Ce que chaque fichier t'apprend

### sensor.py → `pika` / producteur AMQP

```python
# Les 4 lignes essentielles de tout producteur RabbitMQ :
connection = pika.BlockingConnection(pika.ConnectionParameters(host))
channel    = connection.channel()
channel.queue_declare(queue="solar_tasks", durable=True)
channel.basic_publish(exchange="", routing_key="solar_tasks", body=json_msg)
```

**`durable=True`** → la queue survit à un redémarrage de RabbitMQ  
**`delivery_mode=2`** → le message est persisté sur disque

---

### tasks.py → Celery

```python
app = Celery("solar_pipeline", broker="pyamqp://guest@rabbitmq//")

@app.task(bind=True, max_retries=3)
def process_solar_measurement(self, message: dict) -> dict:
    try:
        result = analyze_sky_image(...)
        return result
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)  # retry automatique !
```

**`bind=True`** → `self` donne accès à `self.retry()`, `self.request.id`  
**`max_retries=3`** → 3 tentatives avant d'abandonner  
**`task_acks_late=True`** → ACK envoyé APRÈS succès (fiabilité maximale)

---

### image_processor.py → OpenCV

```python
# Pipeline de traitement :
img_bgr  = cv2.imdecode(buf, cv2.IMREAD_COLOR)        # décoder bytes → array
img_blur = cv2.GaussianBlur(img_bgr, (5,5), 0)        # réduire bruit
img_hsv  = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)  # BGR → HSV

# Détecter nuages (pixels blancs = faible saturation, haute valeur)
mask     = cv2.inRange(img_hsv, [0,0,180], [180,40,255])
cloud_cover = np.sum(mask > 0) / mask.size

# Correction irradiance
irr_pred = irr_raw * (1 - cloud_cover * 0.70)
```

---

## Commandes utiles

```bash
# Voir les logs en temps réel
docker compose logs -f sensor
docker compose logs -f worker

# Entrer dans un container pour débugger
docker compose exec worker bash
docker compose exec worker python -c "from tasks import process_solar_measurement; print('OK')"

# Voir les queues RabbitMQ en CLI
docker compose exec rabbitmq rabbitmqctl list_queues

# Lancer le consommateur bas-niveau (pédagogique)
docker compose exec worker python consumer.py

# Envoyer une tâche Celery manuellement
docker compose exec worker python -c "
from tasks import analyze_image_only
result = analyze_image_only.delay('', 800.0)
print(result.get(timeout=10))
"

# Arrêter proprement
docker compose down

# Tout effacer (volumes compris)
docker compose down -v
```

---

## Aller plus loin

### Étape suivante — Modèle ML réel

Remplace la formule dans `image_processor.py` par un vrai modèle :

```python
import torch
model = torch.load("cloud_net.pt")  # ton modèle entraîné

def predict_irradiance(img_tensor, irr_raw):
    cloud_cover = model(img_tensor).item()
    return irr_raw * (1 - cloud_cover * 0.70)
```

### Ajouter Redis comme backend de résultats

```yaml
# docker-compose.yml
redis:
  image: redis:7-alpine
  ports: ["6379:6379"]
```

```python
# tasks.py
app = Celery(
    broker="pyamqp://guest@rabbitmq//",
    backend="redis://redis:6379/0",   # stockage des résultats
)
```

### Monitorer avec Flower (dashboard Celery)

```yaml
# docker-compose.yml
flower:
  image: mher/flower
  command: celery --broker=pyamqp://guest@rabbitmq// flower
  ports: ["5555:5555"]
```
→ http://localhost:5555

---

## Flux de données complet

```
1. sensor.py génère :
   {
     "id": 42,
     "irradiance_raw": 783.5,
     "cloud_cover_gt": 0.23,
     "image_b64": "/9j/4AAQSkZJRg..."
   }

2. RabbitMQ reçoit le JSON dans la queue "solar_tasks"

3. Celery worker appelle process_solar_measurement(message)

4. image_processor.analyze_sky_image() retourne :
   ImageAnalysis(
     cloud_cover=0.21,
     brightness_mean=198.4,
     cloud_factor=0.853,
     irradiance_pred=668.5
   )

5. Résultat final :
   {
     "irradiance_raw": 783.5,
     "irradiance_pred": 668.5,   ← prédiction corrigée
     "cloud_cover": 0.21,
     "processing_ms": 47.3
   }
```
