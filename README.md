# Pipeline Prédiction Irradiance Solaire : IA & Modèle de Perez

Ce projet est une infrastructure asynchrone complète (End-to-End) permettant de simuler, d'acquérir et de prédire le flux solaire en milieu urbain. Il combine **une architecture distribuée (RabbitMQ/Celery)**, de la **Vision par Intelligence Artificielle (MiDaS)** et de la **physique atmosphérique avancée (Modèle Anisotrope de Perez via `coe_sol`)**.

## 🏗 Architecture de la Pipeline

```text
┌──────────────────┐    celery.send_task()   ┌─────────────┐      Consomme      ┌────────────────────────────┐
│    sensor.py     │ ───────────────────────▶│  RabbitMQ   │ ──────────────────▶│       Celery Worker        │
│  (Pyranomètre)   │                         │  (broker)   │                    │ (Intelligence logicielle)  │
│                  │                         │ queue:      │                    │                            │
│ • Génère image   │   Volume Partagé (I/O)  │ solar_tasks │                    │ • 1. IA MiDaS (Horizon)    │
│ • Simule Scan 360│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │             │                    │ • 2. Fitting Perez (kd)    │
│ • JSON structuré │                         │             │                    │ • 3. Projection sur façade │
└──────────────────┘                         └─────────────┘                    └────────────────────────────┘
```

---

## 🚀 Lancement rapide

```bash
# 1. Cloner le dépôt et se placer dans le dossier
cd solar_pipeline

# 2. Créer les dossiers de volumes vitaux (Images et Cache IA)
mkdir -p shared models torch_cache

# 3. Build ultra-rapide (propulsé par `uv`) + lancement du stack
docker compose up -d --build

# 4. Suivre les calculs du Worker en temps réel
docker compose logs -f worker

# 5. Dashboards de Monitoring :
# → RabbitMQ (Gestion des files) : http://localhost:15672 (guest / guest)
# → Flower (Monitoring des Workers): http://localhost:5555
```

---

## 📂 Structure du Projet

```text
solar_pipeline/
├── docker-compose.yml        # Orchestration (RabbitMQ, Worker, Sensor, Flower)
│
├── sensor/
│   ├── Dockerfile            # Optimisé avec `uv`
│   ├── sensor.py             # Producteur natif Celery (Simule un rotor Pan-Tilt 4 mesures)
│   └── sky_generator.py      # Générateur fisheye avec horizon urbain (faux bâtiments)
│
├── worker/
│   ├── Dockerfile            # Optimisé avec `uv` (Installe PyTorch CPU)
│   ├── tasks.py              # Orchestrateur de l'IA et du Modèle Physique (@app.task)
│   └── coe_sol/              # Package métier (Vision & Physique)
│       ├── horizon.py        # Extraction de la ligne d'horizon sur 360°
│       ├── masking.py        # Segmentation du ciel par IA (Microsoft MiDaS)
│       ├── SolarModel.py     # API haut-niveau de modélisation solaire
│       └── private/ModelKd.py# Cœur mathématique : Modèle de Perez 1990
│
├── shared/                   # Volume partagé : Évite l'encodage B64 des images
└── torch_cache/              # Volume persistant : Évite de retélécharger l'IA à chaque build
```

---

## 🧠 Concepts techniques illustrés

### 1. Producteur Natif Celery (`sensor.py`)
Au lieu d'utiliser un client AMQP bas niveau comme `pika`, le capteur est un **client Celery**. Il encapsule un "scan" complet (1 mesure zénithale + 3 mesures inclinées) et l'envoie de manière transparente et résiliente :
```python
celery_app.send_task("solar.process_measurement", args=[message], queue="solar_tasks")
```

### 2. Worker Asynchrone avec Tolérance aux Pannes (`tasks.py`)
L'inférence IA (MiDaS) et l'optimisation mathématique (Grid Search du $k_d$) sont gourmandes en CPU. Elles sont isolées dans un Worker pour ne pas bloquer l'acquisition.
* **`task_acks_late=True`** : Sécurité maximale. Si le Worker plante pendant le calcul IA, la tâche est remise dans la file.
* **`max_retries=3`** : Le Worker tente 3 fois de résoudre l'équation atmosphérique avant d'abandonner.

### 3. Modélisation Physique et IA Visuelle (`coe_sol`)

1. **Vision :** `masking.py` utilise l'IA **MiDaS** pour estimer la profondeur de la scène et détourer les bâtiments afin de construire le tableau `horizon_profile`.
2. **Reverse Engineering :** `ModelKd.py` teste 100 valeurs de fraction diffuse ($k_d$) pour trouver celle qui correspond aux vraies mesures du capteur.
3. **Projection :** Le modèle anisotrope de Perez décompose la lumière en Direct (BTI), Diffus (DTI) et Réfléchi (RTI) pour projeter l'énergie exacte sur une façade cible virtuelle.

---

## 🛠 Commandes utiles

```bash
# Forcer la recréation complète (utile si modification du requirements.txt)
docker compose up -d --build --force-recreate

# Entrer dans le conteneur Worker pour déboguer l'IA
docker compose exec worker bash
python -c "import coe_sol.SolarModel; print('Package chargé avec succès')"

# Vérifier les files d'attente RabbitMQ en CLI
docker compose exec rabbitmq rabbitmqctl list_queues

# Arrêter le système et nettoyer les volumes (Attention: purge RabbitMQ)
docker compose down -v
```

---

## 🔄 Flux de Données Final (JSON)

**1. `sensor.py` génère un scan 360° simulé :**
```json
{
  "id": 42,
  "timestamp": "2026-04-06 10:15:30.000",
  "image_path": "/app/shared/sky_00042.jpg",
  "origin": { "irradiance": 850.0 },
  "fits":[
    {"azimuth": 180.0, "inclination": 45.0, "irradiance": 910.0},
    {"azimuth": 90.0, "inclination": 60.0, "irradiance": 600.0}
  ]
}
```

**2. Le Worker Celery (`tasks.py`) exécute la pipeline `coe_sol` et retourne :**
```json
{
  "measure_id": 42,
  "input_ghi": 850.0,
  "timestamp_utc": "2026-04-06 10:15:30.000",
  "results":[
    {
      "time": "2026-04-06 10:15:30.000",
      "pyrano-dest-1_value": 720.5    // ← L'énergie projetée exacte sur la façade (W/m²)
    }
  ],
  "processing_time_ms": 612.4,
  "status": "SUCCESS"
}
```
