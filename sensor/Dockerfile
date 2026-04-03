FROM python:3.11-slim

WORKDIR /app

# OpenCV Headless a juste besoin de ces deux bibliothèques légères
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installation de 'uv'
RUN pip install uv --no-cache-dir

# Copie des dépendances et installation ultra-rapide
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copie du code
COPY sensor.py .
COPY sky_generator.py .

CMD ["python", "sensor.py"]