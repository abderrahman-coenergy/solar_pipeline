FROM python:3.11-slim

WORKDIR /app

# 1. Dépendances système pour OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Installation de 'uv'
RUN pip install uv --no-cache-dir

# 3. Copie UNIQUEMENT du fichier des dépendances en premier
COPY requirements.txt .

# 4. Installation ultra-rapide avec uv
RUN uv pip install --system -r requirements.txt


# 6. Copie de TOUT le dossier worker (tasks.py + coe_sol/)
COPY . .
