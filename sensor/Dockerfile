FROM python:3.11-slim

WORKDIR /app

# Dépendances système pour OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installation de uv, poetry et du plugin d'exportation
RUN pip install uv poetry poetry-plugin-export --no-cache-dir
# Copie des fichiers de configuration Poetry depuis la racine
COPY pyproject.toml poetry.lock ./

# Exportation du groupe "sensor" et installation éclair avec uv
RUN poetry export --with sensor --without-hashes -f requirements.txt > requirements.txt \
    && uv pip install --system --index-strategy unsafe-best-match -r requirements.txt

# Copie du code source du capteur
COPY sensor/ ./

CMD ["python", "sensor.py"]