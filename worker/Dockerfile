FROM python:3.11-slim

WORKDIR /app

# 1. Dépendances système (Identique à ton ancienne version)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Installation de uv et poetry
RUN pip install uv poetry poetry-plugin-export --no-cache-dir

# 3. Copie des fichiers de configuration Poetry
COPY pyproject.toml poetry.lock ./

# 4. Installation des dépendances standard avec uv
# On ajoute --index-strategy pour éviter le conflit fsspec/torch que tu as eu tout à l'heure
RUN poetry export --with worker --without-hashes -f requirements.txt > requirements.txt \
    && uv pip install --system --index-strategy unsafe-best-match -r requirements.txt

# 5. Installation de sg2 (On utilise pip ici pour la compatibilité avec le serveur du labo)
RUN pip install sg2 -f https://pip.oie-lab.net/python/ --no-cache-dir

# 6. "CUISSON" DU MODÈLE IA (Pour que l'image soit autonome)
RUN python -c "import torch; \
    torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True); \
    torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)"

# 7. Copie du code source (Dernière étape pour profiter du cache Docker)
COPY worker/ ./