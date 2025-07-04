# -------- base Python officielle --------
FROM python:3.12-slim

# Empêche la génération de .pyc et active le stdout non-bufferisé
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ------- dépendances système pour opencv & scikit-image ------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ------- dossier de travail -------
WORKDIR /app

# Copie des dépendances & installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code application
COPY . .

# -------- port et commande de démarrage ----------
EXPOSE 5000

#   Flask-SocketIO + eventlet = serveur asynchrone robuste
CMD ["python", "app.py"]
