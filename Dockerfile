# ---------- base ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---------- bibliothèques système pour OpenCV & scikit-image ----------
# ▲ libgl1         → libGL.so.1
# ▲ libglib2.0-0   → dépendance cv2.imshow
# ▲ libsm6 libxext6 libxrender1 → rendu X11 logiciel
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ---------- ton appli ----------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
