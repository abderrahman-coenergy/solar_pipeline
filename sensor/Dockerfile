FROM python:3.11-slim

WORKDIR /app
# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install pika numpy pillow celery opencv-python  --no-cache-dir

COPY sensor.py .
COPY sky_generator.py .

CMD ["python", "sensor.py"]
