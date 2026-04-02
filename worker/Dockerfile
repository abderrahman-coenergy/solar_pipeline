FROM python:3.11-slim

WORKDIR /app

# OpenCV a besoin de libGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install celery pika opencv-python-headless numpy pillow --no-cache-dir

COPY tasks.py .
COPY image_processor.py .
