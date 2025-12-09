# Use a slim python image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files and verify stdout is unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for building some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (default index supports both AMD64 and ARM64)
RUN pip install --no-cache-dir torch

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p artifacts

EXPOSE 8000

# Run with Gunicorn using Uvicorn workers
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]
