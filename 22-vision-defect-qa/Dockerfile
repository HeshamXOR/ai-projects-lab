# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Keep Python lean and unbuffered for container logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY core ./core
COPY app.py ./
COPY tests ./tests

EXPOSE 8000

# Default command runs the FastAPI service via uvicorn.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
