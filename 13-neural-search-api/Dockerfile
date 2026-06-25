# Neural Search API -- production image.
# Pure-Python/NumPy core; no model download required at build or run time.
FROM python:3.12-slim

# Keep Python lean and unbuffered for clean container logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY core/ ./core/
COPY static/ ./static/
COPY app.py .

# Where the index is persisted (override with NEURAL_SEARCH_INDEX_DIR).
ENV NEURAL_SEARCH_INDEX_DIR=/data/index
RUN mkdir -p /data/index
VOLUME ["/data/index"]

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
