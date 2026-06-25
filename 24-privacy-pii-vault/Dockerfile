# privacy-pii-vault: python:3.11-slim image.
# The core package depends only on the Python standard library
# (hashlib/hmac/re); FastAPI + uvicorn are only needed for the HTTP service.
FROM python:3.11-slim

# Avoid .pyc clutter and force unbuffered logs for container-friendly output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY core ./core
COPY app.py .

# The vault passphrase and detokenize API key MUST be injected at runtime via
# a secrets manager in production; the defaults are insecure dev values.
ENV VAULT_PASSPHRASE=change-me-in-prod \
    VAULT_API_KEY=change-me-in-prod

EXPOSE 8000

# Run the service. Use a single worker so the in-memory vault is shared; for
# multiple workers you must externalize the vault to a shared store.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
