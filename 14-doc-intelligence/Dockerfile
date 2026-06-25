# python:3.12-slim image for the doc-intelligence service.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY core ./core
COPY app.py .
COPY samples ./samples

EXPOSE 8000

# The extraction core is pure stdlib; only the web layer needs the deps above.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
