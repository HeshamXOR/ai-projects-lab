# finetune-studio — LoRA fine-tuning + eval + serving for LLMs.
FROM python:3.11-slim

# Avoid .pyc clutter and unbuffered logs for clean container output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY . .

EXPOSE 8000

# Serve the FastAPI app.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
