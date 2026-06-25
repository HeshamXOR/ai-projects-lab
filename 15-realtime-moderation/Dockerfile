FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY . .

EXPOSE 8000

# The from-scratch core is stdlib-only; only the API layer needs the deps above.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
