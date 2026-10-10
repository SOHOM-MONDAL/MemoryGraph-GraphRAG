FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install build dependencies and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements with lightweight CPU-only PyTorch
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app/

# Ensure uploads directory exists
RUN mkdir -p /app/data/uploads

EXPOSE 10000

# Default command starts FastAPI API server with dynamic port support (defaults to Render's 10000)
CMD ["sh", "-c", "uvicorn services.api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
