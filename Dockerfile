# MediMind AI — Backend image
# Multi-stage build keeps the runtime image small.

# -------- Build stage --------
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for compiling some Python packages (numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# -------- Runtime stage --------
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY backend ./backend
COPY scripts ./scripts
COPY data ./data

# Make sure models directory exists at runtime
RUN mkdir -p ./data/models ./data/raw ./data/processed

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    DEBUG=false

EXPOSE 8000

# Healthcheck so orchestrators know when the container is ready
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
