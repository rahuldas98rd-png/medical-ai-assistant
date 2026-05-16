# MediMind AI — Backend image
# Multi-stage build keeps the runtime image small.
#
# Build:  docker build -t medimind-backend .
# Run:    docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data medimind-backend

# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (requires a special index URL; must be before the
# rest of requirements.txt so pip doesn't pick up a GPU wheel for it)
RUN pip install --no-cache-dir --user \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# Install everything else
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Tesseract OCR (used by prescription_ocr module)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY backend ./backend
COPY scripts ./scripts

# Persistent data (models + SQLite db) is mounted as a volume at runtime
RUN mkdir -p ./data/models ./data/raw ./data/processed

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    DEBUG=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
