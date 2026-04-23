# Multi-purpose Dockerfile for Fraud Detection MLOps
# Soporta múltiples servicios: FastAPI (puerto 8000), Streamlit (puerto 8501)
# Uso en docker-compose:
#   args:
#     - SERVICE=api       (FastAPI)
#     - SERVICE=streamlit (Streamlit)

ARG SERVICE=api

FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy project files for dependency installation
COPY pyproject.toml ./

# Copy README for build (required by pyproject.toml)
COPY README.md ./

# Install dependencies using uv (system-wide in container)
RUN uv pip install --system .

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/
COPY models/ ./models/
COPY app.py ./
COPY predict_example.py ./

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Service (default)
# ─────────────────────────────────────────────────────────────────────────────
FROM base as service-api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Service
# ─────────────────────────────────────────────────────────────────────────────
FROM base as service-streamlit

# Create Streamlit config directory
RUN mkdir -p /root/.streamlit

# Add Streamlit config
RUN echo "[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
[logger]\n\
level = \"info\"" > /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/ || exit 1

CMD ["streamlit", "run", "app.py"]

# ─────────────────────────────────────────────────────────────────────────────
# Final stage - use appropriate service based on BUILD_ARG
# ─────────────────────────────────────────────────────────────────────────────
FROM service-${SERVICE}

LABEL maintainer="MLOps Team"
LABEL description="Fraud Detection Model - Multi-service Container"
