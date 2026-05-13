# Code Review Agent - Dockerfile
#
# Build from the axiompy-agents repository root (this image installs axiompy from
# your configured package index, then installs this distribution).
#
# Build:
#   docker build -f axiompy/agents/code_review/docker/Dockerfile .

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY axiompy ./axiompy

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir "fastapi>=0.100.0" "uvicorn>=0.23.0"

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=http://ollama:11434 \
    OLLAMA_MODEL=codellama \
    RULES_REPO=varonusmaximus/axiompy \
    RULES_FILE=AGENTS.md \
    LOG_LEVEL=INFO

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "axiompy.agents.code_review.applications.webhook"]
