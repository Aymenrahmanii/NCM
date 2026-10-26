# NCM Lab API with GPU access (needs the NVIDIA Container Toolkit / Docker Desktop + WSL2).
# Weights and data are mounted at runtime (see docker-compose.yml), not baked into the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface NCM_ROOT=/app/third_party/ncm

RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CUDA build of PyTorch (the wheel ships its own CUDA runtime)
RUN pip install torch --index-url https://download.pytorch.org/whl/cu130

# official NCM code, pinned to the commit this project was tested with
ARG NCM_COMMIT=main
RUN git clone https://github.com/HazyResearch/ncm.git third_party/ncm \
    && git -C third_party/ncm checkout ${NCM_COMMIT}

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

RUN useradd --create-home app && mkdir -p /app/.cache && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "ncm_lab.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
