FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    VLLM_CACHE_ROOT=/cache/vllm \
    CUDA_CACHE_PATH=/cache/nv \
    KITTENML_VOICES_DIR=/cache/voices

# Triton compiles small GPU kernels at runtime and needs a C compiler.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/kittenml
COPY pyproject.toml README.md ./
COPY kittenml ./kittenml
RUN pip install --no-cache-dir ".[vllm]"

WORKDIR /workspace
CMD ["python", "-m", "kittenml.kittentts2.docker.server", "--host", "0.0.0.0"]
