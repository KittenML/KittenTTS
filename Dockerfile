FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    VLLM_CACHE_ROOT=/cache/vllm

# Triton compiles small GPU kernels at runtime and needs a C compiler.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/kittenml
COPY pyproject.toml README.md ./
COPY kittenml ./kittenml
RUN pip install --no-cache-dir ".[vllm]"

WORKDIR /workspace
