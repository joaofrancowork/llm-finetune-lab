# syntax=docker/dockerfile:1.7
# Training image for a CUDA GPU (cloud VM, RunPod, Vertex AI custom job).
# docker build -t ft-lab . && docker run --gpus all -v "$PWD/checkpoints:/app/checkpoints" ft-lab train
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_SYSTEM_PYTHON=1
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN uv pip install --system ".[train]"
ENTRYPOINT ["ft-lab"]
CMD ["--help"]
