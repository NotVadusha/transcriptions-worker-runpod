# syntax=docker/dockerfile:1
# =============================================================================
# RunPod Serverless Transcription Worker — Parakeet TDT 0.6B v2 via NeMo
# (SPEC §8.1, RESEARCH.md §1)
# =============================================================================
#
# Base image: the official NVIDIA NeMo NGC container. It ships a pre-validated,
# mutually-compatible PyTorch + CUDA + NeMo triple with Hopper/Blackwell
# support — the lowest-risk way to satisfy "runs on the latest GPUs"
# (Blackwell B200 sm_100 / RTX 5090 sm_120 need CUDA 12.8+ and torch >= 2.7).
#
# PRIMARY PIN (used here): nvcr.io/nvidia/nemo:25.09.02
#   Resolved versions baked into this image (RESEARCH.md §1):
#     NeMo    2.5.0
#     PyTorch 2.8.0a0
#     CUDA    12.9.1
#     Python  3.12 (inferred — MUST-VALIDATE-ON-GPU, RESEARCH.md §5.1)
#   Chosen over 25.11.01 because the NeMo changelog says "NeMo 2 will be
#   deprecated starting 25.11"; 25.09.02 stays one release back from that
#   deprecation edge while still satisfying the Blackwell/CUDA 12.8+ requirement.
#   ALL of the above are MUST-VALIDATE-ON-GPU — confirm inside the pulled image:
#     python3 --version
#     python3 -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_capability())"
#     python3 -c "import nemo; print(nemo.__version__)"
#     ffmpeg -version ; ldconfig -p | grep sndfile
#
# ALTERNATE PIN (newest stable; accept NeMo-2 deprecation-boundary risk):
#     FROM nvcr.io/nvidia/nemo:25.11.01   # NeMo 2.6.0 / torch 2.9.0a0 / CUDA 13.0.1
#   (Do NOT use the 26.02 tag yet — its resolved torch/CUDA/NeMo/Python versions
#    are not in NVIDIA's software-component table as of this research.)
#
# CUDA-BASE FALLBACK (if the ~20GB+ NeMo image is too large / slow to cold-start;
# you then own NeMo/torch compatibility — a real risk on Blackwell):
#     FROM nvcr.io/nvidia/pytorch:25.09-py3        # torch 2.9 / CUDA 13.0 / py3.12
#     RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
#         && rm -rf /var/lib/apt/lists/*
#     RUN pip install --no-cache-dir -U "nemo_toolkit[asr]" runpod httpx
#
# Registry auth: nvcr.io requires `docker login nvcr.io` (user `$oauthtoken`,
# password = your NGC API key) even though the image is publicly pullable. On
# RunPod, set these as the template's container-registry credentials.
# =============================================================================
FROM nvcr.io/nvidia/nemo:25.09.02

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ffmpeg (audio normalize/resample) and libsndfile1 (soundfile/librosa backend)
# are NOT documented as preinstalled in the NeMo image (RESEARCH.md §1, §5.2).
# Install both explicitly so audio decode/resample never silently fails.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Worker-only deps (runpod + httpx). torch/nemo come from the base image.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Bake the model weights into the image so cold starts don't re-download the
# ~2.4 GB checkpoint on every worker boot (SPEC §8.1). This runs BEFORE src/ is
# copied, so prefetch_model.py must NOT import from src — it reads MODEL_NAME
# straight from the environment. Override the model at build time with:
#   docker build --build-arg MODEL_NAME=... .
ARG MODEL_NAME=nvidia/parakeet-tdt-0.6b-v2
ENV MODEL_NAME=${MODEL_NAME}
COPY scripts/prefetch_model.py .
RUN python3 prefetch_model.py

# Application code. Keep the SPEC §8.1 layout: src/ stays a package directory
# under /app (NOT flattened) so the handler's `from src import ...` resolves.
COPY src/ ./src/
COPY handler.py .
COPY tests/test_input.json .

# RunPod entrypoint. The repository-root wrapper exposes the conventional
# `runpod.serverless.start(...)` marker while delegating to src.handler.
CMD ["python3", "-u", "handler.py"]
