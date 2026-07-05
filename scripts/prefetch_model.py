#!/usr/bin/env python3
"""Warm the NeMo / HuggingFace / FunASR caches by downloading every backend
checkpoint at build time, so the worker never pays the download on a cold start
(SPEC §8.1). The worker loads each model lazily per language at runtime; baking
the weights here keeps that first load fast.

IMPORTANT: this script runs in the Dockerfile BEFORE `src/` is copied into the
image, so it MUST NOT import anything from `src`. It reads the model ids from the
environment directly (same defaults as ``src/config.py``).

It also prints the resolved torch / CUDA / NeMo versions so the build log
records exactly what the base image shipped (these are MUST-VALIDATE-ON-GPU per
RESEARCH.md §5 — the build runs on CPU, so CUDA availability here is informational).
"""

import os
import sys


def _env(name: str, default: str) -> str:
    """Read an env var, falling back to ``default`` (kept in sync with config.py)."""
    return (os.environ.get(name, "") or "").strip() or default


# Kept in sync manually with src/config.py — this file cannot import src.config
# (it runs before src/ exists in the image).
PARAKEET_MODEL = _env("MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2")
CANARY_MODEL = _env("CANARY_MODEL", "nvidia/canary-1b-v2")
SENSEVOICE_MODEL = _env("SENSEVOICE_MODEL", "FunAudioLLM/SenseVoiceSmall")
WHISPER_MODEL = _env("WHISPER_MODEL", "openai/whisper-large-v3")

# Back-compat: some tooling/docs still reference MODEL_NAME.
MODEL_NAME = PARAKEET_MODEL


def _print_versions() -> None:
    """Best-effort version banner for the build log."""
    try:
        import torch

        cuda_build = getattr(torch.version, "cuda", None)
        print(f"[prefetch] torch={torch.__version__} cuda(build)={cuda_build}")
        try:
            print(f"[prefetch] torch.cuda.is_available()={torch.cuda.is_available()}")
        except Exception as exc:  # pragma: no cover - depends on build host
            print(f"[prefetch] torch.cuda probe failed (expected on CPU build host): {exc}")
    except Exception as exc:  # pragma: no cover
        print(f"[prefetch] could not import torch: {exc}")

    try:
        import nemo

        print(f"[prefetch] nemo={nemo.__version__}")
    except Exception as exc:  # pragma: no cover
        print(f"[prefetch] could not import nemo: {exc}")

    print(f"[prefetch] python={sys.version.split()[0]}")


def _prefetch_nemo(model_name: str) -> None:
    """Download + cache a NeMo checkpoint (Parakeet / Canary). No GPU needed."""
    print(f"[prefetch] NeMo: downloading {model_name}")
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    try:
        _ = model.cfg  # touch so the object is fully materialized
    except Exception:  # pragma: no cover - non-fatal
        pass
    del model
    print(f"[prefetch] NeMo: cached {model_name}")


def _prefetch_whisper(model_name: str) -> None:
    """Download + cache the Whisper checkpoint via transformers. No GPU needed."""
    print(f"[prefetch] Whisper: downloading {model_name}")
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    AutoProcessor.from_pretrained(model_name)
    AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    print(f"[prefetch] Whisper: cached {model_name}")


def _prefetch_sensevoice(model_name: str) -> None:
    """Download + cache SenseVoiceSmall via FunASR (CPU init populates the cache)."""
    print(f"[prefetch] SenseVoice: downloading {model_name}")
    from funasr import AutoModel

    AutoModel(
        model=model_name,
        hub="hf",
        trust_remote_code=True,
        disable_update=True,
        device="cpu",
    )
    print(f"[prefetch] SenseVoice: cached {model_name}")


def main() -> int:
    _print_versions()

    # Bake every backend's weights so no cold-start download is ever needed.
    _prefetch_nemo(PARAKEET_MODEL)
    _prefetch_nemo(CANARY_MODEL)
    _prefetch_whisper(WHISPER_MODEL)
    _prefetch_sensevoice(SENSEVOICE_MODEL)

    print("[prefetch] done — all backend models cached into the image.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
