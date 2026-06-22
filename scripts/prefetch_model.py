#!/usr/bin/env python3
"""Warm the NeMo / HuggingFace cache by downloading the ASR checkpoint at build
time, so the worker never pays the ~2.4 GB download on a cold start (SPEC §8.1).

IMPORTANT: this script runs in the Dockerfile BEFORE `src/` is copied into the
image, so it MUST NOT import anything from `src`. It reads ``MODEL_NAME`` from
the environment directly (same default as ``src/config.py``) and uses only NeMo.

It also prints the resolved torch / CUDA / NeMo versions so the build log
records exactly what the base image shipped (these are MUST-VALIDATE-ON-GPU per
RESEARCH.md §5 — the build runs on CPU, so CUDA availability here is informational).
"""

import os
import sys

# Same default as src/config.py MODEL_NAME. Kept in sync manually because this
# file cannot import src.config (it runs before src/ exists in the image).
MODEL_NAME = os.environ.get("MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2").strip() \
    or "nvidia/parakeet-tdt-0.6b-v2"


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


def main() -> int:
    _print_versions()

    print(f"[prefetch] downloading model into cache: {MODEL_NAME}")
    # Import NeMo lazily so the version banner still prints even if the ASR
    # import is slow/heavy.
    import nemo.collections.asr as nemo_asr

    # from_pretrained downloads + caches the checkpoint (HF/NeMo cache dirs).
    # We do NOT need a GPU to populate the cache, so this works on a CPU builder.
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)

    # Touch a cheap attribute so the object is fully materialized, then drop it.
    try:
        _ = model.cfg
    except Exception:  # pragma: no cover - non-fatal
        pass
    del model

    print(f"[prefetch] done — {MODEL_NAME} cached into the image.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
