"""Configuration constants for the transcription worker.

All values are parsed from environment variables at import time, with the
defaults defined in SPEC §7. Malformed values raise ``ValueError`` immediately
at startup so the worker fails fast rather than producing confusing runtime
errors deep inside a job.

This module is pure stdlib — it does NOT import torch/nemo/httpx/ffmpeg — so it
can be imported in GPU-free unit tests.
"""

from __future__ import annotations

import os


# --------------------------------------------------------------------------- #
# Private env helpers
# --------------------------------------------------------------------------- #
def _get_str(name: str, default: str) -> str:
    """Return the env var ``name`` as a stripped string, or ``default``.

    An empty/whitespace-only value falls back to the default.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else default


def _get_int(name: str, default: int, *, positive: bool = True) -> int:
    """Return the env var ``name`` parsed as an int, or ``default``.

    Raises ``ValueError`` (at import time) if the value is not a valid integer
    or, when ``positive`` is True, is not strictly greater than zero.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = int(raw.strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid value for env var {name!r}: {raw!r} is not an integer."
            )
    if positive and value <= 0:
        raise ValueError(
            f"Invalid value for env var {name!r}: {value} must be > 0."
        )
    return value


def _get_bool(name: str, default: bool) -> bool:
    """Return the env var ``name`` interpreted as a boolean, or ``default``.

    Truthy: 1/true/yes/on (case-insensitive). Falsy: 0/false/no/off/"".
    Anything else raises ``ValueError`` at import time.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized == "":
        return default
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"Invalid value for env var {name!r}: {raw!r} is not a valid boolean."
    )


# --------------------------------------------------------------------------- #
# Model / language
# --------------------------------------------------------------------------- #
MODEL_NAME: str = _get_str("MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2")

DEFAULT_LANGUAGE: str = "en"
SUPPORTED_LANGUAGES: frozenset = frozenset({"en"})

# --------------------------------------------------------------------------- #
# Audio / inference limits (seconds)
# --------------------------------------------------------------------------- #
MAX_AUDIO_SECONDS: int = _get_int("MAX_AUDIO_SECONDS", 36000)
SINGLE_PASS_MAX_SEC: int = _get_int("SINGLE_PASS_MAX_SEC", 1440)
CHUNK_SEC: int = _get_int("CHUNK_SEC", 1200)
CHUNK_OVERLAP_SEC: int = _get_int("CHUNK_OVERLAP_SEC", 15)

# --------------------------------------------------------------------------- #
# Download limits
# --------------------------------------------------------------------------- #
DOWNLOAD_TIMEOUT_SEC: int = _get_int("DOWNLOAD_TIMEOUT_SEC", 120)
MAX_DOWNLOAD_BYTES: int = _get_int("MAX_DOWNLOAD_BYTES", 2147483648)

# --------------------------------------------------------------------------- #
# Result offload (RunPod payload limits)
# --------------------------------------------------------------------------- #
RESULT_OFFLOAD_THRESHOLD_BYTES: int = _get_int(
    "RESULT_OFFLOAD_THRESHOLD_BYTES", 8000000
)
RESULT_UPLOAD_TIMEOUT_SEC: int = _get_int("RESULT_UPLOAD_TIMEOUT_SEC", 120)

# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
WORKER_VERSION: str = "0.1.0"

# TEST-ONLY escape hatch. When truthy, transcribe.load_model() returns
# immediately without importing NeMo/torch, so handler.py and transcribe.py can
# be imported on a machine without a GPU (e.g. CI). NEVER set this in
# production — the worker would have no model loaded.
SKIP_MODEL_LOAD: bool = _get_bool("SKIP_MODEL_LOAD", False)


# --------------------------------------------------------------------------- #
# Cross-field sanity checks (pragmatic)
# --------------------------------------------------------------------------- #
# A single-pass window longer than the hard max audio length is nonsensical.
if SINGLE_PASS_MAX_SEC > MAX_AUDIO_SECONDS:
    raise ValueError(
        f"SINGLE_PASS_MAX_SEC ({SINGLE_PASS_MAX_SEC}) must be <= "
        f"MAX_AUDIO_SECONDS ({MAX_AUDIO_SECONDS})."
    )

# Overlap must be strictly smaller than the chunk length, otherwise chunked
# planning cannot make forward progress.
if CHUNK_OVERLAP_SEC >= CHUNK_SEC:
    raise ValueError(
        f"CHUNK_OVERLAP_SEC ({CHUNK_OVERLAP_SEC}) must be < "
        f"CHUNK_SEC ({CHUNK_SEC})."
    )


class ErrorCode:
    """Machine-readable error codes (SPEC §3.3).

    Each constant's value equals its own name, so callers can compare against
    either the attribute or the raw string interchangeably.
    """

    MISSING_AUDIO_URL = "MISSING_AUDIO_URL"
    INVALID_URL = "INVALID_URL"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    AUDIO_TOO_LONG = "AUDIO_TOO_LONG"
    FFMPEG_FAILED = "FFMPEG_FAILED"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    RESULT_TOO_LARGE = "RESULT_TOO_LARGE"
    RESULT_UPLOAD_FAILED = "RESULT_UPLOAD_FAILED"
