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
# Per-backend model checkpoints. Language routing (below) picks one per request.
PARAKEET_MODEL: str = _get_str("MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2")
CANARY_MODEL: str = _get_str("CANARY_MODEL", "nvidia/canary-1b-v2")
SENSEVOICE_MODEL: str = _get_str("SENSEVOICE_MODEL", "FunAudioLLM/SenseVoiceSmall")
WHISPER_MODEL: str = _get_str("WHISPER_MODEL", "openai/whisper-large-v3")

# Back-compat alias: the default (English) model. prefetch + meta fallback read it.
MODEL_NAME: str = PARAKEET_MODEL
TRANSCRIPTION_QUALITY: str = _get_str("TRANSCRIPTION_QUALITY", "balanced").lower()
if TRANSCRIPTION_QUALITY not in ("fast", "balanced", "best"):
    raise ValueError(
        "TRANSCRIPTION_QUALITY must be one of: 'fast', 'balanced', 'best'."
    )

_QUALITY_DEFAULTS = {
    "fast": {
        "chunk_sec": 1200,
        "chunk_overlap_sec": 15,
        "gap_retry_enabled": False,
    },
    "balanced": {
        "chunk_sec": 300,
        "chunk_overlap_sec": 20,
        "gap_retry_enabled": True,
    },
    "best": {
        "chunk_sec": 180,
        "chunk_overlap_sec": 30,
        "gap_retry_enabled": True,
    },
}[TRANSCRIPTION_QUALITY]

DEFAULT_LANGUAGE: str = "en"

# Language -> backend routing.
#   Parakeet  : English only.
#   SenseVoice: the Asian languages it officially covers.
#   Canary    : the listed European languages.
#   Whisper   : catch-all for every other language.
PARAKEET_LANGUAGES: frozenset = frozenset({"en"})
SENSEVOICE_LANGUAGES: frozenset = frozenset({"zh", "yue", "ja", "ko"})
CANARY_LANGUAGES: frozenset = frozenset(
    {"de", "fr", "es", "it", "pl", "ro", "da", "sv", "nl", "pt"}
)

MODEL_FOR_BACKEND: dict = {
    "parakeet": PARAKEET_MODEL,
    "canary": CANARY_MODEL,
    "sensevoice": SENSEVOICE_MODEL,
    "whisper": WHISPER_MODEL,
}


def route_backend(language: str) -> str:
    """Return the backend name serving ``language`` (Whisper is the catch-all)."""
    lang = (language or "").strip().lower()
    if lang in PARAKEET_LANGUAGES:
        return "parakeet"
    if lang in SENSEVOICE_LANGUAGES:
        return "sensevoice"
    if lang in CANARY_LANGUAGES:
        return "canary"
    return "whisper"

# --------------------------------------------------------------------------- #
# Audio / inference limits (seconds)
# --------------------------------------------------------------------------- #
MAX_AUDIO_SECONDS: int = _get_int("MAX_AUDIO_SECONDS", 36000)
SINGLE_PASS_MAX_SEC: int = _get_int("SINGLE_PASS_MAX_SEC", 1440)
CHUNK_SEC: int = _get_int("CHUNK_SEC", _QUALITY_DEFAULTS["chunk_sec"])
CHUNK_OVERLAP_SEC: int = _get_int(
    "CHUNK_OVERLAP_SEC", _QUALITY_DEFAULTS["chunk_overlap_sec"]
)
GAP_RETRY_MIN_SEC: int = _get_int("GAP_RETRY_MIN_SEC", 20)
GAP_RETRY_PADDING_SEC: int = _get_int("GAP_RETRY_PADDING_SEC", 5)
GAP_RETRY_MAX_SEC: int = _get_int("GAP_RETRY_MAX_SEC", 300)
GAP_RETRY_ENABLED: bool = _get_bool(
    "GAP_RETRY_ENABLED", _QUALITY_DEFAULTS["gap_retry_enabled"]
)

# --------------------------------------------------------------------------- #
# Input streaming (ffmpeg/ffprobe read the presigned URL directly)
# --------------------------------------------------------------------------- #
# Bounds a stalled network read when streaming the input URL — passed to
# ffmpeg/ffprobe as -rw_timeout (converted to microseconds). Not a total deadline.
DOWNLOAD_TIMEOUT_SEC: int = _get_int("DOWNLOAD_TIMEOUT_SEC", 120)

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

if GAP_RETRY_PADDING_SEC * 2 >= GAP_RETRY_MAX_SEC:
    raise ValueError(
        f"GAP_RETRY_PADDING_SEC ({GAP_RETRY_PADDING_SEC}) must leave a "
        f"positive retry core inside GAP_RETRY_MAX_SEC ({GAP_RETRY_MAX_SEC})."
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
