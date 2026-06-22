"""Request parsing/validation and output building (SPEC §3.1, §3.2).

Pure stdlib + dataclasses — no torch/nemo/httpx/ffmpeg imports — so it is
unit-testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from src import config


class ValidationError(Exception):
    """Raised when a request fails validation (SPEC §3.1).

    Carries a machine-readable ``code`` (one of ``config.ErrorCode``) and a
    human-readable ``message``. ``str(e)`` returns the message.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


@dataclass(frozen=True)
class Request:
    """A validated transcription request."""

    audio_url: str
    return_timestamps: bool
    language: str
    result_upload_url: str | None


def _is_https_url(value: str) -> bool:
    """Return True iff ``value`` is an https URL with a non-empty netloc."""
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def parse_request(job_input: dict) -> Request:
    """Validate ``job_input`` and return a :class:`Request` (SPEC §3.1).

    Rules:
      * ``audio_url`` missing/not a string -> MISSING_AUDIO_URL.
      * ``audio_url`` scheme not https     -> INVALID_URL.
      * ``return_timestamps`` not a bool   -> coerced to True (no raise).
      * ``language`` present and unsupported -> UNSUPPORTED_LANGUAGE
        (defaults to ``config.DEFAULT_LANGUAGE`` when absent).
      * ``result_upload_url`` present and not https -> INVALID_URL
        (defaults to ``None`` when absent).
    """
    if not isinstance(job_input, dict):
        raise ValidationError(
            config.ErrorCode.MISSING_AUDIO_URL,
            "input must be an object containing an audio_url.",
        )

    # --- audio_url -------------------------------------------------------- #
    audio_url = job_input.get("audio_url")
    if not isinstance(audio_url, str) or audio_url.strip() == "":
        raise ValidationError(
            config.ErrorCode.MISSING_AUDIO_URL,
            "audio_url is required and must be a non-empty string.",
        )
    if not _is_https_url(audio_url):
        raise ValidationError(
            config.ErrorCode.INVALID_URL,
            "audio_url must be an https URL.",
        )

    # --- return_timestamps (coerce, never raise) -------------------------- #
    rt = job_input.get("return_timestamps", True)
    # Note: bool is a subclass of int, but we only accept genuine bools here;
    # anything else (including ints/strings) coerces to the default True.
    return_timestamps = rt if isinstance(rt, bool) else True

    # --- language --------------------------------------------------------- #
    language = job_input.get("language", config.DEFAULT_LANGUAGE)
    if language is None:
        language = config.DEFAULT_LANGUAGE
    if language not in config.SUPPORTED_LANGUAGES:
        raise ValidationError(
            config.ErrorCode.UNSUPPORTED_LANGUAGE,
            f"language {language!r} is not supported; v0 supports only "
            f"{sorted(config.SUPPORTED_LANGUAGES)}.",
        )

    # --- result_upload_url ------------------------------------------------ #
    result_upload_url = job_input.get("result_upload_url")
    if result_upload_url is not None:
        if not _is_https_url(result_upload_url):
            raise ValidationError(
                config.ErrorCode.INVALID_URL,
                "result_upload_url must be an https URL.",
            )
    else:
        result_upload_url = None

    return Request(
        audio_url=audio_url,
        return_timestamps=return_timestamps,
        language=language,
        result_upload_url=result_upload_url,
    )


def build_output(
    result: dict, duration: float, processing_time: float, req: Request
) -> dict:
    """Build the SPEC §3.2 success object from a transcribe.run() result.

    ``result`` shape:
        {"text": str, "words": list[dict], "segments": list[dict],
         "chunked": bool, "num_chunks": int}

    ``text`` is always present. ``segments`` and ``words`` are included ONLY
    when ``req.return_timestamps`` is True (the keys are omitted entirely
    otherwise — not ``null``, not ``[]``).
    """
    # rtf is defined as audio_duration_sec / processing_time_sec
    # (a speed factor: higher = faster than real time). Guard against a
    # non-positive processing_time to avoid division by zero.
    rtf = round(duration / processing_time, 3) if processing_time > 0 else 0.0

    output: dict = {"text": result["text"]}

    if req.return_timestamps:
        output["segments"] = result["segments"]
        output["words"] = result["words"]

    output["meta"] = {
        "model": config.MODEL_NAME,
        "language": req.language,
        "audio_duration_sec": round(duration, 3),
        "processing_time_sec": round(processing_time, 3),
        "rtf": rtf,
        "chunked": result["chunked"],
        "num_chunks": result["num_chunks"],
        "worker_version": config.WORKER_VERSION,
    }

    return output
