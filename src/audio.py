"""Audio input streaming and preprocessing (SPEC §4).

The input audio is a caller-supplied presigned HTTPS GET URL. Rather than
downloading it to disk first, ``ffprobe``/``ffmpeg`` read the URL DIRECTLY and
stream it — so the only file we ever write is the normalized WAV (and its
per-chunk windows), never a copy of the raw upload.

Responsibilities:
  * Probe the source duration with ``ffprobe`` reading the URL (the trusted
    duration, used for the ``AUDIO_TOO_LONG`` gate and ``meta.audio_duration_sec``).
  * Normalize the streamed input to 16 kHz mono 16-bit PCM WAV with ``ffmpeg``
    reading the URL (Parakeet expects 16 kHz mono). Video inputs work too —
    ``-vn`` keeps only the audio stream, so ffmpeg does the demux/extract.
  * Extract per-chunk windows from the normalized WAV for the long-audio path.

This module shells out to ``ffmpeg``/``ffprobe`` (subprocess) and does NOT import
torch/nemo, so it can be imported and unit-tested (with the subprocess mocked)
on a GPU-free machine.

Security tradeoff (SPEC §10): streaming with ffmpeg means we no longer enforce a
download byte-cap and ffmpeg may follow HTTP redirects, so the previous
httpx-based size-cap + no-redirect SSRF guard is gone. The ``AUDIO_TOO_LONG``
duration gate (``ffprobe`` runs first) is the remaining resource bound, and the
caller is trusted to pass a direct https presigned S3 URL (README "Security
model"). ``-rw_timeout`` bounds a stalled network read.

Error mapping (SPEC §3.3):
  * A network/HTTP failure fetching the URL (probe or stream) ->
    :class:`DownloadError` (code ``DOWNLOAD_FAILED``).
  * ffmpeg *decode* failure during normalization -> :class:`FFmpegError` with
    code ``UNSUPPORTED_FORMAT`` (the input container/codec could not be decoded).
  * Any other ffmpeg/ffprobe failure -> :class:`FFmpegError` with code
    ``FFMPEG_FAILED``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from src import config
from src.config import ErrorCode


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class AudioError(Exception):
    """Base class for audio download/preprocessing failures.

    Carries a machine-readable ``code`` (one of ``config.ErrorCode``) and a
    human-readable ``message``. ``str(e)`` returns the message.
    """

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class DownloadError(AudioError):
    """Raised when the audio URL can't be fetched/streamed (code ``DOWNLOAD_FAILED``).

    Covers ffprobe/ffmpeg network failures reading the presigned URL: HTTP 4xx/5xx,
    TLS errors, DNS failures, connection refused/reset, or an ``-rw_timeout`` stall.
    """

    def __init__(self, message: str, code: str = ErrorCode.DOWNLOAD_FAILED) -> None:
        super().__init__(message, code)


class FFmpegError(AudioError):
    """Raised when ffmpeg/ffprobe fails.

    The ``code`` must be passed explicitly: ``UNSUPPORTED_FORMAT`` for a decode
    failure during normalization, ``FFMPEG_FAILED`` for everything else.
    """

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message, code)


# --------------------------------------------------------------------------- #
# Temp dir lifecycle
# --------------------------------------------------------------------------- #
def make_workdir(job_id: str) -> str:
    """Create and return a unique temp directory for ``job_id``.

    The job id is embedded in the directory name for easier debugging. A safe
    subset of the id is used so an unusual job id can't escape the temp root.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(job_id))
    return tempfile.mkdtemp(prefix=f"txjob_{safe}_")


def cleanup(work_dir: str) -> None:
    """Recursively remove ``work_dir``; never raises (best-effort cleanup)."""
    if not work_dir:
        return
    shutil.rmtree(work_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# ffprobe / ffmpeg helpers
# --------------------------------------------------------------------------- #
# Substrings that mark an ffmpeg/ffprobe failure as a network/HTTP problem
# (fetching the URL) rather than a decode problem (a bad file). Lowercased match.
_NETWORK_ERROR_MARKERS = (
    "http error",
    "server returned",
    "403 forbidden",
    "404 not found",
    "401 unauthorized",
    "400 bad request",
    "connection refused",
    "connection reset",
    "connection timed out",
    "network is unreachable",
    "failed to resolve",
    "name or service not known",
    "temporary failure in name resolution",
    "timed out",
    "timeout",  # covers "read timeout" and our own -rw_timeout expiry phrasings
    "i/o error",
    "unable to open resource",
    # TLS/SSL handshake + cert failures (gnutls / openssl). Specific phrases only —
    # a bare "tls" substring would false-match codec/container names.
    "error in the pull function",
    "ssl routines",
    "ssl handshake",
    "tls handshake",
    "certificate",
)


def _is_url(source: str) -> bool:
    """True if ``source`` is an http(s) URL (vs a local file path)."""
    return source.startswith("http://") or source.startswith("https://")


def _network_input_opts() -> list[str]:
    """ffmpeg/ffprobe options for robustly streaming a URL input.

    MUST be placed before ``-i``. ``-rw_timeout`` (microseconds) bounds a stalled
    read; the ``-reconnect*`` flags let ffmpeg resume a dropped/erroring S3 GET
    instead of failing the whole job:
      * reconnect / reconnect_streamed  — reconnect at EOF and for non-seekable streams
      * reconnect_on_network_error      — reconnect on a mid-stream read error; THIS is
        the one that matters for seekable S3 URLs (reconnect_streamed does not cover
        a mid-transfer TCP/TLS drop on a seekable source). Needs ffmpeg >= 4.3.
      * reconnect_on_http_error 5xx     — retry transient S3 5xx (503/504 throttling)
    MUST-VALIDATE-IN-CONTAINER: confirm the image's ffmpeg accepts these
    (`ffmpeg -h full | grep reconnect`) — an unknown option fails every job.
    # ponytail: -rw_timeout is per-read only; no total deadline (SPEC §10, trusted caller).
    """
    timeout_usec = max(1, config.DOWNLOAD_TIMEOUT_SEC) * 1_000_000
    return [
        "-rw_timeout",
        str(timeout_usec),
        "-reconnect",
        "1",
        "-reconnect_on_network_error",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_on_http_error",
        "5xx",
        "-reconnect_delay_max",
        "30",
    ]


def _looks_like_network_error(stderr: str) -> bool:
    """Heuristic: does this ffmpeg/ffprobe stderr describe a fetch failure?"""
    low = (stderr or "").lower()
    return any(marker in low for marker in _NETWORK_ERROR_MARKERS)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run ``cmd`` capturing stdout/stderr as text. Never raises on non-zero.

    A missing ffmpeg/ffprobe binary surfaces as ``FileNotFoundError``, which
    callers map to ``FFMPEG_FAILED``.
    """
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def probe_duration(source: str) -> float:
    """Return the audio duration in seconds via ``ffprobe``.

    ``source`` may be a presigned URL (read/streamed directly) or a local path.
    Raises :class:`DownloadError` (``DOWNLOAD_FAILED``) if a URL can't be fetched,
    or :class:`FFmpegError` (``FFMPEG_FAILED``) if ffprobe fails or its output
    cannot be parsed as a positive float.
    """
    cmd = ["ffprobe", "-v", "error"]
    if _is_url(source):
        cmd += _network_input_opts()
    cmd += [
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        source,
    ]
    try:
        proc = _run(cmd)
    except FileNotFoundError as exc:
        raise FFmpegError(
            f"ffprobe binary not found: {exc}", ErrorCode.FFMPEG_FAILED
        ) from exc

    if proc.returncode != 0:
        if _is_url(source) and _looks_like_network_error(proc.stderr):
            raise DownloadError(
                f"Could not read audio from URL: {proc.stderr.strip()}"
            )
        raise FFmpegError(
            f"ffprobe failed to read duration: {proc.stderr.strip()}",
            ErrorCode.FFMPEG_FAILED,
        )

    raw = proc.stdout.strip()
    try:
        duration = float(raw)
    except (TypeError, ValueError) as exc:
        raise FFmpegError(
            f"ffprobe returned an unparseable duration: {raw!r}",
            ErrorCode.FFMPEG_FAILED,
        ) from exc

    if duration <= 0:
        raise FFmpegError(
            f"ffprobe returned a non-positive duration: {duration}",
            ErrorCode.FFMPEG_FAILED,
        )

    return duration


def normalize(source: str, work_dir: str) -> str:
    """Normalize ``source`` to 16 kHz mono 16-bit PCM WAV; return its path.

    ``source`` may be a presigned URL (streamed directly by ffmpeg) or a local
    path. Runs (SPEC §4)::

        ffmpeg -nostdin -y [net opts if URL] -i <source> \
            -ac 1 -ar 16000 -vn -c:a pcm_s16le <work_dir>/audio_16khz_mono.wav

    ``-vn`` drops any video stream, so a video URL is transparently demuxed to
    audio. A network fetch failure maps to ``DOWNLOAD_FAILED``; any other
    non-zero exit maps to ``UNSUPPORTED_FORMAT`` (the container/codec could not
    be decoded). We deliberately do not pre-filter by extension; ffmpeg is the
    arbiter of what is decodable.
    """
    out_path = os.path.join(work_dir, "audio_16khz_mono.wav")
    cmd = ["ffmpeg", "-nostdin", "-y"]
    if _is_url(source):
        cmd += _network_input_opts()
    cmd += [
        "-i",
        source,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        "-c:a",
        "pcm_s16le",
        # Defense-in-depth output-length cap (bounds WAV disk even if a container
        # lies about its duration). A legitimate file already passed the ffprobe
        # AUDIO_TOO_LONG gate, so it is <= MAX_AUDIO_SECONDS and is never truncated.
        "-t",
        str(config.MAX_AUDIO_SECONDS),
        out_path,
    ]
    try:
        proc = _run(cmd)
    except FileNotFoundError as exc:
        # Missing binary is an internal/environment failure, not a bad format.
        raise FFmpegError(
            f"ffmpeg binary not found: {exc}", ErrorCode.FFMPEG_FAILED
        ) from exc

    if proc.returncode != 0:
        if _is_url(source) and _looks_like_network_error(proc.stderr):
            raise DownloadError(
                f"Could not stream audio from URL: {proc.stderr.strip()}"
            )
        raise FFmpegError(
            f"ffmpeg could not decode the audio (unsupported format): "
            f"{proc.stderr.strip()}",
            ErrorCode.UNSUPPORTED_FORMAT,
        )

    return out_path


def extract_window(
    wav_path: str, start: float, duration: float, out_path: str
) -> str:
    """Extract a ``[start, start+duration)`` window of ``wav_path`` to ``out_path``.

    Used by the long-audio path in ``transcribe.py`` to cut per-chunk WAVs from
    the already-normalized 16 kHz mono WAV. Re-encodes to the same PCM format so
    the cut is deterministic at the sample level. A non-zero exit maps to
    ``FFMPEG_FAILED`` (the source is already a known-good WAV at this point, so
    a failure here is an internal error, not a bad-input error).
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        wav_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        out_path,
    ]
    try:
        proc = _run(cmd)
    except FileNotFoundError as exc:
        raise FFmpegError(
            f"ffmpeg binary not found: {exc}", ErrorCode.FFMPEG_FAILED
        ) from exc

    if proc.returncode != 0:
        raise FFmpegError(
            f"ffmpeg failed to extract window "
            f"[{start}, {start + duration}): {proc.stderr.strip()}",
            ErrorCode.FFMPEG_FAILED,
        )

    return out_path
