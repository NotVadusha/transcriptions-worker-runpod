"""Audio download and preprocessing (SPEC §4).

Responsibilities:
  * Download a presigned HTTPS audio URL to a per-job temp dir (streamed to
    disk, size-capped, timeout-bounded).
  * Probe the source duration with ``ffprobe`` (the trusted duration, used for
    the ``AUDIO_TOO_LONG`` gate and ``meta.audio_duration_sec``).
  * Normalize arbitrary input to 16 kHz mono 16-bit PCM WAV with ``ffmpeg``
    (Parakeet expects 16 kHz mono).
  * Extract per-chunk windows from the normalized WAV for the long-audio path.

This module uses ``httpx`` (HTTP) and ``ffmpeg``/``ffprobe`` (subprocess). It
does NOT import torch/nemo, so it can be imported and unit-tested (with those
two dependencies mocked) on a GPU-free machine.

Error mapping (SPEC §3.3):
  * Download problems (non-2xx, timeout, connection error, size cap exceeded)
    -> :class:`DownloadError` (code ``DOWNLOAD_FAILED``).
  * ffmpeg *decode* failure during normalization -> :class:`FFmpegError` with
    code ``UNSUPPORTED_FORMAT`` (a non-zero normalize exit almost always means
    the input container/codec could not be decoded).
  * Any other ffmpeg/ffprobe failure -> :class:`FFmpegError` with code
    ``FFMPEG_FAILED``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import httpx

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
    """Raised when the audio download fails (code ``DOWNLOAD_FAILED``)."""

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
# Download
# --------------------------------------------------------------------------- #
def download(audio_url: str, work_dir: str) -> str:
    """Stream ``audio_url`` to disk inside ``work_dir`` and return the path.

    Enforces ``config.DOWNLOAD_TIMEOUT_SEC`` and aborts with
    :class:`DownloadError` on a non-2xx response, a network/timeout error, or
    when more than ``config.MAX_DOWNLOAD_BYTES`` have been received. The body is
    streamed chunk-by-chunk so a large file is never loaded fully into memory,
    and the size cap is checked mid-stream so we stop early rather than after a
    multi-GB download completes.

    The downloaded file is named ``source_audio`` (no extension): we never trust
    the extension — ffmpeg is the format arbiter (SPEC §4).
    """
    dest_path = os.path.join(work_dir, "source_audio")
    max_bytes = config.MAX_DOWNLOAD_BYTES
    timeout = config.DOWNLOAD_TIMEOUT_SEC

    try:
        # follow_redirects=False on purpose. SPEC §1 guarantees a plain GET on
        # the presigned URL is sufficient, so redirects are not needed — and not
        # chasing them closes an SSRF vector: a presigned host that 30x-redirects
        # to a loopback / link-local / cloud-metadata (169.254.169.254) address
        # is never followed, and the https-only contract (§3.1) cannot be
        # silently bypassed on a redirect hop. A 3xx therefore surfaces as a
        # clean DOWNLOAD_FAILED below rather than being chased.
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            with client.stream("GET", audio_url) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    raise DownloadError(
                        f"Download failed: server returned HTTP "
                        f"{response.status_code}."
                    )

                # Fast-fail on an advertised over-size body before streaming any
                # of it. The mid-stream cap below remains the authoritative guard
                # for a missing or dishonest Content-Length.
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        declared_bytes = int(declared)
                    except ValueError:
                        declared_bytes = None
                    if declared_bytes is not None and declared_bytes > max_bytes:
                        raise DownloadError(
                            f"Download aborted: Content-Length {declared_bytes} "
                            f"exceeds the maximum allowed size of {max_bytes} bytes."
                        )

                bytes_written = 0
                with open(dest_path, "wb") as fh:
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        bytes_written += len(chunk)
                        if bytes_written > max_bytes:
                            # Abort early — don't finish a huge download.
                            raise DownloadError(
                                f"Download aborted: audio exceeds the maximum "
                                f"allowed size of {max_bytes} bytes."
                            )
                        fh.write(chunk)
    except DownloadError:
        raise
    except httpx.TimeoutException as exc:
        raise DownloadError(
            f"Download timed out after {timeout}s: {exc}"
        ) from exc
    except httpx.HTTPError as exc:
        # Connection errors, invalid responses, too many redirects, etc.
        raise DownloadError(f"Download failed: {exc}") from exc

    return dest_path


# --------------------------------------------------------------------------- #
# ffprobe / ffmpeg helpers
# --------------------------------------------------------------------------- #
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


def probe_duration(path: str) -> float:
    """Return the audio duration in seconds via ``ffprobe``.

    Raises :class:`FFmpegError` (code ``FFMPEG_FAILED``) if ffprobe fails or its
    output cannot be parsed as a positive float.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        proc = _run(cmd)
    except FileNotFoundError as exc:
        raise FFmpegError(
            f"ffprobe binary not found: {exc}", ErrorCode.FFMPEG_FAILED
        ) from exc

    if proc.returncode != 0:
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


def normalize(src_path: str, work_dir: str) -> str:
    """Normalize ``src_path`` to 16 kHz mono 16-bit PCM WAV; return its path.

    Runs (SPEC §4)::

        ffmpeg -nostdin -y -i <src> -ac 1 -ar 16000 -vn -c:a pcm_s16le \
            <work_dir>/audio_16khz_mono.wav

    A non-zero exit is mapped to ``UNSUPPORTED_FORMAT`` — the common cause is an
    input container/codec ffmpeg cannot decode. We deliberately do not
    pre-filter by extension; ffmpeg is the arbiter of what is decodable.
    """
    out_path = os.path.join(work_dir, "audio_16khz_mono.wav")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        src_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        "-c:a",
        "pcm_s16le",
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
