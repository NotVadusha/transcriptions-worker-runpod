"""RunPod serverless entrypoint — thin orchestration only (SPEC §6).

Responsibilities, in order, for a single job:

  1. Parse/validate the request (``schemas.parse_request``) — caller errors
     return a structured ``{"error": {...}}`` object (job SUCCEEDED with an
     error payload, per SPEC §3.3 caller-error semantics).
  2. Download the audio, probe its duration, gate on ``MAX_AUDIO_SECONDS``,
     normalize to 16 kHz mono WAV (``audio`` module).
  3. Transcribe (``transcribe.run``) and build the §3.2 output
     (``schemas.build_output``).
  4. Offload the result if it is too large to return inline
     (``storage.maybe_offload``) and return whatever that yields.

Error handling:
  * ``schemas.ValidationError``        -> structured error (caller error).
  * ``AUDIO_TOO_LONG``                 -> structured error (caller error).
  * ``audio.AudioError`` (download /   -> structured error using ``e.code``
    ffmpeg / unsupported format)          (covers DOWNLOAD_FAILED,
                                           FFMPEG_FAILED, UNSUPPORTED_FORMAT).
  * ``transcribe.TranscriptionError``  -> re-raised so RunPod marks the job
                                           FAILED (internal error).
  * ``work_dir`` is always cleaned up in a ``finally`` block.

The heavy model is loaded ONCE at import time (RunPod best practice). Because
``transcribe.load_model`` honors ``config.SKIP_MODEL_LOAD``, importing this
module with ``SKIP_MODEL_LOAD=1`` is safe on a GPU-free machine (tests, CI).
"""

# --------------------------------------------------------------------------- #
# IMPORT SHIM (SPEC §9): the worker is started via ``python src/handler.py``,
# which puts ``src/`` (not the repo root) on sys.path and would break the
# ``from src import ...`` package imports below. Inserting the repo root makes
# ``python src/handler.py``, ``python -m src.handler`` and ``from src import``
# all resolve identically, locally and inside the Docker image.
# --------------------------------------------------------------------------- #
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time  # noqa: E402 - must follow the sys.path shim

import runpod  # noqa: E402

from src import audio, config, schemas, storage, transcribe  # noqa: E402


# Heavy init at import time — load the model once per worker process. A no-op
# under SKIP_MODEL_LOAD (test mode). Load failure propagates (SPEC §3.3).
transcribe.load_model()


def _progress(job, message: str) -> None:
    """Best-effort progress update. Never crashes the job if it fails."""
    try:
        runpod.serverless.progress_update(job, message)
    except Exception:  # noqa: BLE001 - progress is purely informational
        pass


def handler(job) -> dict:
    """Process a single transcription job (SPEC §6).

    Returns the §3.2 success object (possibly offloaded to a reference object),
    or a structured ``{"error": {"code", "message"}}`` for caller errors.
    Internal transcription errors are re-raised so the job is marked FAILED.
    """
    job_input = job.get("input") or {}

    # --- validation (caller error -> structured response, never raises) ----- #
    try:
        req = schemas.parse_request(job_input)
    except schemas.ValidationError as e:
        return {"error": {"code": e.code, "message": str(e)}}

    work_dir = audio.make_workdir(job.get("id", "unknown"))
    try:
        _progress(job, "Downloading audio")
        src_path = audio.download(req.audio_url, work_dir)

        duration = audio.probe_duration(src_path)
        if duration > config.MAX_AUDIO_SECONDS:
            return {
                "error": {
                    "code": config.ErrorCode.AUDIO_TOO_LONG,
                    "message": (
                        f"Audio is {duration:.1f}s long, exceeding the maximum "
                        f"of {config.MAX_AUDIO_SECONDS}s."
                    ),
                }
            }

        _progress(job, "Normalizing audio")
        wav_path = audio.normalize(src_path, work_dir)

        _progress(job, "Transcribing")
        t0 = time.perf_counter()
        result = transcribe.run(
            wav_path, duration, return_timestamps=req.return_timestamps
        )
        proc = time.perf_counter() - t0

        out = schemas.build_output(result, duration, proc, req)
        return storage.maybe_offload(out, req.result_upload_url)

    except transcribe.TranscriptionError:
        # Internal inference failure -> re-raise so RunPod marks the job FAILED.
        raise
    except audio.AudioError as e:
        # Covers DOWNLOAD_FAILED / FFMPEG_FAILED / UNSUPPORTED_FORMAT — caller
        # error semantics: return a structured error rather than failing.
        return {"error": {"code": e.code, "message": str(e)}}
    finally:
        audio.cleanup(work_dir)


if __name__ == "__main__":
    # The RunPod SDK provides --test_input and --rp_serve_api handling itself.
    runpod.serverless.start({"handler": handler})
