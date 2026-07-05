"""GPU-free orchestration tests for ``src.handler`` (SPEC §6).

These tests run on a CPU-only machine: ``SKIP_MODEL_LOAD`` is set BEFORE the
handler is imported (so ``transcribe.load_model()`` at import time is a no-op and
never touches NeMo/torch), and the audio + transcribe boundaries are
monkeypatched so no real download / ffmpeg / inference happens. We exercise the
orchestration and every error path the handler is responsible for.

Requires ``runpod`` to be importable (lightweight, CPU-only) and ``pytest``.
"""

import os

# MUST be set before importing the handler: load_model() runs at module import.
os.environ["SKIP_MODEL_LOAD"] = "1"

import pytest

from src import audio, config, handler, schemas, storage, transcribe


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _job(input_dict, job_id="job-123"):
    return {"id": job_id, "input": input_dict}


def _fake_result(text="hello world"):
    """A transcribe.run()-shaped result."""
    return {
        "text": text,
        "words": [
            {"start": 0.0, "end": 0.5, "word": "hello"},
            {"start": 0.5, "end": 1.0, "word": "world"},
        ],
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
        "chunked": False,
        "num_chunks": 1,
    }


@pytest.fixture
def patched_audio(monkeypatch, tmp_path):
    """Stub out every audio boundary so no real download/ffmpeg runs.

    Returns a dict of mutable call counters / behavior hooks the test can tweak.
    By default: download + normalize succeed, duration is short, cleanup is a
    no-op that records that it was called.
    """
    state = {"cleanup_called": False, "duration": 12.0}

    workdir = str(tmp_path / "work")

    def fake_make_workdir(job_id):
        return workdir

    def fake_probe_duration(source):
        return state["duration"]

    def fake_normalize(source, work_dir):
        return os.path.join(work_dir, "audio_16khz_mono.wav")

    def fake_cleanup(work_dir):
        state["cleanup_called"] = True

    monkeypatch.setattr(audio, "make_workdir", fake_make_workdir)
    monkeypatch.setattr(audio, "probe_duration", fake_probe_duration)
    monkeypatch.setattr(audio, "normalize", fake_normalize)
    monkeypatch.setattr(audio, "cleanup", fake_cleanup)
    return state


@pytest.fixture
def patched_transcribe(monkeypatch):
    """Stub transcribe.run to return a canned result (no NeMo)."""
    calls = {}

    def fake_run(wav_path, duration, return_timestamps=True, language="en"):
        calls["wav_path"] = wav_path
        calls["duration"] = duration
        calls["return_timestamps"] = return_timestamps
        calls["language"] = language
        return _fake_result()

    monkeypatch.setattr(transcribe, "run", fake_run)
    return calls


# --------------------------------------------------------------------------- #
# Validation errors (caller error -> structured response, never raises)
# --------------------------------------------------------------------------- #
def test_missing_audio_url_returns_structured_error(patched_audio):
    out = handler.handler(_job({}))
    assert out["error"]["code"] == config.ErrorCode.MISSING_AUDIO_URL
    assert isinstance(out["error"]["message"], str) and out["error"]["message"]


def test_non_https_audio_url_returns_invalid_url(patched_audio):
    out = handler.handler(_job({"audio_url": "http://example.com/a.wav"}))
    assert out["error"]["code"] == config.ErrorCode.INVALID_URL


def test_non_string_language_returns_structured_error(patched_audio):
    # A non-string language is the only language validation failure now; real
    # codes (fr/de/ja/...) are all routed to a backend.
    out = handler.handler(
        _job({"audio_url": "https://x.com/a.wav", "language": 123})
    )
    assert out["error"]["code"] == config.ErrorCode.UNSUPPORTED_LANGUAGE


def test_routed_language_passed_through_to_transcribe(patched_audio, patched_transcribe):
    out = handler.handler(
        _job({"audio_url": "https://x.com/a.wav", "language": "de"})
    )
    assert "error" not in out
    assert patched_transcribe["language"] == "de"


def test_validation_error_does_not_create_or_clean_workdir(monkeypatch):
    # On a validation error the handler returns before touching the filesystem.
    called = {"make": False, "cleanup": False}
    monkeypatch.setattr(
        audio, "make_workdir", lambda jid: called.__setitem__("make", True)
    )
    monkeypatch.setattr(
        audio, "cleanup", lambda wd: called.__setitem__("cleanup", True)
    )
    handler.handler(_job({}))
    assert called["make"] is False
    assert called["cleanup"] is False


# --------------------------------------------------------------------------- #
# AUDIO_TOO_LONG gate
# --------------------------------------------------------------------------- #
def test_audio_too_long_returns_structured_error(patched_audio, patched_transcribe):
    patched_audio["duration"] = config.MAX_AUDIO_SECONDS + 1
    out = handler.handler(_job({"audio_url": "https://x.com/a.wav"}))
    assert out["error"]["code"] == config.ErrorCode.AUDIO_TOO_LONG
    # transcribe must NOT have been invoked.
    assert "wav_path" not in patched_transcribe
    # cleanup still runs.
    assert patched_audio["cleanup_called"] is True


def test_audio_at_limit_is_allowed(patched_audio, patched_transcribe):
    patched_audio["duration"] = float(config.MAX_AUDIO_SECONDS)
    out = handler.handler(_job({"audio_url": "https://x.com/a.wav"}))
    assert "error" not in out
    assert out["text"] == "hello world"


# --------------------------------------------------------------------------- #
# Download / ffmpeg errors -> structured error using e.code
# --------------------------------------------------------------------------- #
def test_download_error_maps_to_structured_code(
    monkeypatch, patched_audio, patched_transcribe
):
    # A URL fetch failure now surfaces from ffprobe/ffmpeg as DownloadError; the
    # streamed probe runs first, so simulate it there.
    def boom(source):
        raise audio.DownloadError("server returned HTTP 404.")

    monkeypatch.setattr(audio, "probe_duration", boom)
    out = handler.handler(_job({"audio_url": "https://x.com/a.wav"}))
    assert out["error"]["code"] == config.ErrorCode.DOWNLOAD_FAILED
    assert "404" in out["error"]["message"]
    assert patched_audio["cleanup_called"] is True


def test_unsupported_format_maps_to_structured_code(
    monkeypatch, patched_audio, patched_transcribe
):
    def boom(src_path, work_dir):
        raise audio.FFmpegError("bad codec", config.ErrorCode.UNSUPPORTED_FORMAT)

    monkeypatch.setattr(audio, "normalize", boom)
    out = handler.handler(_job({"audio_url": "https://x.com/a.wav"}))
    assert out["error"]["code"] == config.ErrorCode.UNSUPPORTED_FORMAT
    assert patched_audio["cleanup_called"] is True


def test_ffmpeg_probe_failure_maps_to_structured_code(
    monkeypatch, patched_audio, patched_transcribe
):
    def boom(path):
        raise audio.FFmpegError("ffprobe failed", config.ErrorCode.FFMPEG_FAILED)

    monkeypatch.setattr(audio, "probe_duration", boom)
    out = handler.handler(_job({"audio_url": "https://x.com/a.wav"}))
    assert out["error"]["code"] == config.ErrorCode.FFMPEG_FAILED


# --------------------------------------------------------------------------- #
# TranscriptionError -> re-raised (job FAILED), cleanup still runs
# --------------------------------------------------------------------------- #
def test_transcription_error_is_reraised(monkeypatch, patched_audio):
    def boom(wav_path, duration, return_timestamps=True, language="en"):
        raise transcribe.TranscriptionError("inference blew up")

    monkeypatch.setattr(transcribe, "run", boom)
    with pytest.raises(transcribe.TranscriptionError):
        handler.handler(_job({"audio_url": "https://x.com/a.wav"}))
    # finally: cleanup runs even when the error propagates.
    assert patched_audio["cleanup_called"] is True


# --------------------------------------------------------------------------- #
# Happy path shape + return_timestamps handling
# --------------------------------------------------------------------------- #
def test_happy_path_shape_with_timestamps(patched_audio, patched_transcribe):
    out = handler.handler(
        _job({"audio_url": "https://x.com/a.wav", "return_timestamps": True})
    )
    assert "error" not in out
    assert out["text"] == "hello world"
    assert "words" in out and "segments" in out
    assert out["words"][0] == {"start": 0.0, "end": 0.5, "word": "hello"}
    meta = out["meta"]
    assert meta["model"] == config.MODEL_NAME
    assert meta["language"] == "en"
    assert meta["worker_version"] == config.WORKER_VERSION
    assert meta["chunked"] is False
    assert meta["num_chunks"] == 1
    assert meta["audio_duration_sec"] == round(patched_audio["duration"], 3)
    assert meta["processing_time_sec"] >= 0.0
    # transcribe was asked for timestamps (request flag passed through).
    assert patched_transcribe["return_timestamps"] is True
    assert patched_transcribe["duration"] == patched_audio["duration"]


def test_return_timestamps_false_omits_words_and_segments(
    patched_audio, patched_transcribe
):
    out = handler.handler(
        _job({"audio_url": "https://x.com/a.wav", "return_timestamps": False})
    )
    assert "error" not in out
    assert out["text"] == "hello world"
    # Keys omitted entirely (not null, not []).
    assert "words" not in out
    assert "segments" not in out
    assert "meta" in out
    assert patched_transcribe["return_timestamps"] is False


def test_return_timestamps_non_bool_coerces_to_true(patched_audio, patched_transcribe):
    out = handler.handler(
        _job({"audio_url": "https://x.com/a.wav", "return_timestamps": "yes"})
    )
    assert "error" not in out
    assert "words" in out and "segments" in out
    assert patched_transcribe["return_timestamps"] is True


# --------------------------------------------------------------------------- #
# Offload path: handler returns whatever storage.maybe_offload yields
# --------------------------------------------------------------------------- #
def test_offload_result_too_large_passthrough(
    monkeypatch, patched_audio, patched_transcribe
):
    # Force the offload branch by lowering the threshold so the inline result
    # is considered "too large" and no upload URL is given.
    monkeypatch.setattr(config, "RESULT_OFFLOAD_THRESHOLD_BYTES", 1)
    out = handler.handler(_job({"audio_url": "https://x.com/a.wav"}))
    assert out["error"]["code"] == config.ErrorCode.RESULT_TOO_LARGE


def test_missing_job_id_does_not_crash(patched_audio, patched_transcribe):
    # job with no "id" key — handler should default it, not KeyError.
    out = handler.handler({"input": {"audio_url": "https://x.com/a.wav"}})
    assert out["text"] == "hello world"


# --------------------------------------------------------------------------- #
# Module-level guarantees
# --------------------------------------------------------------------------- #
def test_skip_model_load_kept_module_importable():
    # If the import-time load_model() had tried to import NeMo this test file
    # would have failed to import. Reaching here proves the escape hatch works.
    assert transcribe._skip_model_load() is True
    assert callable(handler.handler)


def test_progress_update_failure_never_crashes_job(
    monkeypatch, patched_audio, patched_transcribe
):
    # Even if progress_update raises, the job must complete normally.
    import runpod

    def boom(job, message):
        raise RuntimeError("progress channel down")

    monkeypatch.setattr(runpod.serverless, "progress_update", boom)
    out = handler.handler(_job({"audio_url": "https://x.com/a.wav"}))
    assert out["text"] == "hello world"
