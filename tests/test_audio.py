"""Unit tests for src.audio.

All network (httpx) and subprocess (ffmpeg/ffprobe) interactions are mocked, so
these tests require neither a network connection nor ffmpeg installed and run
under Python 3.13 with only stdlib + pytest + httpx.
"""

from __future__ import annotations

import os
import subprocess
from unittest import mock

import httpx
import pytest

from src import audio, config
from src.config import ErrorCode


# --------------------------------------------------------------------------- #
# make_workdir / cleanup
# --------------------------------------------------------------------------- #
def test_make_workdir_is_unique_and_real(tmp_path):
    d1 = audio.make_workdir("job-123")
    d2 = audio.make_workdir("job-123")
    try:
        assert d1 != d2
        assert os.path.isdir(d1) and os.path.isdir(d2)
        assert "job-123" in os.path.basename(d1)
    finally:
        audio.cleanup(d1)
        audio.cleanup(d2)


def test_make_workdir_sanitizes_job_id():
    d = audio.make_workdir("../../etc/passwd")
    try:
        assert "/" not in os.path.basename(d)
    finally:
        audio.cleanup(d)


def test_cleanup_never_raises_on_missing_dir():
    audio.cleanup("/nonexistent/path/that/does/not/exist")
    audio.cleanup("")  # empty is a no-op


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
class _FakeStreamResponse:
    def __init__(self, status_code, chunks, headers=None):
        self.status_code = status_code
        self._chunks = chunks
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_bytes(self):
        yield from self._chunks


class _FakeClient:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stream(self, method, url):
        if self._exc is not None:
            raise self._exc
        return self._response


def test_download_success(tmp_path):
    resp = _FakeStreamResponse(200, [b"abc", b"def"])
    with mock.patch.object(audio.httpx, "Client", return_value=_FakeClient(resp)):
        path = audio.download("https://example.com/a.wav", str(tmp_path))
    assert os.path.isfile(path)
    with open(path, "rb") as fh:
        assert fh.read() == b"abcdef"


def test_download_non_2xx_raises_download_error(tmp_path):
    resp = _FakeStreamResponse(404, [])
    with mock.patch.object(audio.httpx, "Client", return_value=_FakeClient(resp)):
        with pytest.raises(audio.DownloadError) as ei:
            audio.download("https://example.com/missing", str(tmp_path))
    assert ei.value.code == ErrorCode.DOWNLOAD_FAILED
    assert "404" in str(ei.value)


def test_download_size_cap_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_DOWNLOAD_BYTES", 4)
    resp = _FakeStreamResponse(200, [b"aa", b"bb", b"cc"])  # 6 bytes > 4
    with mock.patch.object(audio.httpx, "Client", return_value=_FakeClient(resp)):
        with pytest.raises(audio.DownloadError) as ei:
            audio.download("https://example.com/big", str(tmp_path))
    assert ei.value.code == ErrorCode.DOWNLOAD_FAILED
    assert "size" in str(ei.value).lower()


def test_download_content_length_over_cap_fails_before_streaming(tmp_path, monkeypatch):
    # An advertised over-size body is rejected up front; the body is never read.
    monkeypatch.setattr(config, "MAX_DOWNLOAD_BYTES", 4)

    streamed = []

    class _RecordingResponse(_FakeStreamResponse):
        def iter_bytes(self):
            streamed.append(True)
            yield from self._chunks

    resp = _RecordingResponse(200, [b"x"], headers={"Content-Length": "999"})
    with mock.patch.object(audio.httpx, "Client", return_value=_FakeClient(resp)):
        with pytest.raises(audio.DownloadError) as ei:
            audio.download("https://example.com/big", str(tmp_path))
    assert ei.value.code == ErrorCode.DOWNLOAD_FAILED
    assert streamed == []  # aborted before reading the body


def test_download_timeout_maps_to_download_error(tmp_path):
    client = _FakeClient(exc=httpx.TimeoutException("boom"))
    with mock.patch.object(audio.httpx, "Client", return_value=client):
        with pytest.raises(audio.DownloadError) as ei:
            audio.download("https://example.com/slow", str(tmp_path))
    assert ei.value.code == ErrorCode.DOWNLOAD_FAILED


def test_download_connection_error_maps_to_download_error(tmp_path):
    client = _FakeClient(exc=httpx.ConnectError("no route"))
    with mock.patch.object(audio.httpx, "Client", return_value=client):
        with pytest.raises(audio.DownloadError) as ei:
            audio.download("https://example.com/x", str(tmp_path))
    assert ei.value.code == ErrorCode.DOWNLOAD_FAILED


# --------------------------------------------------------------------------- #
# probe_duration
# --------------------------------------------------------------------------- #
def _completed(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_probe_duration_success():
    with mock.patch.object(audio, "_run", return_value=_completed(0, "123.45\n")):
        assert audio.probe_duration("/x.wav") == pytest.approx(123.45)


def test_probe_duration_nonzero_exit():
    with mock.patch.object(audio, "_run", return_value=_completed(1, "", "bad")):
        with pytest.raises(audio.FFmpegError) as ei:
            audio.probe_duration("/x.wav")
    assert ei.value.code == ErrorCode.FFMPEG_FAILED


def test_probe_duration_unparseable():
    with mock.patch.object(audio, "_run", return_value=_completed(0, "N/A\n")):
        with pytest.raises(audio.FFmpegError) as ei:
            audio.probe_duration("/x.wav")
    assert ei.value.code == ErrorCode.FFMPEG_FAILED


def test_probe_duration_non_positive():
    with mock.patch.object(audio, "_run", return_value=_completed(0, "0\n")):
        with pytest.raises(audio.FFmpegError) as ei:
            audio.probe_duration("/x.wav")
    assert ei.value.code == ErrorCode.FFMPEG_FAILED


# --------------------------------------------------------------------------- #
# normalize
# --------------------------------------------------------------------------- #
def test_normalize_success(tmp_path):
    with mock.patch.object(audio, "_run", return_value=_completed(0)) as run:
        out = audio.normalize("/src.mp3", str(tmp_path))
    assert out == os.path.join(str(tmp_path), "audio_16khz_mono.wav")
    cmd = run.call_args.args[0]
    assert cmd[0] == "ffmpeg" and "-nostdin" in cmd
    assert "pcm_s16le" in cmd and "16000" in cmd


def test_normalize_decode_failure_is_unsupported_format(tmp_path):
    with mock.patch.object(audio, "_run", return_value=_completed(1, "", "cannot decode")):
        with pytest.raises(audio.FFmpegError) as ei:
            audio.normalize("/src.xyz", str(tmp_path))
    assert ei.value.code == ErrorCode.UNSUPPORTED_FORMAT


def test_normalize_missing_binary_is_ffmpeg_failed(tmp_path):
    with mock.patch.object(audio, "_run", side_effect=FileNotFoundError("ffmpeg")):
        with pytest.raises(audio.FFmpegError) as ei:
            audio.normalize("/src.mp3", str(tmp_path))
    assert ei.value.code == ErrorCode.FFMPEG_FAILED


# --------------------------------------------------------------------------- #
# extract_window
# --------------------------------------------------------------------------- #
def test_extract_window_success(tmp_path):
    out_path = os.path.join(str(tmp_path), "chunk0.wav")
    with mock.patch.object(audio, "_run", return_value=_completed(0)) as run:
        out = audio.extract_window("/full.wav", 10.0, 20.0, out_path)
    assert out == out_path
    cmd = run.call_args.args[0]
    assert "-ss" in cmd and "10.0" in cmd and "-t" in cmd and "20.0" in cmd


def test_extract_window_failure_is_ffmpeg_failed(tmp_path):
    out_path = os.path.join(str(tmp_path), "chunk0.wav")
    with mock.patch.object(audio, "_run", return_value=_completed(1, "", "err")):
        with pytest.raises(audio.FFmpegError) as ei:
            audio.extract_window("/full.wav", 0.0, 5.0, out_path)
    assert ei.value.code == ErrorCode.FFMPEG_FAILED
