"""Unit tests for src.audio.

All subprocess (ffmpeg/ffprobe) interactions are mocked, so these tests require
neither a network connection nor ffmpeg installed and run under Python 3.13 with
only stdlib + pytest.
"""

from __future__ import annotations

import os
import subprocess
from unittest import mock

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
# streaming helpers (URL detection, ffmpeg net opts, error classification)
# --------------------------------------------------------------------------- #
def _completed(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_is_url_vs_path():
    assert audio._is_url("https://x.com/a.wav")
    assert audio._is_url("http://x.com/a.wav")
    assert not audio._is_url("/tmp/a.wav")
    assert not audio._is_url("a.wav")


def test_network_input_opts_uses_rw_timeout_and_reconnect(monkeypatch):
    monkeypatch.setattr(config, "DOWNLOAD_TIMEOUT_SEC", 30)
    opts = audio._network_input_opts()
    assert "-rw_timeout" in opts
    # 30s -> 30_000_000 microseconds
    assert opts[opts.index("-rw_timeout") + 1] == "30000000"
    assert "-reconnect" in opts and "-reconnect_streamed" in opts
    # The flag that actually recovers a mid-stream drop on a seekable S3 URL.
    assert "-reconnect_on_network_error" in opts


@pytest.mark.parametrize(
    "stderr",
    [
        "Server returned 403 Forbidden",
        "HTTP error 404 Not Found",
        "Connection refused",
        "Failed to resolve hostname",
        "Operation timed out",
        "Read timeout on socket",  # our own -rw_timeout expiry must classify as network
        "error:SSL routines:ssl3_read_bytes:handshake failure",
    ],
)
def test_looks_like_network_error_true(stderr):
    assert audio._looks_like_network_error(stderr)


@pytest.mark.parametrize(
    "stderr",
    [
        "Invalid data found when processing input",
        "moov atom not found",
        "Could not find codec parameters",
        "",
    ],
)
def test_looks_like_network_error_false(stderr):
    assert not audio._looks_like_network_error(stderr)


# --------------------------------------------------------------------------- #
# probe_duration
# --------------------------------------------------------------------------- #
def test_probe_duration_success():
    with mock.patch.object(audio, "_run", return_value=_completed(0, "123.45\n")):
        assert audio.probe_duration("/x.wav") == pytest.approx(123.45)


def test_probe_duration_url_adds_network_opts():
    with mock.patch.object(audio, "_run", return_value=_completed(0, "5.0\n")) as run:
        audio.probe_duration("https://s3.example.com/audio.m4a?sig=abc")
    cmd = run.call_args.args[0]
    assert cmd[0] == "ffprobe"
    assert "-rw_timeout" in cmd
    assert cmd[-1] == "https://s3.example.com/audio.m4a?sig=abc"


def test_probe_duration_local_path_has_no_network_opts():
    with mock.patch.object(audio, "_run", return_value=_completed(0, "5.0\n")) as run:
        audio.probe_duration("/x.wav")
    assert "-rw_timeout" not in run.call_args.args[0]


def test_probe_duration_url_network_error_is_download_error():
    with mock.patch.object(
        audio, "_run", return_value=_completed(1, "", "Server returned 403 Forbidden")
    ):
        with pytest.raises(audio.DownloadError) as ei:
            audio.probe_duration("https://s3.example.com/a.wav?sig=x")
    assert ei.value.code == ErrorCode.DOWNLOAD_FAILED


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
    # Defense-in-depth output-length cap == MAX_AUDIO_SECONDS.
    assert cmd[cmd.index("-t") + 1] == str(config.MAX_AUDIO_SECONDS)


def test_normalize_url_streams_directly_with_network_opts(tmp_path):
    with mock.patch.object(audio, "_run", return_value=_completed(0)) as run:
        out = audio.normalize("https://s3.example.com/a.mp4?sig=x", str(tmp_path))
    assert out == os.path.join(str(tmp_path), "audio_16khz_mono.wav")
    cmd = run.call_args.args[0]
    assert cmd[0] == "ffmpeg" and "-rw_timeout" in cmd
    # -vn keeps only audio (video URLs are demuxed to audio by ffmpeg).
    assert "-vn" in cmd
    # The URL is the ffmpeg input.
    assert cmd[cmd.index("-i") + 1] == "https://s3.example.com/a.mp4?sig=x"


def test_normalize_decode_failure_is_unsupported_format(tmp_path):
    with mock.patch.object(audio, "_run", return_value=_completed(1, "", "cannot decode")):
        with pytest.raises(audio.FFmpegError) as ei:
            audio.normalize("/src.xyz", str(tmp_path))
    assert ei.value.code == ErrorCode.UNSUPPORTED_FORMAT


def test_normalize_url_network_error_is_download_error(tmp_path):
    with mock.patch.object(
        audio, "_run", return_value=_completed(1, "", "Connection timed out")
    ):
        with pytest.raises(audio.DownloadError) as ei:
            audio.normalize("https://s3.example.com/a.wav?sig=x", str(tmp_path))
    assert ei.value.code == ErrorCode.DOWNLOAD_FAILED


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
