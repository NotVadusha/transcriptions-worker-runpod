"""Tests for src.storage.maybe_offload (SPEC §8.3).

These tests never touch the network. ``httpx`` is not a test dependency, so we
inject a fake ``httpx`` module into ``sys.modules`` to satisfy the lazy
``import httpx`` inside ``maybe_offload`` and to record/script the PUT call.
"""

from __future__ import annotations

import sys
import types

import pytest

from src import config, storage


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Skip the real retry backoff so the upload-failure tests stay fast."""
    import time

    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)


def _make_output(extra_words: int = 0) -> dict:
    """Build a §3.2-shaped success object; ``extra_words`` pads its size."""
    return {
        "text": "hello world",
        "words": [{"start": 0.0, "end": 0.5, "word": "padding"} for _ in range(extra_words)],
        "segments": [{"start": 0.0, "end": 0.5, "text": "hello world"}],
        "meta": {
            "model": config.MODEL_NAME,
            "language": "en",
            "audio_duration_sec": 1.0,
            "processing_time_sec": 0.1,
            "rtf": 10.0,
            "chunked": False,
            "num_chunks": 1,
            "worker_version": config.WORKER_VERSION,
        },
    }


class _FakeResponse:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        # Mirror httpx semantics: raise on 4xx/5xx, no-op otherwise.
        if self.status_code >= 400:
            raise _FakeHTTPStatusError(f"HTTP {self.status_code}")


class _FakeHTTPStatusError(Exception):
    pass


def _install_fake_httpx(monkeypatch, *, status_code=None, raise_exc=None):
    """Install a fake ``httpx`` module exposing ``put``; return a call recorder.

    The recorder is a list that receives a kwargs dict for each ``put`` call.
    """
    calls: list[dict] = []

    def fake_put(url, **kwargs):
        calls.append({"url": url, **kwargs})
        if raise_exc is not None:
            raise raise_exc
        return _FakeResponse(status_code)

    fake_module = types.ModuleType("httpx")
    fake_module.put = fake_put
    monkeypatch.setitem(sys.modules, "httpx", fake_module)
    return calls


# --------------------------------------------------------------------------- #
# Under-threshold passthrough
# --------------------------------------------------------------------------- #
def test_under_threshold_returns_output_unchanged(monkeypatch):
    # If httpx were touched it would raise (no real module), proving no upload.
    monkeypatch.setitem(sys.modules, "httpx", None)
    output = _make_output()

    result = storage.maybe_offload(output, result_upload_url=None)

    assert result is output  # returned unchanged, same object


def test_under_threshold_with_url_still_inline(monkeypatch):
    calls = _install_fake_httpx(monkeypatch, status_code=200)
    output = _make_output()

    result = storage.maybe_offload(output, "https://bucket.example.com/obj?sig=abc")

    assert result is output
    assert calls == []  # no upload attempted


# --------------------------------------------------------------------------- #
# Over-threshold + no URL -> RESULT_TOO_LARGE
# --------------------------------------------------------------------------- #
def test_over_threshold_no_url_returns_result_too_large(monkeypatch):
    monkeypatch.setattr(config, "RESULT_OFFLOAD_THRESHOLD_BYTES", 50)
    output = _make_output(extra_words=20)  # well over 50 bytes

    result = storage.maybe_offload(output, result_upload_url=None)

    assert result["error"]["code"] == config.ErrorCode.RESULT_TOO_LARGE
    assert "result_upload_url" in result["error"]["message"]
    assert "text" not in result  # not an offloaded reference


# --------------------------------------------------------------------------- #
# Over-threshold + URL + 200 -> offloaded reference
# --------------------------------------------------------------------------- #
def test_over_threshold_with_url_200_offloads(monkeypatch):
    monkeypatch.setattr(config, "RESULT_OFFLOAD_THRESHOLD_BYTES", 50)
    calls = _install_fake_httpx(monkeypatch, status_code=200)
    output = _make_output(extra_words=20)
    url = "https://bucket.example.com/path/obj.json?X-Sig=secret&Expires=123"

    result = storage.maybe_offload(output, url)

    # Reference object shape (query stripped from result_url).
    assert result == {
        "result_url": "https://bucket.example.com/path/obj.json",
        "text": output["text"],
        "meta": output["meta"],
        "offloaded": True,
    }

    # Exactly one PUT of the FULL serialized object with the JSON content type.
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == url
    assert call["headers"]["Content-Type"] == "application/json"
    import json

    sent = json.loads(call["content"].decode("utf-8"))
    assert sent == output  # uploaded JSON is the complete §3.2 object
    assert "words" in sent and "segments" in sent


def test_over_threshold_with_url_201_offloads(monkeypatch):
    # Any 2xx (e.g. S3 PUT returns 200/201) counts as success.
    monkeypatch.setattr(config, "RESULT_OFFLOAD_THRESHOLD_BYTES", 50)
    _install_fake_httpx(monkeypatch, status_code=201)
    output = _make_output(extra_words=20)

    result = storage.maybe_offload(output, "https://bucket.example.com/obj?sig=x")

    assert result["offloaded"] is True


# --------------------------------------------------------------------------- #
# Over-threshold + URL + 500 -> RESULT_UPLOAD_FAILED
# --------------------------------------------------------------------------- #
def test_over_threshold_with_url_500_retries_then_upload_failed(monkeypatch):
    # 5xx may be transient -> retried up to 3 times, then RESULT_UPLOAD_FAILED.
    monkeypatch.setattr(config, "RESULT_OFFLOAD_THRESHOLD_BYTES", 50)
    calls = _install_fake_httpx(monkeypatch, status_code=500)
    output = _make_output(extra_words=20)

    result = storage.maybe_offload(output, "https://bucket.example.com/obj?sig=x")

    assert result["error"]["code"] == config.ErrorCode.RESULT_UPLOAD_FAILED
    assert "result_url" not in result
    assert len(calls) == 3  # retried


def test_over_threshold_with_url_4xx_no_retry(monkeypatch):
    # 4xx is a caller fault (e.g. expired/invalid presigned URL) -> no retry.
    monkeypatch.setattr(config, "RESULT_OFFLOAD_THRESHOLD_BYTES", 50)
    calls = _install_fake_httpx(monkeypatch, status_code=403)
    output = _make_output(extra_words=20)

    result = storage.maybe_offload(output, "https://bucket.example.com/obj?sig=x")

    assert result["error"]["code"] == config.ErrorCode.RESULT_UPLOAD_FAILED
    assert len(calls) == 1  # not retried


def test_over_threshold_with_url_3xx_is_failure_not_silent_success(monkeypatch):
    # We do NOT follow redirects on the PUT, so a 3xx must be reported as a
    # failure rather than silently treated as a successful upload.
    monkeypatch.setattr(config, "RESULT_OFFLOAD_THRESHOLD_BYTES", 50)
    calls = _install_fake_httpx(monkeypatch, status_code=302)
    output = _make_output(extra_words=20)

    result = storage.maybe_offload(output, "https://bucket.example.com/obj?sig=x")

    assert result["error"]["code"] == config.ErrorCode.RESULT_UPLOAD_FAILED
    assert "offloaded" not in result
    assert len(calls) == 1  # 3xx treated like 4xx: not retried


def test_over_threshold_with_url_transport_error_returns_upload_failed(monkeypatch):
    # A connection/timeout error also maps to RESULT_UPLOAD_FAILED, and the
    # presigned signature must NOT leak into the returned message.
    monkeypatch.setattr(config, "RESULT_OFFLOAD_THRESHOLD_BYTES", 50)
    _install_fake_httpx(
        monkeypatch, raise_exc=RuntimeError("connection to ...?X-Sig=topsecret failed")
    )
    output = _make_output(extra_words=20)
    url = "https://bucket.example.com/path/obj.json?X-Sig=topsecret&Expires=123"

    result = storage.maybe_offload(output, url)

    assert result["error"]["code"] == config.ErrorCode.RESULT_UPLOAD_FAILED
    msg = result["error"]["message"]
    # Signature is never echoed back; only the stripped object location appears.
    assert "topsecret" not in msg
    assert "X-Sig" not in msg
    assert "https://bucket.example.com/path/obj.json" in msg
