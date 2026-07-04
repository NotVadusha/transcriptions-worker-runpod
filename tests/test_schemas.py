"""Unit tests for src.schemas — pure logic, no GPU required."""

from __future__ import annotations

import pytest

from src import config
from src.schemas import (
    Request,
    ValidationError,
    build_output,
    parse_request,
)


# --------------------------------------------------------------------------- #
# parse_request — audio_url
# --------------------------------------------------------------------------- #
def test_missing_audio_url_raises_missing_code():
    with pytest.raises(ValidationError) as exc:
        parse_request({})
    assert exc.value.code == config.ErrorCode.MISSING_AUDIO_URL


def test_non_string_audio_url_raises_missing_code():
    with pytest.raises(ValidationError) as exc:
        parse_request({"audio_url": 12345})
    assert exc.value.code == config.ErrorCode.MISSING_AUDIO_URL


def test_empty_audio_url_raises_missing_code():
    with pytest.raises(ValidationError) as exc:
        parse_request({"audio_url": "   "})
    assert exc.value.code == config.ErrorCode.MISSING_AUDIO_URL


def test_non_https_audio_url_raises_invalid_url():
    with pytest.raises(ValidationError) as exc:
        parse_request({"audio_url": "http://example.com/a.wav"})
    assert exc.value.code == config.ErrorCode.INVALID_URL


def test_audio_url_no_netloc_raises_invalid_url():
    with pytest.raises(ValidationError) as exc:
        parse_request({"audio_url": "https:///no-host"})
    assert exc.value.code == config.ErrorCode.INVALID_URL


def test_https_audio_url_ok():
    req = parse_request({"audio_url": "https://example.com/audio.wav"})
    assert isinstance(req, Request)
    assert req.audio_url == "https://example.com/audio.wav"
    # Defaults applied.
    assert req.return_timestamps is True
    assert req.language == config.DEFAULT_LANGUAGE
    assert req.result_upload_url is None


def test_non_dict_input_raises():
    with pytest.raises(ValidationError) as exc:
        parse_request("not a dict")  # type: ignore[arg-type]
    assert exc.value.code == config.ErrorCode.MISSING_AUDIO_URL


# --------------------------------------------------------------------------- #
# parse_request — return_timestamps coercion
# --------------------------------------------------------------------------- #
def test_return_timestamps_false_preserved():
    req = parse_request(
        {"audio_url": "https://x.com/a.wav", "return_timestamps": False}
    )
    assert req.return_timestamps is False


def test_return_timestamps_true_preserved():
    req = parse_request(
        {"audio_url": "https://x.com/a.wav", "return_timestamps": True}
    )
    assert req.return_timestamps is True


@pytest.mark.parametrize("bad", ["false", 0, 1, None, "yes", []])
def test_return_timestamps_non_bool_coerced_to_true(bad):
    # Non-bool values must coerce to True, never raise (SPEC §3.1).
    req = parse_request(
        {"audio_url": "https://x.com/a.wav", "return_timestamps": bad}
    )
    assert req.return_timestamps is True


# --------------------------------------------------------------------------- #
# parse_request — language
# --------------------------------------------------------------------------- #
def test_language_en_ok():
    req = parse_request({"audio_url": "https://x.com/a.wav", "language": "en"})
    assert req.language == "en"


def test_unsupported_language_raises():
    with pytest.raises(ValidationError) as exc:
        parse_request({"audio_url": "https://x.com/a.wav", "language": "fr"})
    assert exc.value.code == config.ErrorCode.UNSUPPORTED_LANGUAGE


def test_language_absent_defaults_to_en():
    req = parse_request({"audio_url": "https://x.com/a.wav"})
    assert req.language == config.DEFAULT_LANGUAGE


def test_language_none_defaults_to_en():
    req = parse_request({"audio_url": "https://x.com/a.wav", "language": None})
    assert req.language == config.DEFAULT_LANGUAGE


# --------------------------------------------------------------------------- #
# parse_request — result_upload_url
# --------------------------------------------------------------------------- #
def test_result_upload_url_https_ok():
    req = parse_request(
        {
            "audio_url": "https://x.com/a.wav",
            "result_upload_url": "https://x.com/put",
        }
    )
    assert req.result_upload_url == "https://x.com/put"


def test_result_upload_url_non_https_raises_invalid_url():
    with pytest.raises(ValidationError) as exc:
        parse_request(
            {
                "audio_url": "https://x.com/a.wav",
                "result_upload_url": "http://x.com/put",
            }
        )
    assert exc.value.code == config.ErrorCode.INVALID_URL


def test_result_upload_url_absent_is_none():
    req = parse_request({"audio_url": "https://x.com/a.wav"})
    assert req.result_upload_url is None


# --------------------------------------------------------------------------- #
# ValidationError behavior
# --------------------------------------------------------------------------- #
def test_validation_error_str_is_message():
    e = ValidationError(config.ErrorCode.INVALID_URL, "bad url")
    assert str(e) == "bad url"
    assert e.code == config.ErrorCode.INVALID_URL
    assert e.message == "bad url"


# --------------------------------------------------------------------------- #
# build_output
# --------------------------------------------------------------------------- #
def _sample_result():
    return {
        "text": "hello world",
        "words": [{"start": 0.0, "end": 0.4, "word": "hello"}],
        "segments": [{"start": 0.0, "end": 0.9, "text": "hello world"}],
        "chunked": False,
        "num_chunks": 1,
    }


def test_build_output_text_always_present_with_timestamps():
    req = Request("https://x.com/a.wav", True, "en", None)
    out = build_output(_sample_result(), duration=10.0, processing_time=2.0, req=req)
    assert out["text"] == "hello world"
    assert "segments" in out
    assert "words" in out
    assert out["words"] == [{"start": 0.0, "end": 0.4, "word": "hello"}]
    assert out["segments"] == [{"start": 0.0, "end": 0.9, "text": "hello world"}]


def test_build_output_omits_timestamps_when_disabled():
    req = Request("https://x.com/a.wav", False, "en", None)
    out = build_output(_sample_result(), duration=10.0, processing_time=2.0, req=req)
    assert out["text"] == "hello world"
    # Keys must be omitted entirely — not null, not [].
    assert "segments" not in out
    assert "words" not in out


def test_build_output_meta_fields_and_rtf_math():
    req = Request("https://x.com/a.wav", True, "en", None)
    out = build_output(
        _sample_result(), duration=120.0, processing_time=8.0, req=req
    )
    meta = out["meta"]
    assert meta["model"] == config.MODEL_NAME
    assert meta["language"] == "en"
    assert meta["audio_duration_sec"] == 120.0
    assert meta["processing_time_sec"] == 8.0
    # rtf = duration / processing_time = 120 / 8 = 15.0
    assert meta["rtf"] == 15.0
    assert meta["chunked"] is False
    assert meta["num_chunks"] == 1
    assert meta["worker_version"] == config.WORKER_VERSION


def test_build_output_rtf_rounding():
    req = Request("https://x.com/a.wav", True, "en", None)
    out = build_output(
        _sample_result(), duration=10.0, processing_time=3.0, req=req
    )
    # 10 / 3 = 3.333... -> rounded to 3 places.
    assert out["meta"]["rtf"] == 3.333


def test_build_output_rtf_zero_processing_time_guard():
    req = Request("https://x.com/a.wav", True, "en", None)
    out = build_output(
        _sample_result(), duration=10.0, processing_time=0.0, req=req
    )
    assert out["meta"]["rtf"] == 0.0


def test_build_output_rtf_negative_processing_time_guard():
    req = Request("https://x.com/a.wav", True, "en", None)
    out = build_output(
        _sample_result(), duration=10.0, processing_time=-1.0, req=req
    )
    assert out["meta"]["rtf"] == 0.0


def test_build_output_propagates_chunked_meta():
    result = _sample_result()
    result["chunked"] = True
    result["num_chunks"] = 5
    req = Request("https://x.com/a.wav", True, "en", None)
    out = build_output(result, duration=3600.0, processing_time=200.0, req=req)
    assert out["meta"]["chunked"] is True
    assert out["meta"]["num_chunks"] == 5


def test_build_output_propagates_gap_retry_meta():
    result = _sample_result()
    result["gap_retry_count"] = 2
    result["gap_retry_recovered"] = 1
    req = Request("https://x.com/a.wav", True, "en", None)
    out = build_output(result, duration=3600.0, processing_time=200.0, req=req)
    assert out["meta"]["gap_retry_count"] == 2
    assert out["meta"]["gap_retry_recovered"] == 1


def test_build_output_duration_rounding():
    req = Request("https://x.com/a.wav", True, "en", None)
    out = build_output(
        _sample_result(), duration=123.456789, processing_time=8.123456, req=req
    )
    assert out["meta"]["audio_duration_sec"] == 123.457
    assert out["meta"]["processing_time_sec"] == 8.123
