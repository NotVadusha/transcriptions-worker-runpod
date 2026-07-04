"""Quality safeguards in src.transcribe — no GPU/model required."""

import os

# Keep transcribe importable on CPU-only test machines.
os.environ["SKIP_MODEL_LOAD"] = "1"

from src import audio, config, transcribe


def _words(*triples):
    return [{"start": s, "end": e, "word": w} for (s, e, w) in triples]


def _segments(*triples):
    return [{"start": s, "end": e, "text": t} for (s, e, t) in triples]


def test_find_internal_gaps_uses_configured_threshold(monkeypatch):
    monkeypatch.setattr(config, "GAP_RETRY_ENABLED", True)
    monkeypatch.setattr(config, "GAP_RETRY_MIN_SEC", 20)

    items = _segments(
        (0.0, 5.0, "first"),
        (8.0, 10.0, "nearby"),
        (35.0, 36.0, "far"),
    )

    assert transcribe._find_internal_gaps(items) == [(10.0, 35.0)]


def test_retry_internal_gaps_splices_only_core_retry_items(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GAP_RETRY_ENABLED", True)
    monkeypatch.setattr(config, "GAP_RETRY_MIN_SEC", 20)
    monkeypatch.setattr(config, "GAP_RETRY_PADDING_SEC", 5)
    monkeypatch.setattr(config, "GAP_RETRY_MAX_SEC", 300)

    extracted = []
    attention_modes = []

    def fake_extract_window(wav_path, start, duration, out_path):
        extracted.append((wav_path, start, duration, out_path))
        return out_path

    def fake_set_attention(mode):
        attention_modes.append(mode)

    def fake_transcribe_one(path):
        assert path.endswith("gap_retry_0000.wav")
        return (
            "ignored",
            _words(
                (2.0, 3.0, "edge-before"),     # abs 7-8, outside core gap
                (15.0, 16.0, "recovered"),     # abs 20-21, inside core gap
                (38.0, 39.0, "edge-after"),    # abs 43-44, outside core gap
            ),
            _segments(
                (2.0, 3.0, "edge before"),
                (15.0, 16.0, "recovered text"),
                (38.0, 39.0, "edge after"),
            ),
        )

    monkeypatch.setattr(audio, "extract_window", fake_extract_window)
    monkeypatch.setattr(transcribe, "_set_attention", fake_set_attention)
    monkeypatch.setattr(transcribe, "_transcribe_one", fake_transcribe_one)

    merged = {
        "text": "left right",
        "words": _words((0.0, 10.0, "left"), (40.0, 50.0, "right")),
        "segments": _segments((0.0, 10.0, "left"), (40.0, 50.0, "right")),
    }

    out = transcribe._retry_internal_gaps(
        "/tmp/source.wav", 100.0, str(tmp_path), merged
    )

    assert extracted == [
        (
            "/tmp/source.wav",
            5.0,
            40.0,
            str(tmp_path / "gap_retry_0000.wav"),
        )
    ]
    assert attention_modes == ["global"]
    assert out["gap_retry_count"] == 1
    assert out["gap_retry_recovered"] == 1
    assert [w["word"] for w in out["words"]] == ["left", "recovered", "right"]
    assert [s["text"] for s in out["segments"]] == [
        "left",
        "recovered text",
        "right",
    ]
    assert out["text"] == "left recovered text right"


def test_chunked_run_uses_global_attention_for_manageable_chunks(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TRANSCRIPTION_QUALITY", "balanced")
    monkeypatch.setattr(config, "SINGLE_PASS_MAX_SEC", 120)
    monkeypatch.setattr(config, "CHUNK_SEC", 60)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_SEC", 10)
    monkeypatch.setattr(config, "GAP_RETRY_ENABLED", False)

    attention_modes = []
    extracted = []

    def fake_set_attention(mode):
        attention_modes.append(mode)

    def fake_extract_window(wav_path, start, duration, out_path):
        extracted.append((start, duration, out_path))
        return out_path

    def fake_transcribe_one(path):
        index = int(os.path.basename(path).split("_")[1].split(".")[0])
        return (
            f"chunk {index}",
            _words((0.0, 1.0, f"w{index}")),
            _segments((0.0, 1.0, f"chunk {index}")),
        )

    monkeypatch.setattr(transcribe, "_set_attention", fake_set_attention)
    monkeypatch.setattr(audio, "extract_window", fake_extract_window)
    monkeypatch.setattr(transcribe, "_transcribe_one", fake_transcribe_one)

    out = transcribe.run(str(tmp_path / "audio.wav"), 130.0, True)

    assert attention_modes == ["global"]
    assert len(extracted) == 3
    assert out["chunked"] is True
    assert out["num_chunks"] == 3
    assert out["gap_retry_count"] == 0


def test_fast_quality_uses_local_attention_for_chunked_path(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TRANSCRIPTION_QUALITY", "fast")
    monkeypatch.setattr(config, "SINGLE_PASS_MAX_SEC", 120)
    monkeypatch.setattr(config, "CHUNK_SEC", 60)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_SEC", 10)
    monkeypatch.setattr(config, "GAP_RETRY_ENABLED", False)

    attention_modes = []

    monkeypatch.setattr(transcribe, "_set_attention", attention_modes.append)
    monkeypatch.setattr(audio, "extract_window", lambda *args: args[-1])
    monkeypatch.setattr(
        transcribe,
        "_transcribe_one",
        lambda path: ("ok", _words((0.0, 1.0, "ok")), _segments((0.0, 1.0, "ok"))),
    )

    transcribe.run(str(tmp_path / "audio.wav"), 130.0, True)

    assert attention_modes == ["local"]
