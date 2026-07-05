"""Non-Parakeet ASR backends: Canary (NeMo), SenseVoice (FunASR), Whisper (HF).

Selected by language in :func:`transcribe.run` via :func:`config.route_backend`.
Each backend is loaded LAZILY on first use for its language and cached in
``_MODELS`` for the life of the worker process (the user's "lazy per-language"
choice) — so a worker that only ever sees English never imports funasr /
transformers or downloads Canary/Whisper/SenseVoice.

Like ``transcribe.py``, every heavy import (torch / nemo / funasr / transformers)
is performed INSIDE a function, so this module imports cleanly on a GPU-free CI
box. Nothing here runs under ``SKIP_MODEL_LOAD`` — the tests exercise the
Parakeet path and the pure routing/chunking logic only.

Contract: ``run(backend, wav, duration, language)`` returns the same result dict
as ``transcribe._run_parakeet`` (minus ``model``, which the caller stamps on).
Long audio reuses the model-agnostic chunk planner/stitcher in ``chunking.py``.

MUST-VALIDATE-ON-GPU (RESEARCH.md §5 convention): the exact model call signatures
below — Canary ``source_lang``/``target_lang``/``pnc``/``timestamps`` kwargs,
SenseVoice ``AutoModel``/``generate`` kwargs + HF hub id, and the Whisper
pipeline ``generate_kwargs`` language token + ``chunks`` return shape — are coded
from the model cards/docs and have NOT been run on a GPU here. Confirm each one
inside the container before trusting it in production.
"""

from __future__ import annotations

import os

from src import audio, chunking, config

__all__ = ["run"]

# One cached instance per backend, loaded on first use and kept warm.
_MODELS: dict = {}


# --------------------------------------------------------------------------- #
# Lazy loaders (one per backend)
# --------------------------------------------------------------------------- #
def _canary():
    """Load nvidia/canary-1b-v2 once (NeMo AED multitask model)."""
    if "canary" in _MODELS:
        return _MODELS["canary"]
    import nemo.collections.asr as nemo_asr  # noqa: WPS433 (intentional local import)

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=config.CANARY_MODEL)
    model.eval()
    _MODELS["canary"] = model
    return model


def _sensevoice():
    """Load FunAudioLLM/SenseVoiceSmall once via FunASR."""
    if "sensevoice" in _MODELS:
        return _MODELS["sensevoice"]
    import torch  # noqa: WPS433
    from funasr import AutoModel  # noqa: WPS433

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel(
        model=config.SENSEVOICE_MODEL,
        hub="hf",  # pull the HF checkpoint id, not the ModelScope one
        trust_remote_code=True,
        disable_update=True,
        device=device,
    )
    _MODELS["sensevoice"] = model
    return model


def _whisper():
    """Load openai/whisper-large-v3 once as a transformers ASR pipeline.

    ``chunk_length_s=30`` is Whisper's own internal long-form chunking; combined
    with our per-chunk extraction it bounds memory on very long audio.
    """
    if "whisper" in _MODELS:
        return _MODELS["whisper"]
    import torch  # noqa: WPS433
    from transformers import pipeline  # noqa: WPS433

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = pipeline(
        "automatic-speech-recognition",
        model=config.WHISPER_MODEL,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=30,
        return_timestamps="word",
    )
    _MODELS["whisper"] = pipe
    return pipe


# --------------------------------------------------------------------------- #
# Per-file transcription (chunk-local seconds, matching transcribe._transcribe_one)
# --------------------------------------------------------------------------- #
def _canary_one(wav_path: str, language: str):
    """Transcribe one file with Canary; reuse the NeMo timestamp mapping."""
    import torch  # noqa: WPS433

    from src import transcribe  # lazy: avoid import cycle (transcribe imports us)

    model = _canary()
    with torch.inference_mode():
        # ASR (not translation): source_lang == target_lang. pnc=True keeps
        # punctuation/capitalization. MUST-VALIDATE-ON-GPU (kwargs vary by NeMo).
        hyps = model.transcribe(
            [wav_path],
            source_lang=language,
            target_lang=language,
            pnc=True,
            timestamps=True,
        )
    if not hyps:
        raise transcribe.TranscriptionError("Canary transcribe() returned nothing.")
    hyp = hyps[0]
    ts = getattr(hyp, "timestamp", None) or {}
    words = transcribe._map_words(ts.get("word"))
    segments = transcribe._map_segments(ts.get("segment"))
    return (hyp.text or ""), words, segments


def _sensevoice_one(wav_path: str, language: str):
    """Transcribe one file with SenseVoice. Text only — no word timestamps."""
    from funasr.utils.postprocess_utils import (  # noqa: WPS433
        rich_transcription_postprocess,
    )

    model = _sensevoice()
    # use_itn=True -> inverse text normalization + punctuation. language codes
    # (zh/yue/ja/ko) map straight through. MUST-VALIDATE-ON-GPU.
    res = model.generate(input=wav_path, cache={}, language=language, use_itn=True)
    text = rich_transcription_postprocess(res[0]["text"]) if res else ""
    return text, [], []


def _whisper_one(wav_path: str, language: str):
    """Transcribe one file with Whisper; map word chunks to our contract."""
    pipe = _whisper()
    out = pipe(
        wav_path,
        return_timestamps="word",
        generate_kwargs={"language": language, "task": "transcribe"},
    )
    words = []
    for chunk in out.get("chunks", []):
        start, end = (chunk.get("timestamp") or (None, None))[:2]
        if start is None:
            continue  # Whisper occasionally emits a null-start fragment; skip it.
        words.append(
            {
                "start": float(start),
                "end": float(end if end is not None else start),
                "word": (chunk.get("text") or "").strip(),
            }
        )
    return (out.get("text") or "").strip(), words, []


_ONE = {
    "canary": _canary_one,
    "sensevoice": _sensevoice_one,
    "whisper": _whisper_one,
}


# --------------------------------------------------------------------------- #
# Dispatch + generic (model-agnostic) long-audio path
# --------------------------------------------------------------------------- #
def run(backend: str, wav_path: str, duration: float, language: str) -> dict:
    """Transcribe ``wav_path`` with ``backend``, chunking long audio generically.

    Mirrors ``transcribe._run_parakeet`` minus Parakeet's attention switching and
    gap-retry (those are Parakeet-tuned).
    ponytail: no per-backend gap-retry; add if long-audio drops surface on these.
    """
    one_fn = _ONE[backend]

    if duration <= config.SINGLE_PASS_MAX_SEC:
        text, words, segments = _call(one_fn, wav_path, language, "single-pass")
        segments = _ensure_segments(text, words, segments, duration)
        return {
            "text": text,
            "words": words,
            "segments": segments,
            "chunked": False,
            "num_chunks": 1,
        }

    plans = chunking.plan_chunks(duration, config.CHUNK_SEC, config.CHUNK_OVERLAP_SEC)
    work_dir = os.path.dirname(os.path.abspath(wav_path))
    chunk_outputs: list[dict] = []
    chunk_paths: list[str] = []
    try:
        for plan in plans:
            chunk_path = os.path.join(work_dir, f"chunk_{plan.index:04d}.wav")
            chunk_paths.append(chunk_path)
            audio.extract_window(
                wav_path, plan.start, plan.end - plan.start, chunk_path
            )
            text, words, segments = _call(
                one_fn, chunk_path, language, f"chunk {plan.index}"
            )
            segments = _ensure_segments(text, words, segments, plan.end - plan.start)
            chunk_outputs.append(
                {"plan": plan, "words": words, "segments": segments}
            )
    finally:
        for path in chunk_paths:
            try:
                os.remove(path)
            except OSError:
                pass

    merged = chunking.stitch(chunk_outputs)
    return {
        "text": merged["text"],
        "words": merged["words"],
        "segments": merged["segments"],
        "chunked": True,
        "num_chunks": len(plans),
    }


def _ensure_segments(text, words, segments, local_end):
    """Synthesize a whole-window segment when a backend gave text but no timestamps.

    SenseVoice returns text only; without this the chunked stitcher (which builds
    the transcript from segments/words) would drop it. For backends that do emit
    timestamps this is a no-op.
    """
    if text and not words and not segments:
        return [{"start": 0.0, "end": float(local_end), "text": text}]
    return segments


def _call(one_fn, wav_path: str, language: str, where: str):
    """Run ``one_fn`` and wrap unexpected errors as TranscriptionError (-> FAILED)."""
    from src.transcribe import TranscriptionError  # lazy: avoid import cycle

    try:
        return one_fn(wav_path, language)
    except TranscriptionError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize any backend failure
        raise TranscriptionError(f"{where} transcription failed: {exc}") from exc
