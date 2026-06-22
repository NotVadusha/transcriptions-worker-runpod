"""NeMo Parakeet TDT transcription — single-pass and chunked paths (SPEC §5).

This is the ONLY module that depends on NeMo/torch. Those imports are LAZY
(performed inside :func:`load_model`), so importing this module never requires a
GPU or the NeMo toolkit. Combined with ``config.SKIP_MODEL_LOAD`` this lets the
handler and its tests import cleanly on a CPU-only / CI machine.

Module state (SPEC §12.3): the only state held across requests is the shared,
read-only model (``_model``) and the current encoder attention mode
(``_attention_mode``). No per-request state is retained.

NeMo API contract (confirmed in RESEARCH.md §2, §3):
  * ``asr_model.transcribe([wav], timestamps=True)`` returns a LIST of
    Hypothesis objects, one per input file. We pass one file -> use ``out[0]``.
  * ``hyp.text`` is the full transcript string.
  * ``hyp.timestamp`` is a dict keyed by level: ``"word"``, ``"segment"``,
    ``"char"`` (we ignore ``"char"`` in v0).
  * Per-entry keys (RESEARCH.md §2 table):
      - word level:    text key ``"word"``;    time keys ``"start"``/``"end"`` (seconds, float)
      - segment level: text key ``"segment"``; time keys ``"start"``/``"end"`` (seconds, float)
    Both levels also carry integer frame offsets ``"start_offset"``/``"end_offset"``.
    We prefer the seconds keys and fall back to frame offsets * time_stride only
    if the seconds keys are absent (defensive — see ``_seconds`` / ``_time_stride``).
"""

from __future__ import annotations

import os

from src import audio, chunking, config

__all__ = ["TranscriptionError", "load_model", "run"]


# Shared, read-only model handle. Loaded once via load_model() (idempotent).
_model = None

# Tracks the encoder attention mode currently configured on the model so the
# (expensive) change_attention_model() call is only issued when the mode
# actually changes. Set to "global" right after a successful load (the model
# loads in its default full-attention config), so a fresh model is NOT
# reconfigured for the first short request. None = no real model (test mode).
_attention_mode: str | None = None

# The encoder attention config captured AS LOADED, so the global path can
# restore exactly what the checkpoint shipped with instead of hardcoding a
# string. Populated in load_model(); used only to revert a local-attention
# switch. Fall back to "rel_pos" (FastConformer default) if the config can't be
# read — MUST-VALIDATE-ON-GPU (RESEARCH.md §5).
_default_self_attention: str | None = None
_default_att_context_size: list | None = None


class TranscriptionError(Exception):
    """Raised on unexpected inference failure (SPEC §3.3 TRANSCRIPTION_FAILED).

    The handler re-raises this so RunPod marks the job FAILED. Carries the
    machine-readable ``code`` and a human-readable ``message`` (``str(e)`` ==
    message).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = config.ErrorCode.TRANSCRIPTION_FAILED
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def load_model() -> None:
    """Load the NeMo ASR model once. Idempotent.

    If ``config.SKIP_MODEL_LOAD`` is truthy (TEST-ONLY escape hatch) this
    returns immediately WITHOUT importing NeMo/torch, so the module is usable on
    a machine with no GPU. Otherwise it imports NeMo + torch lazily, loads the
    pretrained model, switches it to eval mode, and sets the cuDNN benchmark
    flag (SPEC §5.5).

    Model-load failure is allowed to propagate (SPEC §3.3: an internal error at
    load time should fail the worker/job — do NOT swallow it).
    """
    global _model

    if _model is not None:
        return

    # TEST-ONLY: skip NeMo/torch entirely. Must be checked BEFORE importing nemo.
    if config.SKIP_MODEL_LOAD:
        return

    # Lazy, GPU-only imports — kept inside the function on purpose.
    import nemo.collections.asr as nemo_asr  # noqa: WPS433 (intentional local import)
    import torch  # noqa: WPS433

    # RESEARCH.md §2 / SPEC §5.2 — load the pretrained checkpoint once.
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=config.MODEL_NAME)
    model.eval()

    # SPEC §5.5 — throughput tweak for fixed-shape-ish workloads.
    torch.backends.cudnn.benchmark = True

    _model = model

    # Capture the as-loaded attention config so the global path can restore the
    # exact checkpoint default after a local switch (instead of hardcoding a
    # string), and record that the model currently IS in global attention so the
    # first short request issues no redundant change_attention_model() call.
    global _attention_mode, _default_self_attention, _default_att_context_size
    _attention_mode = "global"
    try:
        enc_cfg = model.cfg.encoder
        _default_self_attention = enc_cfg.self_attention_model
        att_ctx = getattr(enc_cfg, "att_context_size", None)
        _default_att_context_size = list(att_ctx) if att_ctx else None
    except Exception:  # noqa: BLE001 - config shape varies; fall back defensively
        _default_self_attention = "rel_pos"
        _default_att_context_size = None


def _set_attention(mode: str) -> None:
    """Switch the encoder attention mode to ``"global"`` or ``"local"``.

    Only issues the (expensive) NeMo call when the mode actually changes
    (tracked in the module global ``_attention_mode``).

      * ``"local"``  — RESEARCH.md §3 PATH A: Transformer-XL-style local windowed
        attention (``rel_pos_local_attn`` with att_context_size ``[128, 128]``)
        plus auto conv-subsampling chunking. This bounds per-chunk VRAM, which is
        essential for long audio on a personal GPU.
      * ``"global"`` — restore default full attention for short audio.
    """
    global _attention_mode

    if mode not in ("global", "local"):
        raise ValueError(f"unknown attention mode: {mode!r}")

    if _attention_mode == mode:
        return

    if config.SKIP_MODEL_LOAD or _model is None:
        # No real model (test mode): just record the requested mode so the
        # bookkeeping stays consistent without touching NeMo.
        _attention_mode = mode
        return

    if mode == "local":
        # RESEARCH.md §3 PATH A (confirmed API). att_context_size = [left, right]
        # in tokens; [128, 128] minimizes VRAM (vs v3's [256, 256]).
        _model.change_attention_model("rel_pos_local_attn", [128, 128])
        # 1 = AUTO-select conv-subsampling chunking factor (cuts conv memory,
        # which can exceed the main forward pass on long inputs).
        _model.change_subsampling_conv_chunking_factor(1)
    else:  # "global" — revert the local-attention switch.
        # Restore the EXACT attention config the checkpoint loaded with (captured
        # in load_model), rather than hardcoding a string. This branch only runs
        # after a real local switch (the mode is initialized to "global" at load,
        # so a fresh model is never reconfigured here).
        restore_model = _default_self_attention or "rel_pos"
        if _default_att_context_size:
            _model.change_attention_model(restore_model, _default_att_context_size)
        else:
            _model.change_attention_model(restore_model)
        # Undo the conv-subsampling chunking enabled for local mode (-1 = disabled),
        # so global mode fully reverts the local-mode mutations.
        _model.change_subsampling_conv_chunking_factor(-1)

    _attention_mode = mode


def _time_stride() -> float:
    """Best-effort seconds-per-frame for converting frame offsets to seconds.

    Only used as a DEFENSIVE fallback when a NeMo build omits the seconds keys
    and provides only ``*_offset`` (frame indices). RESEARCH.md §2 flags that
    the exact multiplier was not captured; we try common config locations and
    fall back to the FastConformer 8x-subsampling default of 0.08 s
    (10 ms frame * 8). This path should not trigger on the confirmed NeMo
    version, which emits seconds keys directly.
    """
    model = _model
    # Try a few documented-ish locations without hard-failing.
    for attr_path in (
        ("cfg", "preprocessor", "window_stride"),
    ):
        obj = model
        try:
            for attr in attr_path:
                obj = obj[attr] if isinstance(obj, dict) else getattr(obj, attr)
            window_stride = float(obj)
            # FastConformer subsamples by 8x; effective stride = window_stride * 8.
            return window_stride * 8.0
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return 0.08


def _seconds(entry: dict, key: str, offset_key: str) -> float:
    """Read a timestamp value in seconds from a NeMo entry.

    Prefers the seconds key (``"start"``/``"end"`` — confirmed present on this
    NeMo version, RESEARCH.md §2). Falls back to ``offset_key`` (frame index)
    converted via the model time stride only if the seconds key is missing.
    """
    if key in entry and entry[key] is not None:
        return float(entry[key])
    if offset_key in entry and entry[offset_key] is not None:
        return float(entry[offset_key]) * _time_stride()
    raise KeyError(
        f"timestamp entry missing both {key!r} and {offset_key!r}: {list(entry.keys())}"
    )


def _map_words(word_stamps) -> list[dict]:
    """Map NeMo word-level timestamps -> our contract {start, end, word}."""
    words: list[dict] = []
    for st in word_stamps or []:
        words.append(
            {
                "start": _seconds(st, "start", "start_offset"),
                "end": _seconds(st, "end", "end_offset"),
                # RESEARCH.md §2: word-level text key is literally "word".
                "word": st["word"],
            }
        )
    return words


def _map_segments(segment_stamps) -> list[dict]:
    """Map NeMo segment-level timestamps -> our contract {start, end, text}.

    NeMo's segment text key is ``"segment"`` (RESEARCH.md §2) — we rename it to
    ``"text"`` for our contract.
    """
    segments: list[dict] = []
    for st in segment_stamps or []:
        segments.append(
            {
                "start": _seconds(st, "start", "start_offset"),
                "end": _seconds(st, "end", "end_offset"),
                "text": st["segment"],
            }
        )
    return segments


def _transcribe_one(wav_path: str) -> tuple[str, list[dict], list[dict]]:
    """Transcribe a single wav file, returning (text, words, segments).

    Always requests ``timestamps=True`` (cheap for short audio; required for
    overlap dedup on the chunked path). Times here are exactly as NeMo emits
    them — for a windowed chunk wav they are CHUNK-LOCAL (the wav starts at 0).
    """
    import torch  # lazy — only reached when a real model is loaded.

    with torch.inference_mode():  # SPEC §5.5
        outputs = _model.transcribe([wav_path], timestamps=True)

    if not outputs:
        raise TranscriptionError("NeMo transcribe() returned no hypotheses.")

    hyp = outputs[0]
    text = hyp.text or ""
    ts = getattr(hyp, "timestamp", None) or {}
    words = _map_words(ts.get("word"))
    segments = _map_segments(ts.get("segment"))
    return text, words, segments


def run(wav_path: str, duration: float, return_timestamps: bool) -> dict:
    """Transcribe ``wav_path`` and return our internal result contract.

    Returns::

        {
            "text": str,
            "words": list[{start, end, word}],     # absolute-time seconds
            "segments": list[{start, end, text}],  # absolute-time seconds
            "chunked": bool,
            "num_chunks": int,
        }

    Path selection (SPEC §5):
      * ``duration <= config.SINGLE_PASS_MAX_SEC`` -> single pass with global
        attention.
      * otherwise -> chunked path: local attention (applied once) + overlapping
        windows extracted with ffmpeg, each transcribed with chunk-local
        timestamps, then stitched to absolute time via ``chunking.stitch``.

    ``return_timestamps`` does NOT change whether we ask NeMo for timestamps —
    we always do (the chunked path needs them for overlap dedup, and they are
    cheap for short audio). The handler/schemas layer decides whether to surface
    them to the client. We return words/segments here unconditionally.

    Unexpected inference errors are wrapped in :class:`TranscriptionError`
    (-> job FAILED). Programming/value errors are allowed to propagate.
    """
    if _model is None and not config.SKIP_MODEL_LOAD:
        # Defensive: the handler loads the model at import. If we get here with
        # no model and not in test mode, something is badly wrong.
        raise TranscriptionError("model is not loaded; call load_model() first.")

    # ----------------------------- single pass --------------------------- #
    if duration <= config.SINGLE_PASS_MAX_SEC:
        _set_attention("global")
        try:
            text, words, segments = _transcribe_one(wav_path)
        except TranscriptionError:
            raise
        except Exception as exc:  # wrap unexpected inference errors
            raise TranscriptionError(f"single-pass transcription failed: {exc}") from exc

        return {
            "text": text,
            "words": words,
            "segments": segments,
            "chunked": False,
            "num_chunks": 1,
        }

    # ------------------------------- chunked ----------------------------- #
    # Local attention + conv chunking, applied ONCE before the loop (RESEARCH.md
    # §3 PATH A). This bounds per-chunk VRAM for long audio.
    _set_attention("local")

    plans = chunking.plan_chunks(duration, config.CHUNK_SEC, config.CHUNK_OVERLAP_SEC)
    work_dir = os.path.dirname(os.path.abspath(wav_path))

    chunk_outputs: list[dict] = []
    chunk_paths: list[str] = []
    try:
        for plan in plans:
            chunk_path = os.path.join(work_dir, f"chunk_{plan.index:04d}.wav")
            chunk_paths.append(chunk_path)
            # extract_window times are seconds into the source wav; the produced
            # chunk wav itself starts at 0, so its timestamps are chunk-local.
            audio.extract_window(
                wav_path, plan.start, plan.end - plan.start, chunk_path
            )
            try:
                text, words, segments = _transcribe_one(chunk_path)
            except TranscriptionError:
                raise
            except Exception as exc:
                raise TranscriptionError(
                    f"chunk {plan.index} transcription failed: {exc}"
                ) from exc

            chunk_outputs.append(
                {"plan": plan, "words": words, "segments": segments}
            )
    finally:
        # Clean up per-chunk temp wavs; never let cleanup mask a real error.
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
