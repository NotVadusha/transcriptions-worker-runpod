"""Benchmark harness: Parakeet vs AssemblyAI vs Whisper large-v3 (SPEC §10).

Compares the transcription systems over the audio files in ``benchmark/datasets/``
and writes one CSV row per (file x system) plus an aggregate summary and a
written verdict template.

What it does (SPEC §10):
  1. Discover ``(audio, reference)`` pairs in ``benchmark/datasets/`` (see
     ``datasets/README.md`` for the expected layout).
  2. Normalize each audio file ONCE to 16 kHz mono WAV (the same WAV is fed to
     every system, so differences aren't preprocessing artifacts — SPEC §10.4).
  3. Run each available system on each WAV, timing wall-clock processing.
  4. Score every result with ``metrics.py`` (WER raw+normalized, punctuation F1,
     timestamp quality, hallucination heuristic) and compute ``audio_hour_cost``.
  5. Write ``results/runs.csv`` (per file x system), ``results/summary.csv``
     (mean/median per metric per system), and ``results/verdict.md`` (template).

Systems degrade gracefully (SPEC §10.4): a system whose dependency or API key is
missing is skipped with a logged reason; rows are still emitted for the others.

Parakeet is exercised through ONE of two backends:
  * ``--parakeet-endpoint <runpod-url>`` + ``RUNPOD_API_KEY`` — calls the deployed
    RunPod ``/runsync`` endpoint (requires each audio file to be reachable via an
    https ``audio_url``; supply a per-file URL in the manifest, see datasets README).
  * ``--parakeet-local`` — imports ``src.transcribe`` and runs the model in-process
    (needs a GPU + NeMo; honors ``SKIP_MODEL_LOAD`` only for smoke wiring tests).
If neither is given, Parakeet is skipped.

Run::

    python benchmark/run_benchmark.py \
        --datasets benchmark/datasets \
        --results benchmark/results \
        --parakeet-endpoint https://api.runpod.ai/v2/<id> \
        --gpu-cost-per-hour 0.79 \
        --assemblyai-price-per-hour 0.37

Environment: ``RUNPOD_API_KEY``, ``ASSEMBLYAI_API_KEY``. Whisper needs no key.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

# Make ``from benchmark import metrics`` / ``import metrics`` work regardless of
# CWD: put the repo root on sys.path, then import the sibling module.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmark import metrics  # noqa: E402

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".webm")
REFERENCE_EXTS = (".txt", ".ref.txt")


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One benchmark item: an audio file plus its human reference transcript."""

    name: str               # stem used as the row's file id
    audio_path: str         # original audio file
    reference_text: str     # human transcript
    audio_url: str | None   # optional https URL for the RunPod endpoint backend
    ref_words: list[dict]   # optional reference word timestamps (may be empty)


def discover_samples(datasets_dir: str) -> list[Sample]:
    """Find ``(audio, reference)`` pairs under ``datasets_dir``.

    Pairing rule: an audio file ``foo.<ext>`` is paired with a reference text
    file ``foo.txt`` in the same directory. An optional ``manifest.json`` maps
    file stems to extra fields (``audio_url`` for the endpoint backend, and
    ``ref_words`` for timestamp scoring). Files without a matching reference are
    skipped with a warning.
    """
    manifest = _load_manifest(datasets_dir)
    samples: list[Sample] = []

    if not os.path.isdir(datasets_dir):
        return samples

    for fname in sorted(os.listdir(datasets_dir)):
        path = os.path.join(datasets_dir, fname)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in AUDIO_EXTS:
            continue

        ref_path = os.path.join(datasets_dir, stem + ".txt")
        if not os.path.isfile(ref_path):
            _warn(f"no reference '{stem}.txt' for audio '{fname}' — skipping")
            continue

        with open(ref_path, "r", encoding="utf-8") as fh:
            reference_text = fh.read().strip()

        entry = manifest.get(stem, {})
        samples.append(
            Sample(
                name=stem,
                audio_path=path,
                reference_text=reference_text,
                audio_url=entry.get("audio_url"),
                ref_words=entry.get("ref_words", []),
            )
        )
    return samples


def _load_manifest(datasets_dir: str) -> dict:
    """Load optional ``manifest.json`` mapping stems -> extra per-file fields."""
    path = os.path.join(datasets_dir, "manifest.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _warn(f"could not read manifest.json: {exc}")
        return {}


# --------------------------------------------------------------------------- #
# Audio normalization (once per file)
# --------------------------------------------------------------------------- #
def normalize_audio(src_path: str, out_dir: str) -> str:
    """Normalize ``src_path`` to 16 kHz mono WAV once; return the WAV path.

    Mirrors the worker's normalization (SPEC §4) so every system transcribes the
    identical signal. Skips re-encoding if the normalized WAV already exists.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, _stem(src_path) + ".16k.wav")
    if os.path.isfile(out_path):
        return out_path
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-i", src_path,
        "-ac", "1", "-ar", "16000", "-vn", "-c:a", "pcm_s16le", out_path,
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to normalize {src_path}: {proc.stderr.strip()}")
    return out_path


def probe_duration(path: str) -> float:
    """Audio duration in seconds via ffprobe (0.0 if it can't be read)."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
    )
    try:
        return float(proc.stdout.strip())
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# System runners — each returns (text, words) or raises to signal skip-this-file
# --------------------------------------------------------------------------- #
@dataclass
class SystemResult:
    """One system's transcription of one file."""

    text: str
    words: list[dict]           # {start, end, word}; may be empty
    processing_time_sec: float
    rtf: float | None           # audio_duration / processing_time (or None)


class ParakeetEndpointRunner:
    """Parakeet via the deployed RunPod endpoint (``/runsync``)."""

    name = "parakeet"

    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key

    @classmethod
    def maybe_create(cls, args) -> "ParakeetEndpointRunner | None":
        if not args.parakeet_endpoint:
            return None
        key = os.environ.get("RUNPOD_API_KEY")
        if not key:
            _warn("parakeet endpoint given but RUNPOD_API_KEY unset — skipping Parakeet")
            return None
        try:
            import httpx  # noqa: F401
        except ImportError:
            _warn("httpx not installed — skipping Parakeet endpoint")
            return None
        return cls(args.parakeet_endpoint, key)

    def transcribe(self, sample: Sample, wav_path: str, duration: float) -> SystemResult:
        if not sample.audio_url:
            raise RuntimeError(
                f"sample '{sample.name}' has no audio_url in manifest.json; "
                "the RunPod endpoint backend needs a reachable https URL"
            )
        import httpx

        payload = {"input": {"audio_url": sample.audio_url, "return_timestamps": True}}
        t0 = time.perf_counter()
        resp = httpx.post(
            f"{self.endpoint}/runsync",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=3600,
        )
        resp.raise_for_status()
        proc = time.perf_counter() - t0

        body = resp.json()
        out = body.get("output", body)
        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(f"worker returned error: {out['error']}")
        text = out.get("text", "")
        words = out.get("words", []) or []
        # Prefer the worker's own rtf/processing_time when present.
        meta = out.get("meta", {}) if isinstance(out, dict) else {}
        rtf = meta.get("rtf")
        if rtf is None:
            rtf = (duration / proc) if proc > 0 else None
        return SystemResult(text=text, words=words, processing_time_sec=proc, rtf=rtf)


class ParakeetLocalRunner:
    """Parakeet via the in-process ``src.transcribe`` module (needs a GPU)."""

    name = "parakeet"

    @classmethod
    def maybe_create(cls, args) -> "ParakeetLocalRunner | None":
        if not args.parakeet_local:
            return None
        try:
            from src import transcribe  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            _warn(f"could not import src.transcribe for local Parakeet: {exc}")
            return None
        return cls()

    def transcribe(self, sample: Sample, wav_path: str, duration: float) -> SystemResult:
        from src import transcribe

        transcribe.load_model()
        t0 = time.perf_counter()
        result = transcribe.run(wav_path, duration, return_timestamps=True)
        proc = time.perf_counter() - t0
        rtf = (duration / proc) if proc > 0 else None
        return SystemResult(
            text=result.get("text", ""),
            words=result.get("words", []) or [],
            processing_time_sec=proc,
            rtf=rtf,
        )


class AssemblyAIRunner:
    """AssemblyAI Universal (incumbent / target to beat)."""

    name = "assemblyai"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @classmethod
    def maybe_create(cls, args) -> "AssemblyAIRunner | None":
        key = os.environ.get("ASSEMBLYAI_API_KEY")
        if not key:
            _warn("ASSEMBLYAI_API_KEY unset — skipping AssemblyAI")
            return None
        try:
            import assemblyai  # noqa: F401
        except ImportError:
            _warn("assemblyai package not installed — skipping AssemblyAI")
            return None
        return cls(key)

    def transcribe(self, sample: Sample, wav_path: str, duration: float) -> SystemResult:
        import assemblyai as aai

        aai.settings.api_key = self.api_key
        config = aai.TranscriptionConfig(punctuate=True, format_text=True)
        transcriber = aai.Transcriber(config=config)
        t0 = time.perf_counter()
        transcript = transcriber.transcribe(wav_path)
        proc = time.perf_counter() - t0
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI error: {transcript.error}")
        text = transcript.text or ""
        # AssemblyAI word times are in MILLISECONDS -> convert to seconds.
        words = [
            {"start": w.start / 1000.0, "end": w.end / 1000.0, "word": w.text}
            for w in (transcript.words or [])
        ]
        rtf = (duration / proc) if proc > 0 else None
        return SystemResult(text=text, words=words, processing_time_sec=proc, rtf=rtf)


class WhisperRunner:
    """Whisper large-v3 (control / sanity baseline). Tries faster-whisper, then openai-whisper."""

    name = "whisper-large-v3"

    def __init__(self, backend, model):
        self.backend = backend  # "faster" | "openai"
        self.model = model

    @classmethod
    def maybe_create(cls, args) -> "WhisperRunner | None":
        # Prefer faster-whisper (CTranslate2) if present, else openai-whisper.
        try:
            from faster_whisper import WhisperModel

            device = "cuda" if _cuda_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
            model = WhisperModel("large-v3", device=device, compute_type=compute)
            return cls("faster", model)
        except ImportError:
            pass
        try:
            import whisper

            model = whisper.load_model("large-v3")
            return cls("openai", model)
        except ImportError:
            _warn("neither faster-whisper nor openai-whisper installed — skipping Whisper")
            return None
        except Exception as exc:  # noqa: BLE001 - model download/load failure
            _warn(f"could not load Whisper large-v3: {exc}")
            return None

    def transcribe(self, sample: Sample, wav_path: str, duration: float) -> SystemResult:
        t0 = time.perf_counter()
        if self.backend == "faster":
            segments, _info = self.model.transcribe(wav_path, word_timestamps=True)
            text_parts: list[str] = []
            words: list[dict] = []
            for seg in segments:
                text_parts.append(seg.text)
                for w in (seg.words or []):
                    words.append({"start": w.start, "end": w.end, "word": w.word.strip()})
            text = "".join(text_parts).strip()
        else:  # openai-whisper
            result = self.model.transcribe(wav_path, word_timestamps=True)
            text = (result.get("text") or "").strip()
            words = []
            for seg in result.get("segments", []):
                for w in seg.get("words", []):
                    words.append(
                        {"start": w["start"], "end": w["end"], "word": w["word"].strip()}
                    )
        proc = time.perf_counter() - t0
        rtf = (duration / proc) if proc > 0 else None
        return SystemResult(text=text, words=words, processing_time_sec=proc, rtf=rtf)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Scoring one (sample x system) result
# --------------------------------------------------------------------------- #
def score(sample: Sample, sys_name: str, res: SystemResult, duration: float, args) -> dict:
    """Produce one CSV row dict for ``(sample, system)`` (SPEC §10.2 metrics)."""
    punct = metrics.punctuation_f1(sample.reference_text, res.text)
    tsq = metrics.timestamp_quality(res.words, sample.ref_words)
    hall = metrics.hallucination_rate(sample.reference_text, res.text, duration)

    # Cost: GPU-derived for the self-hosted models, list price for AssemblyAI.
    if sys_name == "assemblyai":
        cost = metrics.audio_hour_cost_listprice(args.assemblyai_price_per_hour)
    else:
        cost = metrics.audio_hour_cost_from_gpu(args.gpu_cost_per_hour, res.rtf or 0.0)

    return {
        "file": sample.name,
        "system": sys_name,
        "audio_duration_sec": round(duration, 3),
        "processing_time_sec": round(res.processing_time_sec, 3),
        "rtf": round(res.rtf, 3) if res.rtf is not None else "",
        "wer_raw": round(metrics.wer_raw(sample.reference_text, res.text), 4),
        "wer_normalized": round(metrics.wer_normalized(sample.reference_text, res.text), 4),
        "punct_precision": round(punct["precision"], 4),
        "punct_recall": round(punct["recall"], 4),
        "punct_f1": round(punct["f1"], 4),
        "ts_mean_abs_start_err_sec": _round_or_blank(tsq["mean_abs_start_error_sec"], 4),
        "ts_pct_within_200ms": _round_or_blank(tsq["pct_within_tolerance"], 4),
        "ts_matched_words": tsq["matched"],
        "hallucinated_words": hall["hallucinated_words"],
        "hallucination_rate": round(hall["hallucination_rate"], 4),
        "hallucinations_per_min": round(hall["hallucinations_per_min"], 4),
        "audio_hour_cost_usd": round(cost, 4) if cost is not None else "",
        "error": "",
    }


def error_row(sample: Sample, sys_name: str, duration: float, message: str) -> dict:
    """A row recording that a system failed on a file (keeps the matrix complete)."""
    row = {k: "" for k in CSV_FIELDS}
    row.update(
        {
            "file": sample.name,
            "system": sys_name,
            "audio_duration_sec": round(duration, 3),
            "error": message[:500],
        }
    )
    return row


CSV_FIELDS = [
    "file", "system", "audio_duration_sec", "processing_time_sec", "rtf",
    "wer_raw", "wer_normalized",
    "punct_precision", "punct_recall", "punct_f1",
    "ts_mean_abs_start_err_sec", "ts_pct_within_200ms", "ts_matched_words",
    "hallucinated_words", "hallucination_rate", "hallucinations_per_min",
    "audio_hour_cost_usd", "error",
]


# --------------------------------------------------------------------------- #
# Aggregation + verdict
# --------------------------------------------------------------------------- #
# Metrics summarized in summary.csv and the verdict, with the direction that is
# "better" so the verdict can phrase win/lose correctly.
SUMMARY_METRICS = [
    ("wer_raw", "lower"),
    ("wer_normalized", "lower"),
    ("punct_f1", "higher"),
    ("ts_mean_abs_start_err_sec", "lower"),
    ("ts_pct_within_200ms", "higher"),
    ("hallucinations_per_min", "lower"),
    ("rtf", "higher"),
    ("audio_hour_cost_usd", "lower"),
]


def summarize(rows: list[dict], systems: list[str]) -> list[dict]:
    """Per-system mean/median for each summary metric (None/blank skipped)."""
    summary: list[dict] = []
    for sysname in systems:
        sys_rows = [r for r in rows if r["system"] == sysname and not r["error"]]
        entry = {"system": sysname, "n_files": len(sys_rows)}
        for metric, _dir in SUMMARY_METRICS:
            agg = metrics.aggregate([_to_float(r.get(metric)) for r in sys_rows])
            entry[f"{metric}_mean"] = _round_or_blank(agg.mean, 4)
            entry[f"{metric}_median"] = _round_or_blank(agg.median, 4)
        summary.append(entry)
    return summary


def write_verdict(summary: list[dict], path: str) -> None:
    """Write the §10.3 verdict template, pre-filling aggregate numbers."""
    lines: list[str] = []
    lines.append("# Benchmark verdict — Parakeet vs AssemblyAI vs Whisper large-v3\n")
    lines.append(
        "_Auto-generated template (SPEC §10.3). Numbers are aggregate means across "
        "the benchmark set; fill in the qualitative judgement below._\n"
    )

    # Aggregate table.
    lines.append("## Aggregate (mean across files)\n")
    header = "| system | n | WER raw | WER norm | punct F1 | ts err (s) | ts ±200ms | halluc/min | rtf | $/audio-hr |"
    sep = "|" + "---|" * 11
    lines.append(header)
    lines.append(sep)
    for s in summary:
        lines.append(
            "| {system} | {n} | {wer_raw} | {wer_norm} | {pf1} | {tserr} | {tspct} | {hpm} | {rtf} | {cost} |".format(
                system=s["system"],
                n=s["n_files"],
                wer_raw=s.get("wer_raw_mean", ""),
                wer_norm=s.get("wer_normalized_mean", ""),
                pf1=s.get("punct_f1_mean", ""),
                tserr=s.get("ts_mean_abs_start_err_sec_mean", ""),
                tspct=s.get("ts_pct_within_200ms_mean", ""),
                hpm=s.get("hallucinations_per_min_mean", ""),
                rtf=s.get("rtf_mean", ""),
                cost=s.get("audio_hour_cost_usd_mean", ""),
            )
        )
    lines.append("")

    # Head-to-head: Parakeet vs AssemblyAI on the decision metrics.
    para = _find(summary, "parakeet")
    aai = _find(summary, "assemblyai")
    lines.append("## Decision: can Parakeet replace AssemblyAI?\n")
    if para and aai:
        lines.append(_compare_line("Normalized WER", para, aai, "wer_normalized_mean", "lower"))
        lines.append(_compare_line("Punctuation F1", para, aai, "punct_f1_mean", "higher"))
        lines.append(_compare_line("Timestamp err (s)", para, aai, "ts_mean_abs_start_err_sec_mean", "lower"))
        lines.append(_compare_line("$/audio-hour", para, aai, "audio_hour_cost_usd_mean", "lower"))
    else:
        lines.append(
            "_Not enough systems ran to compare (need both Parakeet and AssemblyAI rows)._"
        )
    lines.append("")
    lines.append("## Written verdict (fill in)\n")
    lines.append("- **WER:** Does Parakeet meet/beat AssemblyAI on normalized WER? ______")
    lines.append("- **Cost:** Is Parakeet's $/audio-hour materially lower? ______")
    lines.append("- **Timestamps:** Acceptable mean start error and ±200 ms coverage? ______")
    lines.append("- **Punctuation:** Acceptable F1 vs reference? ______")
    lines.append("- **Hallucinations:** Any concerning per-minute rate on silence/noise? ______")
    lines.append("")
    lines.append("**Recommendation:** ☐ Replace AssemblyAI  ☐ Keep AssemblyAI  ☐ Needs more data\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _compare_line(label: str, para: dict, aai: dict, key: str, better: str) -> str:
    p = _to_float(para.get(key))
    a = _to_float(aai.get(key))
    if p is None or a is None:
        return f"- **{label}:** parakeet={para.get(key, 'n/a')} assemblyai={aai.get(key, 'n/a')} (incomparable)"
    if better == "lower":
        verdict = "Parakeet WINS" if p < a else ("tie" if p == a else "AssemblyAI wins")
    else:
        verdict = "Parakeet WINS" if p > a else ("tie" if p == a else "AssemblyAI wins")
    return f"- **{label}:** parakeet={p} vs assemblyai={a} -> **{verdict}**"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    datasets_dir = os.path.abspath(args.datasets)
    results_dir = os.path.abspath(args.results)
    norm_dir = os.path.join(results_dir, "_normalized_wav")
    os.makedirs(results_dir, exist_ok=True)

    samples = discover_samples(datasets_dir)
    if not samples:
        _warn(
            f"no (audio, reference) pairs found in {datasets_dir}. "
            "See benchmark/datasets/README.md for the expected layout."
        )
        # Still emit empty outputs so downstream tooling has stable files.
        _write_csv(os.path.join(results_dir, "runs.csv"), [], CSV_FIELDS)
        return 0

    # Build the available system runners (each may return None -> skipped).
    runners = []
    parakeet = ParakeetEndpointRunner.maybe_create(args) or ParakeetLocalRunner.maybe_create(args)
    if parakeet:
        runners.append(parakeet)
    aai = AssemblyAIRunner.maybe_create(args)
    if aai:
        runners.append(aai)
    whisper = WhisperRunner.maybe_create(args)
    if whisper:
        runners.append(whisper)

    if not runners:
        _warn("no systems available to run (missing deps/keys). Nothing to benchmark.")
        _write_csv(os.path.join(results_dir, "runs.csv"), [], CSV_FIELDS)
        return 1

    system_names = [r.name for r in runners]
    _log(f"systems: {', '.join(system_names)}  |  files: {len(samples)}")

    rows: list[dict] = []
    for sample in samples:
        try:
            wav = normalize_audio(sample.audio_path, norm_dir)
            duration = probe_duration(wav)
        except Exception as exc:  # noqa: BLE001 - bad file shouldn't kill the run
            _warn(f"[{sample.name}] normalization failed: {exc} — skipping file")
            for r in runners:
                rows.append(error_row(sample, r.name, 0.0, f"normalize failed: {exc}"))
            continue

        for runner in runners:
            try:
                _log(f"[{sample.name}] running {runner.name} ...")
                res = runner.transcribe(sample, wav, duration)
                rows.append(score(sample, runner.name, res, duration, args))
            except Exception as exc:  # noqa: BLE001 - isolate per (file x system)
                _warn(f"[{sample.name}] {runner.name} failed: {exc}")
                rows.append(error_row(sample, runner.name, duration, str(exc)))

    _write_csv(os.path.join(results_dir, "runs.csv"), rows, CSV_FIELDS)

    summary = summarize(rows, system_names)
    _write_summary_csv(os.path.join(results_dir, "summary.csv"), summary)
    write_verdict(summary, os.path.join(results_dir, "verdict.md"))

    _log(f"wrote {os.path.join(results_dir, 'runs.csv')}")
    _log(f"wrote {os.path.join(results_dir, 'summary.csv')}")
    _log(f"wrote {os.path.join(results_dir, 'verdict.md')}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASR benchmark harness (SPEC §10).")
    p.add_argument("--datasets", default=os.path.join("benchmark", "datasets"))
    p.add_argument("--results", default=os.path.join("benchmark", "results"))
    p.add_argument(
        "--parakeet-endpoint",
        default=os.environ.get("PARAKEET_ENDPOINT"),
        help="RunPod endpoint base URL (uses /runsync). Needs RUNPOD_API_KEY and per-file audio_url in manifest.json.",
    )
    p.add_argument(
        "--parakeet-local",
        action="store_true",
        help="Run Parakeet in-process via src.transcribe (needs a GPU + NeMo).",
    )
    p.add_argument(
        "--gpu-cost-per-hour",
        type=float,
        default=float(os.environ.get("GPU_COST_PER_HOUR", "0.79")),
        help="GPU $/hr used for the self-hosted audio_hour_cost (default A10-ish 0.79).",
    )
    p.add_argument(
        "--assemblyai-price-per-hour",
        type=float,
        default=float(os.environ.get("ASSEMBLYAI_PRICE_PER_HOUR", "0.37")),
        help="AssemblyAI list price $/audio-hour for cost comparison.",
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Small IO/util helpers
# --------------------------------------------------------------------------- #
def _write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _write_summary_csv(path: str, summary: list[dict]) -> None:
    if not summary:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("")
        return
    fields = list(summary[0].keys())
    _write_csv(path, summary, fields)


def _round_or_blank(value, ndigits: int):
    return round(value, ndigits) if isinstance(value, (int, float)) else ""


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find(summary: list[dict], system: str) -> dict | None:
    for s in summary:
        if s["system"] == system:
            return s
    return None


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _log(msg: str) -> None:
    print(f"[benchmark] {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"[benchmark][warn] {msg}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
