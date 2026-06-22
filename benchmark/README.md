# Benchmark harness

Compares three transcription systems over a local audio set to decide whether
**Parakeet TDT 0.6B v2** (this worker) can replace **AssemblyAI**, with
**Whisper large-v3** as a control baseline (SPEC §10).

The harness is **benchmark-only** — none of its dependencies (`jiwer`,
`faster-whisper`/`openai-whisper`, `assemblyai`, `pandas`, `soundfile`) are part
of the worker image.

## What it produces

One run writes three files to `benchmark/results/`:

| File | Contents |
|---|---|
| `runs.csv` | One row per **(file × system)** with every metric (and an `error` column when a system failed on a file). |
| `summary.csv` | Per-system **mean & median** of each metric across all files. |
| `verdict.md` | A pre-filled §10.3 verdict template: aggregate table, Parakeet-vs-AssemblyAI head-to-head on the decision metrics, and blanks for the written judgement. |

Audio is normalized to 16 kHz mono WAV **once** per file (cached in
`results/_normalized_wav/`) and the identical WAV is fed to every system.

## Metrics (`metrics.py`, SPEC §10.2)

| Column(s) | Meaning |
|---|---|
| `wer_raw` | WER with text only lowercased (punctuation kept). |
| `wer_normalized` | WER after the **shared normalizer** (lowercase, strip punctuation, fold common number words, collapse whitespace) — applied identically to every system. |
| `punct_precision/recall/f1` | F1 over punctuation marks vs the reference (multiset overlap). |
| `ts_mean_abs_start_err_sec`, `ts_pct_within_200ms`, `ts_matched_words` | Word-start timestamp accuracy vs a reference alignment (`ref_words` in the manifest). Blank when no reference alignment is supplied. |
| `hallucinated_words`, `hallucination_rate`, `hallucinations_per_min` | Coarse heuristic: hypothesis words absent from the reference vocabulary. A screening signal, **not** ground truth — confirm flagged files by ear. |
| `rtf` | `audio_duration_sec / processing_time_sec` (speed factor, higher = faster) — same definition as the worker's `meta.rtf` (SPEC §3.2). |
| `audio_hour_cost_usd` | $ per audio-hour. Self-hosted models: `gpu_cost_per_hour / rtf`. AssemblyAI: flat list price. |

## Install

Separate environment from the worker:

```bash
python -m venv .venv-bench
source .venv-bench/bin/activate
pip install -r benchmark/requirements.txt
```

You can install a subset — a missing package just skips that system.

## Data

Put audio + reference transcripts in `benchmark/datasets/` (gitignored). See
[`datasets/README.md`](datasets/README.md) for the exact layout, the optional
`manifest.json` (per-file `audio_url` and `ref_words`), and an example.

## Systems & how each is exercised

| System | How it runs | Requirements |
|---|---|---|
| **Parakeet** (candidate) | `--parakeet-endpoint <url>` → deployed RunPod `/runsync` (needs per-file `audio_url`), **or** `--parakeet-local` → in-process `src.transcribe`. | endpoint: `RUNPOD_API_KEY` + `httpx`; local: GPU + NeMo. |
| **AssemblyAI** (incumbent) | `assemblyai` SDK on the local WAV. | `ASSEMBLYAI_API_KEY` + `assemblyai`. |
| **Whisper large-v3** (control) | `faster-whisper` if installed, else `openai-whisper`, on the local WAV. | one of the two packages (GPU recommended). |

Each system degrades gracefully: a missing key or package logs a warning, skips
that system, and the run continues for the others (SPEC §10.4).

## Run

```bash
python benchmark/run_benchmark.py \
    --datasets benchmark/datasets \
    --results  benchmark/results \
    --parakeet-endpoint https://api.runpod.ai/v2/<endpoint-id> \
    --gpu-cost-per-hour 0.79 \
    --assemblyai-price-per-hour 0.37
```

Flags / env:

| Flag | Env fallback | Default | Meaning |
|---|---|---|---|
| `--parakeet-endpoint` | `PARAKEET_ENDPOINT` | — | RunPod endpoint base URL (uses `/runsync`). |
| `--parakeet-local` | — | off | Run Parakeet in-process via `src.transcribe`. |
| `--gpu-cost-per-hour` | `GPU_COST_PER_HOUR` | `0.79` | GPU $/hr for self-hosted cost (set to your RunPod GPU price). |
| `--assemblyai-price-per-hour` | `ASSEMBLYAI_PRICE_PER_HOUR` | `0.37` | AssemblyAI list price $/audio-hour. |

> Cost numbers are only as good as the inputs — pass the actual GPU price for the
> RunPod GPU type you benchmarked on, and AssemblyAI's current list price.

## Reading the verdict

`verdict.md` calls Parakeet a winner per metric only on the decision axes
(normalized WER, punctuation F1, timestamp error, $/audio-hour). The final
recommendation is left for a human — automatic metrics (especially the
hallucination heuristic) need a sanity listen before any production switch.
