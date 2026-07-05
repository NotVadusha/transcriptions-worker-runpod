# transcriptions-worker-runpod

A [RunPod](https://www.runpod.io/) **serverless** worker that transcribes audio with
**NVIDIA Parakeet TDT 0.6B v2** (via the [NeMo](https://github.com/NVIDIA/NeMo) ASR
toolkit) and returns transcript text plus word- and segment-level timestamps.

- **Input:** a presigned HTTPS URL to an audio/video file.
- **Output:** transcript `text`, optional `segments`/`words` timestamps, and run `meta`.
- **Range:** a few seconds up to **10 hours** of audio. Short clips run in a single pass;
  long files use an overlapping-chunk path with timestamp stitching.
- **Multilingual, batch (non-streaming).** The `language` code routes to one of four
  backends (see [Model routing](#model-routing)). No diarization, no auth/billing.

The worker stores **no cloud credentials**. It pulls audio from a caller-provided presigned
GET URL and, for large results, PUTs them to a caller-provided presigned PUT URL.

---

## Repo layout

```text
transcriptions-worker-runpod/
├── README.md                  # this file
├── Dockerfile                 # worker image (NeMo base + ffmpeg + runpod)
├── handler.py                 # repository-root RunPod entrypoint wrapper
├── requirements.txt           # only what the NeMo base image lacks
├── pyproject.toml             # pytest config (pythonpath = ["."])
├── .dockerignore
├── .gitignore
├── src/
│   ├── __init__.py            # makes `from src import ...` work
│   ├── handler.py             # worker orchestration implementation
│   ├── transcribe.py          # language routing + Parakeet path (single-pass & chunked)
│   ├── backends.py            # Canary / SenseVoice / Whisper backends (lazy per-language)
│   ├── audio.py               # stream input URL via ffmpeg/ffprobe + normalize
│   ├── chunking.py            # long-audio segmentation + timestamp stitching (pure logic)
│   ├── schemas.py             # request parsing/validation + output building
│   ├── storage.py             # offload large results to a presigned PUT URL
│   └── config.py              # env var parsing, constants, error codes
├── tests/
│   ├── __init__.py
│   ├── test_input.json        # RunPod local test payload
│   ├── test_schemas.py        # request/output contract (no GPU)
│   ├── test_chunking.py       # stitching/offset math (no GPU)
│   ├── test_audio.py          # download/ffmpeg wrappers (no GPU)
│   └── test_storage.py        # offload behavior (no GPU)
├── benchmark/                 # Parakeet vs AssemblyAI vs Whisper large-v3 harness
│   ├── datasets/              # (gitignored) local audio + reference transcripts
│   └── results/               # (gitignored) per-run CSV/JSON output
└── scripts/
    └── smoke_test.sh          # curl against a deployed endpoint
```

Root `handler.py` only starts RunPod and delegates to `src.handler`. The worker
orchestration stays in `src/handler.py`; model logic lives in `transcribe.py` (routing +
Parakeet) and `backends.py` (Canary/SenseVoice/Whisper) — the only modules that import
NeMo/torch/funasr/transformers, all via lazy in-function imports.

---

## Version stack & container

The image is based on the **official NVIDIA NeMo container**. It ships a pre-validated,
mutually-compatible PyTorch + CUDA + NeMo triple with Hopper/Blackwell support — the
lowest-risk way to stay compatible with the latest GPUs (Blackwell B200 / RTX 5090 require
CUDA 12.8+ and PyTorch ≥ 2.7).

### Pinned base image

```dockerfile
FROM nvcr.io/nvidia/nemo:25.09.02
```

This is the conservative-stable pin. Its published software-component table lists:

| Property | Value (baked into the base image) |
|---|---|
| NeMo | 2.5.0 |
| PyTorch | 2.8.0a0 |
| CUDA | 12.9.1 |
| Python | 3.12 *(inferred — must be confirmed on GPU, see below)* |
| Blackwell support | Yes (CUDA 12.9 + PyTorch 2.8) |

> **Why not `25.11.01`?** The NeMo changelog states *"NeMo 2 will be deprecated starting 25.11."*
> `25.11.01` (NeMo 2.6.0 / torch 2.9.0a0 / CUDA 13.0.1) sits exactly on that boundary and may
> carry ASR API churn. `25.09.02` still satisfies the Blackwell / CUDA 12.8+ requirement while
> staying one release back. Do **not** use the `26.02` tag yet — its resolved versions are not in
> NVIDIA's component table.

### Record the *resolved* versions (must validate on GPU)

The table above is what NVIDIA publishes; the numbers are partly inferred. Before trusting the
image in production, run inside the pulled container on the target GPU and **paste the output back
into this section**:

```bash
python3 --version
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); \
            print('capability', torch.cuda.get_device_capability())"
python3 -c "import nemo; print('nemo', nemo.__version__)"
ffmpeg -version | head -1          # confirm ffmpeg present
ldconfig -p | grep sndfile         # confirm libsndfile present
```

```text
# RESOLVED VERSIONS (fill in from the container on the deploy GPU):
# Python  : __________
# torch   : __________   CUDA: __________
# device  : sm_____  (e.g. sm_100 B200 / sm_120 RTX 5090)
# NeMo    : __________
# ffmpeg  : present? ____   libsndfile: present? ____
```

`ffmpeg` and `libsndfile1` presence in the NeMo image is **undocumented**, so the Dockerfile
installs both explicitly — still confirm with the commands above.

### Registry auth

`nvcr.io` requires login even though the NeMo image is publicly pullable:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```

On RunPod, set these under the template's **container registry credentials**. The image is large
(~20 GB+ for this NeMo generation) — expect long cold pulls and size the RunPod container
disk / network volume accordingly.

### CUDA-base fallback

If the full NeMo image is too large or slow to cold-start, fall back to a CUDA/PyTorch base and
install NeMo yourself (must keep CUDA 12.8+ / torch ≥ 2.7 for Blackwell):

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.09-py3
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -U "nemo_toolkit[asr]" runpod
```

You then own NeMo/torch compatibility (a real risk on Blackwell). Note in this README which path
was taken.

---

## Build & deploy

### Build and push the image

```bash
docker login nvcr.io                       # see "Registry auth" above
docker build -t <registry>/transcriptions-worker-runpod:latest .
docker push  <registry>/transcriptions-worker-runpod:latest
```

The Dockerfile bakes the model weights into the image (a prefetch step that calls
`from_pretrained(MODEL_NAME)`) so cold starts don't re-download the ~2.4 GB checkpoint.

### Create the RunPod endpoint

1. RunPod console → **Serverless** → **New Endpoint**.
2. Point it at the pushed image; add the `nvcr.io` registry credentials to the template.
3. **GPU type:** start on A10 / L4 / RTX 4090 (good price/perf for a 0.6B model). The image also
   runs on Blackwell-class GPUs (B200 / RTX 5090) via the base above. Make this configurable and
   capture cost/throughput per GPU in the benchmark so the production choice is data-driven.
4. Set any env overrides from the [env var table](#configuration-env-vars) (defaults are fine for v0).
5. Size container disk for the large image and the baked model.

The worker loads the model **once at container boot** (RunPod best practice) so it persists across
warm invocations.

---

## Local testing

### GPU-free unit tests

The pure-logic modules (`config`, `schemas`, `chunking`, and the `audio`/`storage` wrappers) are
unit-tested **without a GPU or NeMo**. They pass under Python 3.13 with only stdlib + pytest:

```bash
pip install pytest
pytest                 # pyproject.toml sets pythonpath=["."] so `from src import ...` resolves
```

These cover request validation, output shaping, chunk planning, timestamp offset/stitch math,
URL-streaming/ffmpeg error mapping, and result-offload behavior — i.e. acceptance criteria that don't
need a model.

### Running the handler locally

The RunPod SDK loads `tests/test_input.json` automatically:

```bash
# Tests import the worker without a GPU via the test-only escape hatch:
SKIP_MODEL_LOAD=1 python handler.py
```

`tests/test_input.json`:

```json
{ "input": { "audio_url": "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav", "return_timestamps": true } }
```

Or pass input inline (takes precedence over the file):

```bash
python handler.py --test_input '{"input": {"audio_url": "https://.../clip.wav"}}'
```

> `SKIP_MODEL_LOAD=1` is a **test-only** escape hatch: `transcribe.load_model()` returns
> immediately without importing NeMo/torch, so `handler.py` imports on a machine with no GPU.
> An actual transcription still needs a real model — never set this in production.
>
> Note: `runpod.serverless.progress_update(...)` does **not** deliver in pure-local
> `python handler.py` runs; validate progress under `--rp_serve_api` if you depend on it.

---

## Self-host on your own GPU

The RunPod SDK can expose the handler as a local FastAPI server with the same `/run`, `/runsync`,
and `/status` routes the cloud endpoint provides — handy for running the worker on your own GPU box:

```bash
python handler.py --rp_serve_api --rp_api_host 0.0.0.0 --rp_api_port 8000
```

Flags (from the RunPod SDK):

- `--rp_serve_api` — start the local API server.
- `--rp_api_host` — bind address (default `localhost`; use `0.0.0.0` for LAN access).
- `--rp_api_port` — port (default `8000`).
- `--rp_api_concurrency` — worker concurrency (the file must be `main.py` if `> 1`).
- `--rp_log_level` — `ERROR` / `WARN` / `INFO` / `DEBUG`.

Then call it like the cloud endpoint (short clips can use `/runsync`):

```bash
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_url": "https://.../clip.wav", "return_timestamps": true}}'
```

---

## Caller flow (async `/run` + polling, optional webhook)

The primary interface is **async** — a 10-hour job far exceeds any synchronous timeout, so
`/runsync` is **not** a supported path for long audio (use it only for short clips / local testing).

### Submit a job

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"input": {
        "audio_url": "https://<presigned-GET-url>",
        "return_timestamps": true,
        "result_upload_url": "https://<presigned-PUT-url>"
      }}'
# -> { "id": "<JOB_ID>", "status": "IN_QUEUE" }
```

### Poll for the result

```bash
curl https://api.runpod.ai/v2/<ENDPOINT_ID>/status/<JOB_ID> \
  -H "Authorization: Bearer <RUNPOD_API_KEY>"
# status: IN_QUEUE -> IN_PROGRESS -> COMPLETED (output present) | FAILED
```

While running, the worker may emit `progress_update` messages (e.g. `"Transcribed 3/8 chunks"`)
that appear in the `status` response.

### Webhook (optional)

Add a `webhook` URL to the submit body to be notified on completion instead of polling:

```bash
-d '{"input": { ... }, "webhook": "https://your-server.example.com/runpod-callback"}'
```

RunPod POSTs the final job object (same shape as the `/status` payload) to that URL when the job
finishes.

---

## Request / response schemas

### Request (`job["input"]`)

```jsonc
{
  "audio_url": "https://...",          // REQUIRED. Presigned HTTPS GET URL to the audio file.
  "return_timestamps": true,           // OPTIONAL, default true. If false, omit segments & words.
  "language": "en",                    // OPTIONAL, default "en". ISO code; routes to a backend.
  "result_upload_url": "https://..."   // OPTIONAL. Presigned HTTPS PUT URL for large-result offload.
}
```

Validation rules (see `schemas.parse_request`):

| Condition | Result |
|---|---|
| `audio_url` missing / not a string | error `MISSING_AUDIO_URL` |
| `audio_url` scheme not `https` (or empty host) | error `INVALID_URL` |
| `return_timestamps` not a boolean | coerced to `true` (does **not** fail) |
| `language` present but not a non-empty string | error `UNSUPPORTED_LANGUAGE` |
| `result_upload_url` present and not an `https` URL | error `INVALID_URL` |

Any non-empty language code is accepted and routed to a backend (Whisper is the catch-all),
so real ISO codes never fail validation. See [Model routing](#model-routing).

### Success response (inline)

Returned directly when the serialized output is under `RESULT_OFFLOAD_THRESHOLD_BYTES`:

```jsonc
{
  "text": "Full transcript with punctuation and capitalization.",
  "segments": [                                   // present only if return_timestamps=true
    { "start": 0.0, "end": 5.2, "text": "First segment text." }
  ],
  "words": [                                      // present only if return_timestamps=true
    { "start": 0.1, "end": 0.4, "word": "Hello" }
  ],
  "meta": {
    "model": "nvidia/parakeet-tdt-0.6b-v2",
    "language": "en",
    "audio_duration_sec": 123.4,
    "processing_time_sec": 8.1,
    "rtf": 15.235,                                // see "RTF" below
    "chunked": false,                             // true if the long-audio path was used
    "num_chunks": 1,
    "worker_version": "0.1.0"
  }
}
```

- `text` is **always** present.
- `segments` and `words` are present **only** when `return_timestamps` is `true`. When `false`,
  both keys are **omitted entirely** (not `null`, not `[]`).
- Timestamps are floats in **seconds**, relative to the start of the full audio (after stitching for
  chunked runs).

#### RTF

`rtf` is defined as:

```text
rtf = audio_duration_sec / processing_time_sec
```

It is a **speed factor** — higher is faster (e.g. `15.0` means 15× real-time). When
`processing_time_sec <= 0`, `rtf` is reported as `0.0`. The benchmark uses this same convention.
(NVIDIA's "RTFx" uses the same speed-factor direction; the benchmark also reports the inverse
`processing_time / audio_duration` and labels which is which.)

### Offloaded response (large results)

A 10-hour transcript with full word timestamps can exceed RunPod's payload caps
(`/run` = **10 MB**, `/runsync` = **20 MB**). When the serialized output exceeds
`RESULT_OFFLOAD_THRESHOLD_BYTES` (~8 MB, safely under the 10 MB cap) **and** a `result_upload_url`
was provided, the worker PUTs the **full** §3.2 object as JSON to that URL and returns a small
reference instead:

```jsonc
{
  "result_url": "https://...",         // the PUT location, with the query string stripped
  "text": "...",                       // full transcript kept inline (small relative to timestamps)
  "meta": { ... },                     // always inline
  "offloaded": true                    // words & segments live in the uploaded JSON, not inline
}
```

The uploaded JSON file **is** the complete inline success object (with `words`/`segments`).

> `result_url` is the **object location** with the presigned **signature stripped** (the signature is
> a secret and is never echoed back). To fetch the stored JSON, the caller uses its own
> presigned/authenticated GET for that object — it generated the PUT URL, so it can generate a GET.

### Error response

Validation and known failures are returned as the job **output** (the job still completes):

```jsonc
{ "error": { "code": "INVALID_URL", "message": "audio_url must be an https URL." } }
```

Genuinely unexpected failures (e.g. CUDA OOM, model load failure → `TranscriptionError`) are
**raised** so RunPod marks the job `FAILED`.

| Code | When |
|---|---|
| `MISSING_AUDIO_URL` | `audio_url` missing or not a string |
| `INVALID_URL` | `audio_url` or `result_upload_url` not a valid `https` URL |
| `DOWNLOAD_FAILED` | ffmpeg/ffprobe can't fetch the audio URL: HTTP 4xx/5xx, TLS, DNS, connection error, or `-rw_timeout` stall |
| `UNSUPPORTED_FORMAT` | ffmpeg cannot decode the input (normalize failed) |
| `UNSUPPORTED_LANGUAGE` | `language` present but not a non-empty string |
| `AUDIO_TOO_LONG` | probed duration exceeds `MAX_AUDIO_SECONDS` (default 10 h) |
| `FFMPEG_FAILED` | ffprobe/ffmpeg failed for a non-decode reason (e.g. window extraction) |
| `TRANSCRIPTION_FAILED` | inference error — **raised**, job marked `FAILED` |
| `RESULT_TOO_LARGE` | output over threshold but **no** `result_upload_url` provided |
| `RESULT_UPLOAD_FAILED` | the PUT to `result_upload_url` failed: a 4xx/3xx, or a 5xx/transport error after retries (the worker retries transient 5xx/transport failures up to 3× before giving up) |

> **`RESULT_TOO_LARGE`** is returned (not raised) when the result exceeds the offload threshold and
> the caller did not supply a `result_upload_url`. The worker never silently truncates — for
> large/long audio you **must** pass a presigned PUT URL.

---

## Model routing

The `language` code selects the transcription backend. Each model is loaded **lazily on
first use** for its language and kept warm for the life of the worker (a worker that only
sees English never loads the other three). All weights are baked into the image at build
time (`scripts/prefetch_model.py`), so the first request for a language pays no download.

| Language(s) | Backend | Model |
|---|---|---|
| `en` | Parakeet (NeMo) | `nvidia/parakeet-tdt-0.6b-v2` |
| `zh`, `yue`, `ja`, `ko` | SenseVoice (FunASR) | `FunAudioLLM/SenseVoiceSmall` |
| `de`, `fr`, `es`, `it`, `pl`, `ro`, `da`, `sv`, `nl`, `pt` | Canary (NeMo) | `nvidia/canary-1b-v2` |
| everything else | Whisper (transformers) | `openai/whisper-large-v3` |

`meta.model` in the response reports the model actually used. Notes:

- Only the **Parakeet** path uses the attention-switching + gap-retry long-audio machinery.
  The other three reuse the same overlapping-chunk planner/stitcher but without gap-retry.
- **SenseVoice** returns text only (no word timestamps): `words` is empty and `segments`
  are whole-chunk. Word/segment timestamps come from Parakeet, Canary, and Whisper.
- The exact model call signatures (Canary `source_lang`/`target_lang`, SenseVoice/Whisper
  kwargs) are **MUST-VALIDATE-ON-GPU** — coded from the model cards, not yet run on a GPU.

## How long audio is handled

- **Short path** (`duration <= SINGLE_PASS_MAX_SEC`, default 1440 s / 24 min): one
  `transcribe([wav], timestamps=True)` call with the model's default global attention.
- **Long path** (`duration > SINGLE_PASS_MAX_SEC`): split the WAV into overlapping windows
  (`CHUNK_SEC` with `CHUNK_OVERLAP_SEC` overlap), transcribe each, then stitch. Chunks whose
  length is at or below `SINGLE_PASS_MAX_SEC` use the model's default global attention for better
  accuracy in `balanced`/`best` mode. `fast` mode, and larger chunk windows, use local windowed
  attention (`rel_pos_local_attn [128,128]` + conv-chunking factor) to bound VRAM.

**Stitching / dedup rule** (unit-tested in `tests/test_chunking.py`): each chunk's chunk-local
timestamps are offset to absolute time by its start; in the overlap region between chunks, a token
is kept by whichever chunk **owns** that time, where ownership boundaries are the midpoints of the
overlap regions. A token is kept by chunk *i* iff its center `(start+end)/2` falls in chunk *i*'s
ownership window. This guarantees no dropped or doubled words at seams.

After stitching, the worker scans for large internal transcript gaps. If a gap is at least
`GAP_RETRY_MIN_SEC`, it extracts that interval with `GAP_RETRY_PADDING_SEC` of context on both
sides, retranscribes it with global attention, and splices only tokens whose centers fall inside
the original missing interval. The response `meta` includes `gap_retry_count` and
`gap_retry_recovered` for chunked runs.

---

## Configuration (env vars)

All parsed and validated in `config.py` at startup; malformed values raise a clear `ValueError`
immediately (fail fast).

| Var | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | `nvidia/parakeet-tdt-0.6b-v2` | English/Parakeet checkpoint. |
| `CANARY_MODEL` | `nvidia/canary-1b-v2` | Checkpoint for the listed EU languages. |
| `SENSEVOICE_MODEL` | `FunAudioLLM/SenseVoiceSmall` | Checkpoint for `zh`/`yue`/`ja`/`ko`. |
| `WHISPER_MODEL` | `openai/whisper-large-v3` | Catch-all checkpoint for all other languages. |
| `TRANSCRIPTION_QUALITY` | `balanced` | Preset for long-audio quality/speed. `fast` = old 20-min local-attention chunks, no gap retry. `balanced` = 5-min global-attention chunks + gap retry. `best` = 3-min global-attention chunks + gap retry. Explicit chunk/retry env vars override the preset defaults. |
| `MAX_AUDIO_SECONDS` | `36000` | Reject audio longer than this (10 h) → `AUDIO_TOO_LONG`. |
| `SINGLE_PASS_MAX_SEC` | `1440` | `<=` this → single-pass; above → chunked (24 min). |
| `CHUNK_SEC` | `300` in `balanced` | Chunk length for the long-audio path. |
| `CHUNK_OVERLAP_SEC` | `20` in `balanced` | Overlap between chunks (avoids clipping words at seams). |
| `GAP_RETRY_ENABLED` | `true` in `balanced`/`best` | Retry large internal transcript gaps after initial stitching. |
| `GAP_RETRY_MIN_SEC` | `20` | Minimum timestamp gap that triggers a retry. |
| `GAP_RETRY_PADDING_SEC` | `5` | Seconds of context added before/after each retried gap. |
| `GAP_RETRY_MAX_SEC` | `300` | Skip retry windows larger than this to avoid runaway recovery jobs. |
| `DOWNLOAD_TIMEOUT_SEC` | `120` | ffmpeg/ffprobe network read timeout when streaming the input URL (passed as `-rw_timeout`, converted to µs). Not a total deadline. |
| `RESULT_OFFLOAD_THRESHOLD_BYTES` | `8000000` | Offload result above ~8 MB (under the 10 MB `/run` cap). |
| `RESULT_UPLOAD_TIMEOUT_SEC` | `120` | Timeout for the PUT to `result_upload_url`. |
| `SKIP_MODEL_LOAD` | `false` | **Test-only.** Skip NeMo/torch import so the handler imports without a GPU. Never set in production. |

No storage credentials are configured on the worker — result offload always uses the
caller-provided presigned `result_upload_url`.

---

## Security model / trust boundary

v0 treats the **RunPod API caller as trusted** (the system operator). There is no public API
gateway, auth, or billing in front of the worker — those are explicitly out of scope. Within that
boundary the worker still takes the following precautions, and leaves a few residual risks worth
knowing about:

- **Caller-supplied URLs.** Both `audio_url` (GET) and `result_upload_url` (PUT) are caller-controlled
  and must be `https` (validated up front). The **result PUT** goes through `httpx` with
  `follow_redirects=False`, so a 30x to a loopback / link-local / cloud-metadata address
  (e.g. `169.254.169.254`) is never chased — a redirect surfaces as a clean `RESULT_UPLOAD_FAILED`.
  The **audio GET is now streamed directly by ffmpeg/ffprobe** (see below), which *may* follow HTTP
  redirects and has no download byte-cap — a deliberate tradeoff for zero-copy streaming under the
  trusted-caller model.
- **Input streaming (no download to disk).** `ffprobe`/`ffmpeg` read the presigned `audio_url`
  directly and stream it; the only file written is the normalized WAV. This drops the previous
  httpx download's `MAX_DOWNLOAD_BYTES` size-cap and no-redirect guard. Remaining bounds: the
  **`AUDIO_TOO_LONG` duration gate** (`ffprobe` runs first, before the full stream); a **`-t
  MAX_AUDIO_SECONDS` output cap** on the WAV (so a container that lies about its duration can't
  blow up disk); and `-rw_timeout` (`DOWNLOAD_TIMEOUT_SEC`) on a stalled read. ffmpeg's
  `-reconnect_on_network_error` / `-reconnect_on_http_error 5xx` let long streamed jobs survive a
  transient S3 drop or 503.
- **No stored credentials.** The worker holds no cloud keys; all object access goes through the
  caller's presigned URLs. Run it with **no attachable instance/cloud-metadata credentials** and, if
  possible, restricted egress, so even a future SSRF gap has nothing to reach.
- **Presigned signatures are secrets.** They are never logged or returned: `result_url` has its query
  string stripped, and `RESULT_UPLOAD_FAILED` messages report only the object location + an HTTP
  status / exception type, never the signed URL or a raw exception string.
- **Residual risks (v0):** the audio input has **no redirect guard and no byte-cap** (ffmpeg streams
  it); a malicious caller could redirect it or stream an oversized-but-short file. The result-PUT SSRF
  guard is still "don't follow redirects" rather than a full resolve-and-block-private-ranges check.
  `-rw_timeout` is per-read, so a very slow trickle within the duration limit is not bounded by a hard
  total deadline. If you expose this worker to untrusted callers, restore an input byte-cap + redirect
  guard (e.g. stream via httpx into `ffmpeg -i pipe:0`) and add a total deadline.

---

## Acceptance-criteria mapping

| # | Criterion (SPEC §11) | Where it's satisfied |
|---|---|---|
| 1 | Image with weights baked in; model loaded once at boot | `Dockerfile` prefetch step; `transcribe.load_model()` called at `handler.py` import |
| 2 | Short clip returns §3.2 shape with non-empty `text`, monotonic `segments`, in-range `words` | `transcribe.run` single-pass + `schemas.build_output` |
| 3 | `return_timestamps: false` omits `segments` & `words` | `schemas.build_output` (keys omitted) |
| 4 | ~30-min file via chunked path, `chunked=true`, correctly stitched | `transcribe.run` long path + `chunking.stitch` |
| 5 | File > `MAX_AUDIO_SECONDS` → `AUDIO_TOO_LONG` | `handler.py` duration gate |
| 6 | Bad/missing/non-https URL & undecodable file → correct structured error | `schemas.parse_request`, `audio.probe_duration`/`normalize`, `handler.py` catches |
| 7 | Over-threshold output → `result_url` + `offloaded:true`; no URL → `RESULT_TOO_LARGE` | `storage.maybe_offload` |
| 8 | `meta` fully populated; `rtf` matches definition | `schemas.build_output` |
| 9 | `test_chunking.py` & `test_schemas.py` pass without a GPU | `pytest` (this repo) |
| 10 | Benchmark runs end-to-end, emits CSV + summary | `benchmark/` harness |

---

## References

- [Parakeet TDT 0.6B v2 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [NeMo ASR docs (long audio / local attention)](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)
- [RunPod handler functions](https://docs.runpod.io/serverless/workers/handler-functions)
- [RunPod Python SDK](https://github.com/runpod/runpod-python)
