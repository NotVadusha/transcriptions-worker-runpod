# Benchmark datasets (local, gitignored)

Everything in this directory **except this README** is gitignored (SPEC §10.4) —
it likely contains real customer audio. Drop your benchmark files here locally;
they are never committed.

## Expected layout

Put each audio file next to a `.txt` file with the same stem holding its human
reference transcript:

```
benchmark/datasets/
├── README.md            # this file (the only committed thing here)
├── manifest.json        # OPTIONAL — see below
├── call_0001.wav        # audio (any ffmpeg-decodable format: wav/flac/mp3/m4a/mp4/ogg/opus/webm)
├── call_0001.txt        # human reference transcript for call_0001
├── call_0002.mp3
├── call_0002.txt
└── ...
```

Pairing rule: `foo.<audio-ext>` pairs with `foo.txt`. Audio without a matching
`.txt` is skipped with a warning. All audio is normalized to 16 kHz mono WAV
**once** (into `results/_normalized_wav/`) and the same WAV is fed to every
system, so differences aren't preprocessing artifacts.

## `manifest.json` (optional)

Maps a file stem to extra per-file fields:

```json
{
  "call_0001": {
    "audio_url": "https://your-bucket.s3.amazonaws.com/call_0001.wav?X-Amz-...",
    "ref_words": [
      {"start": 0.10, "end": 0.42, "word": "Hello"},
      {"start": 0.45, "end": 0.88, "word": "there"}
    ]
  }
}
```

- `audio_url` — required **only** when benchmarking Parakeet through the deployed
  RunPod endpoint (`--parakeet-endpoint`), which transcribes from an https URL
  rather than a local file. Not needed for `--parakeet-local`, AssemblyAI, or
  Whisper (those read the local WAV).
- `ref_words` — optional word-level reference alignment (`{start, end, word}`,
  seconds). When present, the timestamp-quality metric (mean abs start error,
  % within ±200 ms) is computed; when absent, those columns are left blank.

## Run

```bash
python benchmark/run_benchmark.py \
    --parakeet-endpoint https://api.runpod.ai/v2/<endpoint-id> \
    --gpu-cost-per-hour 0.79 \
    --assemblyai-price-per-hour 0.37
```

Outputs land in `benchmark/results/` (`runs.csv`, `summary.csv`, `verdict.md`).
