# transcriptions-worker-runpod

A [RunPod](https://www.runpod.io/) serverless worker for audio/video transcription.

## Overview

This repo packages a transcription handler that runs as a RunPod serverless
endpoint. It receives a job (e.g. an audio/video URL or base64 payload),
runs transcription, and returns the resulting text.

## Project layout

```
.
├── src/
│   └── handler.py        # RunPod serverless entrypoint
├── Dockerfile            # Container image for the worker
├── requirements.txt      # Python dependencies
└── README.md
```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/handler.py        # runs the handler locally via runpod test mode
```

## Deploy

Build and push the image, then point a RunPod serverless endpoint at it:

```bash
docker build -t <registry>/transcriptions-worker-runpod:latest .
docker push <registry>/transcriptions-worker-runpod:latest
```
