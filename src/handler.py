"""RunPod serverless entrypoint for the transcription worker.

Expected job input (subject to change):
    {
        "input": {
            "audio_url": "https://.../file.mp3"
        }
    }
"""

import runpod


def handler(job):
    """Process a single transcription job.

    Args:
        job: RunPod job dict. The payload is under ``job["input"]``.

    Returns:
        A dict with the transcription result.
    """
    job_input = job.get("input", {})

    # TODO: load the model, fetch the audio, and run transcription.
    audio_url = job_input.get("audio_url")

    return {
        "status": "not_implemented",
        "received": {"audio_url": audio_url},
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
