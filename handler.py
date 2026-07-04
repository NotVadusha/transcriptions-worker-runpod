"""Repository-root RunPod entrypoint.

The implementation lives in ``src.handler``; this wrapper keeps the conventional
root-level handler file that RunPod's GitHub import checks expect to find.
"""

import runpod

from src.handler import handler


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
