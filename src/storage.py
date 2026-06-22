"""Result offloading for outputs that exceed the RunPod payload limit (SPEC §8.3).

RunPod caps job outputs (``/run`` = 10 MB, ``/runsync`` = 20 MB). A long
transcript with full word/segment timestamps can blow past that, so this module
decides — based on the serialized size of the success object — whether to return
it inline or to PUT the full JSON to a caller-provided presigned URL and return
a small reference object instead.

The worker stores no credentials of its own: all offload uploads go through the
caller-supplied presigned ``result_upload_url`` (SPEC §3.1, §8.3).

``httpx`` is imported lazily inside :func:`maybe_offload` so the module can be
imported in GPU-free / dependency-light unit tests (httpx is a worker-image
dependency, not a stdlib module).
"""

from __future__ import annotations

import json
from urllib.parse import urlsplit, urlunsplit

from src import config


def _strip_query(url: str) -> str:
    """Return ``url`` with its query string and fragment removed.

    A presigned PUT URL carries the signature in the query string. We echo back
    the object location (scheme://netloc/path) without leaking that signature.
    """
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def maybe_offload(output: dict, result_upload_url: str | None) -> dict:
    """Return ``output`` inline, or offload it and return a reference (SPEC §8.3).

    Behavior, keyed on the serialized size of ``output``:

    * size <= :data:`config.RESULT_OFFLOAD_THRESHOLD_BYTES` -> return ``output``
      unchanged (the normal, inline case).
    * size over the threshold and ``result_upload_url`` is ``None`` -> return a
      structured ``RESULT_TOO_LARGE`` error: the caller must supply a presigned
      PUT URL for large/long audio. We never silently truncate.
    * size over the threshold and a ``result_upload_url`` is present -> HTTP PUT
      the FULL serialized JSON (``Content-Type: application/json``) to the URL.
      On any non-2xx status or transport error -> return a ``RESULT_UPLOAD_FAILED``
      error. On success -> return a small reference object keeping ``text`` and
      ``meta`` inline, with ``offloaded: True`` and the object location in
      ``result_url`` (query string stripped). The uploaded JSON IS the complete
      §3.2 success object — ``words``/``segments`` live only there.

    This function never raises for an expected failure; it returns an error
    object so the job still completes (SPEC §3.3, caller-error semantics).
    """
    # Serialize once and measure the on-the-wire UTF-8 byte length (not the
    # Python str length) so multi-byte characters are counted correctly.
    serialized = json.dumps(output)
    size = len(serialized.encode("utf-8"))

    if size <= config.RESULT_OFFLOAD_THRESHOLD_BYTES:
        return output

    if result_upload_url is None:
        return {
            "error": {
                "code": config.ErrorCode.RESULT_TOO_LARGE,
                "message": (
                    f"Result is {size} bytes, exceeding the "
                    f"{config.RESULT_OFFLOAD_THRESHOLD_BYTES}-byte inline limit. "
                    "Supply a presigned PUT 'result_upload_url' so the worker can "
                    "offload large/long-audio results."
                ),
            }
        }

    # Lazy import: httpx is a worker-image dependency, not stdlib, so importing
    # it at module top would break dependency-light unit tests.
    import time

    import httpx

    body = serialized.encode("utf-8")
    # Echo back only the object location, never the signed URL — the presigned
    # signature is a secret and must not leak into the response or logs.
    safe_url = _strip_query(result_upload_url)

    attempts = 3
    last_error = "unknown error"
    for attempt in range(1, attempts + 1):
        try:
            response = httpx.put(
                result_upload_url,
                content=body,
                headers={"Content-Type": "application/json"},
                timeout=config.RESULT_UPLOAD_TIMEOUT_SEC,
                # No redirects: a presigned PUT resolves directly. (Same SSRF
                # reasoning as the download path.)
                follow_redirects=False,
            )
        except Exception as exc:  # noqa: BLE001 - see below
            # Broad catch is deliberate and spec-aligned: a failed upload is a
            # DEFINED returned caller-error (RESULT_UPLOAD_FAILED, SPEC §8.3), not
            # a FAILED-worthy internal error, so we convert it rather than let it
            # propagate. Report only the exception TYPE — str(exc) can embed the
            # signed URL, which must never leak into the response or logs.
            last_error = f"transport error ({type(exc).__name__})"
        else:
            status = response.status_code
            if 200 <= status < 300:
                return {
                    "result_url": safe_url,
                    "text": output["text"],
                    "meta": output["meta"],
                    "offloaded": True,
                }
            # A 3xx is NOT success (we don't follow redirects) and is reported as
            # a failure rather than silently dropping the upload. 4xx is a caller
            # fault (e.g. expired/invalid presigned URL) and not worth retrying;
            # 5xx may be transient, so fall through to the retry.
            last_error = f"HTTP {status}"
            if 300 <= status < 500:
                break

        if attempt < attempts:
            time.sleep(1.0 * attempt)  # brief linear backoff between retries

    return {
        "error": {
            "code": config.ErrorCode.RESULT_UPLOAD_FAILED,
            "message": (
                f"Failed to upload result to {safe_url} after {attempts} "
                f"attempt(s): {last_error}."
            ),
        }
    }
