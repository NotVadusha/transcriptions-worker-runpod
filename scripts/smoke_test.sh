#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — end-to-end smoke test against a DEPLOYED RunPod endpoint
# (SPEC §9 / §11). Submits a short clip and a ~30-min clip via the ASYNC /run
# endpoint, polls /status until completion, and checks the response shape +
# word-timestamp monotonicity with jq.
#
# Async /run + polling is used (not /runsync) because long audio exceeds any
# sync timeout (SPEC §12.4).
#
# Requirements: bash, curl, jq.
#
# Env vars:
#   RUNPOD_ENDPOINT_ID   (required)  RunPod serverless endpoint id.
#   RUNPOD_API_KEY       (required)  RunPod API key (Bearer token).
#   SHORT_AUDIO_URL      (optional)  presigned HTTPS URL to a short clip.
#                                    Default: the public LibriSpeech sample.
#   LONG_AUDIO_URL       (optional)  presigned HTTPS URL to a ~30-min clip.
#                                    If unset, the long-audio case is SKIPPED.
#   RESULT_UPLOAD_URL    (optional)  presigned HTTPS PUT URL for result offload
#                                    (needed if the long result exceeds the
#                                    payload threshold -> RESULT_TOO_LARGE).
#   POLL_INTERVAL_SEC    (optional)  seconds between /status polls (default 10).
#   POLL_TIMEOUT_SEC     (optional)  max seconds to wait per job (default 2400).
#
# Usage:
#   RUNPOD_ENDPOINT_ID=xxxx RUNPOD_API_KEY=yyyy ./scripts/smoke_test.sh
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
# Preconditions
# --------------------------------------------------------------------------- #
for bin in curl jq; do
    if ! command -v "$bin" >/dev/null 2>&1; then
        echo "ERROR: required tool '$bin' not found on PATH." >&2
        exit 2
    fi
done

: "${RUNPOD_ENDPOINT_ID:?Set RUNPOD_ENDPOINT_ID to your RunPod endpoint id}"
: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY to your RunPod API key}"

SHORT_AUDIO_URL="${SHORT_AUDIO_URL:-https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav}"
LONG_AUDIO_URL="${LONG_AUDIO_URL:-}"
RESULT_UPLOAD_URL="${RESULT_UPLOAD_URL:-}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-10}"
POLL_TIMEOUT_SEC="${POLL_TIMEOUT_SEC:-2400}"

BASE_URL="https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}"
AUTH_HEADER="Authorization: Bearer ${RUNPOD_API_KEY}"

FAILURES=0

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# submit_job <audio_url> [result_upload_url]  -> echoes the RunPod job id.
submit_job() {
    local audio_url="$1"
    local upload_url="${2:-}"
    local payload

    if [[ -n "$upload_url" ]]; then
        payload=$(jq -n \
            --arg url "$audio_url" \
            --arg up "$upload_url" \
            '{input: {audio_url: $url, return_timestamps: true, result_upload_url: $up}}')
    else
        payload=$(jq -n \
            --arg url "$audio_url" \
            '{input: {audio_url: $url, return_timestamps: true}}')
    fi

    local resp
    resp=$(curl -sS -X POST "${BASE_URL}/run" \
        -H "$AUTH_HEADER" \
        -H "Content-Type: application/json" \
        -d "$payload")

    local job_id
    job_id=$(jq -r '.id // empty' <<<"$resp")
    if [[ -z "$job_id" ]]; then
        echo "ERROR: /run did not return a job id. Response:" >&2
        echo "$resp" >&2
        return 1
    fi
    echo "$job_id"
}

# poll_job <job_id>  -> echoes the final /status JSON, returns non-zero on
# COMPLETED-with-error / FAILED / timeout.
poll_job() {
    local job_id="$1"
    local waited=0
    local resp status

    while :; do
        resp=$(curl -sS "${BASE_URL}/status/${job_id}" -H "$AUTH_HEADER")
        status=$(jq -r '.status // "UNKNOWN"' <<<"$resp")

        case "$status" in
            COMPLETED)
                echo "$resp"
                return 0
                ;;
            FAILED|CANCELLED|TIMED_OUT)
                echo "  job $job_id ended with status=$status" >&2
                echo "$resp" >&2
                return 1
                ;;
            IN_QUEUE|IN_PROGRESS|UNKNOWN)
                if (( waited >= POLL_TIMEOUT_SEC )); then
                    echo "  job $job_id timed out after ${waited}s (last status=$status)" >&2
                    return 1
                fi
                sleep "$POLL_INTERVAL_SEC"
                waited=$(( waited + POLL_INTERVAL_SEC ))
                ;;
            *)
                echo "  job $job_id unexpected status=$status" >&2
                echo "$resp" >&2
                return 1
                ;;
        esac
    done
}

# check_output <label> <status_json>  -> validates response shape + monotonicity.
# Handles BOTH inline §3.2 success objects and the §8.3 offloaded shape.
check_output() {
    local label="$1"
    local status_json="$2"
    local out
    out=$(jq -c '.output' <<<"$status_json")

    if [[ "$out" == "null" || -z "$out" ]]; then
        echo "  [$label] FAIL: no .output in status response" >&2
        echo "$status_json" >&2
        return 1
    fi

    # A structured error object is a failure for the smoke test.
    if jq -e '.error' <<<"$out" >/dev/null 2>&1; then
        echo "  [$label] FAIL: worker returned error: $(jq -c '.error' <<<"$out")" >&2
        return 1
    fi

    # Offloaded result (§8.3): words/segments live in the uploaded JSON, only
    # text + meta + result_url are inline. Validate the offload shape and stop.
    if jq -e '.offloaded == true' <<<"$out" >/dev/null 2>&1; then
        if ! jq -e '(.result_url | type == "string") and (.text | type == "string") and (.meta | type == "object")' \
            <<<"$out" >/dev/null 2>&1; then
            echo "  [$label] FAIL: offloaded result missing result_url/text/meta" >&2
            return 1
        fi
        echo "  [$label] OK (offloaded): result_url present, meta.chunked=$(jq -r '.meta.chunked' <<<"$out")"
        return 0
    fi

    # Inline §3.2 success object: text + meta required.
    if ! jq -e '(.text | type == "string") and (.text | length > 0)' <<<"$out" >/dev/null 2>&1; then
        echo "  [$label] FAIL: missing/empty .text" >&2
        return 1
    fi
    if ! jq -e '.meta | type == "object"' <<<"$out" >/dev/null 2>&1; then
        echo "  [$label] FAIL: missing .meta" >&2
        return 1
    fi

    # meta sanity: required keys present.
    if ! jq -e '.meta | has("model") and has("rtf") and has("chunked") and has("num_chunks") and has("audio_duration_sec")' \
        <<<"$out" >/dev/null 2>&1; then
        echo "  [$label] FAIL: meta missing required keys" >&2
        echo "  meta=$(jq -c '.meta' <<<"$out")" >&2
        return 1
    fi

    # We requested return_timestamps:true, so words + segments must be present.
    if ! jq -e '(.words | type == "array") and (.segments | type == "array")' <<<"$out" >/dev/null 2>&1; then
        echo "  [$label] FAIL: words/segments missing despite return_timestamps=true" >&2
        return 1
    fi

    # Word-timestamp monotonicity: each word.end >= word.start, and word.start
    # is non-decreasing across the list. Also assert all times fall within the
    # probed audio duration (small epsilon).
    local mono
    mono=$(jq '
        .words as $w
        | ($w | length) as $n
        | [ range(0; $n) as $i
            | ($w[$i].start) as $s
            | ($w[$i].end) as $e
            | ( ($e >= $s)
                and ( ($i == 0) or ($s >= $w[$i-1].start - 0.001) ) )
          ]
        | all
    ' <<<"$out")
    if [[ "$mono" != "true" ]]; then
        echo "  [$label] FAIL: word timestamps not monotonic / end<start somewhere" >&2
        return 1
    fi

    local within
    within=$(jq '
        (.meta.audio_duration_sec) as $d
        | [ .words[] | (.start <= $d + 0.5) and (.end <= $d + 0.5) ] | all
    ' <<<"$out")
    if [[ "$within" != "true" ]]; then
        echo "  [$label] FAIL: some word timestamps exceed audio_duration_sec" >&2
        return 1
    fi

    echo "  [$label] OK: words=$(jq '.words | length' <<<"$out"), segments=$(jq '.segments | length' <<<"$out"), chunked=$(jq -r '.meta.chunked' <<<"$out"), rtf=$(jq -r '.meta.rtf' <<<"$out")"
    return 0
}

# run_case <label> <audio_url> [result_upload_url]
run_case() {
    local label="$1"
    local audio_url="$2"
    local upload_url="${3:-}"

    echo "==> [$label] submitting $audio_url"
    local job_id
    if ! job_id=$(submit_job "$audio_url" "$upload_url"); then
        FAILURES=$(( FAILURES + 1 ))
        return
    fi
    echo "    job_id=$job_id ; polling /status (interval=${POLL_INTERVAL_SEC}s, timeout=${POLL_TIMEOUT_SEC}s)"

    local status_json
    if ! status_json=$(poll_job "$job_id"); then
        FAILURES=$(( FAILURES + 1 ))
        return
    fi

    if ! check_output "$label" "$status_json"; then
        FAILURES=$(( FAILURES + 1 ))
    fi
}

# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
echo "RunPod smoke test against endpoint ${RUNPOD_ENDPOINT_ID}"
echo

run_case "short" "$SHORT_AUDIO_URL"

echo
if [[ -n "$LONG_AUDIO_URL" ]]; then
    run_case "long(~30min)" "$LONG_AUDIO_URL" "$RESULT_UPLOAD_URL"
else
    echo "==> [long(~30min)] SKIPPED (set LONG_AUDIO_URL to enable the chunked-path test)"
fi

echo
if (( FAILURES > 0 )); then
    echo "SMOKE TEST FAILED: ${FAILURES} case(s) failed." >&2
    exit 1
fi
echo "SMOKE TEST PASSED."
