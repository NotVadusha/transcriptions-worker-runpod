"""Long-audio segmentation planning and timestamp stitching.

This module is PURE logic (SPEC §5.4). It imports NOTHING that touches a GPU,
NeMo, ffmpeg, or the network — only the Python standard library — so it can be
unit-tested without a model. ``transcribe.py`` uses it to plan overlapping
chunks and to merge per-chunk transcription outputs back into a single,
absolute-timeline transcript.

Dedup rule (the part that must be airtight)
-------------------------------------------
When chunks overlap, the same word/segment can be emitted by two adjacent
chunks. We deduplicate with an *ownership window* per chunk so every token is
kept exactly once and nothing is dropped or doubled at a seam.

For chunk ``i`` (0-indexed) with plan windows ``plan[i]``:

    lo_i = (plan[i].start + plan[i-1].end) / 2   for i >= 1, else -inf
    hi_i = (plan[i+1].start + plan[i].end) / 2   for i <  N-1, else +inf

These are the midpoints of the overlap regions on either side of chunk ``i``.
A token (word or segment) — after being offset to *absolute* time — is owned by
chunk ``i`` iff its CENTER ``(start + end) / 2`` satisfies::

    lo_i <= center < hi_i

Because each adjacent pair of chunks splits their shared overlap at the same
midpoint, and the windows ``[lo_i, hi_i)`` are half-open and contiguous, every
point on the timeline belongs to exactly one chunk. A token therefore survives
in exactly one chunk's filter — no drops, no doubles. Non-overlapping or gapped
chunks degrade gracefully: the midpoints simply fall in the gap/seam and each
token is still owned by exactly one chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

__all__ = ["ChunkPlan", "plan_chunks", "offset_items", "stitch"]


@dataclass(frozen=True)
class ChunkPlan:
    """A single chunk window into the source audio.

    Times are absolute seconds into the source audio. ``end`` is an exclusive
    upper bound, clamped to the audio duration.
    """

    index: int
    start: float
    end: float


def plan_chunks(duration_sec: float, chunk_sec: float, overlap_sec: float) -> list[ChunkPlan]:
    """Plan overlapping chunk windows covering ``[0, duration_sec]``.

    Windows are ``chunk_sec`` long and advance by ``step = chunk_sec - overlap_sec``.
    The final window is clamped to ``duration_sec`` and no zero-length final
    chunk is produced.

    Raises ``ValueError`` if inputs are non-positive or ``overlap_sec >= chunk_sec``.
    """
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0, got {duration_sec}")
    if chunk_sec <= 0:
        raise ValueError(f"chunk_sec must be > 0, got {chunk_sec}")
    if overlap_sec < 0:
        raise ValueError(f"overlap_sec must be >= 0, got {overlap_sec}")
    if overlap_sec >= chunk_sec:
        raise ValueError(
            f"overlap_sec ({overlap_sec}) must be < chunk_sec ({chunk_sec})"
        )

    step = chunk_sec - overlap_sec
    plans: list[ChunkPlan] = []
    start = 0.0
    index = 0
    while start < duration_sec:
        end = min(start + chunk_sec, duration_sec)
        plans.append(ChunkPlan(index=index, start=start, end=end))
        index += 1
        # If this window already reached the end of the audio we are done;
        # advancing further would only produce zero-length / past-the-end chunks.
        if end >= duration_sec:
            break
        start += step

    return plans


def offset_items(items: list[dict], t0: float) -> list[dict]:
    """Return NEW dicts with ``t0`` added to each item's ``start`` and ``end``.

    The input dicts are not mutated. All other keys are copied verbatim.
    """
    shifted: list[dict] = []
    for item in items:
        new_item = dict(item)
        new_item["start"] = item["start"] + t0
        new_item["end"] = item["end"] + t0
        shifted.append(new_item)
    return shifted


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip the ends."""
    return re.sub(r"\s+", " ", text).strip()


def _ownership_window(
    plans: list[ChunkPlan], i: int
) -> tuple[float, float]:
    """Return the half-open ownership window ``[lo, hi)`` for chunk ``i``.

    ``lo`` is the midpoint of the overlap with the previous chunk (or -inf for
    the first chunk); ``hi`` is the midpoint of the overlap with the next chunk
    (or +inf for the last chunk). See the module docstring for why this yields
    a contiguous, non-overlapping partition of the timeline.
    """
    n = len(plans)
    if i >= 1:
        lo = (plans[i].start + plans[i - 1].end) / 2.0
    else:
        lo = -math.inf
    if i < n - 1:
        hi = (plans[i + 1].start + plans[i].end) / 2.0
    else:
        hi = math.inf
    return lo, hi


def _keep_owned(items: list[dict], lo: float, hi: float) -> list[dict]:
    """Keep items whose center ``(start + end) / 2`` lies in ``[lo, hi)``."""
    kept: list[dict] = []
    for item in items:
        center = (item["start"] + item["end"]) / 2.0
        if lo <= center < hi:
            kept.append(item)
    return kept


def stitch(chunk_outputs: list[dict]) -> dict:
    """Merge per-chunk transcription outputs into a single absolute-time result.

    Each element of ``chunk_outputs`` is::

        {
            "plan": ChunkPlan,
            "words": [{"start", "end", "word"}, ...],     # CHUNK-LOCAL times
            "segments": [{"start", "end", "text"}, ...],  # CHUNK-LOCAL times
        }

    where word/segment times are relative to that chunk's own start (i.e. the
    chunk wav began at 0). This function:

    1. Offsets every chunk's words/segments by ``plan.start`` to absolute time.
    2. Keeps only the tokens each chunk *owns* via the ownership-window rule
       (see module docstring) — guaranteeing no dropped/doubled tokens at seams.
    3. Concatenates kept tokens across chunks in order.
    4. Builds ``text`` by joining kept segments' ``text`` (collapsing whitespace);
       if there are no segments at all, joins kept words' ``word`` instead.

    Returns ``{"text": str, "words": list[dict], "segments": list[dict]}``.
    """
    if not chunk_outputs:
        return {"text": "", "words": [], "segments": []}

    plans = [co["plan"] for co in chunk_outputs]

    all_words: list[dict] = []
    all_segments: list[dict] = []

    for i, co in enumerate(chunk_outputs):
        plan = plans[i]
        lo, hi = _ownership_window(plans, i)

        abs_words = offset_items(co.get("words", []), plan.start)
        abs_segments = offset_items(co.get("segments", []), plan.start)

        all_words.extend(_keep_owned(abs_words, lo, hi))
        all_segments.extend(_keep_owned(abs_segments, lo, hi))

    # Ownership windows partition the timeline, but tokens are appended in chunk
    # order. A token that straddles a seam can land out of start-order relative
    # to a neighbour in the adjacent chunk (a later chunk's first token may start
    # inside the prior overlap, before a previous chunk's last kept token). Sort
    # by (start, end) so the emitted timeline is monotonic, as required by
    # SPEC §3.2 and acceptance criteria #2/#4. Python's sort is stable.
    all_words.sort(key=lambda x: (x["start"], x["end"]))
    all_segments.sort(key=lambda x: (x["start"], x["end"]))

    if all_segments:
        text = _collapse_whitespace(" ".join(seg["text"] for seg in all_segments))
    else:
        text = _collapse_whitespace(" ".join(w["word"] for w in all_words))

    return {"text": text, "words": all_words, "segments": all_segments}
