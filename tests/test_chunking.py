"""Unit tests for src/chunking.py — pure logic, no GPU/model/ffmpeg.

Covers (per task spec):
  1. plan_chunks: coverage/overlap/last-chunk clamp, and overlap >= chunk_sec raises.
  2. offset_items: arithmetic and non-mutation.
  3. stitch: monotonic, non-overlapping output with NO dropped and NO doubled
     tokens across a seam — including a word that appears in both chunks'
     overlap, asserted to appear exactly once at the correct absolute time.
  4. A 3-chunk case.
"""

import math

import pytest

from src import chunking
from src.chunking import ChunkPlan, offset_items, plan_chunks, stitch


# ---------------------------------------------------------------------------
# 1. plan_chunks
# ---------------------------------------------------------------------------


def test_plan_chunks_single_window_when_shorter_than_chunk():
    plans = plan_chunks(duration_sec=500.0, chunk_sec=1200.0, overlap_sec=15.0)
    assert plans == [ChunkPlan(index=0, start=0.0, end=500.0)]


def test_plan_chunks_exact_multiple_step_and_clamp():
    # chunk=1200, overlap=15 -> step=1185. duration=2400.
    plans = plan_chunks(duration_sec=2400.0, chunk_sec=1200.0, overlap_sec=15.0)
    assert len(plans) == 3
    assert plans[0] == ChunkPlan(0, 0.0, 1200.0)
    assert plans[1] == ChunkPlan(1, 1185.0, 2385.0)
    # last chunk clamped to duration, not 1185*2 + 1200 = 3570
    assert plans[2] == ChunkPlan(2, 2370.0, 2400.0)


def test_plan_chunks_covers_full_timeline_no_gaps():
    duration = 5000.0
    chunk = 1200.0
    overlap = 15.0
    plans = plan_chunks(duration, chunk, overlap)

    # Starts strictly increasing, indices sequential.
    assert [p.index for p in plans] == list(range(len(plans)))
    starts = [p.start for p in plans]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)

    # First starts at 0, last ends exactly at duration (clamp).
    assert plans[0].start == 0.0
    assert plans[-1].end == pytest.approx(duration)

    # Coverage: every consecutive pair overlaps (next.start < cur.end),
    # so the union is [0, duration] with no gap.
    for cur, nxt in zip(plans, plans[1:]):
        assert nxt.start < cur.end, "gap between chunks would drop audio"
        # advance step is chunk - overlap
        assert nxt.start == pytest.approx(cur.start + (chunk - overlap))


def test_plan_chunks_no_zero_length_final_chunk():
    # Pick a duration that lands a window boundary exactly on duration.
    # chunk=100, overlap=0 -> step=100, duration=300 -> windows [0,100],[100,200],[200,300].
    plans = plan_chunks(duration_sec=300.0, chunk_sec=100.0, overlap_sec=0.0)
    assert plans == [
        ChunkPlan(0, 0.0, 100.0),
        ChunkPlan(1, 100.0, 200.0),
        ChunkPlan(2, 200.0, 300.0),
    ]
    for p in plans:
        assert p.end > p.start


def test_plan_chunks_last_window_clamped_partial():
    # duration=250, chunk=100, overlap=0 -> [0,100],[100,200],[200,250].
    plans = plan_chunks(duration_sec=250.0, chunk_sec=100.0, overlap_sec=0.0)
    assert plans[-1] == ChunkPlan(2, 200.0, 250.0)
    assert plans[-1].end == 250.0


def test_plan_chunks_overlap_equal_to_chunk_raises():
    with pytest.raises(ValueError):
        plan_chunks(duration_sec=1000.0, chunk_sec=100.0, overlap_sec=100.0)


def test_plan_chunks_overlap_greater_than_chunk_raises():
    with pytest.raises(ValueError):
        plan_chunks(duration_sec=1000.0, chunk_sec=100.0, overlap_sec=150.0)


def test_plan_chunks_non_positive_inputs_raise():
    with pytest.raises(ValueError):
        plan_chunks(duration_sec=0.0, chunk_sec=100.0, overlap_sec=10.0)
    with pytest.raises(ValueError):
        plan_chunks(duration_sec=100.0, chunk_sec=0.0, overlap_sec=0.0)
    with pytest.raises(ValueError):
        plan_chunks(duration_sec=100.0, chunk_sec=50.0, overlap_sec=-1.0)


# ---------------------------------------------------------------------------
# 2. offset_items
# ---------------------------------------------------------------------------


def test_offset_items_arithmetic():
    items = [
        {"start": 0.0, "end": 0.5, "word": "Hello"},
        {"start": 1.0, "end": 1.7, "word": "world"},
    ]
    shifted = offset_items(items, 100.0)
    assert shifted == [
        {"start": 100.0, "end": 100.5, "word": "Hello"},
        {"start": 101.0, "end": 101.7, "word": "world"},
    ]


def test_offset_items_does_not_mutate_input_and_preserves_other_keys():
    items = [{"start": 2.0, "end": 3.0, "text": "A segment.", "extra": 42}]
    shifted = offset_items(items, 10.0)
    # original untouched
    assert items[0]["start"] == 2.0
    assert items[0]["end"] == 3.0
    # new dict, shifted, extra keys preserved
    assert shifted[0] is not items[0]
    assert shifted[0]["start"] == 12.0
    assert shifted[0]["end"] == 13.0
    assert shifted[0]["text"] == "A segment."
    assert shifted[0]["extra"] == 42


def test_offset_items_empty():
    assert offset_items([], 5.0) == []


# ---------------------------------------------------------------------------
# 3. stitch — seam dedup (no dropped / no doubled tokens)
# ---------------------------------------------------------------------------


def _words(*triples):
    return [{"start": s, "end": e, "word": w} for (s, e, w) in triples]


def _segments(*triples):
    return [{"start": s, "end": e, "text": t} for (s, e, t) in triples]


def test_stitch_empty():
    assert stitch([]) == {"text": "", "words": [], "segments": []}


def test_stitch_single_chunk_offsets_to_absolute():
    plan = ChunkPlan(0, 0.0, 100.0)
    co = {
        "plan": plan,
        "words": _words((1.0, 1.4, "Hello"), (2.0, 2.5, "world")),
        "segments": _segments((1.0, 2.5, "Hello world")),
    }
    out = stitch([co])
    assert out["text"] == "Hello world"
    assert out["words"] == _words((1.0, 1.4, "Hello"), (2.0, 2.5, "world"))
    assert out["segments"] == _segments((1.0, 2.5, "Hello world"))


def test_stitch_two_chunk_seam_word_appears_exactly_once():
    """A word spoken inside the overlap is emitted by BOTH chunks (chunk-local).
    After stitching it must appear exactly once, at its correct absolute time.

    Layout (absolute seconds):
      chunk0: plan [0, 100],   chunk1: plan [85, 185]   (overlap region [85, 100])
      overlap midpoint = (85 + 100) / 2 = 92.5

    The shared word "seam" is at absolute [90, 91] -> center 90.5 < 92.5,
    so it is owned by chunk0 only.
    Another shared word "bridge" at absolute [95, 96] -> center 95.5 >= 92.5,
    owned by chunk1 only.
    """
    plan0 = ChunkPlan(0, 0.0, 100.0)
    plan1 = ChunkPlan(1, 85.0, 185.0)

    # chunk0 LOCAL times == absolute (starts at 0).
    chunk0 = {
        "plan": plan0,
        "words": _words(
            (10.0, 10.5, "alpha"),
            (90.0, 91.0, "seam"),     # abs center 90.5 -> chunk0 owns
            (95.0, 96.0, "bridge"),   # abs center 95.5 -> chunk1 should own; chunk0 must NOT keep
        ),
        "segments": _segments((10.0, 96.0, "alpha seam bridge")),
    }
    # chunk1 LOCAL times = absolute - 85.
    chunk1 = {
        "plan": plan1,
        "words": _words(
            (90.0 - 85.0, 91.0 - 85.0, "seam"),    # abs [90,91] center 90.5 -> chunk0 owns; chunk1 must NOT keep
            (95.0 - 85.0, 96.0 - 85.0, "bridge"),  # abs [95,96] center 95.5 -> chunk1 owns
            (120.0 - 85.0, 121.0 - 85.0, "omega"), # abs [120,121]
        ),
        "segments": _segments((90.0 - 85.0, 121.0 - 85.0, "seam bridge omega")),
    }

    out = stitch([chunk0, chunk1])

    # Each shared word appears exactly once.
    word_list = [w["word"] for w in out["words"]]
    assert word_list.count("seam") == 1
    assert word_list.count("bridge") == 1
    assert word_list == ["alpha", "seam", "bridge", "omega"]

    # Correct absolute times.
    by_word = {w["word"]: (w["start"], w["end"]) for w in out["words"]}
    assert by_word["alpha"] == (10.0, 10.5)
    assert by_word["seam"] == (90.0, 91.0)
    assert by_word["bridge"] == (95.0, 96.0)
    assert by_word["omega"] == (120.0, 121.0)

    # Monotonic, non-overlapping in absolute time.
    starts = [w["start"] for w in out["words"]]
    assert starts == sorted(starts)
    for a, b in zip(out["words"], out["words"][1:]):
        assert a["end"] <= b["start"]


def test_stitch_no_token_dropped_at_seam_exhaustive_partition():
    """Place words densely across the overlap; assert the kept set is exactly
    the original distinct words (no drop) and has no duplicates (no double)."""
    plan0 = ChunkPlan(0, 0.0, 100.0)
    plan1 = ChunkPlan(1, 80.0, 180.0)  # overlap [80, 100], midpoint 90.0

    # Distinct words at absolute centers 85..95, all inside the overlap.
    abs_words = _words(
        (84.5, 85.5, "w85"),
        (86.5, 87.5, "w87"),
        (88.5, 89.5, "w89"),   # center 89.0 < 90.0 -> chunk0
        (90.5, 91.5, "w91"),   # center 91.0 >= 90.0 -> chunk1
        (92.5, 93.5, "w93"),
        (94.5, 95.5, "w95"),
    )
    # Both chunks "see" all overlap words (chunk-local times).
    chunk0 = {
        "plan": plan0,
        "words": list(abs_words),  # chunk0 local == absolute
        "segments": [],
    }
    chunk1 = {
        "plan": plan1,
        "words": offset_items(abs_words, -80.0),  # express same abs words chunk1-locally
        "segments": [],
    }

    out = stitch([chunk0, chunk1])
    got = [w["word"] for w in out["words"]]
    expected = ["w85", "w87", "w89", "w91", "w93", "w95"]
    assert got == expected  # no drops, no doubles, correct order

    # Verify the partition split exactly at midpoint 90.0.
    by_word = {w["word"]: w for w in out["words"]}
    # words with center < 90 came through with chunk0's offset (0), >= 90 chunk1's (offset cancels)
    assert by_word["w89"]["start"] == 88.5  # absolute preserved
    assert by_word["w91"]["start"] == 90.5


def test_stitch_text_from_segments_collapses_whitespace():
    plan0 = ChunkPlan(0, 0.0, 100.0)
    plan1 = ChunkPlan(1, 90.0, 190.0)
    chunk0 = {
        "plan": plan0,
        "words": [],
        "segments": _segments((1.0, 3.0, "  Hello   there. "), (50.0, 52.0, "Mid chunk.")),
    }
    chunk1 = {
        "plan": plan1,
        "segments": _segments((20.0, 22.0, "End  part."), (5.0, 6.0, "dropped overlap")),
        "words": [],
    }
    out = stitch([chunk0, chunk1])
    # Joined with single spaces, internal runs collapsed, stripped.
    assert "  " not in out["text"]
    assert out["text"].startswith("Hello there. Mid chunk.")


def test_stitch_text_falls_back_to_words_when_no_segments():
    plan = ChunkPlan(0, 0.0, 100.0)
    co = {
        "plan": plan,
        "words": _words((0.0, 0.5, "Hello"), (1.0, 1.5, "world")),
        "segments": [],
    }
    out = stitch([co])
    assert out["text"] == "Hello world"
    assert out["segments"] == []


# ---------------------------------------------------------------------------
# 4. Three-chunk case
# ---------------------------------------------------------------------------


def test_stitch_three_chunks_end_to_end():
    """Three overlapping chunks; each seam word lands deterministically in one
    chunk via the ownership-window rule. Final output is monotonic with every
    distinct word exactly once."""
    # Plans (absolute). overlaps: c0/c1 -> [90,100] mid 95 ; c1/c2 -> [180,200] mid 190.
    plan0 = ChunkPlan(0, 0.0, 100.0)
    plan1 = ChunkPlan(1, 90.0, 200.0)
    plan2 = ChunkPlan(2, 180.0, 280.0)

    # Absolute words, including ones inside each overlap.
    # c0/c1 overlap [90,100], midpoint 95:
    #   "a" center 92  < 95 -> c0
    #   "b" center 97  >= 95 -> c1
    # c1/c2 overlap [180,200], midpoint 190:
    #   "c" center 185 < 190 -> c1
    #   "d" center 195 >= 190 -> c2
    w_start = ("start_word", 5.0, 6.0)
    w_a = ("a", 91.5, 92.5)
    w_b = ("b", 96.5, 97.5)
    w_mid = ("mid", 130.0, 131.0)
    w_c = ("c", 184.5, 185.5)
    w_d = ("d", 194.5, 195.5)
    w_end = ("end_word", 250.0, 251.0)

    def abs_words_for(*specs):
        return [{"start": s, "end": e, "word": w} for (w, s, e) in specs]

    chunk0 = {
        "plan": plan0,
        "words": offset_items(abs_words_for(w_start, w_a, w_b), -plan0.start),
        "segments": [],
    }
    chunk1 = {
        "plan": plan1,
        "words": offset_items(abs_words_for(w_a, w_b, w_mid, w_c, w_d), -plan1.start),
        "segments": [],
    }
    chunk2 = {
        "plan": plan2,
        "words": offset_items(abs_words_for(w_c, w_d, w_end), -plan2.start),
        "segments": [],
    }

    out = stitch([chunk0, chunk1, chunk2])
    got = [w["word"] for w in out["words"]]
    assert got == ["start_word", "a", "b", "mid", "c", "d", "end_word"]

    # No duplicates.
    assert len(got) == len(set(got))

    # Correct absolute times and monotonic, non-overlapping.
    by = {w["word"]: (w["start"], w["end"]) for w in out["words"]}
    assert by["a"] == (91.5, 92.5)
    assert by["b"] == (96.5, 97.5)
    assert by["c"] == (184.5, 185.5)
    assert by["d"] == (194.5, 195.5)

    starts = [w["start"] for w in out["words"]]
    assert starts == sorted(starts)
    for x, y in zip(out["words"], out["words"][1:]):
        assert x["end"] <= y["start"]

    assert out["text"] == "start_word a b mid c d end_word"


def test_ownership_window_endpoints_first_and_last_unbounded():
    plans = [ChunkPlan(0, 0.0, 100.0), ChunkPlan(1, 90.0, 190.0)]
    lo0, hi0 = chunking._ownership_window(plans, 0)
    lo1, hi1 = chunking._ownership_window(plans, 1)
    assert lo0 == -math.inf
    assert hi0 == pytest.approx((90.0 + 100.0) / 2.0)  # 95.0
    assert lo1 == pytest.approx(95.0)
    assert hi1 == math.inf


# ---------------------------------------------------------------------------
# 5. Monotonicity regression (review finding): a wide token owned by the later
#    chunk can start before a short token owned by the earlier chunk. Ownership
#    is decided by token CENTER but the timeline must be ordered by START, so
#    stitch() must sort the merged output (SPEC §3.2 / acceptance #2, #4).
# ---------------------------------------------------------------------------


def _segments(*specs):
    """Build segment dicts from (start, end, text) tuples (local helper alias)."""
    return [{"start": s, "end": e, "text": t} for (s, e, t) in specs]


def test_stitch_sorts_straddling_segment_to_keep_starts_monotonic():
    plan0 = ChunkPlan(0, 0.0, 100.0)
    plan1 = ChunkPlan(1, 85.0, 185.0)  # overlap [85, 100], midpoint 92.5

    # chunk0 owns a short segment near the end of the overlap: center 92.2 < 92.5.
    chunk0 = {
        "plan": plan0,
        "words": [],
        "segments": _segments((92.0, 92.4, "earlier-chunk tail")),  # abs == local
    }
    # chunk1 owns a WIDE segment whose center (92.5) lands on chunk1's side, but
    # whose absolute START (85.5) precedes chunk0's kept segment start (92.0).
    # chunk1-local times = absolute - plan1.start (85.0).
    chunk1 = {
        "plan": plan1,
        "words": [],
        "segments": _segments((85.5 - 85.0, 99.5 - 85.0, "later-chunk wide head")),
    }

    out = stitch([chunk0, chunk1])

    # Both segments survive (no drop), and they come out ordered by start, not by
    # the chunk they were emitted from.
    starts = [s["start"] for s in out["segments"]]
    assert starts == sorted(starts)
    assert starts == [85.5, 92.0]
    assert [s["text"] for s in out["segments"]] == [
        "later-chunk wide head",
        "earlier-chunk tail",
    ]
    # text is reconstructed from the sorted segment order, too.
    assert out["text"] == "later-chunk wide head earlier-chunk tail"
