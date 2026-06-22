"""Benchmark metrics for comparing ASR systems (SPEC §10.2).

This module is benchmark-only. It is NOT imported by the worker image and may
depend on benchmark-only packages (``jiwer``). Everything here operates on
plain Python data structures so it can be unit-tested without audio or a model:

  * a *hypothesis* is the text a system produced (a ``str``);
  * a *reference* is the human transcript (a ``str``);
  * *word timestamps* are lists of ``{"start": float, "end": float, "word": str}``
    — exactly the worker's §3.2 ``words`` shape (Parakeet path). The benchmark
    adapts AssemblyAI / Whisper word lists into the same shape before calling in.

Metrics implemented (SPEC §10.2):
  * WER — raw and normalized, with a single shared normalizer applied identically
    to every system so differences aren't normalization artifacts.
  * Punctuation F1 — precision/recall/F1 over punctuation marks vs the reference.
  * Timestamp quality — mean absolute word-START error vs a reference alignment,
    plus the fraction of words within ±200 ms.
  * Hallucination heuristic — rate and per-minute count of hypothesis content with
    no support in the reference (a coarse, automatic flag; SPEC notes manual review).
  * audio_hour_cost — $ per audio-hour, either from GPU $/hr and throughput (RTF)
    for a self-hosted model, or a flat list price for a hosted API.

``jiwer`` is imported lazily inside the WER functions so the rest of the module
(normalization, punctuation, timestamps, cost) works even if ``jiwer`` is absent.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# Marks we score for punctuation quality. Kept small and unambiguous.
PUNCTUATION_MARKS = ".,?!;:-'\""

# Number words used by the shared normalizer to fold a few common spellings so
# "two" vs "2" doesn't inflate WER. Deliberately small — full number expansion
# is out of scope; the goal is only that all systems are normalized identically.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}


# --------------------------------------------------------------------------- #
# Shared text normalizer
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """Normalize ``text`` identically for every system (SPEC §10.2).

    Steps: lowercase -> strip punctuation -> fold a few number words to digits
    -> collapse whitespace. Applied to BOTH reference and hypothesis before
    *normalized* WER so the comparison is fair across systems.
    """
    if not text:
        return ""
    text = text.lower()
    # Replace punctuation with spaces (so "well,done" -> "well done").
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [_NUMBER_WORDS.get(tok, tok) for tok in text.split()]
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# WER (raw + normalized)
# --------------------------------------------------------------------------- #
def _wer(reference: str, hypothesis: str) -> float:
    """Word error rate via jiwer; falls back to a stdlib edit-distance WER.

    Returns a float in [0, inf) (insertions can push WER above 1.0). An empty
    reference returns 0.0 when the hypothesis is also empty, else 1.0.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    try:
        import jiwer  # benchmark-only; lazy so the module imports without it.

        return float(jiwer.wer(reference, hypothesis))
    except ImportError:
        # Stdlib Levenshtein over word lists, normalized by reference length.
        return _word_edit_distance(ref_words, hyp_words) / len(ref_words)


def _word_edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance between two token lists (substitution cost 1)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def wer_raw(reference: str, hypothesis: str) -> float:
    """WER on lightly-touched text (lowercased only, punctuation kept).

    "Raw" still lowercases so casing differences alone don't dominate; it keeps
    punctuation attached to words, which penalizes punctuation/casing-sensitive
    differences relative to :func:`wer_normalized`.
    """
    return _wer((reference or "").lower(), (hypothesis or "").lower())


def wer_normalized(reference: str, hypothesis: str) -> float:
    """WER after the shared :func:`normalize_text` is applied to both sides."""
    return _wer(normalize_text(reference), normalize_text(hypothesis))


# --------------------------------------------------------------------------- #
# Punctuation F1
# --------------------------------------------------------------------------- #
def _punct_counts(text: str) -> Counter:
    """Multiset (Counter) of punctuation marks in ``text``."""
    return Counter(ch for ch in (text or "") if ch in PUNCTUATION_MARKS)


def punctuation_f1(reference: str, hypothesis: str) -> dict:
    """Precision / recall / F1 over punctuation marks vs the reference (SPEC §10.2).

    Treats punctuation as a multiset: true positives are the per-mark overlap
    ``min(ref_count, hyp_count)``; precision is over the hypothesis's marks,
    recall over the reference's. Returns
    ``{"precision": float, "recall": float, "f1": float}``.

    Degenerate cases: if neither side has punctuation, F1 is 1.0 (perfect
    agreement on "no punctuation"). If exactly one side has punctuation, the
    corresponding precision/recall is 0.0.
    """
    ref_c = _punct_counts(reference)
    hyp_c = _punct_counts(hypothesis)
    ref_total = sum(ref_c.values())
    hyp_total = sum(hyp_c.values())

    if ref_total == 0 and hyp_total == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = sum(min(ref_c[m], hyp_c[m]) for m in (set(ref_c) | set(hyp_c)))
    precision = tp / hyp_total if hyp_total else 0.0
    recall = tp / ref_total if ref_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------- #
# Timestamp quality
# --------------------------------------------------------------------------- #
def timestamp_quality(
    hyp_words: list[dict],
    ref_words: list[dict],
    tolerance_sec: float = 0.2,
) -> dict:
    """Word-start timestamp accuracy vs a reference alignment (SPEC §10.2).

    Both ``hyp_words`` and ``ref_words`` are word-timestamp lists in the worker's
    §3.2 shape: ``{"start": float, "end": float, "word": str}``. Hypothesis words
    are matched to reference words *positionally* over the normalized word
    sequence (the benchmark feeds in references whose word order matches), and
    the absolute start-time error is measured for each matched pair.

    Returns::

        {
          "mean_abs_start_error_sec": float,   # mean |hyp.start - ref.start|
          "pct_within_tolerance": float,       # fraction within ±tolerance_sec
          "matched": int,                      # number of matched word pairs
        }

    If there is no reference alignment (empty ``ref_words``) the metric is not
    computable and returns NaN-ish sentinels (error=None, pct=None, matched=0)
    so the caller can record "n/a" rather than a misleading 0.
    """
    if not ref_words or not hyp_words:
        return {
            "mean_abs_start_error_sec": None,
            "pct_within_tolerance": None,
            "matched": 0,
        }

    # Positional alignment over the shorter common prefix. We compare only words
    # whose normalized text matches, which keeps a single insertion/deletion
    # from cascading every subsequent pair into a mismatch via a greedy walk.
    errors: list[float] = []
    within = 0
    i = j = 0
    while i < len(hyp_words) and j < len(ref_words):
        hw = _norm_word(hyp_words[i].get("word", ""))
        rw = _norm_word(ref_words[j].get("word", ""))
        if hw == rw:
            err = abs(
                float(hyp_words[i]["start"]) - float(ref_words[j]["start"])
            )
            errors.append(err)
            if err <= tolerance_sec:
                within += 1
            i += 1
            j += 1
        else:
            # Greedy resync: advance whichever side keeps us closest. Look one
            # ahead on each side; advance the one that re-aligns soonest.
            if _resync_advance_hyp(hyp_words, ref_words, i, j):
                i += 1
            else:
                j += 1

    if not errors:
        return {
            "mean_abs_start_error_sec": None,
            "pct_within_tolerance": None,
            "matched": 0,
        }

    return {
        "mean_abs_start_error_sec": sum(errors) / len(errors),
        "pct_within_tolerance": within / len(errors),
        "matched": len(errors),
    }


def _norm_word(word: str) -> str:
    """Lowercase and strip non-word chars from a single token for matching."""
    return re.sub(r"[^\w]", "", (word or "").lower())


def _resync_advance_hyp(
    hyp_words: list[dict], ref_words: list[dict], i: int, j: int
) -> bool:
    """Decide which side to advance on a mismatch (cheap one-step lookahead)."""
    hw = _norm_word(hyp_words[i].get("word", ""))
    rw = _norm_word(ref_words[j].get("word", ""))
    # If the next ref word equals the current hyp word, the ref had an extra
    # token: advance ref. Otherwise advance hyp.
    if j + 1 < len(ref_words) and _norm_word(ref_words[j + 1].get("word", "")) == hw:
        return False  # advance ref (j)
    if i + 1 < len(hyp_words) and _norm_word(hyp_words[i + 1].get("word", "")) == rw:
        return True  # advance hyp (i)
    return True  # default: advance hyp


# --------------------------------------------------------------------------- #
# Hallucination heuristic
# --------------------------------------------------------------------------- #
def hallucination_rate(
    reference: str, hypothesis: str, audio_duration_sec: float
) -> dict:
    """Coarse automatic hallucination flag (SPEC §10.2).

    Heuristic: words present in the hypothesis but NOT in the reference vocabulary
    are candidate "inserted" content with no acoustic basis. This is intentionally
    crude (SPEC says manual or heuristic) — it over-counts legitimate
    substitutions and under-counts plausible-sounding hallucinations, so treat it
    as a screening signal, not ground truth.

    Returns::

        {
          "hallucinated_words": int,        # hyp words absent from ref vocab
          "hallucination_rate": float,      # hallucinated / total hyp words
          "hallucinations_per_min": float,  # hallucinated words per audio minute
        }
    """
    ref_vocab = set(normalize_text(reference).split())
    hyp_tokens = normalize_text(hypothesis).split()
    if not hyp_tokens:
        return {
            "hallucinated_words": 0,
            "hallucination_rate": 0.0,
            "hallucinations_per_min": 0.0,
        }

    hallucinated = sum(1 for t in hyp_tokens if t not in ref_vocab)
    minutes = (audio_duration_sec / 60.0) if audio_duration_sec and audio_duration_sec > 0 else 0.0
    return {
        "hallucinated_words": hallucinated,
        "hallucination_rate": hallucinated / len(hyp_tokens),
        "hallucinations_per_min": (hallucinated / minutes) if minutes > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
def audio_hour_cost_from_gpu(gpu_cost_per_hour: float, rtf: float) -> float | None:
    """$ per audio-hour for a self-hosted model (SPEC §10.2).

    ``rtf`` is the worker's definition (audio_duration_sec / processing_time_sec
    = speed factor, higher = faster). Processing one audio-hour therefore takes
    ``1 / rtf`` wall-clock hours, so::

        audio_hour_cost = gpu_cost_per_hour / rtf

    Returns ``None`` if ``rtf`` is non-positive (can't compute throughput).
    """
    if not rtf or rtf <= 0:
        return None
    return gpu_cost_per_hour / rtf


def audio_hour_cost_listprice(price_per_hour: float) -> float:
    """$ per audio-hour for a hosted API quoted as a flat list price."""
    return float(price_per_hour)


# --------------------------------------------------------------------------- #
# Aggregation helper
# --------------------------------------------------------------------------- #
@dataclass
class Aggregate:
    """Mean/median of a numeric metric across files (None values skipped)."""

    mean: float | None
    median: float | None
    count: int


def aggregate(values: list[float | None]) -> Aggregate:
    """Mean/median over ``values``, ignoring ``None`` (uncomputable) entries."""
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return Aggregate(mean=None, median=None, count=0)
    nums_sorted = sorted(nums)
    n = len(nums_sorted)
    mid = n // 2
    median = (
        nums_sorted[mid]
        if n % 2 == 1
        else (nums_sorted[mid - 1] + nums_sorted[mid]) / 2.0
    )
    return Aggregate(mean=sum(nums) / n, median=median, count=n)
