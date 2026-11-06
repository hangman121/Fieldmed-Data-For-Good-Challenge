"""Confidence for an answer, with the reasons behind it.

Signals: how relevant the best source is (cross-encoder score as a probability), how many
sources are strongly relevant, and how much of the summary is backed by citations.
"""
import math
import re

from .config import RERANK_THRESHOLD


def prob(logit: float) -> float:
    return 1 / (1 + math.exp(-logit))


THRESHOLD = prob(RERANK_THRESHOLD)  # sources below this relevance are discarded


def assess(sources: list[dict], answer: str | None = None, invalid: list[int] = ()) -> dict:
    if not sources:
        return {"level": "none", "score": 0.0, "threshold": THRESHOLD,
                "reasons": [f"No passage reached the relevance threshold ({THRESHOLD:.0%}), so no answer was given."]}
    rel = [s["relevance"] for s in sources]
    strong = sum(r >= 0.5 for r in rel)
    score = 0.6 * rel[0] + 0.4 * min(strong, 3) / 3
    reasons = [f"Best source is {rel[0]:.0%} relevant (threshold {THRESHOLD:.0%}).",
               f"{strong} of {len(rel)} sources are strongly relevant (50% or more)."]
    if answer is not None:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if len(s.split()) >= 4]
        cited = [s for s in sentences if re.search(r"\[\d+\]", s)]
        coverage = len(cited) / len(sentences) if sentences else 0
        score = 0.7 * score + 0.3 * coverage
        reasons.append(f"{len(cited)} of {len(sentences)} summary sentences cite a source.")
        if invalid:
            score *= 0.7
            reasons.append(f"The summary cited sources that do not exist: {invalid}.")
        if "do not answer this question" in answer.lower():
            score = min(score, 0.2)
            reasons.append("The model reported that the sources do not answer the question.")
    level = "high" if score >= 0.7 else "medium" if score >= 0.45 else "low"
    return {"level": level, "score": round(score, 3), "threshold": THRESHOLD, "reasons": reasons}
