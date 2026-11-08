"""Metrics of optimization_4 on top of research.opt2.metrics.

Completeness of compound answers: for an answerable question with two or more scorer facts
(answer_must_contain), the share of those facts found in the published text. The scorer
counts a case correct only when every fact is found; completeness shows partial answers
(one part of a multi-part question answered, the other trimmed or missing).
"""

from __future__ import annotations

from typing import Any

from eval_dataset import contains_fact

from research.opt2.metrics import paired, pct, summarize
from research.opt3 import data


def compound(case_id: str) -> bool:
    case = data.cases()[case_id]
    return case.expected_behavior == "answer" and len(case.answer_must_contain) >= 2


def facts_found(case_id: str, text: str | None) -> tuple[int, int]:
    facts = data.cases()[case_id].answer_must_contain
    if not text:
        return 0, len(facts)
    return sum(contains_fact(text, f) for f in facts), len(facts)


def completeness(published: dict[str, str | None]) -> dict[str, Any]:
    """Mean fact share and complete answers over the compound questions of `published`."""
    shares, complete, n = [], 0, 0
    for case_id, text in published.items():
        if not compound(case_id):
            continue
        found, total = facts_found(case_id, text)
        shares.append(found / total)
        complete += found == total
        n += 1
    return {
        "compound_questions": n,
        "complete": complete,
        "mean_fact_share": round(sum(shares) / n, 4) if n else None,
    }


def block(outcomes: list[dict[str, Any]], base: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline metrics of a variant and its paired comparison with the baseline."""
    s = summarize(outcomes)
    return {
        "correct": pct(s["factrag_correct"]),
        "correct_k": s["factrag_correct"]["k"],
        "hallucination": pct(s["hallucination"]),
        "hallucination_k": s["hallucination"]["k"],
        "false_refusal": pct(s["false_refusal"]),
        "false_refusal_k": s["false_refusal"]["k"],
        "selective_accuracy": pct(s["selective_accuracy"]),
        "paired": paired(base, outcomes),
    }


def regressions(outcomes: list[dict[str, Any]], base: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Cases that got worse: correct before and not now; a new hallucination."""
    lost, new_hallucinations = [], []
    for o, b in zip(outcomes, base, strict=True):
        if b["factrag"]["correct"] and not o["factrag"]["correct"]:
            lost.append(o["id"])
        if o["factrag"]["hallucination"] and not b["factrag"]["hallucination"]:
            new_hallucinations.append(o["id"])
    return {"lost": lost, "new_hallucinations": new_hallucinations}
