"""Metrics of the optimization_2 research: headline rates, decision quality and uncertainty.

An *outcome* is the scored result of one case under one configuration:
``{"id", "dataset", "split", "expected", "vanilla": score, "factrag": score, "published": bool}``
where score is ``evaluate_suite.score_answer`` (correct / refused / abstained / hallucination).

The verifier is also measured as a binary classifier of its own decision: among cases whose
draft is an answer (not a canonical refusal), "positive" = the draft is correct, "predicted
positive" = the answer was published. Precision, recall, macro-F1, balanced accuracy and MCC
of that decision show directly how often the filter withholds a correct draft (damage) and
how often it lets a wrong one through.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any

from evaluate_suite import mcnemar_exact, score_answer, wilson

Outcome = dict[str, Any]


def outcome(record: Any, *, vanilla_text: str | None, published_text: str | None) -> Outcome:
    """Score a case: Vanilla is the first draft, FactRAG the published text (None = withheld)."""
    case = record.case
    return {
        "id": case.id,
        "dataset": "xquad-full" if case.id.startswith("V3-XQF") else "v3",
        "split": record.split,
        "category": case.category,
        "expected": case.expected_behavior,
        "vanilla": score_answer(case, vanilla_text),
        "factrag": score_answer(case, published_text),
        "published": published_text is not None,
    }


def ratio(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": wilson(k, n)}


def safe(a: float, b: float) -> float:
    return a / b if b else 0.0


def binary(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision, recall = safe(tp, tp + fp), safe(tp, tp + fn)
    f1_pos = safe(2 * precision * recall, precision + recall)
    npv, specificity = safe(tn, tn + fn), safe(tn, tn + fp)
    f1_neg = safe(2 * npv * specificity, npv + specificity)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1_pos,
        "macro_f1": (f1_pos + f1_neg) / 2,
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": (tp * tn - fp * fn) / denom if denom else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def summarize(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    """Headline and decision metrics of one configuration over `outcomes`."""
    answerable = [o for o in outcomes if o["expected"] == "answer"]
    defined = [o for o in outcomes if o["factrag"]["hallucination"] is not None]
    drafted = [o for o in outcomes if not o["vanilla"]["refused"]]
    tp = sum(o["vanilla"]["correct"] and o["published"] for o in drafted)
    fp = sum(not o["vanilla"]["correct"] and o["published"] for o in drafted)
    fn = sum(o["vanilla"]["correct"] and not o["published"] for o in drafted)
    tn = sum(not o["vanilla"]["correct"] and not o["published"] for o in drafted)
    van = sum(o["vanilla"]["correct"] for o in outcomes)
    fac = sum(o["factrag"]["correct"] for o in outcomes)
    corrected = sum(o["factrag"]["correct"] and not o["vanilla"]["correct"] for o in outcomes)
    damaged = sum(o["vanilla"]["correct"] and not o["factrag"]["correct"] for o in outcomes)
    published_answers = [o for o in answerable if not o["factrag"]["refused"]]
    return {
        "cases": len(outcomes),
        "factrag_correct": ratio(fac, len(outcomes)),
        "vanilla_correct": ratio(van, len(outcomes)),
        "hallucination": ratio(sum(o["factrag"]["hallucination"] for o in defined), len(defined)),
        "vanilla_hallucination": ratio(
            sum(bool(o["vanilla"]["hallucination"]) for o in defined), len(defined)
        ),
        "false_refusal": ratio(sum(o["factrag"]["refused"] for o in answerable), len(answerable)),
        "coverage": ratio(len(published_answers), len(answerable)),
        "selective_accuracy": ratio(
            sum(o["factrag"]["correct"] for o in published_answers), len(published_answers)
        ),
        "net_contribution": fac - van,
        "correction": corrected,
        "damage": damaged,
        "filter": binary(tp, fp, fn, tn),
    }


def by(outcomes: Sequence[Outcome], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Outcome]] = {}
    for o in outcomes:
        groups.setdefault(str(o[key]), []).append(o)
    return {name: summarize(items) for name, items in sorted(groups.items())}


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = lambda v: sum(v) / len(v),
    *,
    rounds: int = 2000,
    seed: int = 20261031,
) -> list[float]:
    """Percentile 95% bootstrap interval of `statistic` (cases resampled with replacement)."""
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(rounds))
    return [stats[int(0.025 * rounds)], stats[int(0.975 * rounds) - 1]]


def paired(a: Sequence[Outcome], b: Sequence[Outcome], key: str = "factrag") -> dict[str, Any]:
    """Paired comparison of two configurations on the same cases (b minus a).

    Returns the discordant counts, the exact McNemar p-value and a paired bootstrap 95%
    interval of the difference in correct answers (in percentage points).
    """
    left = {o["id"]: o for o in a}
    pairs = [(left[o["id"]][key]["correct"], o[key]["correct"]) for o in b if o["id"] in left]
    only_b = sum(not x and y for x, y in pairs)
    only_a = sum(x and not y for x, y in pairs)
    diffs = [float(y) - float(x) for x, y in pairs]
    return {
        "n": len(pairs),
        "gained": only_b,
        "lost": only_a,
        "delta_pp": 100 * sum(diffs) / len(diffs) if diffs else 0.0,
        "ci95_pp": [100 * v for v in bootstrap_ci(diffs)] if diffs else None,
        "mcnemar_p": mcnemar_exact(only_a, only_b),
    }


def ece(confidences: Sequence[float], labels: Sequence[bool], bins: int = 10) -> dict[str, Any]:
    """Expected calibration error with equal-width bins, plus the reliability table."""
    table = []
    total = len(confidences)
    error = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        idx = [
            i
            for i, c in enumerate(confidences)
            if (low <= c < high) or (b == bins - 1 and c == 1.0)
        ]
        if not idx:
            continue
        conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        error += len(idx) / total * abs(conf - acc)
        table.append({"bin": [low, high], "n": len(idx), "confidence": conf, "accuracy": acc})
    brier = sum((c - float(y)) ** 2 for c, y in zip(confidences, labels, strict=True)) / total
    return {"ece": error, "brier": brier, "n": total, "bins": table}


def pct(block: dict[str, Any]) -> str:
    rate = block.get("rate")
    return "-" if rate is None else f"{100 * rate:.1f}% ({block['k']}/{block['n']})"
