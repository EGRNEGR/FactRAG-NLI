"""Claim-level effect of a variant on the logged drafts of run 1222Z (v3 and XQuAD).

Every logged attempt (draft, repair, thinking, recovery) of every question is verified by the
run's verifier with and without the overrides; a claim whose status differs is listed with
its judge score, reason and whether a cited source holds a Markdown table. This shows what
a variant changes below the question level: claims it publishes or withholds in attempts
the replayed pipeline never reaches.

Usage:  python -m research.optimization_6.claims T12      (names of research.optimization_6.offline)
Output: reports/optimization_6/offline/claims_<name>.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any

import research.optimization_6  # noqa: F401 - selects the baseline run and the output folder
from grounding import markdown_tables
from research.opt3 import data
from research.opt3.replay import verifier, verify
from research.opt4.offline import CONFIG, RESULTS
from research.optimization_6.offline import REGISTRY


def _has_table(attempt: data.Attempt, ids: tuple[str, ...]) -> bool:
    return any(
        key in attempt.contexts and markdown_tables(attempt.contexts[key]["text"]) for key in ids
    )


def compare(name: str) -> dict[str, Any]:
    base, variant = verifier(CONFIG), verifier(CONFIG, **REGISTRY[name].overrides)
    changed: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for record in data.records().values():
        for attempt in record.attempts[CONFIG]:
            a, b = verify(base, attempt), verify(variant, attempt)
            totals["attempts"] += 1
            for ca, cb in zip(a.claims, b.claims, strict=True):
                if ca.kind != "fact":
                    continue
                totals["claims"] += 1
                table = _has_table(attempt, ca.source_ids)
                totals["table_claims"] += table
                if ca.status == cb.status:
                    continue
                changed.append(
                    {
                        "id": record.id,
                        "split": record.split,
                        "dataset": data.dataset_of(record.id),
                        "role": attempt.role,
                        "claim": ca.claim,
                        "table": table,
                        "base": {"status": ca.status.value, "reason": ca.reason,
                                 "score": ca.confidence, "premise": ca.evidence[0].premise_text
                                 if ca.evidence else None},
                        "variant": {"status": cb.status.value, "reason": cb.reason,
                                    "score": cb.confidence,
                                    "premise": max(cb.evidence, key=lambda e: e.judge_score or 0)
                                    .premise_text if cb.evidence else None},
                    }
                )
    flips = Counter(f"{c['base']['status']} -> {c['variant']['status']}" for c in changed)
    return {"variant": name, "overrides": REGISTRY[name].overrides, "totals": dict(totals),
            "flips": dict(flips), "changed": changed}


def main(names: list[str]) -> None:
    for name in names:
        result = compare(name)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"claims_{name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(name, result["totals"], result["flips"])
        for c in result["changed"]:
            print(f"  {c['id']} {c['split']} {c['role']} table={c['table']} "
                  f"{c['base']['status']}({c['base']['score']:.3f}) -> "
                  f"{c['variant']['status']}({c['variant']['score']:.3f}) | {c['claim'][:100]}")


if __name__ == "__main__":
    main(sys.argv[1:])
