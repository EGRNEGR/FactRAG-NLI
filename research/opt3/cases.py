"""Per-question table of run 0749Z: both configurations side by side, every attempt scored.

For each case and configuration: the published outcome (from the run's cases.jsonl), the
attempt path (draft/repair/think/think_repair/recovery), the refusal reason and, for every
logged attempt, whether its raw text would have been correct had it been published
(evaluate_suite.score_answer) and whether the run's verifier withheld it. This is the base
of the error taxonomy: "a correct draft existed but was withheld" vs "no correct draft".

Usage:  python -m research.opt3.cases      (writes reports/optimization_3/error_analysis/)
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from typing import Any

from evaluate_suite import score_answer
from generator_verifier import REFUSAL

from research.opt3 import data


def attempt_score(record: data.CaseRecord, attempt: data.Attempt) -> dict[str, Any]:
    score = score_answer(record.case, attempt.raw_text)
    cleaned = score_answer(record.case, None if attempt.refused else attempt.cleaned_text)
    return {
        "role": attempt.role,
        "llm_refusal": attempt.raw_text.strip() == REFUSAL,
        "draft_correct": bool(score["correct"]),
        "draft_hallucination": score["hallucination"],
        "published": not attempt.refused,
        "published_correct": bool(cleaned["correct"]),
        "trimmed": not attempt.refused and attempt.cleaned_text.strip() != attempt.raw_text.strip(),
        "statuses": [c["status"] for c in attempt.claims],
        "grades": [c.get("grade") for c in attempt.claims],
    }


def case_row(record: data.CaseRecord, config: str) -> dict[str, Any]:
    row = record.rows[config]
    attempts = [attempt_score(record, a) for a in record.attempts[config]]
    any_correct = any(a["draft_correct"] for a in attempts)
    withheld_correct = [a["role"] for a in attempts if a["draft_correct"] and not a["published"]]
    return {
        "correct": bool(row["factrag"]["correct"]),
        "vanilla_correct": bool(row["vanilla"]["correct"]),
        "refused": bool(row["factrag"]["refused"]),
        "hallucination": row["factrag"]["hallucination"],
        "reason": row["refusal_reason"],
        "path": "+".join(a["role"] for a in attempts) or "gate",
        "any_draft_correct": any_correct,
        "withheld_correct": "+".join(withheld_correct),
        "attempts": attempts,
        "answer": row["answer"],
        "gen_ms": row["execution_stats"]["generation_ms"],
        "total_ms": row["wall_ms"],
    }


def error_class(record: data.CaseRecord, c: dict[str, Any]) -> str:
    """Why a case is not correct (one label; first matching rule wins)."""
    if c["correct"]:
        return "ok"
    expected = record.case.expected_behavior
    if expected != "answer":
        if c["hallucination"]:
            return "unanswerable_answered"
        return "unanswerable_other"
    if c["reason"] == "relevance_threshold_not_met":
        return "gate_refusal"
    if not c["refused"]:
        published = [a for a in c["attempts"] if a["published"]]
        if published and published[-1]["draft_correct"] and not published[-1]["published_correct"]:
            return "verifier_trimmed_correct"
        return "published_wrong"
    if c["withheld_correct"]:
        return "verifier_blocked_correct"
    roles = [a for a in c["attempts"]]
    if roles and all(a["llm_refusal"] for a in roles):
        return "llm_refusal_only"
    return "no_correct_draft"


def build() -> list[dict[str, Any]]:
    table = []
    for record in data.records().values():
        entry: dict[str, Any] = {
            "id": record.id,
            "dataset": data.dataset_of(record.id),
            "split": record.split,
            "category": record.case.category,
            "subtype": record.case.subtype,
            "expected": record.case.expected_behavior,
            "question": record.case.question,
            "top_rerank": round(record.top_rerank("judge"), 4),
            "gold_in_sources": record.rows["judge"].get("gold_in_sources"),
        }
        for config in data.CONFIGS:
            c = case_row(record, config)
            c["class"] = error_class(record, c)
            entry[config] = c
        table.append(entry)
    return table


def flat(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for e in table:
        row = {k: v for k, v in e.items() if k not in data.CONFIGS}
        for config in data.CONFIGS:
            c = e[config]
            for key in ("correct", "vanilla_correct", "reason", "path", "class", "withheld_correct"):
                row[f"{config}_{key}"] = c[key]
            row[f"{config}_answer"] = c["answer"]
        rows.append(row)
    return rows


def main() -> None:
    table = build()
    out = data.OUT / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for e in table:
            handle.write(json.dumps(e, ensure_ascii=False) + "\n")
    rows = flat(table)
    with (out / "cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for config in data.CONFIGS:
        for dataset in data.DATASETS:
            sub = [e for e in table if e["dataset"] == dataset]
            counts = Counter(e[config]["class"] for e in sub)
            print(config, dataset, len(sub), dict(counts.most_common()))
    disagree = [e for e in table if e["rules"]["correct"] != e["judge"]["correct"]]
    print("rules/judge disagree:", len(disagree))
    for e in disagree:
        print(
            " ",
            e["id"],
            "rules" if e["rules"]["correct"] else "judge",
            "| rules:",
            e["rules"]["path"],
            e["rules"]["class"],
            "| judge:",
            e["judge"]["path"],
            e["judge"]["class"],
        )


if __name__ == "__main__":
    main()
