"""Paired comparison of an e2e run on selected questions with the baseline run 1310Z.

`ids <set>` prints the question ids of a population (one per line) for
evaluate_suite.py --ids:
  E1   questions where a logged claim was withheld for a missing anchor
       (anchor_not_in_source, any attempt) - the only ones nli_anchor_scope=document can change
  controls  a fixed sample of 20 questions untouched by every variant (determinism of e2e)

`run <dir>` compares <dir>/<dataset>/e2e_guard_on/cases.jsonl with the baseline's cases on
the same ids: correct, hallucination, false refusal, completeness of compound answers, and
the regressions (correct before and not now; new hallucinations). Exit code 1 on any
regression. Results: <dir>/comparison.json.

Usage:  python -m research.opt4.compare ids E1 > ids.txt
        python -m research.opt4.compare run reports/optimization_4/live/<run>/e2e
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import research.opt4  # noqa: F401 - selects the baseline run
from evaluate_suite import score_answer

from research.opt3 import data
from research.opt4 import metrics

CONFIG = "judge"


def anchor_population() -> list[str]:
    ids = []
    for record in data.records().values():
        if any(
            c.get("reason") == "anchor_not_in_source"
            for a in record.attempts[CONFIG]
            for c in a.claims
        ):
            ids.append(record.id)
    return ids


def controls(n: int = 20) -> list[str]:
    """Fixed v3 questions with one logged draft, published fully verified, no anchor drop."""
    picked = []
    for record in sorted(data.records().values(),
                         key=lambda r: hashlib.sha1(("c" + r.id).encode()).hexdigest()):
        attempts = record.attempts[CONFIG]
        if data.dataset_of(record.id) != "v3" or len(attempts) != 1 or attempts[0].refused:
            continue
        if any(c["status"] != "verified" for c in attempts[0].claims):
            continue
        picked.append(record.id)
        if len(picked) == n:
            break
    return picked


POPULATIONS = {"E1": anchor_population, "controls": controls}


def outcome(record: data.CaseRecord, text: str | None, first: str | None) -> dict[str, Any]:
    return {
        "id": record.id,
        "dataset": data.dataset_of(record.id),
        "split": record.split,
        "category": record.case.category,
        "expected": record.case.expected_behavior,
        "vanilla": score_answer(record.case, first),
        "factrag": score_answer(record.case, text),
        "published": text is not None,
    }


def run_rows(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in run_dir.glob("*/e2e_guard_on/cases.jsonl"):
        with path.open(encoding="utf-8") as handle:
            rows.update((r["id"], r) for r in map(json.loads, handle) if r.get("ok"))
    return rows


def compare(run_dir: Path) -> dict[str, Any]:
    records = data.records()
    rows = run_rows(run_dir)
    if not rows:
        raise SystemExit(f"no e2e cases under {run_dir}")
    ids = sorted(rows)
    base, variant, changed = [], [], []
    for cid in ids:
        record, row = records[cid], rows[cid]
        logged = record.rows[CONFIG]
        before = None if logged["refused"] else logged["answer"]
        after = None if row["refused"] else row["answer"]
        base.append(outcome(record, before, logged["raw_answer"]))
        variant.append(outcome(record, after, row["raw_answer"]))
        if (before or "") != (after or ""):
            changed.append({"id": cid, "before": before, "after": after,
                            "correct_before": base[-1]["factrag"]["correct"],
                            "correct_after": variant[-1]["factrag"]["correct"]})
    result: dict[str, Any] = {
        "run": str(run_dir),
        "baseline": str(data.RUN),
        "questions": len(ids),
        "identical_published": len(ids) - len(changed),
        "all": metrics.block(variant, base),
        "completeness": {
            "variant": metrics.completeness(
                {cid: None if rows[cid]["refused"] else rows[cid]["answer"] for cid in ids}
            ),
            "baseline": metrics.completeness(
                {cid: None if records[cid].rows[CONFIG]["refused"]
                 else records[cid].rows[CONFIG]["answer"] for cid in ids}
            ),
        },
        "regressions": metrics.regressions(variant, base),
        "changed": changed,
        "seconds": round(sum(r["wall_ms"] for r in rows.values()) / 1000, 1),
    }
    (run_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return result


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "ids":
        print("\n".join(POPULATIONS[argv[1]]()))
        return 0
    if len(argv) == 2 and argv[0] == "run":
        result = compare(Path(argv[1]))
        p = result["all"]["paired"]
        print(f"{result['questions']} questions, identical answers {result['identical_published']}, "
              f"correct {result['all']['correct']} (+{p['gained']} -{p['lost']}), "
              f"hallucination {result['all']['hallucination']}, "
              f"false refusal {result['all']['false_refusal']}, "
              f"regressions {result['regressions']}, {result['seconds']} s")
        for c in result["changed"]:
            print("  ", c["id"], c["correct_before"], "->", c["correct_after"])
        reg = result["regressions"]
        return 1 if reg["lost"] or reg["new_hallucinations"] else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
