"""Per-question analysis of a steps run of scripts/linux/run_experiments.sh (G3, G4).

Reads <run>/steps/<experiment>/cases.jsonl and writes <run>/analysis.json:
  G3  the trigger reasons, how many questions passed the relevance gate after translation and
      how many of them the LLM refused again (answerable / unanswerable), every replaced
      answer with its question, translation, gold answer and score, and gold answers with
      Latin letters inside Cyrillic words (the scorer's literal rules cannot match them)
  G4  how many questions the normalization left unchanged and how many changed questions
      were refused again
Both: the overlap with the 52 persistent XQuAD refusals of DIAGNOSTIC_REPORT.md.

With --steps, <run> is an e2e run of PLAN=g3 (query_translation in the pipeline) and is
checked against the G3 steps of <steps run>: the same translated population and texts, the
same published answers, unchanged controls, measured time.

Usage:  python -m research.opt4.live_analysis reports/optimization_4/live/<run>
        python -m research.opt4.live_analysis live/<g3 run> --steps live/<quick run>
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import research.opt4  # noqa: F401 - selects the baseline run
from evaluate_suite import score_answer

from research.opt3 import data
from research.opt4 import steps
from research.opt4.diagnose import OUT

# A word with both Cyrillic and Latin letters ("бoльшим" with a Latin o).
_MIXED_WORD = re.compile(r"\b(?=\w*[а-яё])(?=\w*[a-z])\w+\b", re.I)


def rows(run: Path, experiment: str) -> list[dict[str, Any]]:
    path = run / "steps" / experiment / "cases.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def persistent_refusals() -> set[str]:
    path = OUT / "diagnostics" / "xquad_refusals.jsonl"
    with path.open(encoding="utf-8") as handle:
        return {json.loads(line)["id"] for line in handle if line.strip()}


def mixed_script(text: str) -> list[str]:
    return _MIXED_WORD.findall(text)


def replaced(row: dict[str, Any], persistent: set[str]) -> dict[str, Any]:
    record = data.records()[row["id"]]
    score = score_answer(record.case, row["text"])
    return {
        "id": row["id"],
        "split": record.split,
        "trigger": steps.reason(record),
        "persistent_refusal": row["id"] in persistent,
        "question": record.case.question,
        "translation": (row.get("translation") or row.get("normalized") or {}).get("text"),
        "answer": row["text"],
        "gold": record.case.ground_truth_answer,
        "correct": bool(score["correct"]),
        "gold_mixed_script": mixed_script(record.case.ground_truth_answer),
    }


def g3(run: Path, persistent: set[str]) -> dict[str, Any]:
    found = rows(run, "G3")
    records = data.records()
    by_outcome: Counter[str] = Counter()
    for row in found:
        expected = records[row["id"]].case.expected_behavior
        stage = (
            "no translation" if not row["translation"]["text"]
            else "gate refused" if not row.get("gate_passed")
            else "published" if row["replace"]
            else "LLM refused"
        )
        by_outcome[f"{expected}: {stage}"] += 1
    return {
        "population": len(found),
        "trigger": dict(Counter(steps.reason(records[r["id"]]) for r in found)),
        "dataset": dict(Counter(data.dataset_of(r["id"]) for r in found)),
        "persistent_refusals": sum(r["id"] in persistent for r in found),
        "outcome": dict(sorted(by_outcome.items())),
        "replaced": [replaced(r, persistent) for r in found if r["replace"]],
        "seconds": round(sum(r.get("seconds", 0.0) for r in found), 1),
    }


def g4(run: Path, persistent: set[str]) -> dict[str, Any]:
    found = rows(run, "G4")
    changed = [r for r in found if not r.get("skipped")]
    return {
        "population": len(found),
        "persistent_refusals": sum(r["id"] in persistent for r in found),
        "unchanged_question": sum(bool(r.get("skipped")) for r in found),
        "changed_question": len(changed),
        "changed_refused_again": sum(not r["replace"] for r in changed),
        "replaced": [replaced(r, persistent) for r in found if r["replace"]],
        "seconds": round(sum(r.get("seconds", 0.0) for r in found), 1),
    }


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def e2e_vs_steps(run: Path, steps_run: Path) -> dict[str, Any]:
    """The e2e G3 run (query_translation in the pipeline) against the G3 steps of steps_run:
    the same translations, the same published texts, unchanged controls, the extra time."""
    from research.opt4.compare import controls

    cases: dict[str, dict[str, Any]] = {}
    responses: dict[str, dict[str, Any]] = {}
    for folder in sorted((run / "e2e").glob("*/e2e_guard_on")):
        cases.update((r["id"], r) for r in jsonl(folder / "cases.jsonl") if r.get("ok"))
        responses.update(
            (r["id"], r["response"]) for r in jsonl(folder.parent / "full" / "e2e_guard_on.jsonl")
        )
    logged = data.case_rows(steps.CONFIG)
    full = data.responses(steps.CONFIG)
    step = {r["id"]: r for r in rows(steps_run, "G3")}
    translated = [cid for cid, r in cases.items() if r.get("translation_pass")]
    published = [cid for cid in translated if not cases[cid]["refused"]]
    ctrl = controls()
    extra = sorted(
        (responses[cid]["execution_stats"]["total_ms"] - full[cid]["execution_stats"]["total_ms"])
        / 1000
        for cid in translated
    )
    return {
        "questions": len(cases),
        "translated": len(translated),
        "translated_equals_steps_population": set(translated) == set(step),
        "translation_text_as_steps": sum(
            (responses[cid].get("translation") or {}).get("question")
            == step[cid]["translation"]["text"]
            for cid in translated if cid in step
        ),
        "translated_outcome": dict(Counter(
            (responses[cid].get("translation") or {}).get("refusal_reason") or "published"
            for cid in translated
        )),
        "published": published,
        "published_text_as_steps": sum(
            step.get(cid, {}).get("replace") and cases[cid]["answer"] == step[cid]["text"]
            for cid in published
        ),
        "controls_unchanged": sum(
            (cases[c]["answer"], cases[c]["refused"]) == (logged[c]["answer"], logged[c]["refused"])
            for c in ctrl if c in cases
        ),
        "controls": len([c for c in ctrl if c in cases]),
        "minutes_e2e": round(sum(r["wall_ms"] for r in cases.values()) / 60000, 1),
        "minutes_baseline_same_questions": round(
            sum(full[cid]["execution_stats"]["total_ms"] for cid in cases) / 60000, 1
        ),
        "extra_seconds_per_translated_question": {
            "median": round(extra[len(extra) // 2], 1) if extra else None,
            "max": round(extra[-1], 1) if extra else None,
        },
    }


def main(argv: list[str]) -> int:
    if len(argv) == 1:
        run = Path(argv[0])
        persistent = persistent_refusals()
        result = {"run": run.name, "G3": g3(run, persistent), "G4": g4(run, persistent)}
    elif len(argv) == 3 and argv[1] == "--steps":
        run = Path(argv[0])
        result = {"run": run.name, "steps_run": Path(argv[2]).name,
                  "G3_e2e": e2e_vs_steps(run, Path(argv[2]))}
    else:
        print(__doc__)
        return 2
    (run / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
