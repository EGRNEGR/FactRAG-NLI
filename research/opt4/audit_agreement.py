"""Audit of the v3 reference labels and of the automatic scorer against an expert's review.

Two steps, both on the final run 20261104T1222Z (research.review selects it):

  sheet   writes reports/review/audit/v3_audit_sheet.csv: one row per v3 question with the
          question, reference answer, required facts, evidence quote, the published answer of
          1222Z and the scorer's verdict. The expert fills three columns:
            expert_label_ok   yes / no   the reference answer and its evidence are correct
            expert_outcome    correct / wrong   the published answer (or refusal) is correct
            expert_comment    free text
  agree   reads the filled sheet and reports, over the rows with an expert verdict:
          percentage agreement and Cohen's kappa between the scorer and the expert
          (correct / wrong), the confusion table, the share of reference labels confirmed,
          and whether the review covers all 220 questions. Nothing is assumed for an empty row.

Usage:  python -m research.opt4.audit_agreement sheet
        python -m research.opt4.audit_agreement agree [path/to/filled.csv]
Results: reports/review/audit/agreement.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from research.review import OUT  # noqa: I001 - selects the run first
from research.opt3 import data
from research.opt3.replay import logged

AUDIT = OUT / "audit"
SHEET = AUDIT / "v3_audit_sheet.csv"
EXPERT = ("expert_label_ok", "expert_outcome", "expert_comment")


def scorer_verdict(score: dict[str, Any]) -> str:
    if score["correct"]:
        return "correct"
    if score.get("hallucination"):
        return "wrong (hallucination)"
    if score.get("refused"):
        return "wrong (refused)"
    return "wrong (required fact missing)"


def sheet() -> Path:
    rows = []
    for record in data.records().values():
        case = record.case
        if case.id.startswith("V3-XQF"):
            continue
        o = logged(record, "judge")
        row = record.rows["judge"]
        evidence = case.evidence[0].quote if case.evidence else ""
        rows.append(
            {
                "id": case.id,
                "category": case.category,
                "language": case.language,
                "expected_behavior": case.expected_behavior,
                "document": case.document,
                "question": case.question,
                "ground_truth_answer": case.ground_truth_answer,
                "answer_must_contain": json.dumps(case.answer_must_contain, ensure_ascii=False),
                "evidence_quote": evidence,
                "published_1222Z": "" if row["refused"] else row["answer"],
                "refusal_reason": row.get("refusal_reason") or "",
                "scorer_verdict": scorer_verdict(o["factrag"]),
                **{key: "" for key in EXPERT},
            }
        )
    AUDIT.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return SHEET


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if not n:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def agree(path: Path = SHEET) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    reviewed = [r for r in rows if r["expert_outcome"].strip()]
    scorer = ["correct" if r["scorer_verdict"] == "correct" else "wrong" for r in reviewed]
    expert = [r["expert_outcome"].strip().lower() for r in reviewed]
    bad = sorted({e for e in expert} - {"correct", "wrong"})
    if bad:
        raise ValueError(f"expert_outcome must be correct / wrong, found {bad}")
    labels = [r["expert_label_ok"].strip().lower() for r in rows if r["expert_label_ok"].strip()]
    result = {
        "sheet": str(path),
        "questions": len(rows),
        "reviewed": len(reviewed),
        "complete": len(reviewed) == len(rows),
        "agreement": (
            sum(s == e for s, e in zip(scorer, expert, strict=True)) / len(reviewed)
            if reviewed
            else None
        ),
        "cohen_kappa": cohen_kappa(scorer, expert),
        "confusion": {
            f"scorer {s} / expert {e}": n for (s, e), n in sorted(Counter(zip(scorer, expert)).items())
        },
        "disagreements": [
            {"id": r["id"], "scorer": s, "expert": e, "comment": r["expert_comment"]}
            for r, s, e in zip(reviewed, scorer, expert, strict=True)
            if s != e
        ],
        "labels_reviewed": len(labels),
        "labels_confirmed": sum(x in ("yes", "да") for x in labels),
    }
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "agreement.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return result


def main(args: list[str]) -> None:
    if not args or args[0] == "sheet":
        print("sheet:", sheet())
    elif args[0] == "agree":
        r = agree(Path(args[1]) if len(args) > 1 else SHEET)
        print(json.dumps({k: v for k, v in r.items() if k != "disagreements"}, ensure_ascii=False,
                         indent=1))
        for d in r["disagreements"]:
            print("  ", d)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
