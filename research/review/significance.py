"""Remark 4: paired significance of the full contour against the generator's own draft.

Per question of run 20261104T1222Z, three answers are compared by the exact two-sided
McNemar test:

  draft     the first draft of the generator, published as is (Vanilla);
  verified  the same draft filtered by the decision layer (no adaptive pass: the verifier's
            own effect, replayed offline with the cached NLI predictions);
  full      the answer the run published (verification + thinking, recovery, translation).

draft -> full is the effect of the whole contour; draft -> verified isolates the verifier;
verified -> full the adaptive passes. Also: the smallest p an exact two-sided test can reach
with k discordant pairs all in one direction (2 * 0.5**k), which bounds the single
optimization steps of the article (+5/-0 -> 0.0625, +4/-0 -> 0.125).

Usage:  python -m research.review.significance      Results: reports/review/significance.json
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from evaluate_suite import mcnemar_exact

from research.review import OUT  # noqa: I001 - selects the run first
from research.opt2.metrics import outcome
from research.opt3 import data
from research.opt3.replay import logged
from research.review import verifiers as V
from research.review.compare import first_draft, verify

PARTS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "all": lambda o: True,
    "v3": lambda o: o["dataset"] == "v3",
    "xquad-full": lambda o: o["dataset"] == "xquad-full",
    "dev": lambda o: o["split"] == "dev",
    "test": lambda o: o["split"] == "test",
}


def answers() -> list[dict[str, Any]]:
    """Per case: correctness and hallucination of draft, verified draft and full answer."""
    judge = V.build("judge")
    rows = []
    for record in data.records().values():
        full = logged(record, "judge")
        draft = first_draft(record)
        if draft is None:
            verified = outcome(record, vanilla_text=None, published_text=None)
        else:
            checked = verify(judge, draft)
            verified = outcome(
                record,
                vanilla_text=draft.raw_text,
                published_text=None if checked.refused else checked.cleaned_text,
            )
        rows.append(
            {
                "id": record.id,
                "dataset": full["dataset"],
                "split": full["split"],
                "expected": full["expected"],
                "draft": full["vanilla"],
                "verified": verified["factrag"],
                "full": full["factrag"],
            }
        )
    return rows


def paired(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    gained = sum(r[b]["correct"] and not r[a]["correct"] for r in rows)
    lost = sum(r[a]["correct"] and not r[b]["correct"] for r in rows)
    return {
        a: sum(r[a]["correct"] for r in rows),
        b: sum(r[b]["correct"] for r in rows),
        "gained": gained,
        "lost": lost,
        "mcnemar_p": mcnemar_exact(gained, lost),
    }


def hallucinations(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    defined = [r for r in rows if r["full"]["hallucination"] is not None]
    return {"k": sum(bool(r[key]["hallucination"]) for r in defined), "n": len(defined)}


def main() -> None:
    rows = answers()
    result: dict[str, Any] = {"run": data.RUN.name, "parts": {}}
    for part, keep in PARTS.items():
        sel = [r for r in rows if keep(r)]
        result["parts"][part] = {
            "n": len(sel),
            "draft_vs_full": paired(sel, "draft", "full"),
            "draft_vs_verified": paired(sel, "draft", "verified"),
            "verified_vs_full": paired(sel, "verified", "full"),
            "hallucination": {k: hallucinations(sel, k) for k in ("draft", "verified", "full")},
            "false_refusal": {
                k: sum(r[k]["refused"] for r in sel if r["expected"] == "answer")
                for k in ("draft", "verified", "full")
            },
        }
    result["min_p_one_sided_discordance"] = {k: mcnemar_exact(k, 0) for k in range(1, 11)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "significance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for part, r in result["parts"].items():
        line = [f"{part:10} n={r['n']}"]
        for key in ("draft_vs_full", "draft_vs_verified", "verified_vs_full"):
            p = r[key]
            line.append(f"{key} +{p['gained']}/-{p['lost']} p={p['mcnemar_p']:.2g}")
        h = r["hallucination"]
        line.append("halluc " + " ".join(f"{k} {v['k']}/{v['n']}" for k, v in h.items()))
        print(" | ".join(line))
    print("min p:", {k: round(v, 4) for k, v in result["min_p_one_sided_discordance"].items()})


if __name__ == "__main__":
    main()
