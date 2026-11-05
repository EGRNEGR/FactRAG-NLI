"""Remark 3: the verifier on a second generator (Qwen2.5-14B-Instruct) on the v3 set.

The prototype run 20261026T2139Z_v3 (RTX 5060 Laptop, Qwen2.5-14B-Instruct Q4_K_M, parser v1,
BGE-Reranker-Large, top-5 sources) logged the first draft of every v3 question with the exact
sources the generator saw. Those drafts and the first drafts of the final run 1222Z
(Qwen3.8-27B, parser v2, bge-reranker-v2-m3, top-10) are filtered offline in three modes:

  vanilla     the draft as is;
  nli_single  the single-NLI filter of the early verifier (mDeBERTa-v3, whole chunk,
              P(e) > 0.40 and P(c) < 0.20);
  judge       the decision layer of the final configuration (judge + anchor guard).

Both generators are scored by the same scorer on the same questions and templates (the v3
file of 1222Z; it differs from the prototype's only in document bindings and gold chunks,
which the scorer does not read). Retrieval differs between the generators, so the table
compares the verifier's effect within each generator, not the generators with each other.
The live result of each run (with its own repair or adaptive passes) is reported alongside.

Usage:  python -m research.review.second_generator     Results: reports/review/second_generator.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from evaluate_suite import mcnemar_exact

from research.review import OUT  # noqa: I001 - selects the run first
from research.opt2.metrics import outcome
from research.opt3 import data
from research.opt3.replay import logged
from research.review import verifiers as V
from research.review.compare import first_draft, verify

PROTOTYPE = Path(
    os.environ.get(
        "REVIEW_PROTOTYPE_RUN",
        r"C:\Hakatons\FactRAG-NLI\reports\runs\20261026T2139Z_v3",
    )
)
MODES = ("vanilla", "nli_single", "judge")


def prototype_records() -> tuple[dict[str, data.CaseRecord], dict[str, str | None]]:
    """v3 cases of 1222Z with the prototype's logged attempts, and its published answers."""
    cases, labels = data.cases(), data.split()
    records: dict[str, data.CaseRecord] = {}
    published: dict[str, str | None] = {}
    with (PROTOTYPE / "full" / "e2e_guard_on.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            response = row["response"]
            record = data.CaseRecord(case=cases[row["id"]], split=labels[row["id"]])
            record.attempts["judge"] = data.attempts_of(response)
            records[row["id"]] = record
            published[row["id"]] = None if response["refused"] else response["answer"]
    return records, published


def mode_outcomes(records: dict[str, data.CaseRecord], mode: str) -> list[dict[str, Any]]:
    v = None if mode == "vanilla" else V.build(mode)
    result = []
    for record in records.values():
        draft = first_draft(record)
        text = draft.raw_text if draft else None
        if draft is not None and v is not None:
            checked = verify(v, draft)
            text = None if checked.refused else checked.cleaned_text
        result.append(
            outcome(record, vanilla_text=draft.raw_text if draft else None, published_text=text)
        )
    return result


def block(items: list[dict[str, Any]], base: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [o for o in items if o["expected"] == "answer"]
    defined = [o for o in items if o["factrag"]["hallucination"] is not None]
    gained = sum(o["factrag"]["correct"] and not b["factrag"]["correct"] for o, b in zip(items, base))
    lost = sum(b["factrag"]["correct"] and not o["factrag"]["correct"] for o, b in zip(items, base))
    return {
        "n": len(items),
        "correct": sum(o["factrag"]["correct"] for o in items),
        "hallucination": sum(bool(o["factrag"]["hallucination"]) for o in defined),
        "hallucination_n": len(defined),
        "false_refusal": sum(o["factrag"]["refused"] for o in answerable),
        "false_refusal_n": len(answerable),
        "vs_vanilla": {"gained": gained, "lost": lost, "mcnemar_p": mcnemar_exact(gained, lost)},
        "hallucination_vs_vanilla": {
            "removed": (removed := sum(
                bool(b["factrag"]["hallucination"]) and not o["factrag"]["hallucination"]
                for o, b in zip(items, base) if b["factrag"]["hallucination"] is not None)),
            "added": (added := sum(
                bool(o["factrag"]["hallucination"]) and not b["factrag"]["hallucination"]
                for o, b in zip(items, base) if o["factrag"]["hallucination"] is not None)),
            "mcnemar_p": mcnemar_exact(removed, added),
        },
    }


def generator(records: dict[str, data.CaseRecord], live: list[dict[str, Any]]) -> dict[str, Any]:
    vanilla = mode_outcomes(records, "vanilla")
    result = {mode: block(mode_outcomes(records, mode), vanilla) for mode in MODES}
    result["live"] = block(live, vanilla)
    return result


def main() -> None:
    proto, proto_published = prototype_records()
    proto_live = [
        outcome(r, vanilla_text=None, published_text=proto_published[cid])
        for cid, r in proto.items()
    ]
    final = {cid: r for cid, r in data.records().items() if not cid.startswith("V3-XQF")}
    final = {cid: final[cid] for cid in proto}
    final_live = [logged(r, "judge") for r in final.values()]
    result = {
        "questions": len(proto),
        "Qwen2.5-14B-Instruct (prototype 20261026T2139Z_v3)": generator(proto, proto_live),
        "Qwen3.8-27B (final run 20261104T1222Z)": generator(final, final_live),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "second_generator.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for name, rows in result.items():
        if not isinstance(rows, dict):
            continue
        print(name)
        for mode, r in rows.items():
            p = r["vs_vanilla"]
            print(f"  {mode:11} correct {r['correct']}/{r['n']} halluc {r['hallucination']}/"
                  f"{r['hallucination_n']} FR {r['false_refusal']}/{r['false_refusal_n']} "
                  f"vs vanilla +{p['gained']}/-{p['lost']} p={p['mcnemar_p']:.3g} "
                  f"halluc -{r['hallucination_vs_vanilla']['removed']}"
                  f"/+{r['hallucination_vs_vanilla']['added']} "
                  f"p={r['hallucination_vs_vanilla']['mcnemar_p']:.3g}")


if __name__ == "__main__":
    main()
