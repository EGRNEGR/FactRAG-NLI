"""Offline experiments of optimization_4: verifier variants replayed on the 1310Z logs.

The baseline is run 20261103T1310Z itself (judge column, logged outcomes). Every variant
re-verifies the logged drafts of all 1307 questions in pipeline order (research.opt3.replay:
draft -> repair -> thinking after a refusal or after a trimmed answer -> recovery samples),
with the run's settings plus one override. A step the variant needs but the run never
generated counts as "missing" (the case keeps the last available, withheld result:
pessimistic). NLI predictions go through the persistent caches of research.opt2.nli_models
(computed on CUDA on a miss), so a variant differs from the baseline by decision logic only.

  B00  replay of the run with the current code, no override (must reproduce 1307/1307)
  E1   nli_anchor_scope=document (enhancement designators, same-document identifiers)
  E2   nli_premise_unwrap_tables=true (one-column tables as running text)

Results: reports/optimization_4/experiments/<name>.json and EXPERIMENTS.csv.

Usage:  python -m research.opt4.offline [B00 E1 E2]
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any

import research.opt4  # noqa: F401 - selects the baseline run
from research.opt3 import data
from research.opt3.replay import logged, replay, scored, verifier
from research.opt4 import metrics
from research.opt4.diagnose import OUT

CONFIG = "judge"
# Results of another baseline (optimization_5) go elsewhere: OPT4_EXPERIMENTS_OUT.
RESULTS = Path(os.environ.get("OPT4_EXPERIMENTS_OUT") or OUT)


@dataclass(frozen=True)
class Experiment:
    name: str
    hypothesis: str
    text: str
    overrides: dict[str, Any] = field(default_factory=dict)


REGISTRY = {
    e.name: e
    for e in (
        Experiment("B00", "-", "replay of the run, current code, no override"),
        Experiment(
            "E1",
            "H2",
            "control enhancement as one identifier; identifiers may come from a same-document chunk",
            {"nli_anchor_scope": "document"},
        ),
        Experiment(
            "E2",
            "H3",
            "one-column Markdown tables of a premise joined into running text before NLI",
            {"nli_premise_unwrap_tables": True},
        ),
    )
}


def thinking_drafts(record: data.CaseRecord) -> list[data.Attempt] | None:
    """The logged thinking drafts: the run thinks after a refusal and after a trimmed answer
    with the same prompt, so a variant that trims where the run refused (or the reverse)
    reuses them."""
    drafts = [a for a in record.attempts[CONFIG] if a.role in ("think", "think_repair")]
    return drafts or None


def run(e: Experiment) -> dict[str, Any]:
    records = data.records()
    v = verifier(CONFIG, **e.overrides)
    started = time.perf_counter()
    outcomes, base, texts, base_texts, missing = [], [], {}, {}, []
    for record in records.values():
        r = replay(record, v, source=CONFIG, trim_drafts=thinking_drafts)
        o = scored(record, r, CONFIG)
        b = logged(record, CONFIG)
        outcomes.append(o)
        base.append(b)
        texts[record.id] = r.published
        row = record.rows[CONFIG]
        base_texts[record.id] = None if row["refused"] else row["answer"]
        if r.missing:
            missing.append({"id": record.id, "role": r.missing})
    result: dict[str, Any] = {
        "experiment": e.name,
        "hypothesis": e.hypothesis,
        "text": e.text,
        "overrides": e.overrides,
        "baseline": f"run {data.RUN.name}, judge, logged",
        "seconds": round(time.perf_counter() - started, 1),
        "missing": missing,
    }
    parts: dict[str, Callable[[dict[str, Any]], bool]] = {
        "all": lambda o: True,
        "v3": lambda o: o["dataset"] == "v3",
        "xquad-full": lambda o: o["dataset"] == "xquad-full",
        "dev": lambda o: o["split"] == "dev",
        "test": lambda o: o["split"] == "test",
    }
    for part, keep in parts.items():
        idx = [i for i, o in enumerate(outcomes) if keep(o)]
        result[part] = metrics.block([outcomes[i] for i in idx], [base[i] for i in idx])
    result["completeness"] = {
        "variant": metrics.completeness(texts),
        "baseline": metrics.completeness(base_texts),
    }
    result["regressions"] = metrics.regressions(outcomes, base)
    result["changed"] = [
        {
            "id": o["id"],
            "split": o["split"],
            "baseline": {"correct": b["factrag"]["correct"], "published": base_texts[o["id"]]},
            "variant": {"correct": o["factrag"]["correct"], "path": o["path"],
                        "missing": o["missing"], "published": texts[o["id"]]},
        }
        for o, b in zip(outcomes, base, strict=True)
        if (texts[o["id"]] or "") != (base_texts[o["id"]] or "")
        or o["factrag"]["correct"] != b["factrag"]["correct"]
    ]
    return result


def save(result: dict[str, Any]) -> None:
    out = RESULTS / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{result['experiment']}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    path = RESULTS / "EXPERIMENTS.csv"
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = {row["experiment"]: row for row in csv.DictReader(handle)}
    a, p = result["all"], result["all"]["paired"]
    c = result["completeness"]
    rows[result["experiment"]] = {
        "experiment": result["experiment"],
        "kind": "offline replay",
        "hypothesis": result["hypothesis"],
        "text": result["text"],
        "overrides": json.dumps(result["overrides"], ensure_ascii=False),
        "correct": a["correct_k"],
        "baseline_correct": a["correct_k"] - p["gained"] + p["lost"],
        "gained": p["gained"],
        "lost": p["lost"],
        "mcnemar_p": round(p["mcnemar_p"], 4),
        "dev_gained_lost": f"{result['dev']['paired']['gained']}/{result['dev']['paired']['lost']}",
        "test_gained_lost": f"{result['test']['paired']['gained']}/{result['test']['paired']['lost']}",
        "hallucination": a["hallucination_k"],
        "new_hallucinations": len(result["regressions"]["new_hallucinations"]),
        "false_refusal": a["false_refusal_k"],
        "complete_compound": f"{c['variant']['complete']}/{c['variant']['compound_questions']}",
        "baseline_complete_compound": f"{c['baseline']['complete']}/{c['baseline']['compound_questions']}",
        "changed_texts": len(result["changed"]),
        "missing": len(result["missing"]),
        "seconds": result["seconds"],
    }
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(rows.values()))))
        writer.writeheader()
        writer.writerows(rows.values())


def report(result: dict[str, Any]) -> None:
    a, p = result["all"], result["all"]["paired"]
    c = result["completeness"]
    print(
        f"{result['experiment']}: correct {a['correct']} halluc {a['hallucination']} "
        f"FR {a['false_refusal']} | +{p['gained']} -{p['lost']} p={p['mcnemar_p']:.3f} "
        f"dev {result['dev']['paired']['gained']}/{result['dev']['paired']['lost']} "
        f"test {result['test']['paired']['gained']}/{result['test']['paired']['lost']} "
        f"complete {c['variant']['complete']}/{c['variant']['compound_questions']} "
        f"(base {c['baseline']['complete']}) changed {len(result['changed'])} "
        f"missing {len(result['missing'])} regressions {result['regressions']}",
        flush=True,
    )
    for ch in result["changed"]:
        print("   ", ch["id"], ch["split"], ch["baseline"]["correct"], "->",
              ch["variant"]["correct"], ch["variant"]["path"], ch["variant"]["missing"])


def main(names: list[str]) -> None:
    for name in names or list(REGISTRY):
        result = run(REGISTRY[name])
        save(result)
        report(result)


if __name__ == "__main__":
    main(sys.argv[1:])
