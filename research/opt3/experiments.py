"""Registry of verifier experiments on the logs of run 0749Z (offline replay).

Every experiment replays the logged drafts of one configuration ("source") under a verifier
("rules" | "judge") with settings overrides and compares it, case by case, with the run's own
result of that configuration (exact McNemar test, paired bootstrap interval; overall, dev and
test). Results go to reports/optimization_3/experiments/<name>.json and EXPERIMENTS.csv.

The code under test is the working tree: an experiment that measures a code change is run
after the change; its "baseline" is the logged run, which the unchanged code reproduces
1307/1307 (python -m research.opt3.replay).

Usage:  python -m research.opt3.experiments [name ...]
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from research.opt2.metrics import paired, pct, summarize
from research.opt3 import data
from research.opt3.replay import logged, replay, scored, verifier


@dataclass(frozen=True)
class Experiment:
    name: str
    text: str
    verifier: str  # rules | judge
    source: str  # drafts of: rules | judge
    overrides: dict[str, Any] = field(default_factory=dict)
    recovery_min_rerank: float = 0.5


REGISTRY: dict[str, Experiment] = {}


def register(*experiments: Experiment) -> None:
    for e in experiments:
        REGISTRY[e.name] = e


register(
    Experiment("X00_rules_replay", "rules verifier, rules drafts, current code", "rules", "rules"),
    Experiment("X00_judge_replay", "judge verifier, judge drafts, current code", "judge", "judge"),
    Experiment("X01_judge_on_rules", "judge verifier on the rules drafts", "judge", "rules"),
    Experiment("X01_rules_on_judge", "rules verifier on the judge drafts", "rules", "judge"),
    Experiment(
        "X02_judge_anchor_document",
        "identifier anchors may come from a same-document retrieved chunk",
        "judge",
        "judge",
        {"nli_anchor_scope": "document"},
    ),
    Experiment(
        "X02_rules_anchor_document",
        "identifier anchors may come from a same-document retrieved chunk",
        "rules",
        "rules",
        {"nli_anchor_scope": "document"},
    ),
)


def headline(s: dict[str, Any]) -> dict[str, Any]:
    return {
        "correct": s["factrag_correct"]["k"],
        "correct_rate": s["factrag_correct"]["rate"],
        "hallucination": s["hallucination"]["k"],
        "false_refusal": s["false_refusal"]["k"],
        "coverage": s["coverage"]["rate"],
        "selective_accuracy": s["selective_accuracy"]["rate"],
        "damage": s["damage"],
        "correction": s["correction"],
    }


def evaluate(e: Experiment) -> dict[str, Any]:
    records = data.records()
    v = verifier(e.verifier, **e.overrides)
    started = time.perf_counter()
    outcomes, base, missing = [], [], []
    for record in records.values():
        r = replay(record, v, source=e.source, recovery_min_rerank=e.recovery_min_rerank)
        o = scored(record, r, e.source)
        outcomes.append(o)
        base.append(logged(record, e.source))
        if r.missing:
            missing.append({"id": record.id, "role": r.missing})
    result: dict[str, Any] = {
        "experiment": e.name,
        "text": e.text,
        "verifier": e.verifier,
        "source": e.source,
        "overrides": e.overrides,
        "recovery_min_rerank": e.recovery_min_rerank,
        "seconds": round(time.perf_counter() - started, 1),
        "missing": missing,
        "baseline": f"logged {e.source}",
    }
    for part in ("all", "dev", "test"):
        keep = [i for i, o in enumerate(outcomes) if part == "all" or o["split"] == part]
        variant = [outcomes[i] for i in keep]
        reference = [base[i] for i in keep]
        result[part] = {
            "variant": headline(summarize(variant)),
            "baseline": headline(summarize(reference)),
            "paired": paired(reference, variant),
        }
    for dataset in data.DATASETS:
        keep = [i for i, o in enumerate(outcomes) if o["dataset"] == dataset]
        result[dataset] = {
            "variant": headline(summarize([outcomes[i] for i in keep])),
            "baseline": headline(summarize([base[i] for i in keep])),
        }
    changed = []
    for o, b in zip(outcomes, base, strict=True):
        if o["factrag"]["correct"] != b["factrag"]["correct"] or o["published"] != b["published"]:
            changed.append(
                {
                    "id": o["id"],
                    "split": o["split"],
                    "baseline": {"correct": b["factrag"]["correct"], "path": b["path"],
                                 "hallucination": b["factrag"]["hallucination"]},
                    "variant": {"correct": o["factrag"]["correct"], "path": o["path"],
                                "hallucination": o["factrag"]["hallucination"],
                                "missing": o["missing"]},
                }
            )
    result["changed"] = changed
    result["summary"] = summarize(outcomes)
    return result


def save(result: dict[str, Any]) -> None:
    out = data.OUT / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{result['experiment']}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    path = data.OUT / "EXPERIMENTS.csv"
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = {row["experiment"]: row for row in csv.DictReader(handle)}
    a, p = result["all"], result["all"]["paired"]
    rows[result["experiment"]] = {
        "experiment": result["experiment"],
        "text": result["text"],
        "verifier": result["verifier"],
        "source": result["source"],
        "overrides": json.dumps(result["overrides"], ensure_ascii=False),
        "correct": a["variant"]["correct"],
        "baseline_correct": a["baseline"]["correct"],
        "dev_delta": result["dev"]["variant"]["correct"] - result["dev"]["baseline"]["correct"],
        "test_delta": result["test"]["variant"]["correct"] - result["test"]["baseline"]["correct"],
        "gained": p["gained"],
        "lost": p["lost"],
        "mcnemar_p": round(p["mcnemar_p"], 4),
        "ci95_pp": " .. ".join(f"{x:+.2f}" for x in p["ci95_pp"]),
        "hallucination": a["variant"]["hallucination"],
        "baseline_hallucination": a["baseline"]["hallucination"],
        "false_refusal": a["variant"]["false_refusal"],
        "baseline_false_refusal": a["baseline"]["false_refusal"],
        "missing": len(result["missing"]),
        "seconds": result["seconds"],
    }
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(rows.values()))))
        writer.writeheader()
        writer.writerows(rows.values())


def report(result: dict[str, Any]) -> None:
    s = result["summary"]
    p = result["all"]["paired"]
    print(
        f"{result['experiment']}: {pct(s['factrag_correct'])} halluc {pct(s['hallucination'])} "
        f"FR {pct(s['false_refusal'])} | vs {result['baseline']}: +{p['gained']} -{p['lost']} "
        f"p={p['mcnemar_p']:.3f} dev {result['dev']['paired']['gained']}/"
        f"{result['dev']['paired']['lost']} test {result['test']['paired']['gained']}/"
        f"{result['test']['paired']['lost']} missing {len(result['missing'])}"
    )
    for c in result["changed"]:
        print("   ", c["id"], c["split"], c["baseline"], "->", c["variant"])


def main(names: list[str]) -> None:
    for name in names or list(REGISTRY):
        result = evaluate(REGISTRY[name])
        save(result)
        report(result)


if __name__ == "__main__":
    main(sys.argv[1:])
