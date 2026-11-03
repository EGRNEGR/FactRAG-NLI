"""Final comparison of the optimization_3 candidates on run 0749Z (both columns).

  B0  the run as logged (code 816c1de)
  C1  + anchor grouping fix (grounding._ungroup)
  C2  C1 + llm_think_on_trim (thinking drafts of research.opt3.trim_think)
  C3  C2 + recovery without the premise rule before the logged samples (candidate only)

Every candidate is compared with B0 of the same column, case by case (McNemar, paired
bootstrap), overall, per dataset and per dev/test half. Generation time adds the extra calls
at the median RTX 3090 time of that call type in the column's own logs.

Usage:  python -m research.opt3.final
"""

from __future__ import annotations

import csv
import json
import statistics
from typing import Any

from research.opt2.metrics import paired, pct, summarize
from research.opt3 import data, recovery, trim_think
from research.opt3.replay import logged, replay, scored, verifier


def call_ms(config: str) -> dict[str, float]:
    times: dict[str, list[float]] = {}
    for record in data.records().values():
        for a in record.attempts[config]:
            times.setdefault(a.role, []).append(a.metadata.get("generation_ms") or 0.0)
    return {role: statistics.median(v) for role, v in times.items()}


def hours(config: str, outcomes: list[dict[str, Any]], base: list[dict[str, Any]]) -> float:
    ms = call_ms(config)
    total = sum(
        a.metadata.get("generation_ms") or 0.0
        for r in data.records().values()
        for a in r.attempts[config]
    )
    for o, b in zip(outcomes, base, strict=True):
        extra = o["path"].split("+") if o["path"] != "gate" else []
        before = b["path"].split("+") if b["path"] != "gate" else []
        for role in set(extra) | set(before):
            total += (extra.count(role) - before.count(role)) * ms.get(role, ms["draft"])
    return total / 3.6e6


def block(outcomes: list[dict[str, Any]], base: list[dict[str, Any]]) -> dict[str, Any]:
    s = summarize(outcomes)
    return {
        "correct": pct(s["factrag_correct"]),
        "hallucination": pct(s["hallucination"]),
        "false_refusal": pct(s["false_refusal"]),
        "coverage": pct(s["coverage"]),
        "selective_accuracy": pct(s["selective_accuracy"]),
        "paired": paired(base, outcomes),
    }


def main() -> dict[str, Any]:
    records = data.records()
    rec_store = recovery.load_store()
    trim_store = trim_think.load_store()
    report: dict[str, Any] = {}
    rows = []
    for config in data.CONFIGS:
        v = verifier(config)
        base = [logged(r, config) for r in records.values()]
        candidates = {
            "B0": base,
            "C1": [scored(r, replay(r, v, source=config), config) for r in records.values()],
            "C2": [
                scored(
                    r,
                    replay(r, v, source=config,
                           trim_drafts=trim_think.drafts_of(trim_store, config)),
                    config,
                )
                for r in records.values()
            ],
            "C3": [
                scored(
                    r,
                    replay(
                        r, v, source=config,
                        trim_drafts=trim_think.drafts_of(trim_store, config),
                        recovery_drafts=recovery.drafts_of("relaxed", rec_store, config, True),
                    ),
                    config,
                )
                for r in records.values()
            ],
        }
        for name, outcomes in candidates.items():
            entry: dict[str, Any] = {"generation_hours": round(hours(config, outcomes, base), 3)}
            for part, keep in (
                ("all", lambda o: True),
                ("v3", lambda o: o["dataset"] == "v3"),
                ("xquad-full", lambda o: o["dataset"] == "xquad-full"),
                ("dev", lambda o: o["split"] == "dev"),
                ("test", lambda o: o["split"] == "test"),
            ):
                idx = [i for i, o in enumerate(outcomes) if keep(o)]
                entry[part] = block([outcomes[i] for i in idx], [base[i] for i in idx])
            report.setdefault(config, {})[name] = entry
            a = entry["all"]
            rows.append({
                "column": config,
                "candidate": name,
                "correct": a["correct"],
                "v3": entry["v3"]["correct"],
                "xquad": entry["xquad-full"]["correct"],
                "hallucination": a["hallucination"],
                "false_refusal": a["false_refusal"],
                "selective_accuracy": a["selective_accuracy"],
                "gained": a["paired"]["gained"],
                "lost": a["paired"]["lost"],
                "mcnemar_p": round(a["paired"]["mcnemar_p"], 4),
                "ci95_pp": " .. ".join(f"{x:+.2f}" for x in a["paired"]["ci95_pp"]),
                "dev_gained_lost": f"{entry['dev']['paired']['gained']}/{entry['dev']['paired']['lost']}",
                "test_gained_lost": f"{entry['test']['paired']['gained']}/{entry['test']['paired']['lost']}",
                "generation_hours": entry["generation_hours"],
            })
            print(rows[-1], flush=True)
    out = data.OUT / "final"
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with (data.OUT / "METRICS_COMPARISON.csv").open("w", encoding="utf-8-sig", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


if __name__ == "__main__":
    main()
