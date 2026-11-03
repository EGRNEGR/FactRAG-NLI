"""SUMMARY.md of one run of scripts/linux/run_experiments.sh (RTX 3090).

Reads <out>/plan.json, <out>/e2e/comparison.json and <out>/steps/summary.json and writes
<out>/SUMMARY.md: per experiment the population, the paired result against the baseline
(gained / lost, McNemar p), hallucination, false refusal, completeness of compound answers,
the regressions and the measured time. Exit code 1 if any experiment has a regression.

Usage:  python -m research.opt4.live_report reports/optimization_4/live/<run>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    start = text.find("{")  # plan.json is the tee of a log line block
    return json.loads(text[start:]) if start >= 0 else None


def row(name: str, s: dict[str, Any], minutes: float | None) -> str:
    a = s.get("all") or {}
    p = a.get("paired") or {}
    c = s.get("completeness") or {}
    reg = s.get("regressions") or {}
    cv, cb = c.get("variant") or {}, c.get("baseline") or {}
    verdict = "REGRESSION" if reg.get("lost") or reg.get("new_hallucinations") else "ok"
    return (
        f"| {name} | {s.get('population', s.get('questions', 0))} | {s.get('replaced', '-')} "
        f"| +{p.get('gained', 0)} / -{p.get('lost', 0)} | {p.get('mcnemar_p', 1):.3f} "
        f"| {a.get('hallucination', '-')} | {a.get('false_refusal', '-')} "
        f"| {cb.get('complete', '-')} -> {cv.get('complete', '-')} of {cv.get('compound_questions', '-')} "
        f"| {', '.join(reg.get('lost', [])) or '-'} | {', '.join(reg.get('new_hallucinations', [])) or '-'} "
        f"| {minutes if minutes is not None else '-'} | {verdict} |"
    )


def main(out: Path) -> int:
    plan = load(out / "plan.json") or {}
    e1 = load(out / "e2e" / "comparison.json")
    flags_file = out / "e2e" / "flags.txt"
    flags = flags_file.read_text(encoding="utf-8").strip() if flags_file.exists() else ""
    steps = load(out / "steps" / "summary.json") or {}
    lines = [
        f"# Experiments {out.name}",
        "",
        "Baseline: run 20261103T1310Z_rtx3090_main (judge), paired per question on the same ids.",
        "Population = questions the experiment's trigger selects; untriggered questions keep",
        "the baseline answer by construction. Estimate from the per-call medians of 1310Z:",
        f"{plan.get('estimate_total_min', '-')} min for the steps.",
        "",
        "| Experiment | Questions | Replaced | Paired | McNemar p | Hallucination | False refusal "
        "| Complete compound | Lost | New hallucinations | Minutes | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    failed = False
    if e1:
        name = f"e2e, {flags or 'baseline settings'} (real pipeline)"
        lines.append(row(name, e1, round(e1["seconds"] / 60, 1)))
        failed |= bool(e1["regressions"]["lost"] or e1["regressions"]["new_hallucinations"])
    for name, s in steps.items():
        if name == "D0":
            continue
        lines.append(row(f"{name} ({s.get('hypothesis')})", s, s.get("minutes")))
        reg = s.get("regressions") or {}
        failed |= bool(reg.get("lost") or reg.get("new_hallucinations"))
    d0 = steps.get("D0")
    if d0:
        lines += ["", f"D0 determinism: {d0['identical']} of {d0['checked']} first drafts "
                  "identical to the log."]
    projected = [
        f"- {name}: all 1307 questions {s['projected_all']['correct']} "
        f"(+{s['projected_all']['paired']['gained']} / -{s['projected_all']['paired']['lost']}), "
        f"hallucination {s['projected_all']['hallucination']}, "
        f"false refusal {s['projected_all']['false_refusal']}"
        for name, s in steps.items()
        if name != "D0" and s.get("projected_all")
    ]
    if projected:
        lines += ["", "Projected on all questions (baseline 1184/1307):", *projected]
    if e1 and e1.get("changed"):
        lines += ["", "e2e changed answers:"]
        lines += [f"- {c['id']}: {c['correct_before']} -> {c['correct_after']}" for c in e1["changed"]]
    lines += ["", "Per-question results: e2e/*/e2e_guard_on, steps/<experiment>/cases.jsonl.", ""]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
