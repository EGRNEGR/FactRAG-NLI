"""Is the decision-layer threshold (0.3, fitted on run 0508Z) still right for run 0749Z?

For each candidate threshold: the replayed judge column (dev half only for the choice) and
the acceptance of the optimization_2 stress claims (false claims built from real drafts:
answer swap, number swap, negation, transplant; dev half). A lower threshold may publish a
correct answer the run withheld (V3-TC-016 0.289, V3-XQFEN-171 0.292, both test) only if
dev shows the gain without more false-claim acceptance. Test is reported once.

Usage:  python -m research.opt3.threshold
"""

from __future__ import annotations

import json
from typing import Any

from research.opt2 import data as opt2_data
from research.opt2 import stress
from research.opt2.metrics import paired, summarize
from research.opt3 import data
from research.opt3.experiments import headline
from research.opt3.replay import logged, replay, scored, verifier

GRID = (0.2, 0.25, 0.28, 0.3, 0.32, 0.35, 0.4)


def main() -> dict[str, Any]:
    records = data.records()
    base = {r.id: logged(r, "judge") for r in records.values()}
    stress_items = stress.load()
    stress_records = opt2_data.records(("base", "think"))
    report: dict[str, Any] = {}
    for t in GRID:
        v = verifier("judge", judge={"threshold": t})
        outcomes = [scored(r, replay(r, v, source="judge"), "judge") for r in records.values()]
        row: dict[str, Any] = {}
        for part in ("dev", "test"):
            sel = [o for o in outcomes if o["split"] == part]
            row[part] = headline(summarize(sel))
            row[part]["paired"] = paired([base[o["id"]] for o in sel], sel)
            row[part]["published_wrong_answerable"] = sum(
                o["expected"] == "answer" and o["published"] and not o["factrag"]["correct"]
                for o in sel
            )
            row[part]["missing"] = sum(bool(o["missing"]) for o in sel)
        acceptance = stress.evaluate(v, stress_items, stress_records)
        row["stress"] = acceptance
        report[str(t)] = row
        d = row["dev"]
        print(
            t, "dev correct", d["correct"], "+%d -%d" % (d["paired"]["gained"], d["paired"]["lost"]),
            "halluc", d["hallucination"], "pub_wrong", d["published_wrong_answerable"],
            "missing", d["missing"], "| stress dev:", stress.summary_line(acceptance, "dev"),
            flush=True,
        )
    (data.OUT / "experiments").mkdir(parents=True, exist_ok=True)
    (data.OUT / "experiments" / "judge_threshold.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    main()
