"""Offline experiments of optimization_6: table row windows and bit cells on the 1222Z logs.

The engine is research.opt4.offline (pipeline-order replay of the logged drafts with the
run's settings plus overrides, persistent NLI caches). B00 is the replay without overrides;
every variant is compared with the logged run and, for the decision, with B00 (the replay
lacks the G3 translation pass, so B00 itself differs from the log on those questions).

  B00  replay of 1222Z, current code, no override
  T1   nli_table_row_windows=true
  T2   nli_anchor_bit_cells=true
  T12  both

Results: reports/optimization_6/offline/experiments/<name>.json, EXPERIMENTS.csv and
VS_B00.json (variant against B00 per question).

Usage:  python -m research.optimization_6.offline [B00 T1 T2 T12]
"""

from __future__ import annotations

import json
import sys
from typing import Any

import research.optimization_6  # noqa: F401 - selects the baseline run and the output folder
from research.opt4 import offline
from research.opt4.offline import Experiment

ROWS = {"nli_table_row_windows": True}
BITS = {"nli_anchor_bit_cells": True}
REGISTRY = {
    e.name: e
    for e in (
        Experiment("B00", "-", "replay of run 1222Z, current code, no override"),
        Experiment("T1", "H7", "one NLI window per table row with its column headers", ROWS),
        Experiment("T2", "H7", "bit patterns over table cells match claim anchors", BITS),
        Experiment("T12", "H7", "table row windows and bit cells", {**ROWS, **BITS}),
    )
}


def versus(result: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """Questions whose correctness differs between a variant and B00."""
    changed = {c["id"]: c for c in result["changed"]}
    base_changed = {c["id"]: c for c in base["changed"]}
    rows = []
    for cid in sorted(set(changed) | set(base_changed)):
        v, b = changed.get(cid), base_changed.get(cid)
        # A question absent from a "changed" list kept the logged outcome there.
        v_ok = v["variant"]["correct"] if v else (b or v)["baseline"]["correct"]
        b_ok = b["variant"]["correct"] if b else (v or b)["baseline"]["correct"]
        split = (v or b)["split"]
        v_text = v["variant"]["published"] if v else (b or v)["baseline"]["published"]
        b_text = b["variant"]["published"] if b else (v or b)["baseline"]["published"]
        if v_ok != b_ok or (v_text or "") != (b_text or ""):
            rows.append({"id": cid, "split": split, "b00": b_ok, "variant": v_ok,
                         "b00_published": b_text, "variant_published": v_text})
    return {
        "gained": [r["id"] for r in rows if r["variant"] and not r["b00"]],
        "lost": [r["id"] for r in rows if r["b00"] and not r["variant"]],
        "text_changed": [r["id"] for r in rows if r["variant"] == r["b00"]],
        "rows": rows,
    }


def main(names: list[str]) -> None:
    results = {}
    for name in names or list(REGISTRY):
        result = offline.run(REGISTRY[name])
        offline.save(result)
        offline.report(result)
        results[name] = result
    path = offline.RESULTS / "experiments"
    base = results.get("B00") or json.loads((path / "B00.json").read_text("utf-8"))
    summary = {n: versus(r, base) for n, r in results.items() if n != "B00"}
    target = offline.RESULTS / "VS_B00.json"
    if target.exists():
        summary = {**json.loads(target.read_text("utf-8")), **summary}
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for n, s in summary.items():
        split = {r["id"]: r["split"] for r in s["rows"]}
        print(f"{n} vs B00: gained {[(i, split[i]) for i in s['gained']]} "
              f"lost {[(i, split[i]) for i in s['lost']]} text changed {len(s['text_changed'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
