"""The claim stress set of optimization_2 under the 1222Z verifier with each variant.

6000 claims verified against the logged contexts of run 0508Z drafts (research.opt2.stress:
positive, answer_swap, number_swap, negation, transplant). Those contexts come from parser
v1 and XQuAD, so the set checks that a variant leaves the verifier unchanged outside the
tables it targets. The verifier is the one of run 1222Z (judge, its settings) plus overrides.

Usage:  python -m research.optimization_6.stress [B00 T1 T2 T12]
Output: reports/optimization_6/offline/stress_opt2.json
"""

from __future__ import annotations

import json
import sys

import research.optimization_6  # noqa: F401 - selects the baseline run and the output folder
from research.opt2 import data as opt2_data
from research.opt2 import stress
from research.opt3.replay import verifier
from research.opt4.offline import CONFIG, RESULTS
from research.optimization_6.offline import REGISTRY


def main(names: list[str]) -> None:
    items = stress.load()
    records = opt2_data.records(("base", "think"))
    target = RESULTS / "stress_opt2.json"
    result = json.loads(target.read_text("utf-8")) if target.exists() else {}
    for name in names or list(REGISTRY):
        result[name] = stress.evaluate(verifier(CONFIG, **REGISTRY[name].overrides), items, records)
        print(f"{name}: {stress.summary_line(result[name])}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
