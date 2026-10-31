"""PHASE 4: fit the calibrated decision layer (ClaimJudge) on the dev half only.

1. Features: the verifier itself (nli_decision=judge, a zero judge, feature_sink) scores every
   stress-set claim against the logged contexts of its draft with the primary (mDeBERTa) and
   the secondary NLI model; one row per (claim, cited source).
2. Model: L2 logistic regression on the dev rows (label: positive claim vs any synthetic
   negative), C chosen by 5-fold cross-validated log-loss on dev, inverse-frequency class
   weights so the negative kinds do not swamp the positives.
3. Threshold: replay the 0508Z pipeline on the dev cases for each candidate threshold; keep
   the thresholds at which every negative kind is accepted on dev no more often than by
   verifier v2 (V00) and take the one with the most correct dev answers (ties: the higher
   threshold). Test cases are not touched here.

Writes data/nli_judge/<secondary>.json (the file RAG_NLI_JUDGE_PATH points to) and
reports/optimization_2/judge/<secondary>.json with the fitting record.

Usage:  python -m research.opt2.judge bge-m3-zeroshot
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from generator_verifier import ClaimJudge, NLIFactVerifier, RawGenerationResult

from research.opt2 import data, stress
from research.opt2.experiments import EXPERIMENTS
from research.opt2.metrics import ece, summarize
from research.opt2.nli_models import CachedNLI
from research.opt2.replay import on_withheld, run, settings

FEATURES = (
    "a_e_max",
    "a_c_at_a",
    "a_c_max",
    "b_e_max",
    "b_c_max",
    "ab_e_max",
    "a_full_e",
    "a_full_c",
    "b_full_e",
    "overlap",
    "code_span",
    "anchors_ok",
    "has_anchors",
)
JUDGE_DIR = data.ROOT / "data" / "nli_judge"
NEGATIVES = ("answer_swap", "number_swap", "negation", "transplant")


def zero_judge() -> ClaimJudge:
    return ClaimJudge(
        {"features": FEATURES, "coef": [0.0] * len(FEATURES), "intercept": 0.0, "threshold": 0.5}
    )


def judge_verifier(secondary: str, judge: ClaimJudge, **overrides: Any) -> NLIFactVerifier:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(judge.spec, handle)
    s = settings(nli_decision="judge", nli_judge_path=handle.name, **overrides)
    v = NLIFactVerifier(s, backend=CachedNLI("mdeberta"), secondary=CachedNLI(secondary), judge=judge)
    Path(handle.name).unlink()
    return v


def capture(secondary: str, records: dict[str, data.CaseRecord]) -> list[dict[str, Any]]:
    """Feature rows of every stress claim (one per cited source)."""
    v = judge_verifier(secondary, zero_judge())
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(stress.load()):
        draft = records[item["id"]].drafts[item["config"]][0]
        v.feature_sink = []
        v.verify(
            RawGenerationResult.model_validate(
                {"text": item["text"], "contexts": draft.contexts, "metadata": draft.metadata}
            )
        )
        for entry in v.feature_sink:
            rows.append(
                {
                    "id": item["id"],
                    "item": index,
                    "split": item["split"],
                    "kind": item["kind"],
                    "label": int(item["kind"] == "positive"),
                    "features": entry["features"],
                }
            )
    return rows


def fit(rows: list[dict[str, Any]]) -> tuple[list[float], float, dict[str, Any]]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score

    dev = [r for r in rows if r["split"] == "dev"]
    x = np.array([[r["features"][f] for f in FEATURES] for r in dev])
    y = np.array([r["label"] for r in dev])
    groups = np.array([r["id"] for r in dev])
    scores = {}
    for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
        scores[c] = float(
            cross_val_score(model, x, y, cv=GroupKFold(5), groups=groups, scoring="neg_log_loss").mean()
        )
    best_c = max(scores, key=lambda c: scores[c])
    model = LogisticRegression(C=best_c, class_weight="balanced", max_iter=5000).fit(x, y)
    return list(map(float, model.coef_[0])), float(model.intercept_[0]), {
        "cv_neg_log_loss": scores,
        "C": best_c,
        "dev_rows": len(dev),
        "dev_label_counts": dict(Counter(int(v) for v in y)),
    }


def main(secondary: str) -> None:
    records = data.records(("base", "think"))
    rows = capture(secondary, records)
    coef, intercept, record = fit(rows)
    spec = {
        "version": 1,
        "primary_model": "mdeberta-nli (MoritzLaurer mDeBERTa-v3-base multilingual-nli-2mil7)",
        "secondary_model": secondary,
        "features": FEATURES,
        "coef": coef,
        "intercept": intercept,
        "threshold": 0.5,
        "contradiction": 0.5,
        "disagreement": 0.6,
        "veto": 0.9,
        "fitted_on": "stress claims of the dev half of run 20261031T0508Z (research/opt2/judge.py)",
    }
    baseline = json.loads((EXPERIMENTS / "V00_baseline.json").read_text(encoding="utf-8"))
    limits = {k: baseline["stress"][k]["dev"]["rate"] for k in NEGATIVES}
    dev_items = [i for i in stress.load() if i["split"] == "dev"]
    dev_ids = [cid for cid, r in records.items() if r.split == "dev"]
    sweep = []
    for threshold in [round(0.05 + 0.05 * i, 2) for i in range(19)]:
        spec["threshold"] = threshold
        spec["partial_threshold"] = threshold / 2
        judge = ClaimJudge(spec)
        v = judge_verifier(secondary, judge)
        st = stress.evaluate(v, dev_items, records)
        cases = run(records, v, on_withheld, ids=dev_ids)
        s = summarize([c.outcome for c in cases])
        point = {
            "threshold": threshold,
            "dev_factrag": s["factrag_correct"]["k"],
            "dev_halluc": s["hallucination"]["k"],
            "stress": {k: st[k]["dev"]["rate"] for k in st},
            "admissible": all(st[k]["dev"]["rate"] <= limits[k] + 1e-9 for k in NEGATIVES),
        }
        sweep.append(point)
        print(point, flush=True)
    admissible = [p for p in sweep if p["admissible"]]
    chosen = max(admissible or sweep, key=lambda p: (p["dev_factrag"], p["threshold"]))
    spec["threshold"] = chosen["threshold"]
    spec["partial_threshold"] = chosen["threshold"] / 2
    judge = ClaimJudge(spec)
    dev_scores = [
        (judge.probability(r["features"]), r["label"]) for r in rows if r["split"] == "dev"
    ]
    record.update(
        sweep=sweep,
        limits_from_V00_dev=limits,
        chosen=chosen,
        calibration_dev=ece([p for p, _ in dev_scores], [bool(y) for _, y in dev_scores]),
    )
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    (JUDGE_DIR / f"{secondary}.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    out = data.OUT / "judge"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{secondary}.json").write_text(json.dumps({"spec": spec, **record}, indent=1), "utf-8")
    print("chosen", chosen, "C", record["C"])
    print("coef", dict(zip(FEATURES, [round(c, 3) for c in coef])), "intercept", round(intercept, 3))
    print("ECE dev", round(record["calibration_dev"]["ece"], 4), "logit", math.isfinite(intercept))


if __name__ == "__main__":
    main(sys.argv[1])
