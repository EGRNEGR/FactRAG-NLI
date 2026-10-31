"""PHASE 7: calibration of claim support scores and of the published-answer confidence.

Claim level (stress set, one score per claim = max over its cited sources): raw mDeBERTa
best-window entailment, raw bge-m3-zeroshot entailment, their mean, and the decision-layer
probability; each fitted on dev with temperature scaling, Platt scaling and isotonic
regression, and evaluated on test (ECE with 10 equal-width bins, Brier, log-loss, ROC-AUC).

Answer level: for published answers of the replayed hybrid (verifier v2 and the decision
layer), confidence = the lowest claim confidence, label = answer correct; ECE on test before
and after an isotonic map fitted on dev.

Usage:  python -m research.opt2.calibration
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from typing import Any

from generator_verifier import ClaimJudge

from research.opt2 import data
from research.opt2.judge import capture
from research.opt2.metrics import ece
from research.opt2.replay import on_withheld, run
from research.opt2.routing import judge_verifier


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-max(-60.0, min(60.0, z))))


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def per_claim(rows: list[dict[str, Any]], judge: ClaimJudge) -> list[dict[str, Any]]:
    """Collapse source rows to claims (rows of one stress item are consecutive)."""
    claims: list[dict[str, Any]] = []
    previous = None
    for r in rows:
        f = r["features"]
        scores = {
            "mdeberta": sigmoid(f["a_e_max"]),
            "bge-m3-zeroshot": sigmoid(f["b_e_max"]),
            "mean": sigmoid(f["ab_e_max"]),
            "decision_layer": judge.probability(f),
        }
        key = (r["id"], r["kind"], r.get("item"))
        if previous == key:
            for name, value in scores.items():
                claims[-1][name] = max(claims[-1][name], value)
            continue
        previous = key
        claims.append({"split": r["split"], "label": r["label"], **scores})
    return claims


def fit_calibrators(x: Sequence[float], y: Sequence[int]) -> dict[str, Callable[[float], float]]:
    import numpy as np
    from scipy.optimize import minimize_scalar
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    z = np.array([logit(v) for v in x])
    t = np.array(y, dtype=float)

    def nll(temperature: float) -> float:
        p = 1 / (1 + np.exp(-z / temperature))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))

    temperature = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
    platt = LogisticRegression(C=1e6, max_iter=5000).fit(z.reshape(-1, 1), t)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, t)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    return {
        "none": lambda v: v,
        f"temperature (T={temperature:.2f})": lambda v: sigmoid(logit(v) / temperature),
        "platt": lambda v: sigmoid(a * logit(v) + b),
        "isotonic": lambda v: float(iso.predict([v])[0]),
    }


def scores(p: Sequence[float], y: Sequence[int]) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    clipped = [min(max(v, 1e-7), 1 - 1e-7) for v in p]
    e = ece(list(p), [bool(v) for v in y])
    return {
        "ece": e["ece"],
        "brier": e["brier"],
        "log_loss": -sum(
            t * math.log(q) + (1 - t) * math.log(1 - q) for q, t in zip(clipped, y, strict=True)
        )
        / len(y),
        "roc_auc": float(roc_auc_score(y, p)),
        "n": len(y),
    }


def main() -> None:
    records = data.records(("base", "think"))
    spec = json.loads(
        (data.ROOT / "data" / "nli_judge" / "bge-m3-zeroshot.json").read_text("utf-8")
    )
    judge = ClaimJudge(spec)
    rows = capture("bge-m3-zeroshot", records)
    claims = per_claim(rows, judge)
    report: dict[str, Any] = {"claim_level": {}, "answer_level": {}}
    for name in ("mdeberta", "bge-m3-zeroshot", "mean", "decision_layer"):
        dev = [c for c in claims if c["split"] == "dev"]
        test = [c for c in claims if c["split"] == "test"]
        calibrators = fit_calibrators([c[name] for c in dev], [c["label"] for c in dev])
        report["claim_level"][name] = {
            method: scores([fn(c[name]) for c in test], [c["label"] for c in test])
            for method, fn in calibrators.items()
        }
        for method, s in report["claim_level"][name].items():
            print(
                f"claim {name:16s} {method:22s} ECE {s['ece']:.4f} Brier {s['brier']:.4f} AUC {s['roc_auc']:.4f}"
            )
    for name in ("none", "bge-m3-zeroshot"):
        cases = run(records, judge_verifier(name), on_withheld)
        points = []
        for c in cases:
            if c.final.refused or c.final.verified is None:
                continue
            facts = [x for x in c.final.verified.claims if x.kind == "fact"]
            conf = min((x.confidence for x in facts), default=0.0)
            points.append((c.outcome["split"], conf, int(c.outcome["factrag"]["correct"])))
        dev = [(p, y) for s, p, y in points if s == "dev"]
        test = [(p, y) for s, p, y in points if s == "test"]
        calibrators = fit_calibrators([p for p, _ in dev], [y for _, y in dev])
        report["answer_level"][name] = {
            method: scores([fn(p) for p, _ in test], [y for _, y in test])
            for method, fn in calibrators.items()
            if method in ("none", "isotonic")
        }
        for method, s in report["answer_level"][name].items():
            print(
                f"answer {name:15s} {method:10s} ECE {s['ece']:.4f} Brier {s['brier']:.4f} AUC {s['roc_auc']:.4f} n {s['n']}"
            )
    out = data.OUT / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
