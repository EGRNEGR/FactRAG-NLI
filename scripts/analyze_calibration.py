"""Replay ranking parameters on cached real scores; never replay generation as measured."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_quality import ranking_metrics


def sweep(report: dict[str, Any]) -> list[dict[str, Any]]:
    examples = {e["id"]: e for e in report["dataset"]["examples"]}
    config = report["configuration"]
    bound = (float(config.get("dense_weight", 1)) + float(config.get("sparse_weight", 1))) / (
        float(config.get("rrf_k", 60)) + 1
    )
    results = []
    for alpha in (0.0, 0.25, 0.5, 0.8, 1.0):
        for threshold in (0.0, 0.01, 0.05, 0.1, 0.5):
            scores = []
            early_negative = 0
            for row in report["rows"]:
                example = examples[row["id"]]
                candidates = row["response"]["retrieval_diagnostics"]
                eligible = [c for c in candidates if c["rerank_score"] >= threshold]
                ranked = sorted(
                    eligible,
                    key=lambda c: (
                        -(alpha * c["rrf_score"] / bound + (1 - alpha) * c["rerank_score"]),
                        -c["rrf_score"],
                        c["chunk_id"],
                    ),
                )
                if example["is_negative"]:
                    early_negative += not ranked
                else:
                    if any(j["chunk_id"] is None for j in example["relevant"]):
                        raise ValueError("This replay requires explicit chunk judgments")
                    grades = {j["chunk_id"]: j["grade"] for j in example["relevant"]}
                    scores.append(
                        ranking_metrics([c["chunk_id"] for c in ranked], grades, report["k"])
                    )
            results.append(
                {
                    "alpha": alpha,
                    "threshold": threshold,
                    "positive_cases": len(scores),
                    "negative_early_rejections": early_negative,
                    "means": {key: sum(s[key] for s in scores) / len(scores) for key in scores[0]},
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.output.write_text(
        json.dumps(
            {
                "role": "in_sample_ranking_calibration_not_heldout",
                "source_report": str(args.report),
                "generation_replayed": False,
                "sweep": sweep(report),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
