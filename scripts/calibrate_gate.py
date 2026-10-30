"""Calibrate RAG_RERANK_SCORE_THRESHOLD (the relevance gate) for another reranker.

The gate refuses a question when no substantive chunk reaches the threshold. Its value 0.05
was tuned for bge-reranker-large; a different reranker has a different score scale. This
script reads top_rerank_score from retrieval runs (evaluate_suite.py retrieval) and picks,
on the dev half of the dataset only, the threshold that rejects the most unanswerable
questions while rejecting no larger share of answerable ones than the reference reranker
does at its own threshold. The test half is reported as the overfitting check.

Usage:
  python scripts/calibrate_gate.py --run-dir .artifacts/smoke/v3_v2m3 \\
      --reference-run .artifacts/smoke/v3_large --reference-threshold 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_dataset import load_dataset  # noqa: E402
from evaluate_suite import DATASET, dataset_split  # noqa: E402


def load_scores(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "retrieval" / "cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if rows and "top_rerank_score" not in rows[0]:
        raise SystemExit(f"{path} has no top_rerank_score: rerun the retrieval mode")
    return {row["id"]: row for row in rows}


def gate(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Rejection shares at a threshold; a missing score (no substantive chunk) is rejected."""

    def rejected(row: dict[str, Any]) -> bool:
        score = row["top_rerank_score"]
        return score is None or score < threshold

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    frr = sum(map(rejected, answerable)) / len(answerable) if answerable else 0.0
    catch = sum(map(rejected, unanswerable)) / len(unanswerable) if unanswerable else 0.0
    return {
        "threshold": round(threshold, 4),
        "false_rejection_answerable": round(frr, 4),
        "rejected_unanswerable": round(catch, 4),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
    }


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cli.add_argument("--dataset", type=Path, default=DATASET)
    cli.add_argument("--run-dir", type=Path, required=True)
    cli.add_argument("--reference-run", type=Path, help="Retrieval run of the reference reranker")
    cli.add_argument("--reference-threshold", type=float, default=0.05)
    cli.add_argument(
        "--max-frr", type=float, help="Allowed false rejection share (default: reference dev)"
    )
    args = cli.parse_args(argv)
    dataset = load_dataset(args.dataset)
    split = dataset_split(dataset.cases)
    scores = load_scores(args.run_dir)
    halves = {
        name: [scores[c.id] for c in dataset.cases if split[c.id] == name and c.id in scores]
        for name in ("dev", "test")
    }
    report: dict[str, Any] = {"run": str(args.run_dir)}
    target = args.max_frr
    if args.reference_run is not None:
        reference = load_scores(args.reference_run)
        ref = {
            name: gate(
                [reference[c.id] for c in dataset.cases if split[c.id] == name],
                args.reference_threshold,
            )
            for name in ("dev", "test")
        }
        report["reference"] = {"run": str(args.reference_run), **ref}
        if target is None:
            target = ref["dev"]["false_rejection_answerable"]
    if target is None:
        target = 0.025
    observed = {r["top_rerank_score"] for r in halves["dev"] if r["top_rerank_score"] is not None}
    candidates = sorted({0.0} | observed)
    # Each observed score as a threshold rejects exactly the cases strictly below it.
    feasible = [
        gate(halves["dev"], t)
        for t in candidates
        if gate(halves["dev"], t)["false_rejection_answerable"] <= target
    ]
    if not feasible:
        raise SystemExit(f"No threshold keeps dev false rejection <= {target}")
    best = max(feasible, key=lambda g: (g["rejected_unanswerable"], -g["threshold"]))
    # Midpoint to the next lower observed score keeps the dev decision and adds a margin.
    lower = [t for t in candidates if t < best["threshold"]]
    chosen = round((best["threshold"] + lower[-1]) / 2, 4) if lower else best["threshold"]
    report.update(
        {
            "max_false_rejection_dev": target,
            "chosen_threshold": chosen,
            "dev": gate(halves["dev"], chosen),
            "test": gate(halves["test"], chosen),
            "current_0.05": {name: gate(rows, 0.05) for name, rows in halves.items()},
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
