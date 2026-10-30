"""Comparison table of an RTX 3090 benchmark run (scripts/linux/run_benchmark.sh).

Reads <run>/<profile>+<reranker>/{v3,xquad}/e2e_guard_on/summary.json,
<run>/retrieval/<dataset>+<reranker>/retrieval/summary.json and <run>/nli/xquad, prints a
Markdown report and writes it to <run>/COMPARISON.md. The live laptop run 20261026T2139Z_v3
(RTX 5060 Laptop 8 GB, Qwen2.5-14B, verifier v1) is listed as the reference for v3.

Usage:  python scripts/compare_runs.py reports/runs/<run_id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reports" / "runs" / "20261026T2139Z_v3" / "e2e_guard_on" / "summary.json"


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(block: dict[str, Any] | None) -> str:
    if not block or block.get("rate") is None:
        return "-"
    return f"{100 * block['rate']:.1f}% ({block['k']}/{block['n']})"


def seconds(stats: dict[str, Any] | None) -> str:
    return f"{stats['p50'] / 1000:.1f}" if stats and stats.get("p50") is not None else "-"


def e2e_row(label: str, dataset: str, summary: dict[str, Any]) -> str:
    vanilla, factrag = summary["vanilla"], summary["factrag"]
    generated = summary["latency_ms"]["generated_queries"]
    return (
        f"| {label} | {dataset} | {summary['cases']} | {summary['errors']} "
        f"| {pct(vanilla['correct'])} | {pct(factrag['correct'])} "
        f"| {pct(factrag['hallucination'])} | {pct(factrag['false_refusal_on_answerable'])} "
        f"| {seconds(generated.get('generation_ms'))} | {seconds(generated.get('total_ms'))} |"
    )


def report(run: Path) -> str:
    lines = [
        f"# FactRAG-NLI on RTX 3090: {run.name}",
        "",
        "## End to end (profile guard_on)",
        "",
        "Vanilla = LLM draft before NLI filtering; FactRAG = published answer (same LLM call).",
        "Hallucination: share of answers asserting facts on unanswerable/adversarial questions;",
        "false refusal: refusals on answerable questions. Time: median seconds per generated query.",
        "",
        "| Configuration | Dataset | Cases | Errors | Vanilla correct | FactRAG correct "
        "| Hallucination | False refusal | Generation, s | Total, s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    reference = load(REFERENCE)
    if reference:
        lines.append(
            e2e_row("reference: RTX 5060 8 GB, Qwen2.5-14B (2139Z, verifier v1)", "v3", reference)
        )
    for config in sorted(p for p in run.iterdir() if p.is_dir() and "+" in p.name):
        for dataset in ("v3", "xquad"):
            summary = load(config / dataset / "e2e_guard_on" / "summary.json")
            if summary:
                lines.append(e2e_row(config.name, dataset, summary))
    lines += [
        "",
        "Verifier v2 replay of the reference drafts (20261027T0447Z_v3_verifier2): FactRAG 78.6%",
        "on v3. The 3090 runs use verifier v2, so compare v3 rows against 78.6% for the verifier",
        "and against the vanilla column for the LLM itself.",
        "",
        "## Retrieval (no LLM)",
        "",
        "| Dataset + reranker | Recall@5 | MRR@10 | Gate rejects unanswerable | Gate rejects answerable |",
        "|---|---|---|---|---|",
    ]
    retrieval = run / "retrieval"
    for path in sorted(retrieval.glob("*/retrieval/summary.json")) if retrieval.exists() else []:
        summary = load(path) or {}
        hybrid = summary.get("overall", {}).get("hybrid_rerank", {})
        gate = summary.get("relevance_gate", {})
        lines.append(
            f"| {path.parents[1].name} | {hybrid.get('recall@5', 0):.3f} | {hybrid.get('mrr@10', 0):.3f} "
            f"| {pct(gate.get('rejected_unanswerable'))} | {pct(gate.get('rejected_answerable'))} |"
        )
    nli = load(run / "nli" / "xquad" / "nli" / "summary.json")
    if nli:
        rule = nli["overall"]["decision_rule"]
        lines += [
            "",
            "## NLI probes, XQuAD",
            "",
            f"Catch rate of neutral probes {rule['hallucination_catch_rate']:.3f}, false rejection of "
            f"entailed probes {rule['false_rejection_rate']:.3f}, status agreement "
            f"{rule['status_agreement']:.3f} (n = {rule['n']}).",
        ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    run = Path(args[0])
    run = run if run.is_absolute() else ROOT / run
    text = report(run)
    (run / "COMPARISON.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
