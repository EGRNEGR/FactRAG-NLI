"""Comparison table of an RTX 3090 benchmark run (scripts/linux/run_benchmark.sh).

Reads <run>/<profile>+<reranker>/{v3,xquad,xquad-full}/e2e_guard_on/summary.json,
<run>/retrieval/<dataset>+<reranker>/retrieval/summary.json and <run>/nli/{xquad,xquad-full},
prints a Markdown report and writes it to <run>/COMPARISON.md. A configuration evaluated on
several datasets also gets a pooled "all" row (rates over the summed counts). The live laptop run 20261026T2139Z_v3
(RTX 5060 Laptop 8 GB, Qwen2.5-14B, verifier v1) is listed as the reference for v3.

With both qwen38-27b+<reranker> and qwen38-27b-think+<reranker> logged, a "thinking only on
refusal" row is computed per case from the two logs: a case whose non-thinking first draft
was a refusal takes the outcome of the thinking configuration (the same retrieval, prompt and
seed, so its first thinking call is the call such a hybrid would make).

Usage:  python scripts/compare_runs.py reports/runs/<run_id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reports" / "runs" / "20261026T2139Z_v3" / "e2e_guard_on" / "summary.json"
DATASETS = ("v3", "xquad", "xquad-full")


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


def pooled(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summed counts of several E2E summaries; latencies are not pooled."""

    def block(*keys: str) -> dict[str, Any]:
        k = n = 0
        for summary in summaries:
            value: Any = summary
            for key in keys:
                value = value.get(key) or {}
            k += value.get("k", 0)
            n += value.get("n", 0)
        return {"k": k, "n": n, "rate": k / n if n else None}

    return {
        "cases": sum(s["cases"] for s in summaries),
        "errors": sum(s["errors"] for s in summaries),
        "vanilla": {"correct": block("vanilla", "correct")},
        "factrag": {
            "correct": block("factrag", "correct"),
            "hallucination": block("factrag", "hallucination"),
            "false_refusal_on_answerable": block("factrag", "false_refusal_on_answerable"),
        },
        "latency_ms": {"generated_queries": {}},
    }


def case_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {row["id"]: row for row in rows}


def think_on_refusal(
    base: dict[str, dict[str, Any]], think: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], int] | None:
    """Summary of the hybrid over the cases logged by both configurations, and its trigger count."""
    shared = [case_id for case_id in base if case_id in think]
    if not shared:
        return None
    rows, triggered = [], 0
    for case_id in shared:
        row = base[case_id]
        if row.get("refusal_reason") == "llm_refusal" and row.get("attempts") == 1:
            row, triggered = think[case_id], triggered + 1
        rows.append(row)
    ok = [row for row in rows if row.get("ok")]

    def block(items: Sequence[bool]) -> dict[str, Any]:
        k, n = sum(bool(x) for x in items), len(items)
        return {"k": k, "n": n, "rate": k / n if n else None}

    answer = [row for row in ok if row["expected_behavior"] == "answer"]
    other = [row for row in ok if row["expected_behavior"] != "answer"]
    summary = {
        "cases": len(rows),
        "errors": len(rows) - len(ok),
        "vanilla": {"correct": block([row["vanilla"]["correct"] for row in ok])},
        "factrag": {
            "correct": block([row["factrag"]["correct"] for row in ok]),
            "hallucination": block([row["factrag"]["hallucination"] for row in other]),
            "false_refusal_on_answerable": block([row["factrag"]["refused"] for row in answer]),
        },
        "latency_ms": {"generated_queries": {}},
    }
    return summary, triggered


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
        found = []
        for dataset in DATASETS:
            summary = load(config / dataset / "e2e_guard_on" / "summary.json")
            if summary:
                lines.append(e2e_row(config.name, dataset, summary))
                found.append(summary)
        if len(found) > 1:
            lines.append(e2e_row(config.name, "all", pooled(found)))
    hybrid_notes = []
    for think_dir in sorted(run.glob("qwen38-27b-think+*")):
        reranker = think_dir.name.split("+", 1)[1]
        base_dir = run / f"qwen38-27b+{reranker}"
        found = []
        for dataset in DATASETS:
            log = Path(dataset) / "e2e_guard_on" / "cases.jsonl"
            result = think_on_refusal(case_rows(base_dir / log), case_rows(think_dir / log))
            if result:
                summary, triggered = result
                label = f"qwen38-27b+{reranker}, thinking only on refusal (computed)"
                lines.append(e2e_row(label, dataset, summary))
                found.append(summary)
                hybrid_notes.append(
                    f"{dataset}: thinking call for {triggered} of {summary['cases']} cases "
                    f"(non-thinking first draft was a refusal)."
                )
        if len(found) > 1:
            lines.append(e2e_row(label, "all", pooled(found)))
    if hybrid_notes:
        lines += [
            "",
            "Thinking only on refusal, computed from the two logs: " + " ".join(hybrid_notes),
        ]
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
    for dataset in DATASETS[1:]:
        nli = load(run / "nli" / dataset / "nli" / "summary.json")
        if not nli:
            continue
        rule = nli["overall"]["decision_rule"]
        lines += [
            "",
            f"## NLI probes, {dataset}",
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
