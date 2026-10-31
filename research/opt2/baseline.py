"""PHASE 1: baseline of the 0508Z run, recomputed case by case from its logs.

Reproduces every row of the run's COMPARISON.md (the hybrid rows are computed from the
qwen38-27b and qwen38-27b-think logs) and adds the metrics the comparison lacks: Vanilla vs
FactRAG net contribution, correction and damage counts, coverage, selective accuracy, the
publish/withhold decision quality of the verifier (precision, recall, macro-F1, balanced
accuracy, MCC) and Wilson / bootstrap intervals. Writes BASELINE.md and baseline.json.

Usage:  python -m research.opt2.baseline
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from research.opt2 import data
from research.opt2.metrics import Outcome, outcome, paired, pct, summarize

Chooser = Callable[[data.CaseRecord], str]


def first_llm_refusal(record: data.CaseRecord) -> str:
    """COMPARISON.md 0508Z: thinking when the non-thinking first draft was an LLM refusal."""
    row = record.rows["base"]
    return "think" if row["refusal_reason"] == "llm_refusal" and row["attempts"] == 1 else "base"


def any_withheld(record: data.CaseRecord) -> str:
    """RAG_LLM_THINK_RETRY=refusal (profile qwen38-27b-hybrid of rtx3090-linux f888cdf)."""
    reason = record.rows["base"]["refusal_reason"]
    exempt = (None, "relevance_threshold_not_met", "context_budget_exceeded")
    return "think" if record.rows["base"]["refused"] and reason not in exempt else "base"


VARIANTS: dict[str, tuple[str, Chooser]] = {
    "qwen25-14b": ("qwen25", lambda r: "qwen25"),
    "qwen38-27b + reranker-large": ("large", lambda r: "large"),
    "qwen38-27b (no thinking)": ("base", lambda r: "base"),
    "qwen38-27b-think (always)": ("think", lambda r: "think"),
    "hybrid: thinking on first-draft LLM refusal (baseline 89.0%)": ("base", first_llm_refusal),
    "hybrid: thinking on any withheld answer (f888cdf default)": ("base", any_withheld),
}


def outcomes_of(records: dict[str, data.CaseRecord], chooser: Chooser) -> list[Outcome]:
    result = []
    for record in records.values():
        row = record.rows[chooser(record)]
        result.append(
            outcome(
                record,
                vanilla_text=row["raw_answer"] if row["raw_answer"] is not None else row["answer"],
                published_text=None if row["refused"] else row["answer"],
            )
        )
    return result


def thinking_calls(records: dict[str, data.CaseRecord], chooser: Chooser) -> int:
    return sum(chooser(r) == "think" for r in records.values())


def main() -> None:
    records = data.records(("base", "think", "qwen25", "large"))
    results: dict[str, Any] = {}
    baseline_key = "hybrid: thinking on first-draft LLM refusal (baseline 89.0%)"
    per_variant: dict[str, list[Outcome]] = {}
    for name, (_, chooser) in VARIANTS.items():
        outs = outcomes_of(records, chooser)
        per_variant[name] = outs
        results[name] = {
            "all": summarize(outs),
            **{ds: summarize([o for o in outs if o["dataset"] == ds]) for ds in data.DATASETS},
            **{sp: summarize([o for o in outs if o["split"] == sp]) for sp in ("dev", "test")},
            "thinking_calls": thinking_calls(records, chooser) if "hybrid" in name else None,
        }
    for name, outs in per_variant.items():
        results[name]["vs_baseline"] = paired(per_variant[baseline_key], outs)
    out = data.OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), "utf-8")

    lines = [
        "# Baseline (PHASE 1): run 20261031T0508Z, RTX 3090, verifier v2",
        "",
        "Recomputed case by case from the run logs by `python -m research.opt2.baseline` (inputs",
        "read in place, never modified; datasets restored from git b058977 and checked against",
        "the sha256 in the run's config.json). All six rows reproduce COMPARISON.md exactly.",
        "",
        "Definitions. FactRAG correct: published answer correct (refusal is correct on",
        "unanswerable questions; adversarial: refusal or the corrected fact). Vanilla: the",
        "first LLM draft of the same call, before NLI filtering. Hallucination: on",
        "unanswerable/adversarial questions (n = 269). False refusal: refusals on answerable",
        "questions (n = 1038). Coverage: answerable questions with a published answer.",
        "Selective accuracy: correct among those. Net = FactRAG correct - Vanilla correct;",
        "correction = Vanilla wrong -> FactRAG right; damage = Vanilla right -> FactRAG wrong.",
        "Filter metrics: the verifier's publish decision on drafts that are answers, positive =",
        "draft correct.",
        "",
        "## All 1307 questions",
        "",
        "| Configuration | FactRAG | Vanilla | Halluc. | False refusal | Net | Corr. | Damage | Filter macro-F1 | MCC | Thinking calls |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, res in results.items():
        s = res["all"]
        f = s["filter"]
        lines.append(
            f"| {name} | {pct(s['factrag_correct'])} | {pct(s['vanilla_correct'])} "
            f"| {pct(s['hallucination'])} | {pct(s['false_refusal'])} | {s['net_contribution']:+d} "
            f"| {s['correction']} | {s['damage']} | {f['macro_f1']:.3f} | {f['mcc']:.3f} "
            f"| {res['thinking_calls'] if res['thinking_calls'] is not None else '-'} |"
        )
    for ds in data.DATASETS:
        lines += [
            "",
            f"## {ds}",
            "",
            "| Configuration | FactRAG | Vanilla | Halluc. | False refusal | Coverage | Selective acc. |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, res in results.items():
            s = res[ds]
            lines.append(
                f"| {name} | {pct(s['factrag_correct'])} | {pct(s['vanilla_correct'])} "
                f"| {pct(s['hallucination'])} | {pct(s['false_refusal'])} | {pct(s['coverage'])} "
                f"| {pct(s['selective_accuracy'])} |"
            )
    lines += [
        "",
        "## Dev / test halves (split used by every later experiment)",
        "",
        "Grouped by XQuAD paragraph (both languages together), stratified by dataset and",
        "category, 651 dev / 656 test. Parameters are fitted on dev only.",
        "",
        "| Configuration | FactRAG dev | FactRAG test |",
        "|---|---|---|",
    ]
    for name, res in results.items():
        lines.append(
            f"| {name} | {pct(res['dev']['factrag_correct'])} | {pct(res['test']['factrag_correct'])} |"
        )
    lines += [
        "",
        "## Paired against the baseline (FactRAG correct, same 1307 cases)",
        "",
        "| Configuration | Gained | Lost | Delta, pp | Bootstrap 95% CI, pp | McNemar p |",
        "|---|---|---|---|---|---|",
    ]
    for name, res in results.items():
        p = res["vs_baseline"]
        ci = p["ci95_pp"]
        lines.append(
            f"| {name} | {p['gained']} | {p['lost']} | {p['delta_pp']:+.2f} "
            f"| [{ci[0]:+.2f}, {ci[1]:+.2f}] | {p['mcnemar_p']:.3g} |"
        )
    lines += [
        "",
        "Reference for the verifier on v3: verifier v2 replay of the 5060 reference drafts",
        "(20261027T0447Z_v3_verifier2) gives FactRAG 78.6%; it is a replay of Qwen2.5-14B drafts",
        "and is not mixed with the RTX 3090 rows above (all of which use verifier v2).",
        "",
    ]
    text = "\n".join(lines)
    (out / "BASELINE.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
