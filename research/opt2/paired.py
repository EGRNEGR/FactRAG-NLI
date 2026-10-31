"""PHASE 2: paired Vanilla/FactRAG error analysis of the 0508Z baseline, case by case.

Baseline = the "thinking only on refusal (computed)" row of the run's COMPARISON.md: the
qwen38-27b outcome, replaced by the qwen38-27b-think outcome when the non-thinking first
draft was an LLM refusal (refusal_reason llm_refusal after one attempt). 1163/1307 correct.

Every case gets a quadrant (A: Vanilla correct -> FactRAG correct, B: correct -> incorrect,
C: incorrect -> correct, D: incorrect -> incorrect) and a root cause derived from the logged
claims of its verification attempts. Writes error_analysis/paired_cases.csv and
PAIRED_ERROR_ANALYSIS.md (also copied to the top of reports/optimization_2/).

Usage:  python -m research.opt2.paired
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

from research.opt2 import data
from research.opt2.metrics import by, outcome, pct, summarize

CONTRADICTION, ENTAILMENT = 0.2, 0.4  # verifier v2 thresholds of the run (config.json)


def baseline_config(record: data.CaseRecord) -> str:
    row = record.rows["base"]
    return "think" if row["refusal_reason"] == "llm_refusal" and row["attempts"] == 1 else "base"


def evidence_of(claim: dict[str, Any]) -> list[dict[str, Any]]:
    return list(claim.get("evidence") or [])


def supports(item: dict[str, Any]) -> bool:
    p = item.get("probabilities")
    if item.get("support"):
        return True
    return bool(p) and p["p_entailment"] > ENTAILMENT and p["p_contradiction"] < CONTRADICTION


def claim_cause(claim: dict[str, Any], gold: set[str]) -> str:
    """Why one fact claim was not verified, from its logged evidence."""
    reason = claim.get("reason")
    if reason in {"missing_citation", "invalid_citation", "unknown_citation", "empty_claim"}:
        return f"format:{reason}"
    if reason == "anchor_not_in_source":
        return "preprocessing:anchor_not_in_source"
    evidence = evidence_of(claim)
    if any(item.get("reason") == "pair_exceeds_nli_context" for item in evidence):
        return "preprocessing:pair_exceeds_nli_context"
    cited = {item.get("chunk_id") for item in evidence}
    if gold and not cited & gold:
        return "citation:not_gold_chunk"
    probs = [item["probabilities"] for item in evidence if item.get("probabilities")]
    if len(evidence) > 1 and any(supports(i) for i in evidence):
        return "decision:multi_source_veto"
    if claim["status"] == "contradiction":
        return "model:false_contradiction"
    if probs and max(p["p_entailment"] for p in probs) > ENTAILMENT:
        return "decision:entailment_below_strong"
    return "model:missed_entailment"


def multi_source_conflict(attempts: Sequence[data.Draft]) -> bool:
    for draft in attempts:
        for claim in draft.claims:
            evidence = evidence_of(claim)
            if (
                len(evidence) > 1
                and any(supports(i) for i in evidence)
                and any(
                    (i.get("probabilities") or {}).get("p_contradiction", 0) >= CONTRADICTION
                    and not i.get("support")
                    for i in evidence
                )
            ):
                return True
    return False


def facts(draft: data.Draft) -> list[dict[str, Any]]:
    return [c for c in draft.claims if c.get("kind", "fact") == "fact"]


def root_cause(record: data.CaseRecord, config: str, o: dict[str, Any]) -> tuple[str, str]:
    """(cause, detail) of the case outcome."""
    row = record.rows[config]
    drafts = record.drafts[config]
    gold = set(record.case.gold_chunk_ids)
    van, fac = o["vanilla"], o["factrag"]
    reason = row["refusal_reason"]
    if reason == "relevance_threshold_not_met":
        return ("ok:gate_refusal" if fac["correct"] else "retrieval:gate", "")
    first = drafts[0] if drafts else None
    if van["correct"] and fac["correct"]:
        if len(drafts) > 1:
            return ("ok:after_repair", "")
        rejected = [c for c in facts(first) if c["status"] != "verified"] if first else []
        return ("ok:claims_dropped" if rejected else "ok:clean", "")
    if not van["correct"] and fac["correct"]:
        if van["hallucination"] and not fac["hallucination"]:
            return ("prevented:hallucination", "")
        if o["expected"] != "answer":
            return ("prevented:wrong_or_unsupported", "")
        return ("fixed:repair_attempt", "")
    if van["correct"] and not fac["correct"]:
        if fac["refused"]:
            if reason == "context_budget_exceeded":
                return ("damage:context_budget", "")
            if first and first.metadata.get("finish_reason") not in {"stop", "eos_token"}:
                return ("damage:truncated", "")
            claims = facts(first) if first else []
            verified = [c for c in claims if c["status"] == "verified"]
            if reason == "llm_refusal" and len(drafts) > 1:
                causes = Counter(claim_cause(c, gold) for c in claims if c["status"] != "verified")
                top = causes.most_common(1)[0][0] if causes else "unknown"
                return (f"damage:{top}", "repair draft was an LLM refusal")
            if verified:
                return (
                    "damage:aggregation_majority_or_guard",
                    f"{len(verified)}/{len(claims)} verified",
                )
            causes = Counter(claim_cause(c, gold) for c in claims)
            top = causes.most_common(1)[0][0] if causes else "unknown"
            return (f"damage:{top}", json.dumps(dict(causes)))
        if len(drafts) > 1:
            return ("damage:repair_changed_answer", "")
        return ("damage:aggregation_dropped_key_claim", "")
    # D: both incorrect
    if o["expected"] == "answer" and record.rows[config].get("gold_in_sources") is False:
        return ("both:retrieval_miss", "")
    if fac["refused"]:
        if van["refused"]:
            return ("both:llm_refusal", "")
        if not van["facts_ok"] and o["expected"] == "answer":
            return ("both:wrong_draft_withheld", "")
        return ("both:withheld", "")
    if fac["hallucination"]:
        return ("both:hallucination_published", "")
    return ("both:wrong_answer_published", "")


def analyse() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = data.records(("base", "think"))
    rows, outcomes = [], []
    for case_id, record in records.items():
        config = baseline_config(record)
        row = record.rows[config]
        o = outcome(
            record,
            vanilla_text=row["raw_answer"] if row["raw_answer"] is not None else row["answer"],
            published_text=None if row["refused"] else row["answer"],
        )
        # The log's own scores must agree with the frozen dataset.
        assert o["factrag"]["correct"] == row["factrag"]["correct"], case_id
        assert o["vanilla"]["correct"] == row["vanilla"]["correct"], case_id
        quadrant = (
            ("A" if o["factrag"]["correct"] else "B")
            if o["vanilla"]["correct"]
            else ("C" if o["factrag"]["correct"] else "D")
        )
        cause, detail = root_cause(record, config, o)
        drafts = record.drafts[config]
        first = drafts[0] if drafts else None
        rejected = [
            {
                "claim": c["claim"],
                "status": c["status"],
                "reason": c.get("reason"),
                "p": [
                    {k: round(v, 3) for k, v in (i.get("probabilities") or {}).items()}
                    for i in evidence_of(c)
                ],
                "gold": bool(
                    {i.get("chunk_id") for i in evidence_of(c)} & set(record.case.gold_chunk_ids)
                ),
            }
            for c in (facts(first) if first else [])
            if c["status"] != "verified"
        ]
        o.update(quadrant=quadrant, cause=cause)
        outcomes.append(o)
        rows.append(
            {
                "id": case_id,
                "dataset": o["dataset"],
                "split": record.split,
                "category": record.case.category,
                "subtype": record.case.subtype,
                "language": record.case.language,
                "expected_behavior": o["expected"],
                "baseline_config": config,
                "attempts": row["attempts"],
                "quadrant": quadrant,
                "root_cause": cause,
                "cause_detail": detail,
                "vanilla_correct": o["vanilla"]["correct"],
                "factrag_correct": o["factrag"]["correct"],
                "factrag_refused": o["factrag"]["refused"],
                "refusal_reason": row["refusal_reason"] or "",
                "vanilla_hallucination": o["vanilla"]["hallucination"],
                "factrag_hallucination": o["factrag"]["hallucination"],
                "gold_in_sources": row.get("gold_in_sources"),
                "multi_source_conflict": multi_source_conflict(drafts),
                "question": record.case.question,
                "ground_truth": record.case.ground_truth_answer,
                "first_draft": (first.raw_text if first else "").replace("\n", " / "),
                "published": (row["answer"] if not row["refused"] else "").replace("\n", " / "),
                "rejected_claims": json.dumps(rejected, ensure_ascii=False),
            }
        )
    return rows, outcomes


def table(counter: Counter[str], total: int, title: str) -> list[str]:
    lines = [f"| {title} | Cases | Share |", "|---|---|---|"]
    for key, n in counter.most_common():
        lines.append(f"| {key} | {n} | {100 * n / total:.1f}% |")
    return lines


def report(rows: list[dict[str, Any]], outcomes: list[dict[str, Any]], notes: str) -> str:
    total = len(rows)
    summary = summarize(outcomes)
    quad = Counter(r["quadrant"] for r in rows)
    lines = [
        "# Paired error analysis: Vanilla vs FactRAG, baseline 0508Z",
        "",
        "Baseline: `qwen38-27b + bge-reranker-v2-m3, thinking only on refusal (computed)` from",
        "`20261031T0508Z_rtx3090_full/COMPARISON.md`, verifier v2, 1307 questions (v3 220 +",
        "XQuAD full 1087). Vanilla = first LLM draft of the configuration that answered; FactRAG =",
        "published text. Scored with the dataset versions the run used (sha256 in config.json).",
        "Generated by `python -m research.opt2.paired`; per-case data in",
        "`error_analysis/paired_cases.csv`.",
        "",
        f"FactRAG correct {pct(summary['factrag_correct'])}, Vanilla correct "
        f"{pct(summary['vanilla_correct'])}, hallucination {pct(summary['hallucination'])}, false "
        f"refusal {pct(summary['false_refusal'])}. Verifier net contribution "
        f"{summary['net_contribution']:+d} (correction {summary['correction']}, damage "
        f"{summary['damage']}).",
        "",
        "## Quadrants",
        "",
        "| Quadrant | Meaning | Cases | Share | v3 | XQuAD |",
        "|---|---|---|---|---|---|",
    ]
    meaning = {
        "A": "Vanilla correct -> FactRAG correct",
        "B": "Vanilla correct -> FactRAG incorrect (damage)",
        "C": "Vanilla incorrect -> FactRAG correct (correction)",
        "D": "Vanilla incorrect -> FactRAG incorrect",
    }
    for q in "ABCD":
        n = quad[q]
        v3 = sum(r["quadrant"] == q and r["dataset"] == "v3" for r in rows)
        lines.append(f"| {q} | {meaning[q]} | {n} | {100 * n / total:.1f}% | {v3} | {n - v3} |")
    for q in "ABCD":
        subset = [r for r in rows if r["quadrant"] == q]
        lines += ["", f"### {q}: {meaning[q]}", ""]
        lines += table(Counter(r["root_cause"] for r in subset), total, "Root cause")
    flags = {
        "Correct answer (answerable question) replaced by a refusal": sum(
            r["vanilla_correct"] and r["factrag_refused"] and r["expected_behavior"] == "answer"
            for r in rows
        ),
        "Correct draft with at least one claim not verified": sum(
            r["vanilla_correct"] and r["rejected_claims"] != "[]" for r in rows
        ),
        "Wrong answer published (incl. hallucinations)": sum(
            not r["factrag_correct"] and not r["factrag_refused"] for r in rows
        ),
        "Hallucination prevented (draft hallucinated, published did not)": sum(
            bool(r["vanilla_hallucination"]) and r["factrag_hallucination"] is False for r in rows
        ),
        "NLI misread a correct claim citing the gold chunk": sum(
            r["vanilla_correct"]
            and any(
                c["gold"] and c["reason"] == "cited_support_not_established"
                for c in json.loads(r["rejected_claims"])
            )
            for r in rows
        ),
        "Retrieval: gold chunk not in the sources (answerable)": sum(
            r["expected_behavior"] == "answer" and r["gold_in_sources"] is False for r in rows
        ),
        "Claim with conflicting cited sources (one supports, one contradicts)": sum(
            r["multi_source_conflict"] for r in rows
        ),
        "Aggregation changed a correct result (majority rule, guard, dropped fact)": sum(
            r["root_cause"]
            in {"damage:aggregation_majority_or_guard", "damage:aggregation_dropped_key_claim"}
            for r in rows
        ),
    }
    lines += [
        "",
        "## Flags (a case may carry several)",
        "",
        "| Flag | Cases | Share |",
        "|---|---|---|",
    ]
    for name, n in flags.items():
        lines.append(f"| {name} | {n} | {100 * n / total:.1f}% |")
    lines += ["", "## Rejected claims of correct drafts by cause (claim level)", ""]
    claim_causes: Counter[str] = Counter()
    records = data.records(("base", "think"))
    for r in rows:
        if not r["vanilla_correct"]:
            continue
        record = records[r["id"]]
        drafts = record.drafts[r["baseline_config"]]
        if drafts:
            for c in facts(drafts[0]):
                if c["status"] != "verified":
                    claim_causes[claim_cause(c, set(record.case.gold_chunk_ids))] += 1
    lines += table(claim_causes, sum(claim_causes.values()) or 1, "Claim cause")
    lines += [
        "",
        "## By dataset and category",
        "",
        "| Group | FactRAG | Vanilla | Net | Correction | Damage |",
        "|---|---|---|---|---|---|",
    ]
    for key in ("dataset", "category"):
        for name, s in by(outcomes, key).items():
            lines.append(
                f"| {name} | {pct(s['factrag_correct'])} | {pct(s['vanilla_correct'])} "
                f"| {s['net_contribution']:+d} | {s['correction']} | {s['damage']} |"
            )
    f = summary["filter"]
    lines += [
        "",
        "## The verifier as a publish/withhold classifier",
        "",
        "Cases whose first draft is an answer; positive = draft correct, predicted positive =",
        "published.",
        "",
        f"TP {f['tp']}, FP {f['fp']}, FN {f['fn']}, TN {f['tn']}; precision {f['precision']:.3f}, "
        f"recall {f['recall']:.3f}, specificity {f['specificity']:.3f}, macro-F1 "
        f"{f['macro_f1']:.3f}, balanced accuracy {f['balanced_accuracy']:.3f}, MCC {f['mcc']:.3f}.",
        "",
    ]
    lines += [notes.rstrip(), ""]
    return "\n".join(lines)


def main() -> None:
    rows, outcomes = analyse()
    target = data.OUT / "error_analysis"
    target.mkdir(parents=True, exist_ok=True)
    with (target / "paired_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    notes_path = data.ROOT / "research" / "opt2" / "notes" / "paired_manual.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    text = report(rows, outcomes, notes)
    (target / "PAIRED_ERROR_ANALYSIS.md").write_text(text, encoding="utf-8")
    (data.OUT / "PAIRED_ERROR_ANALYSIS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
