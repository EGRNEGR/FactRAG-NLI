"""PHASES 9-10: candidate architectures compared on the same 1307 cases (replay of run 0508Z).

A candidate = verifier (verifier v2 rules | decision layer) x thinking router x recovery.
Recovery (llm_recovery) uses the drafts of research.opt2.recovery; its trigger threshold
(best reranker score) is chosen on dev: the most correct dev answers with no more dev
hallucinations than the same candidate without recovery. Generation time is the logged RTX
3090 time; a recovery draft is costed at the case's own non-thinking draft time.

Writes METRICS_COMPARISON.csv and final/final_comparison.json, prints a Markdown table.

Usage:  python -m research.opt2.final
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from typing import Any

from research.opt2 import data, recovery
from research.opt2.metrics import Outcome, outcome, paired, pct, summarize
from research.opt2.routing import ROUTERS, Case, prepare
from research.opt2.routing import judge_verifier as build_verifier

THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)


def run_candidate(
    cases: list[Case],
    router: Callable[[Case], bool],
    v: Any,
    mode: str | None,
    threshold: float | None,
    store: dict[Any, Any],
) -> dict[str, Any]:
    outs: list[Outcome] = []
    calls = recovered = 0
    ms = 0.0
    for c in cases:
        routed = router(c)
        final = c.think if routed else c.base
        config = "think" if routed else "base"
        published = final.published
        ms += c.base_ms + (c.think_ms if routed else 0.0)
        if (
            mode
            and threshold is not None
            and final.reason == "llm_refusal"
            and recovery.top_rerank(c.record) >= threshold
        ):
            first_ms = c.record.drafts["base"][0].metadata.get("generation_ms") or 0.0
            ms += first_ms * len(recovery.MODES[mode])
            text = recovery.recovered(v, c.record, mode, store)
            if text is not None:
                published, recovered = text, recovered + 1
                config = "base"
        drafts = c.record.drafts[config]
        outs.append(
            outcome(
                c.record,
                vanilla_text=drafts[0].raw_text if drafts else None,
                published_text=published,
            )
        )
        calls += routed
    return {
        "outcomes": outs,
        "thinking_calls": calls,
        "recovered": recovered,
        "generation_hours_3090": ms / 3.6e6,
    }


def choose_threshold(cases, router, v, mode, store) -> tuple[float | None, list[dict[str, Any]]]:
    plain = run_candidate(cases, router, v, None, None, store)
    dev_h = sum(
        bool(o["factrag"]["hallucination"]) for o in plain["outcomes"] if o["split"] == "dev"
    )
    sweep, best = [], None
    for t in THRESHOLDS:
        res = run_candidate(cases, router, v, mode, t, store)
        dev = [o for o in res["outcomes"] if o["split"] == "dev"]
        correct = sum(o["factrag"]["correct"] for o in dev)
        halluc = sum(bool(o["factrag"]["hallucination"]) for o in dev)
        ok = halluc <= dev_h
        sweep.append(
            {"threshold": t, "dev_correct": correct, "dev_halluc": halluc, "admissible": ok}
        )
        if ok and (best is None or correct > best[0] or (correct == best[0] and t > best[1])):
            best = (correct, t)
    return (best[1] if best else None), sweep


def main() -> None:
    store = recovery.load_store()
    verifiers = {"rules": build_verifier("none"), "judge": build_verifier("bge-m3-zeroshot")}
    prepared = {name: prepare(v) for name, v in verifiers.items()}
    plan = [
        (
            "C0 baseline: verifier v2, thinking on first-draft LLM refusal",
            "rules",
            "A1_first_llm_refusal",
            None,
        ),
        ("C1 verifier v2, thinking on any withheld answer (f888cdf)", "rules", "A_withheld", None),
        ("C2 decision layer, thinking on any withheld answer", "judge", "A_withheld", None),
        ("C3 C2 + recovery: evidence prompt", "judge", "A_withheld", "evidence_prompt"),
        ("C4 C2 + recovery: 2 samples T=0.7", "judge", "A_withheld", "sampling"),
        ("C5 C1 + recovery: 2 samples T=0.7", "rules", "A_withheld", "sampling"),
        ("C6 C1 + recovery: evidence prompt", "rules", "A_withheld", "evidence_prompt"),
        ("R1 qwen38-27b without thinking, verifier v2", "rules", "A0_never", None),
        ("R2 qwen38-27b-think always, verifier v2", "rules", "Z_always", None),
        ("R3 qwen38-27b-think always, decision layer", "judge", "Z_always", None),
    ]
    results: dict[str, Any] = {}
    baseline: list[Outcome] | None = None
    for name, vname, router_name, mode in plan:
        router = ROUTERS[router_name][1]
        threshold, sweep = (None, [])
        if mode:
            threshold, sweep = choose_threshold(
                prepared[vname], router, verifiers[vname], mode, store
            )
        res = run_candidate(prepared[vname], router, verifiers[vname], mode, threshold, store)
        outs = res.pop("outcomes")
        if baseline is None:
            baseline = outs
        res.update(
            verifier=vname,
            router=router_name,
            recovery=mode,
            recovery_threshold=threshold,
            threshold_sweep=sweep,
            all=summarize(outs),
            dev=summarize([o for o in outs if o["split"] == "dev"]),
            test=summarize([o for o in outs if o["split"] == "test"]),
            **{ds: summarize([o for o in outs if o["dataset"] == ds]) for ds in data.DATASETS},
            vs_baseline_all=paired(baseline, outs),
            vs_baseline_test=paired(
                [o for o in baseline if o["split"] == "test"],
                [o for o in outs if o["split"] == "test"],
            ),
            missing_repair=sum(
                (c.base.missing_repair) or (router(c) and c.think.missing_repair)
                for c in prepared[vname]
            ),
        )
        results[name] = res
        s, t = res["all"], res["test"]
        p = res["vs_baseline_test"]
        print(
            f"{name[:58]:58s} all {s['factrag_correct']['k']} test {t['factrag_correct']['k']} "
            f"hal {s['hallucination']['k']} FR {s['false_refusal']['k']} think {res['thinking_calls']} "
            f"rec {res['recovered']}@{threshold} h {res['generation_hours_3090']:.2f} "
            f"test d {p['delta_pp']:+.2f} [{p['ci95_pp'][0]:+.2f},{p['ci95_pp'][1]:+.2f}] p {p['mcnemar_p']:.3g} "
            f"miss {res['missing_repair']}",
            flush=True,
        )
    out = data.OUT / "final"
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_comparison.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), "utf-8"
    )
    fields = [
        "configuration",
        "verifier",
        "router",
        "recovery",
        "recovery_threshold",
        "factrag_all",
        "factrag_dev",
        "factrag_test",
        "factrag_v3",
        "factrag_xquad_full",
        "vanilla_all",
        "hallucination_all",
        "hallucination_v3",
        "false_refusal_all",
        "false_refusal_v3",
        "false_refusal_xquad_full",
        "coverage",
        "selective_accuracy",
        "net_contribution",
        "correction",
        "damage",
        "filter_precision",
        "filter_recall",
        "filter_macro_f1",
        "filter_balanced_accuracy",
        "filter_mcc",
        "thinking_calls",
        "recovered",
        "generation_hours_3090",
        "delta_test_pp",
        "ci95_test_pp",
        "mcnemar_p_test",
        "delta_all_pp",
        "ci95_all_pp",
        "mcnemar_p_all",
        "missing_repair",
    ]
    with (data.OUT / "METRICS_COMPARISON.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, r in results.items():
            a, f = r["all"], r["all"]["filter"]
            writer.writerow(
                {
                    "configuration": name,
                    "verifier": r["verifier"],
                    "router": r["router"],
                    "recovery": r["recovery"] or "",
                    "recovery_threshold": r["recovery_threshold"]
                    if r["recovery_threshold"] is not None
                    else "",
                    "factrag_all": pct(a["factrag_correct"]),
                    "factrag_dev": pct(r["dev"]["factrag_correct"]),
                    "factrag_test": pct(r["test"]["factrag_correct"]),
                    "factrag_v3": pct(r["v3"]["factrag_correct"]),
                    "factrag_xquad_full": pct(r["xquad-full"]["factrag_correct"]),
                    "vanilla_all": pct(a["vanilla_correct"]),
                    "hallucination_all": pct(a["hallucination"]),
                    "hallucination_v3": pct(r["v3"]["hallucination"]),
                    "false_refusal_all": pct(a["false_refusal"]),
                    "false_refusal_v3": pct(r["v3"]["false_refusal"]),
                    "false_refusal_xquad_full": pct(r["xquad-full"]["false_refusal"]),
                    "coverage": pct(a["coverage"]),
                    "selective_accuracy": pct(a["selective_accuracy"]),
                    "net_contribution": a["net_contribution"],
                    "correction": a["correction"],
                    "damage": a["damage"],
                    "filter_precision": round(f["precision"], 4),
                    "filter_recall": round(f["recall"], 4),
                    "filter_macro_f1": round(f["macro_f1"], 4),
                    "filter_balanced_accuracy": round(f["balanced_accuracy"], 4),
                    "filter_mcc": round(f["mcc"], 4),
                    "thinking_calls": r["thinking_calls"],
                    "recovered": r["recovered"],
                    "generation_hours_3090": round(r["generation_hours_3090"], 2),
                    "delta_test_pp": round(r["vs_baseline_test"]["delta_pp"], 2),
                    "ci95_test_pp": [round(x, 2) for x in r["vs_baseline_test"]["ci95_pp"]],
                    "mcnemar_p_test": round(r["vs_baseline_test"]["mcnemar_p"], 4),
                    "delta_all_pp": round(r["vs_baseline_all"]["delta_pp"], 2),
                    "ci95_all_pp": [round(x, 2) for x in r["vs_baseline_all"]["ci95_pp"]],
                    "mcnemar_p_all": round(r["vs_baseline_all"]["mcnemar_p"], 4),
                    "missing_repair": r["missing_repair"],
                }
            )


if __name__ == "__main__":
    main()
