"""Data of the article figures, recomputed from the run logs and the optimization reports.

Every number a figure shows comes from here (reports/article/figure_data.json); the plotting
script (scripts/article/figures.py) only draws it. Sources:

* COMPARISON.md of the full RTX 3090 runs 0749Z, 1310Z, 1929Z and 1222Z (accuracy, latency);
* the logs of run 1310Z (research.opt3.data): reranker score of the gold chunk for XQuAD
  questions, every verified claim with both NLI models and the judge score, per-pass latency;
* reports/optimization_4: the 52 persistent XQuAD refusals and their hand annotations
  (research/opt4/annotations.json), experiment E3 (reference translation of EN questions),
  the G3 runs 1801Z (steps) and 1839Z (real pipeline).

Usage (project environment):  python scripts/article/figure_data.py
Run folders: ARTICLE_RUN_<ID> overrides the default laptop paths below.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_HAKATONS = Path(r"C:\Hakatons")
RUNS = {
    "0749Z": _HAKATONS / "FactRAG-NLI-3090" / "reports" / "runs" / "20261102T0749Z_rtx3090_compare",
    "1310Z": _HAKATONS / "20261103T1310Z_rtx3090_main",
    "1929Z": _HAKATONS / "20261103T1310Z_rtx3090_main" / "20261103T1929Z_rtx3090_main",
    "1222Z": _HAKATONS
    / "20261103T1310Z_rtx3090_main"
    / "20261103T1929Z_rtx3090_main"
    / "20261104T1222Z_rtx3090_main",
}
RUNS = {k: Path(os.environ.get(f"ARTICLE_RUN_{k}", str(v))) for k, v in RUNS.items()}
# research.opt3.data reads the run named by OPT3_RUN (its default is run 0749Z).
os.environ["OPT3_RUN"] = str(RUNS["1310Z"])

OPT4 = ROOT / "reports" / "optimization_4"
OUT = ROOT / "reports" / "article" / "figure_data.json"
JUDGE = "qwen38-27b-hybrid+bge-reranker-v2-m3+judge"
_ROW = re.compile(r"^\| (?P<config>[^|]+) \| (?P<dataset>v3|xquad-full|all) \| (?P<rest>.*)\|\s*$")
_FRACTION = re.compile(r"\((\d+)/(\d+)\)")


def comparison(run: Path) -> dict[str, dict[str, Any]]:
    """Judge rows of a run's COMPARISON.md: counts and median latency per dataset."""
    result: dict[str, dict[str, Any]] = {}
    for line in (run / "COMPARISON.md").read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if not m or m["config"].strip() != JUDGE:
            continue
        cells = [c.strip() for c in m["rest"].split("|")]
        fractions = [tuple(map(int, f)) for f in _FRACTION.findall(m["rest"])]
        vanilla, correct, halluc, refusal = fractions[:4]
        result[m["dataset"]] = {
            "vanilla": vanilla, "correct": correct, "hallucination": halluc,
            "false_refusal": refusal,
            "generation_s": None if cells[-2] == "-" else float(cells[-2]),
            "total_s": None if cells[-1] == "-" else float(cells[-1]),
        }
    return result


def gold_scores() -> dict[str, Any]:
    """Reranker score of the gold chunk, XQuAD answerable questions of run 1310Z: the 52
    persistent refusals against the rest, per question language."""
    from research.opt3 import data

    refusals = {
        json.loads(line)["id"]
        for line in (OPT4 / "diagnostics" / "xquad_refusals.jsonl").open(encoding="utf-8")
    }
    responses = data.responses("judge")
    groups: dict[str, list[float]] = {}
    for cid, case in data.cases().items():
        if data.dataset_of(cid) != "xquad-full" or not case.answerable:
            continue
        r = responses[cid]
        gold = set(case.gold_chunk_ids)
        scores = [
            d.get("rerank_score") or 0.0
            for d in r.get("retrieval_diagnostics") or []
            if d["chunk_id"] in gold
        ] or [s.get("rerank_score") or 0.0 for s in r["sources"] if s["chunk_id"] in gold]
        key = f"{case.language}_{'refused' if cid in refusals else 'other'}"
        groups.setdefault(key, []).append(max(scores, default=0.0))
    return {
        k: {"n": len(v), "median": round(statistics.median(v), 3), "values": [round(x, 4) for x in v]}
        for k, v in sorted(groups.items())
    }


def translation_e3() -> dict[str, Any]:
    e3 = json.loads((OPT4 / "experiments" / "E3_translation_retrieval.json").read_text("utf-8"))
    rows = [r for r in e3["rows"] if r["answerable"]]
    before = [r["top_before"] for r in rows]
    after = [r["top_after"] for r in rows]
    s = e3["summary"]
    return {
        "n": len(rows),
        "median_before": round(statistics.median(before), 3),
        "median_after": round(statistics.median(after), 3),
        "before": before,
        "after": after,
        "gate_answerable": s["gate_passes_answerable"],
        "gate_unanswerable": s["gate_passes_unanswerable"],
        "gold_at_1": s["gold_at_1"],
        "answerable": s["en_answerable"],
        "unanswerable": s["en_unanswerable"],
    }


def taxonomy() -> dict[str, Any]:
    """Primary (first) annotation code of each of the 52 persistent XQuAD refusals."""
    annotations = json.loads((ROOT / "research" / "opt4" / "annotations.json").read_text("utf-8"))
    ids = [
        json.loads(line)["id"]
        for line in (OPT4 / "diagnostics" / "xquad_refusals.jsonl").open(encoding="utf-8")
    ]
    primary: Counter[str] = Counter()
    for cid in ids:
        entry = annotations["cases"][cid]
        primary[entry["code"][0]] += 1
    return {"cases": len(ids), "primary": dict(primary.most_common())}


def claims_and_latency() -> dict[str, Any]:
    """Every fact claim of every logged attempt of run 1310Z (judge): mDeBERTa and bge-m3
    entailment of the best-scored cited source, judge score, status and reason; and the
    per-pass generation time of the thinking pass after a trimmed answer."""
    from research.opt3 import data

    think_ids = {
        json.loads(line)["id"]
        for line in (OPT4 / "diagnostics" / "think_on_trim.jsonl").open(encoding="utf-8")
    }
    marked = {"V3-MH-002", "V3-MH-018", "V3-MH-024"}
    points, trim_ms = [], []
    stage: dict[str, dict[str, list[float]]] = {}
    for cid, response in data.responses("judge").items():
        dataset = data.dataset_of(cid)
        stats = response.get("execution_stats") or {}
        # As COMPARISON.md: medians over the questions that reached generation.
        if stats.get("generation_ms", 0) > 0:
            for key in ("search_ms", "rerank_ms", "generation_ms", "nli_ms", "total_ms"):
                if key in stats:
                    stage.setdefault(dataset, {}).setdefault(key, []).append(stats[key])
        attempts = response.get("verification_attempts") or []
        roles = data.roles(attempts)
        if cid in think_ids:
            trim_ms.append(
                sum(
                    a["generation_metadata"]["generation_ms"]
                    for a, role in zip(attempts, roles, strict=True)
                    if role in ("think", "think_repair")
                )
            )
        for attempt, role in zip(attempts, roles, strict=True):
            for claim in attempt["claims"]:
                if claim.get("kind", "fact") != "fact" or not claim.get("evidence"):
                    continue
                best = max(claim["evidence"], key=lambda e: e.get("judge_score") or 0.0)
                if not best.get("probabilities") or not best.get("secondary"):
                    continue
                points.append({
                    "id": cid,
                    "role": role,
                    "mdeberta_e": round(best["probabilities"]["p_entailment"], 4),
                    "mdeberta_c": round(best["probabilities"]["p_contradiction"], 4),
                    "bge_e": round(best["secondary"]["p_entailment"], 4),
                    "judge": round(best.get("judge_score") or 0.0, 4),
                    "status": claim["status"],
                    "reason": claim.get("reason"),
                    "co_verified": claim.get("co_verified", False),
                    "marked": cid in marked and claim["status"] != "verified",
                })
    medians = {
        ds: {k: round(statistics.median(v) / 1000, 2) for k, v in keys.items()}
        for ds, keys in stage.items()
    }
    return {
        "points": points,
        "stage_median_s": medians,
        "think_on_trim": {
            "cases": len(think_ids),
            "calls": len(trim_ms),
            "mean_s": round(statistics.mean(trim_ms) / 1000, 1) if trim_ms else None,
            "median_s": round(statistics.median(trim_ms) / 1000, 1) if trim_ms else None,
            "values_s": [round(x / 1000, 2) for x in trim_ms],
        },
    }


def g3_cost() -> dict[str, Any]:
    """Per-question time of the translated questions in 1839Z against the same questions in
    1310Z (real pipeline, the 151 of the G3 population)."""
    from research.opt3 import data

    base = data.responses("judge")
    rows = [
        json.loads(line)
        for ds in ("v3", "xquad-full")
        for line in (OPT4 / "live" / "20261103T1839Z_opt4_g3" / "e2e" / ds / "e2e_guard_on" / "cases.jsonl")
        .open(encoding="utf-8")
        if (OPT4 / "live" / "20261103T1839Z_opt4_g3" / "e2e" / ds / "e2e_guard_on" / "cases.jsonl").exists()
    ]
    def total(stats: dict[str, float]) -> float:
        return stats["total_ms"] - stats.get("model_init_ms", 0.0)

    deltas = []
    for row in rows:
        if not row.get("translation_pass"):
            continue
        deltas.append((total(row["execution_stats"]) - total(base[row["id"]]["execution_stats"])) / 1000)
    analysis = json.loads(
        (OPT4 / "live" / "20261103T1839Z_opt4_g3" / "analysis.json").read_text("utf-8")
    )["G3_e2e"]
    # The optimization_4 report called the median (4.1 s) a mean; both are kept here.
    return {
        "translated": len(deltas),
        "mean_extra_s": round(statistics.mean(deltas), 1) if deltas else None,
        "median_extra_s": round(statistics.median(deltas), 1) if deltas else None,
        "max_extra_s": round(max(deltas), 1) if deltas else None,
        "total_extra_min": round(sum(deltas) / 60, 1) if deltas else None,
        "values_s": [round(x, 2) for x in deltas],
        "minutes_e2e": analysis["minutes_e2e"],
        "minutes_baseline_same_questions": analysis["minutes_baseline_same_questions"],
    }


def final_run() -> dict[str, Any]:
    """The final configuration (run 1222Z) read straight from its logs: every fact claim of
    every attempt, stage medians over the questions that reached generation, the thinking
    pass after a trimmed draft, the translation pass, and the reranker score of the gold
    chunk for the 52 persistent refusals of 1310Z (XQuAD retrieval is the same in both)."""
    from research.opt3 import data

    run = RUNS["1222Z"]
    responses, rows = {}, {}
    for ds in ("v3", "xquad-full"):
        for line in (run / JUDGE / ds / "full" / "e2e_guard_on.jsonl").open(encoding="utf-8"):
            x = json.loads(line)
            responses[x["id"]] = x["response"]
        for line in (run / JUDGE / ds / "e2e_guard_on" / "cases.jsonl").open(encoding="utf-8"):
            x = json.loads(line)
            rows[x["id"]] = x
    marked = {"V3-MH-002", "V3-TC-021", "V3-TC-022", "V3-TC-025"}
    points, trim = [], []
    stage: dict[str, Any] = {}
    for cid, r in responses.items():
        dataset = data.dataset_of(cid)
        stats = r.get("execution_stats") or {}
        if stats.get("generation_ms", 0) > 0:
            for key in ("search_ms", "rerank_ms", "generation_ms", "nli_ms", "total_ms"):
                if key in stats:
                    stage.setdefault(dataset, {}).setdefault(key, []).append(stats[key])
        attempts = r.get("verification_attempts") or []
        roles = data.roles(attempts)
        first = attempts[0] if attempts else None
        if (
            first is not None and roles[0] == "draft" and not first["refused"]
            and any(c["status"] != "verified" and c.get("kind", "fact") == "fact"
                    for c in first["claims"])
            and "think" in roles
        ):
            trim.append(sum(a["generation_metadata"]["generation_ms"]
                            for a, role in zip(attempts, roles, strict=True)
                            if role in ("think", "think_repair")))
        for attempt, role in zip(attempts, roles, strict=True):
            for claim in attempt["claims"]:
                if claim.get("kind", "fact") != "fact" or not claim.get("evidence"):
                    continue
                best = max(claim["evidence"], key=lambda e: e.get("judge_score") or 0.0)
                if not best.get("probabilities") or not best.get("secondary"):
                    continue
                points.append({
                    "id": cid, "role": role,
                    "mdeberta_e": round(best["probabilities"]["p_entailment"], 4),
                    "mdeberta_c": round(best["probabilities"]["p_contradiction"], 4),
                    "bge_e": round(best["secondary"]["p_entailment"], 4),
                    "judge": round(best.get("judge_score") or 0.0, 4),
                    "status": claim["status"], "reason": claim.get("reason"),
                    "marked": cid in marked and claim["status"] != "verified",
                })
    refusals = {
        json.loads(line)["id"]
        for line in (OPT4 / "diagnostics" / "xquad_refusals.jsonl").open(encoding="utf-8")
    }
    gold: dict[str, list[float]] = {}
    for cid, case in data.cases().items():
        if data.dataset_of(cid) != "xquad-full" or not case.answerable:
            continue
        ids = set(case.gold_chunk_ids)
        scores = [d.get("rerank_score") or 0.0
                  for d in responses[cid].get("retrieval_diagnostics") or [] if d["chunk_id"] in ids]
        key = f"{case.language}_{'refused' if cid in refusals else 'other'}"
        gold.setdefault(key, []).append(max(scores, default=0.0))
    translated = [r for r in rows.values() if r.get("translation_pass")]
    return {
        "points": points,
        "stage_median_s": {
            ds: {k: round(statistics.median(v) / 1000, 2) for k, v in keys.items()}
            for ds, keys in stage.items()
        },
        "think_on_trim": {
            "calls": len(trim),
            "median_s": round(statistics.median(trim) / 1000, 1),
            "mean_s": round(statistics.mean(trim) / 1000, 1),
            "values_s": [round(x / 1000, 2) for x in trim],
        },
        "translation": {
            "translated": len(translated),
            "published": sum(not r["refused"] for r in translated),
            "translation_call_median_s": round(
                statistics.median(r["execution_stats"]["translation_ms"] for r in translated) / 1000, 2),
        },
        "gold_medians": {k: round(statistics.median(v), 3) for k, v in sorted(gold.items())},
    }


REVIEW = ROOT / "reports" / "review"
FALSE_KINDS = ("number_swap", "negation", "answer_swap", "transplant")


def review() -> dict[str, Any]:
    """Figure 8: acceptance of correct and distorted claims per verifier on the test split of
    the stress set built from the first drafts of 1222Z (research.review.compare)."""
    stress = json.loads((REVIEW / "stress.json").read_text("utf-8"))
    baselines = json.loads((REVIEW / "baselines.json").read_text("utf-8"))["rows"]
    result = {}
    for label, rates in stress.items():
        if not isinstance(rates, dict) or "positive/test" not in rates:
            continue
        false_k = sum(rates[f"{k}/test"]["k"] for k in FALSE_KINDS)
        false_n = sum(rates[f"{k}/test"]["n"] for k in FALSE_KINDS)
        result[label] = {
            "threshold": baselines[label]["threshold"],
            "true_accepted": rates["positive/test"]["rate"],
            "true_n": rates["positive/test"]["n"],
            "false_accepted": false_k / false_n,
            "false_n": false_n,
            "by_kind": {k: rates[f"{k}/test"]["rate"] for k in FALSE_KINDS},
        }
    return result


def main() -> None:
    result = {
        "runs": {k: comparison(v) for k, v in RUNS.items()},
        "gold_scores": gold_scores(),
        "translation_e3": translation_e3(),
        "taxonomy": taxonomy(),
        **claims_and_latency(),
        "g3_cost": g3_cost(),
        "final": final_run(),
        "review": review(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    short = {k: v for k, v in result.items() if k not in ("points", "final")}
    final = result["final"]
    short["final"] = {k: v for k, v in final.items() if k != "points"}
    short["final"]["think_on_trim"] = {k: v for k, v in final["think_on_trim"].items()
                                       if k != "values_s"}
    print("final claims:", len(final["points"]),
          Counter((p["status"], p["reason"]) for p in final["points"]))
    for k in ("gold_scores",):
        short[k] = {g: {"n": x["n"], "median": x["median"]} for g, x in result[k].items()}
    short["translation_e3"] = {k: v for k, v in result["translation_e3"].items() if k not in ("before", "after")}
    print(json.dumps(short, ensure_ascii=False, indent=1))
    print("claims:", len(result["points"]), Counter((p["status"], p["reason"]) for p in result["points"]))


if __name__ == "__main__":
    main()
