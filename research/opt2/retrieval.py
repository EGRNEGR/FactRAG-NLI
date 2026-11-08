"""PHASE 6: adaptive retrieval, replayed on the 40 logged candidates of every question.

The qwen38-27b runs with bge-reranker-v2-m3 and with bge-reranker-large retrieved the same
hybrid (dense + BM25, RRF) candidate pool, and both logged every candidate's RRF and reranker
scores (retrieval_diagnostics). Joining them per chunk replays reranker fusion, reranker
routing by a pre-answer feature (the question's script), the relevance gate and an adaptive
context size without re-running any model. Gold = the case's evidence units.

Metrics (answerable questions with gold chunks): Recall@5, MRR@10, Hit@K of the context
actually sent (K = 10, or adaptive), mean context size, gold share of the context, and the
gate's rejection rate on answerable and unanswerable questions.

Usage:  python -m research.opt2.retrieval
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from statistics import mean
from typing import Any

from evaluate_suite import evidence_metrics

from research.opt2 import data
from research.opt2.replay import settings

_CYRILLIC = re.compile(r"[а-яё]", re.I)


def pools() -> list[dict[str, Any]]:
    base, large = data.responses("base"), data.responses("large")
    result = []
    for case_id, case in data.cases().items():
        a = {r["chunk_id"]: r for r in base[case_id].get("retrieval_diagnostics") or []}
        b = {r["chunk_id"]: r for r in large[case_id].get("retrieval_diagnostics") or []}
        rows = [
            {
                "chunk_id": cid,
                "rrf": float(a[cid]["rrf_score"]),
                "v2": float(a[cid]["rerank_score"]),
                "large": float(b[cid]["rerank_score"]),
            }
            for cid in a
            if cid in b and not a[cid].get("heading_only")
        ]
        units = [list(e.chunk_ids) for e in case.evidence] if case.evidence else []
        result.append(
            {
                "id": case_id,
                "dataset": data.dataset_of(case_id),
                "split": data.split()[case_id],
                "answerable": case.expected_behavior != "refusal",
                "units": units,
                "question": case.question,
                "rows": rows,
                "overlap": len(rows) / max(1, len(a)),
            }
        )
    return result


def mixer(scorer: Callable[[dict[str, Any]], float]) -> Callable[[dict[str, Any]], float]:
    s = settings()
    scale = (s.dense_weight + s.sparse_weight) / (s.rrf_k + 1)
    alpha = s.rerank_rrf_alpha
    return lambda r: alpha * r["rrf"] / scale + (1 - alpha) * scorer(r)


def rank_fusion(pool: dict[str, Any]) -> dict[str, float]:
    rows = pool["rows"]
    fused = {r["chunk_id"]: 0.0 for r in rows}
    for key in ("v2", "large"):
        for rank, r in enumerate(sorted(rows, key=lambda x: -x[key]), 1):
            fused[r["chunk_id"]] += 1 / (60 + rank)
    return fused


Strategy = Callable[[dict[str, Any]], list[tuple[str, float]]]
STRATEGIES: dict[str, Strategy] = {}


def strategy(name: str) -> Callable[[Strategy], Strategy]:
    def wrap(fn: Strategy) -> Strategy:
        STRATEGIES[name] = fn
        return fn

    return wrap


def ordered(
    pool: dict[str, Any], score: Callable[[dict[str, Any]], float], gate_key: str
) -> list[tuple[str, float]]:
    """Candidates by mixing score, each with the reranker score the gate reads."""
    rows = sorted(pool["rows"], key=lambda r: (-score(r), r["chunk_id"]))
    return [(r["chunk_id"], r[gate_key] if gate_key in r else score(r)) for r in rows]


@strategy("S0 bge-reranker-v2-m3 (production)")
def s0(pool: dict[str, Any]) -> list[tuple[str, float]]:
    return ordered(pool, mixer(lambda r: r["v2"]), "v2")


@strategy("S1 bge-reranker-large")
def s1(pool: dict[str, Any]) -> list[tuple[str, float]]:
    return ordered(pool, mixer(lambda r: r["large"]), "large")


@strategy("S2 mean of both rerankers")
def s2(pool: dict[str, Any]) -> list[tuple[str, float]]:
    for r in pool["rows"]:
        r["mean"] = (r["v2"] + r["large"]) / 2
    return ordered(pool, mixer(lambda r: r["mean"]), "mean")


@strategy("S3 max of both rerankers")
def s3(pool: dict[str, Any]) -> list[tuple[str, float]]:
    for r in pool["rows"]:
        r["max"] = max(r["v2"], r["large"])
    return ordered(pool, mixer(lambda r: r["max"]), "max")


@strategy("S4 reciprocal-rank fusion of both rerankers")
def s4(pool: dict[str, Any]) -> list[tuple[str, float]]:
    fused = rank_fusion(pool)
    rows = sorted(pool["rows"], key=lambda r: (-fused[r["chunk_id"]], r["chunk_id"]))
    return [(r["chunk_id"], r["v2"]) for r in rows]


@strategy("S5 route by question script: Latin -> mean, Cyrillic -> v2-m3")
def s5(pool: dict[str, Any]) -> list[tuple[str, float]]:
    return s0(pool) if _CYRILLIC.search(pool["question"]) else s2(pool)


@strategy("S6 hybrid RRF only (no reranker)")
def s6(pool: dict[str, Any]) -> list[tuple[str, float]]:
    rows = sorted(pool["rows"], key=lambda r: (-r["rrf"], r["chunk_id"]))
    return [(r["chunk_id"], r["v2"]) for r in rows]


def evaluate(name: str, gate: float, top_k: int = 10, rho: float | None = None) -> dict[str, Any]:
    fn = STRATEGIES[name]
    per: dict[str, list[dict[str, Any]]] = {}
    for pool in POOLS:
        ranked = fn(pool)
        if not ranked:
            continue
        top = ranked[0][1]
        gated = top < gate
        context = ranked[:top_k]
        if rho is not None:
            context = [c for c in context if c[1] >= max(gate, rho * top)] or context[:1]
        ids = [c[0] for c in context]
        row: dict[str, Any] = {"gated": gated, "answerable": pool["answerable"], "k": len(ids)}
        if pool["answerable"] and pool["units"]:
            m = evidence_metrics([c[0] for c in ranked], pool["units"])
            gold = {cid for unit in pool["units"] for cid in unit}
            row.update(
                recall5=m["recall@5"],
                mrr10=m["mrr@10"],
                context_hit=float(not gated and bool(gold & set(ids))),
                gold_share=sum(cid in gold for cid in ids) / len(ids),
            )
        for key in ("all", pool["dataset"], pool["split"]):
            per.setdefault(key, []).append(row)
    out = {}
    for key, rows in per.items():
        ans = [r for r in rows if "recall5" in r]
        una = [r for r in rows if not r["answerable"]]
        out[key] = {
            "recall@5": mean(r["recall5"] for r in ans),
            "mrr@10": mean(r["mrr10"] for r in ans),
            "context_hit": mean(r["context_hit"] for r in ans),
            "gold_share": mean(r["gold_share"] for r in ans),
            "mean_k": mean(r["k"] for r in rows),
            "gate_rejects_answerable": mean(r["gated"] for r in rows if r["answerable"]),
            "gate_rejects_unanswerable": mean(r["gated"] for r in una) if una else None,
        }
    return out


POOLS: list[dict[str, Any]] = []


def main() -> None:
    POOLS.extend(pools())
    print(
        "candidate pool overlap between the two runs:", round(mean(p["overlap"] for p in POOLS), 4)
    )
    report: dict[str, Any] = {"strategies": {}, "adaptive_k": {}}
    for name in STRATEGIES:
        res = evaluate(name, gate=0.005)
        report["strategies"][name] = res
        for ds in ("v3", "xquad-full"):
            r = res[ds]
            print(
                f"{name[:44]:44s} {ds:10s} R@5 {r['recall@5']:.3f} MRR {r['mrr@10']:.3f} "
                f"ctx-hit {r['context_hit']:.3f} gate ans {r['gate_rejects_answerable']:.3f} "
                f"una {r['gate_rejects_unanswerable'] if r['gate_rejects_unanswerable'] is None else round(r['gate_rejects_unanswerable'], 3)}"
            )
    base = "S0 bge-reranker-v2-m3 (production)"
    for rho in (None, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5):
        res = evaluate(base, gate=0.005, rho=rho)
        report["adaptive_k"][str(rho)] = res
        r = res["all"]
        print(
            f"adaptive k rho={rho}: mean k {r['mean_k']:.2f} ctx-hit {r['context_hit']:.4f} "
            f"gold share {r['gold_share']:.3f}"
        )
    for gate in (0.001, 0.005, 0.01, 0.02, 0.05):
        res = evaluate(base, gate=gate)
        report.setdefault("gate", {})[str(gate)] = res
        for ds in ("v3", "xquad-full"):
            r = res[ds]
            print(
                f"gate {gate} {ds}: rejects answerable {r['gate_rejects_answerable']:.3f} "
                f"unanswerable {r['gate_rejects_unanswerable']:.3f}"
            )
    out = data.OUT / "retrieval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_replay.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
