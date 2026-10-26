# -*- coding: utf-8 -*-
"""Real A/B evaluation of data/extended_dataset_v2.json against a live stack.

Baseline profile reproduces the pre-change pipeline:
  premise_guard=False, nli_strong_entailment_enabled=False, max_tokens=1024.
Improved profile activates U1 (premise guard), U2 (strong single-source
entailment), U3 (max_tokens=512). The runner indexes the corpus in two
isolated Qdrant directories, runs each profile once (single pass, no
threshold tuning between arms), and records paired per-case outcomes with
Wilson confidence intervals and an exact McNemar test.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import RAGPipeline
from settings import RAGSettings
DATASET = ROOT / "data" / "extended_dataset_v2.json"
OUT = ROOT / "extended_evaluation.json"
CORPUS = ROOT / "data" / "test_docs"

PROFILES: dict[str, dict[str, Any]] = {
    "baseline": {
        "premise_guard": False,
        "nli_strong_entailment_enabled": False,
        "max_tokens": 1024,
    },
    "improved": {
        "premise_guard": True,
        "nli_strong_entailment_enabled": True,
        "nli_strong_entailment_threshold": 0.7,
        "max_tokens": 512,
    },
}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


import re


def document_text(path: Path) -> str:
    """Extract plain text for unsupported extensions via explicit text ingestion."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"[ \t]+", " ", raw)


def documents() -> list[Path]:
    suffixes = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}
    return [p for p in sorted(CORPUS.iterdir()) if p.suffix.lower() in suffixes]


def run_profile(name: str, overrides: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    index_dir = ROOT / ".artifacts" / f"ab-{name}-index"
    if index_dir.exists():
        for item in index_dir.iterdir():
            item.unlink()
        index_dir.rmdir()
    values: dict[str, Any] = {
        "mode": "production",
        "document_root": CORPUS,
        "qdrant_path": index_dir,
        "llm_api_base": "http://127.0.0.1:8080/v1",
        "llm_tokenizer_mode": "llama_server",
        "llm_api_timeout": 300.0,
        "embedding_batch_size": 2,
        "rerank_batch_size": 2,
    }
    values.update(overrides)
    settings = RAGSettings.model_validate(values)
    results: list[dict[str, Any]] = []
    stats = {
        "positive_answered": 0,
        "negative_refused": 0,
        "adversarial_correct": 0,
        "errors": 0,
        "refusals_by_reason": {},
        "claims_total": 0,
        "claims_verified": 0,
    }
    with RAGPipeline(settings) as pipeline:
        for path in documents():
            if path.suffix.lower() in {".txt", ".md", ".docx"}:
                pipeline.ingest_path(path.name)
            else:
                text = document_text(path)
                if text.strip():
                    pipeline.ingest_text(text, source=path.name)
        indexed_chunks = sum(len(d.chunks) for d in pipeline.documents.values())
        for i, row in enumerate(rows, 1):
            print(f"[{name}] {row['test_id']} ({i}/{len(rows)})", flush=True)
            started = time.perf_counter()
            record: dict[str, Any] = {k: row[k] for k in
                                      ("test_id", "question_type", "question",
                                       "expected_behavior")}
            try:
                response = pipeline.query(row["question"])
                record["answer"] = response.answer
                record["refused"] = response.refused
                record["refusal_reason"] = response.refusal_reason
                record["faithfulness"] = response.faithfulness_score
                record["sources"] = [
                    {"chunk_id": s.chunk_id, "rerank_score": s.rerank_score}
                    for s in response.sources
                ]
                record["claims"] = [
                    {"claim": c.claim, "status": c.status.value, "confidence": c.confidence}
                    for c in response.claims
                ]
                stats["claims_total"] += len(response.claims)
                stats["claims_verified"] += sum(
                    c.status.value == "verified" for c in response.claims
                )
                qtype = row["question_type"]
                if qtype == "Positive":
                    ok = not response.refused and bool(response.answer.strip())
                elif qtype == "Negative":
                    ok = response.refused
                else:  # Adversarial: refusal OR a correction (any non-refused answer
                    # counts via the NLI gate: verified claims only survive it)
                    ok = response.refused or bool(response.claims) and all(
                        c.status.value == "verified" for c in response.claims
                    )
                record["correct"] = ok
                if response.refused:
                    reason = response.refusal_reason or "unknown"
                    stats["refusals_by_reason"][reason] = (
                        stats["refusals_by_reason"].get(reason, 0) + 1
                    )
                if qtype == "Positive" and not response.refused:
                    stats["positive_answered"] += 1
                if qtype == "Negative" and response.refused:
                    stats["negative_refused"] += 1
                if qtype == "Adversarial" and ok:
                    stats["adversarial_correct"] += 1
            except Exception as exc:  # technical error: scored as failure
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["refused"] = None
                record["correct"] = False
                stats["errors"] += 1
            record["wall_ms"] = (time.perf_counter() - started) * 1000
            results.append(record)
    return {"stats": stats, "results": results, "indexed_chunks": indexed_chunks}


def main() -> int:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    rows = data["cases"]
    report: dict[str, Any] = {
        "kind": "real_ab_evaluation",
        "dataset": "data/extended_dataset_v2.json",
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "profiles": PROFILES,
        "llama_server_flags": "-ngl 35 -c 4096 -b 128 -ub 64 -np 1",
    }
    for name, overrides in PROFILES.items():
        print(f"=== PROFILE {name} ===", flush=True)
        report[name] = run_profile(name, overrides, rows)
    # aggregate metrics
    summary: dict[str, Any] = {}
    for name in PROFILES:
        arm = report[name]
        st = arm["stats"]
        n_pos = sum(1 for r in rows if r["question_type"] == "Positive")
        n_neg = sum(1 for r in rows if r["question_type"] == "Negative")
        n_adv = sum(1 for r in rows if r["question_type"] == "Adversarial")
        correct = sum(1 for r in arm["results"] if r["correct"])
        summary[name] = {
            "positive_answer_rate": [st["positive_answered"], n_pos],
            "negative_refusal_rate": [st["negative_refused"], n_neg],
            "adversarial_correct_rate": [st["adversarial_correct"], n_adv],
            "overall_correct": [correct, len(rows)],
            "faithfulness_micro": [
                st["claims_verified"], st["claims_total"]
            ] if st["claims_total"] else [0, 0],
            "errors": st["errors"],
            "median_wall_ms": sorted(r["wall_ms"] for r in arm["results"])[
                len(arm["results"]) // 2
            ],
            "refusals_by_reason": st["refusals_by_reason"],
        }
    # paired McNemar over per-case correctness
    base = [r["correct"] for r in report["baseline"]["results"]]
    imp = [r["correct"] for r in report["improved"]["results"]]
    b_only = sum(1 for x, y in zip(base, imp) if x and not y)
    c_only = sum(1 for x, y in zip(base, imp) if y and not x)
    summary["paired_mcnemar"] = {
        "baseline_only_correct": b_only,
        "improved_only_correct": c_only,
        "exact_p": mcnemar_exact(b_only, c_only),
    }
    summary["wilson_95"] = {
        name: {
            metric: wilson(value[0], value[1])
            for metric, value in (
                ("positive_answer_rate", summary[name]["positive_answer_rate"]),
                ("negative_refusal_rate", summary[name]["negative_refusal_rate"]),
                ("adversarial_correct_rate", summary[name]["adversarial_correct_rate"]),
                ("overall_correct", summary[name]["overall_correct"]),
            )
        }
        for name in PROFILES
    }
    report["summary"] = summary
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
