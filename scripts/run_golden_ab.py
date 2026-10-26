# -*- coding: utf-8 -*-
"""Run evaluate_quality on golden_dataset for both A/B profiles (real metrics)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_quality import GoldenDataset, evaluate
from pipeline import RAGPipeline
from settings import RAGSettings

PROFILES = {
    "baseline": {"premise_guard": False, "nli_strong_entailment_enabled": False,
                 "max_tokens": 1024},
    "improved": {"premise_guard": True, "nli_strong_entailment_enabled": True,
                 "nli_strong_entailment_threshold": 0.7, "max_tokens": 512},
}

OUT = ROOT / "docs" / "evidence" / "golden_ab_report.json"


def main() -> int:
    dataset_path = ROOT / "data" / "golden_dataset.json"
    dataset = GoldenDataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    root = (dataset_path.parent / dataset.corpus_root).resolve(strict=True)
    summary = {}
    for name, overrides in PROFILES.items():
        index = ROOT / ".artifacts" / f"golden-ab-{name}"
        if index.exists():
            for item in index.iterdir():
                item.unlink()
            index.rmdir()
        values = {
            "document_root": root,
            "qdrant_path": index,
            "llm_api_base": "http://127.0.0.1:8080/v1",
            "llm_tokenizer_mode": "llama_server",
            "llm_api_timeout": 300.0,
            "embedding_batch_size": 2,
            "rerank_batch_size": 2,
            **dataset.parser_settings,
            **overrides,
        }
        settings = RAGSettings.model_validate(values)
        started = time.perf_counter()
        with RAGPipeline(settings) as pipeline:
            report = evaluate(pipeline, dataset, 5)
        report["profile"] = name
        report["wall_s"] = time.perf_counter() - started
        summary[name] = report
        print(f"[{name}] answered/non-refused positives:",
              report["summary"]["positive_answer"])
        print(f"[{name}] negative refusal:", report["summary"]["negative_refusal"])
        for metric, v in report["summary"]["metrics"].items():
            if v["mean"] is not None:
                print(f"  {metric}: {v['mean']:.4f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
