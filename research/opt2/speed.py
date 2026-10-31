"""PHASE 8: cost of the decision layer - NLI throughput, pairs per question, CPU query encoding.

Pairs come from the logged drafts: for 300 first drafts (fixed sample, seed 7) every fact claim
is paired with the texts the verifier scores for each cited source (chunk, sentence windows,
title-framed variants). Each model scores the same pairs in float16 on CUDA (no cache), with
the production batch sizes; peak VRAM is torch.cuda.max_memory_allocated. The BGE-M3 query
encoder is timed on the CPU (fp32) for 200 questions, as RAG_EMBEDDING_DEVICE=cpu runs it.

Usage:  python -m research.opt2.speed
"""

from __future__ import annotations

import json
import random
import statistics
import time
from typing import Any

from generator_verifier import _ANY_CITATION, SourceContext
from grounding import cited_claims

from research.opt2 import data
from research.opt2.nli_models import SPECS, HFClassifier, normalize_pair
from research.opt2.replay import settings, verifier


def sample_pairs() -> tuple[list[tuple[str, str]], list[int]]:
    records = data.records(("base",))
    drafts = [r.drafts["base"][0] for r in records.values() if r.drafts["base"]]
    random.Random(7).shuffle(drafts)
    s = settings()
    from research.opt2.nli_models import CachedNLI

    # Only its _variants are used: the same premise texts the verifier scores.
    helper = verifier(CachedNLI("mdeberta", load=False))
    pairs: list[tuple[str, str]] = []
    per_draft: list[int] = []
    for draft in drafts[:300]:
        count = 0
        for item in cited_claims(draft.raw_text, s.claim_segmentation, inherit=True):
            hypothesis = _ANY_CITATION.sub("", item.text).strip()
            for key in item.source_ids:
                if key not in draft.contexts:
                    continue
                premise, variants = helper._variants(
                    SourceContext.model_validate(draft.contexts[key])
                )
                for text in [premise, *(t for t, _ in variants)]:
                    pairs.append(normalize_pair(text, hypothesis))
                    count += 1
        per_draft.append(count)
    return pairs, per_draft


def throughput(name: str, pairs: list[tuple[str, str]]) -> dict[str, Any]:
    import torch

    torch.cuda.reset_peak_memory_stats()
    model = HFClassifier(SPECS[name])
    model.predict(pairs[:64])  # warm-up
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.predict(pairs)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    result = {
        "pairs": len(pairs),
        "seconds": seconds,
        "pairs_per_second": len(pairs) / seconds,
        "batch_size": SPECS[name].batch_size,
        "max_length": SPECS[name].max_length,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "attention": getattr(model.model.config, "_attn_implementation", None),
    }
    del model
    torch.cuda.empty_cache()
    return result


def cpu_encoding() -> dict[str, Any]:
    from hybrid_retriever import SentenceTransformerEncoder

    encoder = SentenceTransformerEncoder(settings(embedding_device="cpu"))
    questions = [c.question for c in list(data.cases().values())[:200]]
    encoder.encode(questions[:5])
    times = []
    for q in questions:
        started = time.perf_counter()
        encoder.encode([q])
        times.append(time.perf_counter() - started)
    return {
        "questions": len(times),
        "median_ms": 1000 * statistics.median(times),
        "p90_ms": 1000 * sorted(times)[int(0.9 * len(times))],
    }


def main() -> None:
    import torch

    pairs, per_draft = sample_pairs()
    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "pairs_per_draft": {
            "median": statistics.median(per_draft),
            "mean": statistics.mean(per_draft),
            "p90": sorted(per_draft)[int(0.9 * len(per_draft))],
        },
        "models": {name: throughput(name, pairs) for name in ("mdeberta", "bge-m3-zeroshot")},
        "bge_m3_query_on_cpu": cpu_encoding(),
    }
    for name, r in report["models"].items():
        r["ms_per_median_draft"] = (
            1000 * report["pairs_per_draft"]["median"] / r["pairs_per_second"]
        )
    out = data.OUT / "speed"
    out.mkdir(parents=True, exist_ok=True)
    (out / "speed_laptop.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
