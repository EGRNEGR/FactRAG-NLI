"""Reproducible local GPU latency/stress diagnostics, not a quality benchmark."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from generator_verifier import (
    GenerationMetadata,
    RawGenerationResult,
    SourceContext,
    _HTTPBackend,
)
from pipeline import RAGPipeline
from settings import RAGSettings


def percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {f"p{p}_ms": ordered[math.ceil(len(ordered) * p / 100) - 1] for p in (50, 95)}


class VRAMMonitor:
    """Sample whole-device free memory; PyTorch peaks exclude llama-server."""

    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.stop = threading.Event()
        free, self.total = torch.cuda.mem_get_info()
        self.min_free = free
        self.error: str | None = None
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        try:
            while not self.stop.wait(0.05):
                free, _ = self.torch.cuda.mem_get_info()
                self.min_free = min(self.min_free, free)
        except Exception as exc:
            self.error = str(exc)

    def __enter__(self) -> VRAMMonitor:
        self.torch.cuda.reset_peak_memory_stats()
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop.set()
        self.thread.join()

    def report(self) -> dict[str, Any]:
        return {
            "total_bytes": self.total,
            "sampled_min_free_bytes": self.min_free,
            "sampled_peak_device_used_bytes": self.total - self.min_free,
            "pytorch_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
            "pytorch_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(),
            "sample_interval_ms": 50,
            "sampling_error": self.error,
        }


def query_record(pipeline: RAGPipeline, question: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = pipeline.query(question)
        return {
            "question": question,
            "ok": True,
            "response": response.model_dump(mode="json"),
            "wall_ms": (time.perf_counter() - started) * 1000,
        }
    except Exception as exc:
        return {
            "question": question,
            "ok": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "wall_ms": (time.perf_counter() - started) * 1000,
        }


def edge_cases(pipeline: RAGPipeline) -> list[dict[str, Any]]:
    """Controlled generation inputs isolate real NLI handling of edge conditions."""
    verifier = pipeline._fact_verifier
    if verifier is None:
        raise RuntimeError("Warm-up must initialize the real NLI classifier")
    contexts = {
        "S1": SourceContext(
            document_id="a",
            chunk_id="a",
            text="Давление равно 10 МПа.",
            citation="Синтетический источник A",
        ),
        "S2": SourceContext(
            document_id="b",
            chunk_id="b",
            text="Давление равно 99 МПа, а не 10 МПа.",
            citation="Синтетический источник B",
        ),
    }
    records = []
    for name, text in (
        ("empty_answer", ""),
        ("contradictory_cited_sources", "Давление равно 10 МПа [S1] [S2]."),
    ):
        started = time.perf_counter()
        result = verifier.verify(
            RawGenerationResult(
                text=text,
                contexts=contexts,
                metadata=GenerationMetadata(backend="controlled_edge_input", finish_reason="stop"),
            )
        )
        records.append(
            {
                "case": name,
                "expected_refusal": True,
                "passed": result.refused,
                "wall_ms": (time.perf_counter() - started) * 1000,
                "result": result.model_dump(mode="json"),
            }
        )
    record = query_record(pipeline, "Как приготовить шоколадный торт?")
    record["case"] = "off_topic"
    record["passed"] = bool(record["ok"] and record["response"]["refused"])
    records.append(record)
    return records


def stress_llm(backend: _HTTPBackend, target: int, repeats: int) -> list[dict[str, Any]]:
    """Use llama.cpp's local tokenizer; measure actual chat token usage, never estimate it."""
    system = "Ответь кратко по документу. Укажи допустимое давление."
    unit = "Допустимое давление составляет 10 МПа. Контроль выполняют перед запуском.\n"
    low, high = 1, target
    while low < high:
        middle = (low + high + 1) // 2
        response = backend.client.post("../tokenize", json={"content": system + unit * middle})
        response.raise_for_status()
        count = len(response.json()["tokens"])
        if count <= target:
            low = middle
        else:
            high = middle - 1
    results = []
    for i in range(repeats):
        started = time.perf_counter()
        reply = backend.complete(system, unit * low)
        if reply.prompt_tokens is None or reply.completion_tokens is None:
            raise RuntimeError("Server omitted usage: cannot validate 4k stress context")
        total = reply.prompt_tokens + reply.completion_tokens
        results.append(
            {
                "iteration": i,
                "wall_ms": (time.perf_counter() - started) * 1000,
                "reply": reply.model_dump(mode="json"),
                "total_tokens": total,
                "passed": target - 100 <= reply.prompt_tokens and total <= 4096,
            }
        )
    return results


def run(settings: RAGSettings, workdir: Path, repeats: int, documents: int) -> dict[str, Any]:
    import torch

    if repeats < 1 or documents < 1:
        raise ValueError("repeats and documents must be positive")
    if settings.allow_fallback or not torch.cuda.is_available():
        raise RuntimeError("Real benchmark requires CUDA and allow_fallback=False")
    # Dedicated new directory prevents accidental mutation of a user's corpus/index.
    workdir.mkdir(parents=True, exist_ok=False)
    root = workdir / "documents"
    root.mkdir()
    config = RAGSettings.model_validate(
        {
            **settings.model_dump(),
            "model_device": "cuda",
            "llm_api_stream": True,
            "document_root": root,
            "qdrant_path": workdir / "qdrant",
            "llm_context_window": 4096,
            "max_tokens": 128,
            "embedding_batch_size": 2,
            "rerank_batch_size": 2,
        }
    )
    report: dict[str, Any] = {
        "kind": "synthetic_gpu_stress_not_quality_evaluation",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "batch_size": 2,
        "repeats": repeats,
        "ok": False,
    }
    with VRAMMonitor() as monitor:
        try:
            with RAGPipeline(config) as pipeline:
                short = root / "control.md"
                short.write_text(
                    "# Требования\nДопустимое давление составляет 10 МПа.", encoding="utf-8"
                )
                report["initial_index"] = pipeline.index_documents([short]).model_dump()
                question = "Допустимое давление составляет 10 МПа?"
                report["cold_query"] = query_record(pipeline, question)
                if pipeline._fact_verifier is None or pipeline._generator is None:
                    raise RuntimeError("Warm-up did not reach generation/NLI")
                models = [
                    pipeline.retriever.encoder._model,  # type: ignore[attr-defined]
                    pipeline.retriever.reranker._model,  # type: ignore[attr-defined]
                    pipeline._fact_verifier.backend.model,  # type: ignore[attr-defined]
                ]
                report["models"] = [
                    {
                        "device": str(next(m.parameters()).device),
                        "dtype": str(next(m.parameters()).dtype),
                    }
                    for m in models
                ]
                report["edges"] = edge_cases(pipeline)
                paths = []
                for i in range(documents):
                    path = root / f"long-{i}.md"
                    path.write_text(
                        f"# Документ {i}\n"
                        + (
                            "Допустимое давление составляет 10 МПа. Контроль выполняют перед запуском.\n"
                            * 300
                        ),
                        encoding="utf-8",
                    )
                    paths.append(path)
                report["long_document_index"] = pipeline.index_documents(paths).model_dump()
                report["warm_queries"] = [query_record(pipeline, question) for _ in range(repeats)]
                backend = pipeline._generator._backend
                if not isinstance(backend, _HTTPBackend):
                    raise RuntimeError("Stress requires local llama-server HTTP backend")
                report["context_stress"] = stress_llm(backend, 3800, repeats)
                report["warm_latency"] = percentiles([r["wall_ms"] for r in report["warm_queries"]])
                report["ok"] = (
                    report["cold_query"]["ok"]
                    and all(r["ok"] for r in report["warm_queries"])
                    and all(r["passed"] for r in report["edges"] + report["context_stress"])
                    and all(m["device"].startswith("cuda") for m in report["models"])
                )
        except Exception as exc:
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["vram"] = monitor.report()
    if monitor.error:
        report["ok"] = False
    (workdir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".artifacts") / f"gpu-benchmark-{time.time_ns()}",
        help="New isolated run directory",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--documents", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        report = run(RAGSettings(), args.output.resolve(), args.repeats, args.documents)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
