"""FactRAG-NLI evaluation suite for dataset v3: retrieval ablation, NLI probes, end-to-end A/B.

Modes (run sequentially; the embedded Qdrant index allows one owner process):
  retrieval  dense-only, BM25-only, hybrid RRF and hybrid+rerank rankings of the same query
             (no LLM): Recall@K, Hit@K, nDCG@K over gold evidence units, MRR@10.
  nli        the production claim rule (NLIFactVerifier._claim) on gold premise/hypothesis
             probes in EN and RU: 3-class confusion matrix, per-class P/R/F1, catch rate of
             contradiction/neutral probes, false rejection of entailed probes, threshold sweep.
  e2e        the full pipeline per question. Vanilla RAG = LLM draft before NLI filtering
             (RAGResponse.raw_answer); FactRAG-NLI = published answer. Both come from the same
             LLM call, so the comparison is paired. Profiles: guard_on (production), guard_off.

Every run writes reports/runs/<run_id>/: config.json, <mode>/cases.jsonl, <mode>/summary.json,
<mode>/resources.json; heavy traces go to reports/runs/<run_id>/full/ (git-ignored).

Usage:
  python evaluate_suite.py --run-dir reports/runs/<run_id> retrieval
  python evaluate_suite.py --run-dir reports/runs/<run_id> nli
  python evaluate_suite.py --run-dir reports/runs/<run_id> e2e --profile guard_on
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from eval_dataset import EvalCase, EvalDatasetV3, contains_fact, load_dataset

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "eval_dataset_v3.json"
INDEX = ROOT / ".artifacts" / "eval-v3-index"
KS = (1, 3, 5, 10)
LABELS = ("entailment", "neutral", "contradiction")
EXPECTED_STATUS = {
    "entailment": "verified",
    "neutral": "unverified",
    "contradiction": "contradiction",
}
# Hardware invariants of the 8 GB stack (docs/evidence/improvement_report.md): with llama-server
# holding ~6.6 GB, larger embedding/rerank batches overflow VRAM into shared memory on Windows
# and slow generation 2-3x (observed with the code defaults 16/8).
RUNTIME: dict[str, Any] = {"embedding_batch_size": 2, "rerank_batch_size": 2}
PROFILES: dict[str, dict[str, Any]] = {
    "guard_on": {},
    "guard_off": {"premise_guard": False},
}
NO_INFO = re.compile(
    r"(не\s+(указ|определ|опис|упомина|содерж|привод|рассматрива|регламентир|зада[её]т|"
    r"устанавлива|называ|перечисля|предусматрива)|нет\s+(сведени|информаци|данных|упоминани)|"
    r"отсутству|not\s+(specified|defined|mentioned|described|covered)|"
    r"does\s+not\s+(define|specify|mention|describe|cover))",
    re.IGNORECASE,
)
CORE_FILES = (
    "pipeline.py",
    "generator_verifier.py",
    "hybrid_retriever.py",
    "document_parser.py",
    "settings.py",
    "eval_dataset.py",
    "evaluate_suite.py",
)


# --------------------------------------------------------------------------- metrics (pure)


def evidence_metrics(
    ranked: Sequence[str], evidence: Sequence[Sequence[str]], ks: Sequence[int] = KS
) -> dict[str, float]:
    """Unit-level metrics: an evidence unit is hit by any of its chunks (overlap duplicates).

    Recall@K is the share of evidence units hit in the top K; nDCG@K assigns each ranked
    position to at most one still-unsatisfied unit (binary gains), so it never exceeds 1.
    """
    if not evidence:
        raise ValueError("evidence units required")
    order = list(dict.fromkeys(ranked))
    units = [set(unit) for unit in evidence]
    first = [next((i for i, cid in enumerate(order, 1) if cid in unit), None) for unit in units]
    assigned: list[int] = []
    open_units = list(range(len(units)))
    for position, cid in enumerate(order, 1):
        match = next((u for u in open_units if cid in units[u]), None)
        if match is not None:
            assigned.append(position)
            open_units.remove(match)
    out: dict[str, float] = {}
    for k in ks:
        hits = [rank is not None and rank <= k for rank in first]
        out[f"recall@{k}"] = sum(hits) / len(units)
        out[f"hit@{k}"] = float(any(hits))
        dcg = sum(1 / math.log2(p + 1) for p in assigned if p <= k)
        ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(units)) + 1))
        out[f"ndcg@{k}"] = dcg / ideal
    reached = [rank for rank in first if rank is not None and rank <= 10]
    out["mrr@10"] = 1 / min(reached) if reached else 0.0
    return out


def rerank_order(diagnostics: Sequence[dict[str, Any]], settings: Any) -> list[str]:
    """Order of all reranked candidates by the production mixing score, before the threshold."""
    scale = (settings.dense_weight + settings.sparse_weight) / (settings.rrf_k + 1)
    alpha = settings.rerank_rrf_alpha

    def score(row: dict[str, Any]) -> float:
        return float(
            alpha * float(row["rrf_score"]) / scale + (1 - alpha) * float(row["rerank_score"])
        )

    rows = [row for row in diagnostics if not row.get("heading_only")]
    return [row["chunk_id"] for row in sorted(rows, key=lambda r: (-score(r), r["chunk_id"]))]


def classification_report(gold: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted lengths differ")
    matrix = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, predicted, strict=True):
        matrix[g][p] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = matrix[label][label]
        predicted_n = sum(matrix[g][label] for g in LABELS)
        support = sum(matrix[label].values())
        precision = tp / predicted_n if predicted_n else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    n = len(gold)
    present = [label for label in LABELS if per_class[label]["support"]]
    return {
        "n": n,
        "accuracy": sum(matrix[lab][lab] for lab in LABELS) / n if n else 0.0,
        "macro_f1": sum(float(per_class[lab]["f1"]) for lab in present) / len(present)
        if present
        else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def decision_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """rows: {'label': gold label, 'status': verified|unverified|contradiction}."""

    def share(selected: Sequence[dict[str, Any]], predicate: Any) -> float | None:
        return sum(bool(predicate(r)) for r in selected) / len(selected) if selected else None

    bad = [r for r in rows if r["label"] != "entailment"]
    con = [r for r in rows if r["label"] == "contradiction"]
    neu = [r for r in rows if r["label"] == "neutral"]
    ent = [r for r in rows if r["label"] == "entailment"]
    status_matrix = {g: Counter(r["status"] for r in rows if r["label"] == g) for g in LABELS}
    return {
        "n": len(rows),
        "hallucination_catch_rate": share(bad, lambda r: r["status"] != "verified"),
        "contradiction_catch_rate": share(con, lambda r: r["status"] != "verified"),
        "neutral_catch_rate": share(neu, lambda r: r["status"] != "verified"),
        "contradiction_labelled_as_contradiction": share(
            con, lambda r: r["status"] == "contradiction"
        ),
        "false_rejection_rate": share(ent, lambda r: r["status"] != "verified"),
        "status_agreement": share(rows, lambda r: r["status"] == EXPECTED_STATUS[r["label"]]),
        "status_matrix": {g: dict(c) for g, c in status_matrix.items()},
    }


def single_source_status(
    p_ent: float, p_con: float, *, entail: float, contra: float, strong: float | None
) -> str:
    """Production rule for one cited source (consensus or strong entailment)."""
    if p_con >= contra:
        return "contradiction"
    if p_ent > entail or (strong is not None and p_ent >= strong):
        return "verified"
    return "unverified"


def threshold_sweep(
    rows: Sequence[dict[str, Any]],
    entail_grid: Iterable[float],
    contra_grid: Iterable[float],
    strong: float | None,
) -> list[dict[str, Any]]:
    scored = [r for r in rows if r.get("p_entailment") is not None]
    out = []
    for e in entail_grid:
        for c in contra_grid:
            decided = [
                {
                    "label": r["label"],
                    "status": single_source_status(
                        r["p_entailment"], r["p_contradiction"], entail=e, contra=c, strong=strong
                    ),
                }
                for r in scored
            ]
            m = decision_metrics(decided)
            out.append(
                {
                    "entailment_threshold": e,
                    "contradiction_threshold": c,
                    "hallucination_catch_rate": m["hallucination_catch_rate"],
                    "false_rejection_rate": m["false_rejection_rate"],
                }
            )
    return out


def latency_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    data = sorted(float(v) for v in values)
    if not data:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}

    def quantile(q: float) -> float:
        pos = (len(data) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return data[lo] + (data[hi] - data[lo]) * (pos - lo)

    return {
        "n": len(data),
        "mean": sum(data) / len(data),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "min": data[0],
        "max": data[-1],
    }


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b, c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def is_canonical_refusal(text: str | None) -> bool:
    from generator_verifier import REFUSAL

    if text is None or not text.strip():
        return True
    head = " ".join(REFUSAL.split()[:6]).casefold()
    return head in " ".join(text.split()).casefold()


def score_answer(case: EvalCase, text: str | None) -> dict[str, Any]:
    """Case-level outcome of a published text (FactRAG answer or Vanilla draft).

    refused: canonical refusal (or nothing published). abstained: refused or an explicit
    no-information statement. Hallucination is defined where the dataset fixes the truth:
    any non-abstaining answer to an unanswerable question, and an adversarial answer that
    neither abstains nor states the correct fact.
    """
    refused = is_canonical_refusal(text)
    abstained = refused or bool(NO_INFO.search(text or ""))
    facts = (
        bool(case.answer_must_contain)
        and not refused
        and all(contains_fact(text or "", fact) for fact in case.answer_must_contain)
    )
    if case.expected_behavior == "answer":
        correct, hallucination = facts, None
    elif case.expected_behavior == "refusal":
        correct, hallucination = abstained, not abstained
    else:
        correct = abstained or facts
        hallucination = not correct
    return {
        "refused": refused,
        "abstained": abstained,
        "facts_ok": facts,
        "correct": correct,
        "hallucination": hallucination,
    }


def rate(items: Sequence[bool]) -> dict[str, Any]:
    k, n = sum(bool(x) for x in items), len(items)
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": wilson(k, n)}


def summarize_variant(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    """Aggregate scores of one variant ('vanilla' or 'factrag') over e2e case rows."""
    ok = [r for r in rows if r.get("ok")]
    answer = [r for r in ok if r["expected_behavior"] == "answer"]
    defined = [r for r in ok if r[key]["hallucination"] is not None]
    by_category = {}
    for category in sorted({r["category"] for r in ok}):
        sub = [r for r in ok if r["category"] == category]
        by_category[category] = {
            "correct": rate([r[key]["correct"] for r in sub]),
            "hallucination": rate(
                [r[key]["hallucination"] for r in sub if r[key]["hallucination"] is not None]
            ),
            "refused": rate([r[key]["refused"] for r in sub]),
        }
    by_language = {
        lang: rate([r[key]["correct"] for r in ok if r["language"] == lang])
        for lang in sorted({r["language"] for r in ok})
    }
    return {
        "correct": rate([r[key]["correct"] for r in ok]),
        "hallucination": rate([r[key]["hallucination"] for r in defined]),
        "false_refusal_on_answerable": rate([r[key]["refused"] for r in answer]),
        "key_fact_accuracy_on_answerable": rate([r[key]["facts_ok"] for r in answer]),
        "by_category": by_category,
        "by_language": by_language,
    }


# --------------------------------------------------------------------------- environment


def sha256_file(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = limit
        while True:
            block = stream.read(1 << 20 if remaining is None else min(1 << 20, remaining))
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
                if remaining <= 0:
                    break
    return digest.hexdigest()


def model_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "missing": True}
    if path.is_file():
        return {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256_first_64MiB": sha256_file(path, 64 << 20),
        }
    files = sorted(p for p in path.rglob("*") if p.is_file())
    listing = "\n".join(f"{p.relative_to(path).as_posix()}:{p.stat().st_size}" for p in files)
    config = path / "config.json"
    return {
        "path": path.as_posix(),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "listing_sha256": hashlib.sha256(listing.encode()).hexdigest(),
        "config_sha256": sha256_file(config) if config.exists() else None,
    }


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty_files": run("status", "--porcelain").splitlines(),
    }


def llama_props(api_base: str | None) -> dict[str, Any] | None:
    if not api_base:
        return None
    import httpx

    root = api_base.rstrip("/").removesuffix("/v1")
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            props = client.get(f"{root}/props").json()
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        return {"error": f"{type(exc).__name__}: {exc}"}
    settings = props.get("default_generation_settings", {})
    cmdline = None
    try:
        import psutil

        for proc in psutil.process_iter(["name", "cmdline"]):
            if "llama-server" in (proc.info.get("name") or "").lower():
                cmdline = [
                    Path(part).name if i < 3 else part
                    for i, part in enumerate(proc.info["cmdline"] or [])
                ]
                break
    except Exception:  # noqa: BLE001 - optional provenance only
        cmdline = None
    return {
        "cmdline": cmdline,
        "n_ctx": settings.get("n_ctx"),
        "model_path": props.get("model_path"),
        "total_slots": props.get("total_slots"),
        "build_info": props.get("build_info"),
    }


def gpu_name() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


class ResourceMonitor:
    """Samples whole-card GPU memory (nvidia-smi) and process RSS once per second."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _gpu_used(self) -> float | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            return float(out.splitlines()[0])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def _run(self) -> None:
        import psutil

        me = psutil.Process()
        while not self._stop.is_set():
            llama = 0.0
            for proc in psutil.process_iter(["name", "memory_info"]):
                name = (proc.info.get("name") or "").lower()
                if "llama-server" in name and proc.info.get("memory_info"):
                    llama += proc.info["memory_info"].rss / 2**20
            sample = {
                "t": time.time(),
                "process_rss_mib": me.memory_info().rss / 2**20,
                "llama_server_rss_mib": llama,
            }
            gpu = self._gpu_used()
            if gpu is not None:
                sample["gpu_used_mib"] = gpu
            self.samples.append(sample)
            self._stop.wait(self.interval)

    def __enter__(self) -> ResourceMonitor:
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"samples": len(self.samples)}
        for key in ("gpu_used_mib", "process_rss_mib", "llama_server_rss_mib"):
            values = [s[key] for s in self.samples if key in s]
            out[key] = {"peak": max(values), "mean": sum(values) / len(values)} if values else None
        try:
            import torch

            if torch.cuda.is_available():
                out["torch_max_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
                out["torch_max_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        except Exception:  # noqa: BLE001 - torch is optional for the monitor
            pass
        return out


# --------------------------------------------------------------------------- runtime


def build_settings(dataset: EvalDatasetV3, overrides: dict[str, Any]) -> Any:
    from settings import RAGSettings

    base = RAGSettings()
    values = base.model_dump()
    values.update(
        {
            "mode": "production",
            "allow_fallback": False,
            "document_root": ROOT / dataset.corpus_root,
            "qdrant_path": INDEX,
            "chunk_target_tokens": int(dataset.chunking["chunk_target_tokens"]),
            "chunk_max_tokens": int(dataset.chunking["chunk_max_tokens"]),
            "chunk_overlap_tokens": int(dataset.chunking["chunk_overlap_tokens"]),
        }
    )
    values.update(RUNTIME)
    values.update(overrides)
    return RAGSettings.model_validate(values)


def settings_record(settings: Any) -> dict[str, Any]:
    data: dict[str, Any] = settings.model_dump(mode="json")
    data.pop("api_tokens", None)
    return data


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def write_config(run_dir: Path, dataset_path: Path, settings: Any, extra: dict[str, Any]) -> None:
    path = run_dir / "config.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "sentence-transformers", "qdrant-client", "rank-bm25"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
    record.update(
        {
            "dataset": {
                "path": dataset_path.relative_to(ROOT).as_posix()
                if dataset_path.is_relative_to(ROOT)
                else str(dataset_path),
                "sha256": sha256_file(dataset_path),
            },
            "git": git_state(),
            "code_sha256": {
                name: sha256_file(ROOT / name) for name in CORE_FILES if (ROOT / name).exists()
            },
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "gpu": gpu_name(),
            "packages": versions,
            "models": {
                "embedding": model_fingerprint(settings.embedding_model_path),
                "reranker": model_fingerprint(settings.reranker_model_path),
                "nli": model_fingerprint(settings.nli_model_path),
                "llm": model_fingerprint(settings.llm_model_path),
            },
            "llama_server": llama_props(settings.llm_api_base),
            "sampling": {
                "temperature": settings.llm_temperature,
                "top_p": settings.llm_top_p,
                "max_tokens": settings.llm_max_tokens,
                "seed": settings.llm_seed,
                "context_window": settings.llm_context_window,
            },
            "runtime": {
                "embedding_batch_size": settings.embedding_batch_size,
                "rerank_batch_size": settings.rerank_batch_size,
            },
            "nli_thresholds": {
                "entailment": settings.nli_entailment_threshold,
                "contradiction": settings.nli_contradiction_threshold,
                "strong_entailment": settings.nli_strong_entailment_threshold,
                "strong_enabled": settings.nli_strong_entailment_enabled,
            },
        }
    )
    record.setdefault("modes", {}).update(extra)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def open_pipeline(dataset: EvalDatasetV3, settings: Any) -> Any:
    """Open the pipeline on the shared v3 index; (re)build it only if it does not match."""
    from pipeline import RAGPipeline

    expected = sum(doc.chunks for doc in dataset.documents)
    gold = {cid for case in dataset.cases for cid in case.gold_chunk_ids}
    gold |= {case.nli_probe.premise_chunk_id for case in dataset.cases}
    for attempt in range(2):
        pipeline = RAGPipeline(settings)
        retriever = pipeline.retriever
        size = retriever.size
        missing = [cid for cid in gold if retriever.get_chunk(cid) is None]
        if size == expected and not missing:
            print(f"Index ready: {size} chunks", file=sys.stderr, flush=True)
            return pipeline
        if attempt == 1 or size:
            pipeline.close()
            if attempt == 1:
                raise RuntimeError(
                    f"Index mismatch after rebuild: {size} chunks, {len(missing)} gold chunks missing"
                )
            shutil.rmtree(INDEX, ignore_errors=True)
            continue
        started = time.perf_counter()
        print(
            f"Building index for {len(dataset.documents)} documents ...",
            file=sys.stderr,
            flush=True,
        )
        stats = pipeline.index_documents(
            [Path(settings.document_root) / d.path for d in dataset.documents]
        )
        print(
            f"Indexed {stats.indexed_chunks} chunks in {time.perf_counter() - started:.0f} s",
            file=sys.stderr,
            flush=True,
        )
        pipeline.close()
    raise RuntimeError("unreachable")


def normalize_question(question: str) -> str:
    return unicodedata.normalize("NFKC", question).strip()


class CaseLog:
    """Append-only JSONL with resume support."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        if path.exists():
            self.rows = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
            ]

    @property
    def done(self) -> set[str]:
        return {row["id"] for row in self.rows}

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- modes


def run_retrieval(
    dataset: EvalDatasetV3, dataset_path: Path, run_dir: Path, limit: int | None
) -> dict[str, Any]:
    settings = build_settings(dataset, {})
    out = run_dir / "retrieval"
    log = CaseLog(out / "cases.jsonl")
    cases = dataset.cases[:limit] if limit else dataset.cases
    with ResourceMonitor() as monitor:
        pipeline = open_pipeline(dataset, settings)
        try:
            retriever = pipeline.retriever
            for i, case in enumerate(cases, 1):
                if case.id in log.done:
                    continue
                print(f"[retrieval] {case.id} ({i}/{len(cases)})", file=sys.stderr, flush=True)
                results, timing, hybrid, diagnostics = retriever.search_trace(
                    normalize_question(case.question), top_k=10
                )
                lists = retriever.candidate_lists()
                rankings = {
                    "dense": lists["dense"],
                    "sparse": lists["sparse"],
                    "hybrid_rrf": hybrid,
                    "hybrid_rerank_unfiltered": rerank_order(diagnostics, settings),
                    "hybrid_rerank": [hit.chunk.chunk_id for hit in results],
                }
                row: dict[str, Any] = {
                    "id": case.id,
                    "category": case.category,
                    "language": case.language,
                    "answerable": case.answerable,
                    "gate_passed": bool(results),
                    "timing_ms": timing,
                    "top5": {mode: ids[:5] for mode, ids in rankings.items()},
                    "metrics": {},
                }
                if case.evidence:
                    units = [e.chunk_ids for e in case.evidence]
                    row["metrics"] = {
                        mode: evidence_metrics(ids, units) for mode, ids in rankings.items()
                    }
                log.append(row)
        finally:
            pipeline.close()
    summary = summarize_retrieval(log.rows)
    summary["resources"] = monitor.summary()
    dump_json(out / "summary.json", summary)
    write_config(
        run_dir, dataset_path, settings, {"retrieval": {"cases": len(log.rows), "top_k": 10}}
    )
    return summary


def summarize_retrieval(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r["metrics"]]
    modes = sorted({m for r in scored for m in r["metrics"]})

    def mean_metrics(subset: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
        result = {}
        for mode in modes:
            names = scored[0]["metrics"][mode].keys() if scored else []
            result[mode] = (
                {
                    name: sum(r["metrics"][mode][name] for r in subset) / len(subset)
                    for name in names
                }
                if subset
                else {}
            )
        return result

    categories = sorted({r["category"] for r in scored})
    gate_unanswerable = [not r["gate_passed"] for r in rows if not r["answerable"]]
    gate_answerable = [not r["gate_passed"] for r in rows if r["answerable"]]
    return {
        "cases": len(rows),
        "scored_cases": len(scored),
        "overall": mean_metrics(scored),
        "by_category": {
            c: mean_metrics([r for r in scored if r["category"] == c]) for c in categories
        },
        "by_language": {
            lang: mean_metrics([r for r in scored if r["language"] == lang])
            for lang in sorted({r["language"] for r in scored})
        },
        "relevance_gate": {
            "rejected_unanswerable": rate(gate_unanswerable),
            "rejected_answerable": rate(gate_answerable),
        },
        "latency_ms": {
            "search": latency_stats([r["timing_ms"]["search_ms"] for r in rows]),
            "rerank": latency_stats([r["timing_ms"]["rerank_ms"] for r in rows]),
        },
    }


def run_nli(
    dataset: EvalDatasetV3, dataset_path: Path, run_dir: Path, limit: int | None
) -> dict[str, Any]:
    from generator_verifier import NLIFactVerifier, SourceContext

    settings = build_settings(dataset, {})
    out = run_dir / "nli"
    log = CaseLog(out / "pairs.jsonl")
    cases = dataset.cases[:limit] if limit else dataset.cases
    with ResourceMonitor() as monitor:
        pipeline = open_pipeline(dataset, settings)
        try:
            retriever = pipeline.retriever
            verifier = NLIFactVerifier(settings)
            for i, case in enumerate(cases, 1):
                chunk = retriever.get_chunk(case.nli_probe.premise_chunk_id)
                if chunk is None:
                    raise RuntimeError(f"Premise chunk missing for {case.id}")
                context = SourceContext(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    citation="S1",
                )
                for lang in ("en", "ru"):
                    pair_id = f"{case.id}:{lang}"
                    if pair_id in log.done:
                        continue
                    hypothesis = getattr(case.nli_probe, f"hypothesis_{lang}")
                    started = time.perf_counter()
                    claim = verifier._claim(f"{hypothesis} [S1]", {"S1": context})
                    elapsed = (time.perf_counter() - started) * 1000
                    probs = claim.evidence[0].probabilities if claim.evidence else None
                    log.append(
                        {
                            "id": pair_id,
                            "case_id": case.id,
                            "category": case.category,
                            "subtype": case.subtype,
                            "hypothesis_language": lang,
                            "label": case.nli_probe.label,
                            "hypothesis": hypothesis,
                            "status": claim.status.value,
                            "reason": claim.evidence[0].reason if claim.evidence else claim.reason,
                            "p_entailment": probs.p_entailment if probs else None,
                            "p_neutral": probs.p_neutral if probs else None,
                            "p_contradiction": probs.p_contradiction if probs else None,
                            "latency_ms": elapsed,
                        }
                    )
                print(f"[nli] {case.id} ({i}/{len(cases)})", file=sys.stderr, flush=True)
            if hasattr(verifier, "close"):
                verifier.close()
        finally:
            pipeline.close()
    summary = summarize_nli(log.rows, settings)
    summary["resources"] = monitor.summary()
    dump_json(out / "summary.json", summary)
    write_config(run_dir, dataset_path, settings, {"nli": {"pairs": len(log.rows)}})
    return summary


def summarize_nli(rows: Sequence[dict[str, Any]], settings: Any) -> dict[str, Any]:
    scored = [r for r in rows if r["p_entailment"] is not None]

    def argmax(r: dict[str, Any]) -> str:
        return max(LABELS, key=lambda label: r[f"p_{label}"])

    def block(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        with_probs = [r for r in subset if r["p_entailment"] is not None]
        return {
            "pairs": len(subset),
            "pairs_exceeding_context": len(subset) - len(with_probs),
            "argmax": classification_report(
                [r["label"] for r in with_probs], [argmax(r) for r in with_probs]
            ),
            "decision_rule": decision_metrics(subset),
        }

    strong = (
        settings.nli_strong_entailment_threshold if settings.nli_strong_entailment_enabled else None
    )
    return {
        "overall": block(rows),
        "by_hypothesis_language": {
            lang: block([r for r in rows if r["hypothesis_language"] == lang])
            for lang in ("en", "ru")
        },
        "by_category": {
            c: block([r for r in rows if r["category"] == c])
            for c in sorted({r["category"] for r in rows})
        },
        "threshold_sweep": threshold_sweep(
            scored,
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.1, 0.2, 0.3, 0.5, 0.7],
            strong,
        ),
        "latency_ms_per_pair": latency_stats([r["latency_ms"] for r in rows]),
    }


def run_e2e(
    dataset: EvalDatasetV3, dataset_path: Path, run_dir: Path, profile: str, limit: int | None
) -> dict[str, Any]:
    settings = build_settings(dataset, PROFILES[profile])
    seed_everything(settings.llm_seed)
    out = run_dir / f"e2e_{profile}"
    log = CaseLog(out / "cases.jsonl")
    full = CaseLog(run_dir / "full" / f"e2e_{profile}.jsonl")
    cases = dataset.cases[:limit] if limit else dataset.cases
    by_id = {c.id: c for c in dataset.cases}
    with ResourceMonitor() as monitor:
        pipeline = open_pipeline(dataset, settings)
        try:
            for i, case in enumerate(cases, 1):
                if case.id in log.done:
                    continue
                print(f"[e2e:{profile}] {case.id} ({i}/{len(cases)})", file=sys.stderr, flush=True)
                started = time.perf_counter()
                row: dict[str, Any] = {
                    "id": case.id,
                    "category": case.category,
                    "subtype": case.subtype,
                    "language": case.language,
                    "expected_behavior": case.expected_behavior,
                    "ok": False,
                }
                try:
                    response = pipeline.query(case.question)
                except Exception as exc:  # noqa: BLE001 - recorded per case, never fabricated
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    row["wall_ms"] = (time.perf_counter() - started) * 1000
                    log.append(row)
                    continue
                draft = response.raw_answer if response.raw_answer is not None else response.answer
                claims = [
                    {
                        "claim": c.claim,
                        "status": c.status.value,
                        "reason": c.reason,
                        "probabilities": [
                            e.probabilities.model_dump() if e.probabilities else None
                            for e in c.evidence
                        ],
                        "chunks": [e.chunk_id for e in c.evidence],
                        "reasons": [e.reason for e in c.evidence],
                    }
                    for c in response.claims
                ]
                gold = set(case.gold_chunk_ids)
                row.update(
                    {
                        "ok": True,
                        "answer": response.answer,
                        "raw_answer": response.raw_answer,
                        "refused": response.refused,
                        "refusal_reason": response.refusal_reason,
                        "claims": claims,
                        "sources": [s.chunk_id for s in response.sources],
                        "gold_in_sources": bool(gold & {s.chunk_id for s in response.sources})
                        if gold
                        else None,
                        "execution_stats": response.execution_stats,
                        "finish_reason": response.generation_metadata.finish_reason
                        if response.generation_metadata
                        else None,
                        "wall_ms": (time.perf_counter() - started) * 1000,
                        "factrag": score_answer(
                            case, response.answer if not response.refused else None
                        ),
                        "vanilla": score_answer(case, draft),
                    }
                )
                log.append(row)
                full.append({"id": case.id, "response": response.model_dump(mode="json")})
        finally:
            pipeline.close()
    summary = summarize_e2e(log.rows, by_id)
    summary["resources"] = monitor.summary()
    dump_json(out / "summary.json", summary)
    write_config(
        run_dir,
        dataset_path,
        settings,
        {
            f"e2e_{profile}": {
                "cases": len(log.rows),
                "overrides": PROFILES[profile],
                "settings": settings_record(settings),
            }
        },
    )
    return summary


def resummarize_e2e(dataset: EvalDatasetV3, run_dir: Path, profile: str) -> dict[str, Any]:
    """Recompute an E2E summary from logged rows without querying the models again.

    Claim-level reasons missing from older light logs are projected from ``full/`` when
    available; measured resources of the original run are kept as recorded.
    """
    out = run_dir / f"e2e_{profile}"
    rows = CaseLog(out / "cases.jsonl").rows
    if not rows:
        raise FileNotFoundError(f"No logged cases in {out}")
    full_path = run_dir / "full" / f"e2e_{profile}.jsonl"
    if full_path.exists():
        responses = {row["id"]: row["response"] for row in CaseLog(full_path).rows}
        for row in rows:
            raw = responses.get(row["id"], {}).get("claims", [])
            if row.get("ok") and len(raw) == len(row["claims"]):
                for claim, source in zip(row["claims"], raw, strict=True):
                    claim.setdefault("reason", source.get("reason"))
        (out / "cases.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    previous = out / "summary.json"
    resources = (
        json.loads(previous.read_text(encoding="utf-8")).get("resources")
        if previous.exists()
        else None
    )
    summary = summarize_e2e(rows, {c.id: c for c in dataset.cases})
    summary["resources"] = resources
    dump_json(previous, summary)
    return summary


def stage_values(stats: Sequence[dict[str, Any]], stage: str) -> list[float]:
    # TTFT is recorded only with streaming (RAG_LLM_API_STREAM); absence is not 0 ms.
    if stage == "llm_ttft_ms":
        return [s[stage] for s in stats if s.get(stage) is not None]
    return [s.get(stage, 0.0) for s in stats]


def summarize_e2e(rows: Sequence[dict[str, Any]], by_id: dict[str, EvalCase]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("ok")]
    vanilla, factrag = summarize_variant(rows, "vanilla"), summarize_variant(rows, "factrag")
    lost = [r for r in ok if r["vanilla"]["correct"] and not r["factrag"]["correct"]]
    b = len(lost)
    c = sum(r["factrag"]["correct"] and not r["vanilla"]["correct"] for r in ok)
    hv, hf = vanilla["hallucination"]["rate"], factrag["hallucination"]["rate"]
    claims = [claim for r in ok for claim in r["claims"]]
    statuses = Counter(claim["status"] for claim in claims)
    reasons = Counter(f"{claim['status']}:{claim.get('reason') or 'none'}" for claim in claims)
    stats = [r["execution_stats"] for r in ok]
    # Lazy model loading happens on the first generated query; later ones report ~0 ms.
    warm = [s for s in stats if s.get("model_init_ms", 0.0) < 50.0]
    stages = ("search_ms", "rerank_ms", "generation_ms", "nli_ms", "total_ms", "llm_ttft_ms")
    generated = [s for s in warm if s.get("generation_ms")]
    return {
        "cases": len(rows),
        "errors": len(rows) - len(ok),
        "vanilla": vanilla,
        "factrag": factrag,
        "paired_correctness": {
            "vanilla_only_correct": b,
            "factrag_only_correct": c,
            "mcnemar_exact_p": mcnemar_exact(b, c),
            # Why the filter turned a correct draft into a wrong answer.
            "vanilla_only_correct_by_cause": dict(
                Counter(r["refusal_reason"] if r["refused"] else "filtered_answer" for r in lost)
            ),
        },
        "hallucination_relative_reduction": (hv - hf) / hv if hv else None,
        "hallucination_ids": {
            "vanilla_only": [
                r["id"]
                for r in ok
                if r["vanilla"]["hallucination"] and not r["factrag"]["hallucination"]
            ],
            "factrag_only": [
                r["id"]
                for r in ok
                if r["factrag"]["hallucination"] and not r["vanilla"]["hallucination"]
            ],
            "both": [
                r["id"]
                for r in ok
                if r["factrag"]["hallucination"] and r["vanilla"]["hallucination"]
            ],
        },
        "claims": {
            "total": len(claims),
            "by_status": dict(statuses),
            "by_status_reason": dict(reasons),
            "pairs_exceeding_nli_context": sum(
                reason == "pair_exceeds_nli_context"
                for claim in claims
                for reason in claim.get("reasons", [])
            ),
            "faithfulness_micro": statuses.get("verified", 0) / len(claims) if claims else None,
        },
        "refusal_reasons": dict(Counter(r["refusal_reason"] or "none" for r in ok if r["refused"])),
        "latency_ms": {
            "all_queries": {stage: latency_stats(stage_values(warm, stage)) for stage in stages},
            "generated_queries": {
                stage: latency_stats(stage_values(generated, stage)) for stage in stages
            },
            "model_init_ms": [
                s["model_init_ms"] for s in stats if s.get("model_init_ms", 0.0) >= 50.0
            ],
        },
        "gold_in_sources": rate(
            [bool(r["gold_in_sources"]) for r in ok if r["gold_in_sources"] is not None]
        ),
    }


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cli.add_argument("--dataset", type=Path, default=DATASET)
    cli.add_argument("--run-dir", type=Path, required=True)
    cli.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N cases (smoke test)"
    )
    sub = cli.add_subparsers(dest="mode", required=True)
    sub.add_parser("retrieval")
    sub.add_parser("nli")
    e2e = sub.add_parser("e2e")
    e2e.add_argument("--profile", choices=sorted(PROFILES), default="guard_on")
    again = sub.add_parser("resummarize", help="Recompute an E2E summary from existing logs")
    again.add_argument("--profile", choices=sorted(PROFILES), default="guard_on")
    args = cli.parse_args(argv)
    dataset_path = args.dataset.resolve()
    dataset = load_dataset(dataset_path)
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "retrieval":
        summary = run_retrieval(dataset, dataset_path, run_dir, args.limit)
        print(json.dumps(summary["overall"].get("hybrid_rerank", {}), indent=1))
    elif args.mode == "nli":
        summary = run_nli(dataset, dataset_path, run_dir, args.limit)
        print(json.dumps(summary["overall"]["decision_rule"], indent=1))
    elif args.mode == "resummarize":
        summary = resummarize_e2e(dataset, run_dir, args.profile)
        print(json.dumps(summary["paired_correctness"], indent=1))
    else:
        summary = run_e2e(dataset, dataset_path, run_dir, args.profile, args.limit)
        print(
            json.dumps({k: summary[k] for k in ("cases", "errors", "paired_correctness")}, indent=1)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
