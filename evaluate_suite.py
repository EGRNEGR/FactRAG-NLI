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
  replay     no LLM, no retrieval: the logged drafts and the NLI probes of an earlier run are
             re-verified under each verifier ablation step (ablation.json); the last step is
             the current profile. Isolates verifier changes from generation changes.

Every run writes reports/runs/<run_id>/: config.json, <mode>/cases.jsonl, <mode>/summary.json,
<mode>/resources.json; heavy traces go to reports/runs/<run_id>/full/ (git-ignored).

Usage:
  python evaluate_suite.py --run-dir reports/runs/<run_id> retrieval
  python evaluate_suite.py --run-dir reports/runs/<run_id> nli
  python evaluate_suite.py --run-dir reports/runs/<run_id> e2e --profile guard_on
  python evaluate_suite.py --run-dir reports/runs/<run_id> replay --source reports/runs/<old_run>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shlex
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


def _env_path(name: str, default: Path) -> Path:
    """Path from the environment (relative paths resolve against ROOT), else the default."""
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


DATASET = ROOT / "data" / "eval_dataset_v3.json"
# Each dataset needs its own index and parse cache: open_pipeline rebuilds a mismatching index.
INDEX = _env_path("FACTRAG_INDEX", ROOT / ".artifacts" / "eval-v3-index")
KS = (1, 3, 5, 10)
LABELS = ("entailment", "neutral", "contradiction")
EXPECTED_STATUS = {
    "entailment": "verified",
    "neutral": "unverified",
    "contradiction": "contradiction",
}
# Hardware invariants of the 8 GB stack (docs/evidence/improvement_report.md): with llama-server
# holding ~6.6 GB, larger embedding/rerank batches overflow VRAM into shared memory on Windows
# and slow generation 2-3x (observed with the code defaults 16/8). A 24 GB card raises them via
# FACTRAG_EMBEDDING_BATCH / FACTRAG_RERANK_BATCH (scripts/linux/env.rtx3090.sh).
RUNTIME: dict[str, Any] = {
    "embedding_batch_size": int(os.environ.get("FACTRAG_EMBEDDING_BATCH", "2")),
    "rerank_batch_size": int(os.environ.get("FACTRAG_RERANK_BATCH", "2")),
}
PROFILES: dict[str, dict[str, Any]] = {
    "guard_on": {},
    "guard_off": {"premise_guard": False},
}
# llama-server watchdog of long e2e stages (scripts/linux/run_benchmark.sh sets the command):
# FACTRAG_LLM_RECOVER_CMD brings the server back, e.g. "bash scripts/linux/llama_server.sh ensure".
RECOVER_COMMAND = os.environ.get("FACTRAG_LLM_RECOVER_CMD", "")
MAX_CONSECUTIVE_ERRORS = int(os.environ.get("FACTRAG_MAX_CONSECUTIVE_ERRORS", "5"))
EXIT_ABORTED = 3
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
    "grounding.py",
    "hybrid_retriever.py",
    "document_parser.py",
    "settings.py",
    "eval_dataset.py",
    "evaluate_suite.py",
)
PARSE_CACHE = _env_path("FACTRAG_PARSE_CACHE", ROOT / ".artifacts" / "eval_v3_parse_cache")
# Verifier switches of the baseline run reports/runs/20261026T1410Z_v3 (bf16 NLI, whole-chunk
# premises, strict multi-source rule; the last two switches came after 20261026T2139Z_v3).
# The replay ablation releases them one step at a time in ABLATION order; the last step is
# the current profile without any override.
BASELINE_VERIFIER: dict[str, Any] = {
    "nli_precision": "bfloat16",
    "claim_segmentation": "v1",
    "claim_citation_inheritance": False,
    "nli_premise_cleanup": False,
    "nli_window_scoring": False,
    "nli_aggregation": "strict",
    "nli_anchor_guard": False,
    "nli_anchor_windows": False,
    "nli_code_span_min_chars": 0,
    "abstention_guard": False,
    "nli_premise_title": False,
    "nli_co_verification": False,
    "nli_strip_verdict_prefix": False,
}
ABLATION: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("baseline", ()),
    ("nli_float16", ("nli_precision",)),
    ("segmentation_v2", ("claim_segmentation",)),
    ("citation_inheritance", ("claim_citation_inheritance",)),
    ("premise_cleanup", ("nli_premise_cleanup",)),
    ("sentence_windows", ("nli_window_scoring",)),
    ("support_first", ("nli_aggregation",)),
    ("anchor_guard", ("nli_anchor_guard", "nli_anchor_windows")),
    ("code_span", ("nli_code_span_min_chars",)),
    ("abstention_guard", ("abstention_guard",)),
    ("document_titles", ("nli_premise_title",)),
    ("co_verification", ("nli_co_verification",)),
    ("verdict_prefix", ("nli_strip_verdict_prefix",)),
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


def dataset_split(cases: Iterable[EvalCase]) -> dict[str, str]:
    """Calibration (dev) and held-out (test) halves of the dataset.

    Within each category the case ids are ordered by SHA-1 and alternate dev/test, so the
    split is fixed by the ids alone. Verifier thresholds are tuned on dev; test is reported
    separately as the check against overfitting.
    """
    by_category: dict[str, list[str]] = {}
    for case in cases:
        by_category.setdefault(case.category, []).append(case.id)
    split: dict[str, str] = {}
    for ids in by_category.values():
        ordered = sorted(ids, key=lambda cid: hashlib.sha1(cid.encode()).hexdigest())
        split.update({cid: "dev" if i % 2 == 0 else "test" for i, cid in enumerate(ordered)})
    return split


def headline(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    """Correctness, false refusal on answerable questions and hallucination of one variant."""
    ok = [r for r in rows if r.get("ok")]
    return {
        "correct": rate([r[key]["correct"] for r in ok]),
        "false_refusal_on_answerable": rate(
            [r[key]["refused"] for r in ok if r["expected_behavior"] == "answer"]
        ),
        "hallucination": rate(
            [r[key]["hallucination"] for r in ok if r[key]["hallucination"] is not None]
        ),
    }


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
                    # Best reranker probability of a substantive chunk: the relevance gate
                    # passes iff it reaches rerank_score_threshold (scripts/calibrate_gate.py).
                    "top_rerank_score": max(
                        (d["rerank_score"] for d in diagnostics if not d["heading_only"]),
                        default=None,
                    ),
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


def probe_rows(verifier: Any, case: EvalCase, context: Any) -> list[dict[str, Any]]:
    """The production claim rule on the gold premise for the EN and RU probe hypotheses.

    Probabilities are those of the deciding evidence: the supporting sentence window when
    window scoring supports the claim, otherwise the whole premise chunk.
    """
    rows = []
    for lang in ("en", "ru"):
        hypothesis = getattr(case.nli_probe, f"hypothesis_{lang}")
        started = time.perf_counter()
        claim = verifier._claim(f"{hypothesis} [S1]", {"S1": context})
        elapsed = (time.perf_counter() - started) * 1000
        evidence = claim.evidence[0] if claim.evidence else None
        probs = evidence.probabilities if evidence else None
        rows.append(
            {
                "id": f"{case.id}:{lang}",
                "case_id": case.id,
                "category": case.category,
                "subtype": case.subtype,
                "hypothesis_language": lang,
                "label": case.nli_probe.label,
                "hypothesis": hypothesis,
                "status": claim.status.value,
                "reason": (evidence.reason if evidence else None) or claim.reason,
                "support": evidence.support if evidence else None,
                "p_entailment": probs.p_entailment if probs else None,
                "p_neutral": probs.p_neutral if probs else None,
                "p_contradiction": probs.p_contradiction if probs else None,
                "latency_ms": elapsed,
            }
        )
    return rows


def run_nli(
    dataset: EvalDatasetV3, dataset_path: Path, run_dir: Path, limit: int | None
) -> dict[str, Any]:
    from document_parser import document_title
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
                if {f"{case.id}:en", f"{case.id}:ru"} <= log.done:
                    continue
                chunk = retriever.get_chunk(case.nli_probe.premise_chunk_id)
                if chunk is None:
                    raise RuntimeError(f"Premise chunk missing for {case.id}")
                context = SourceContext(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    citation="S1",
                    section_path=chunk.section_path,
                    document_title=document_title(chunk.source),
                )
                for row in probe_rows(verifier, case, context):
                    if row["id"] not in log.done:
                        log.append(row)
                print(f"[nli] {case.id} ({i}/{len(cases)})", file=sys.stderr, flush=True)
            if hasattr(verifier, "close"):
                verifier.close()
        finally:
            pipeline.close()
    summary = summarize_nli(log.rows, settings, dataset_split(dataset.cases))
    summary["resources"] = monitor.summary()
    dump_json(out / "summary.json", summary)
    write_config(run_dir, dataset_path, settings, {"nli": {"pairs": len(log.rows)}})
    return summary


def summarize_nli(
    rows: Sequence[dict[str, Any]], settings: Any, split: dict[str, str] | None = None
) -> dict[str, Any]:
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
        "by_split": {
            name: decision_metrics(subset)
            for name in ("dev", "test")
            if (subset := [r for r in rows if (split or {}).get(r["case_id"]) == name])
        },
        "threshold_sweep": threshold_sweep(
            scored,
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.1, 0.2, 0.3, 0.5, 0.7],
            strong,
        ),
        "latency_ms_per_pair": latency_stats([r["latency_ms"] for r in rows]),
    }


def case_row(case: EvalCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "category": case.category,
        "subtype": case.subtype,
        "language": case.language,
        "expected_behavior": case.expected_behavior,
        "ok": False,
    }


def claim_rows(claims: Sequence[Any]) -> list[dict[str, Any]]:
    """Light projection of VerifiedClaim objects for the case logs."""
    return [
        {
            "claim": c.claim,
            "status": c.status.value,
            "reason": c.reason,
            "kind": c.kind,
            "citation_inherited": c.citation_inherited,
            "probabilities": [
                e.probabilities.model_dump() if e.probabilities else None for e in c.evidence
            ],
            "chunks": [e.chunk_id for e in c.evidence],
            "reasons": [e.reason for e in c.evidence],
            "support": [e.support for e in c.evidence],
            "co_verified": c.co_verified,
        }
        for c in claims
    ]


def outcome(
    case: EvalCase,
    *,
    answer: str,
    raw_answer: str | None,
    refused: bool,
    refusal_reason: str | None,
    claims: Sequence[Any],
    sources: list[str],
    finish_reason: str | None,
    attempts: int = 1,
    final_draft: str | None = None,
) -> dict[str, Any]:
    """Scored fields of one answered case; Vanilla is the first draft, FactRAG the published text.

    With repair attempts, `final_draft` is the regenerated draft the published answer came from.
    """
    gold = set(case.gold_chunk_ids)
    return {
        "ok": True,
        "answer": answer,
        "raw_answer": raw_answer,
        "attempts": attempts,
        "final_draft": final_draft,
        "refused": refused,
        "refusal_reason": refusal_reason,
        "claims": claim_rows(claims),
        "sources": sources,
        "gold_in_sources": bool(gold & set(sources)) if gold else None,
        "finish_reason": finish_reason,
        "factrag": score_answer(case, answer if not refused else None),
        "vanilla": score_answer(case, raw_answer if raw_answer is not None else answer),
    }


def archive_failed(log: CaseLog, archive_path: Path) -> list[str]:
    """Move failed rows of `log` to `archive_path` so a resumed run executes them again."""
    failed = [row for row in log.rows if not row.get("ok")]
    if not failed:
        return []
    archive = CaseLog(archive_path)
    for row in failed:
        archive.append(row)
    log.rows = [row for row in log.rows if row.get("ok")]
    log.path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in log.rows),
        encoding="utf-8",
    )
    return [row["id"] for row in failed]


class LLMWatchdog:
    """Keeps the external LLM server alive during an e2e stage.

    A dead llama-server turns every later case into an error, retry stages included. With a
    recovery command the stage checks /health before the first case and after each failure,
    brings the server back and runs the failed case once more.
    """

    def __init__(self, api_base: str | None, command: str) -> None:
        base = (api_base or "").rstrip("/")
        self.health_url = (base.removesuffix("/v1") + "/health") if base else None
        self.command = shlex.split(command) if command else []
        self.recoveries = 0

    def healthy(self) -> bool:
        if self.health_url is None:
            return True
        import httpx

        try:
            return httpx.get(self.health_url, timeout=10, trust_env=False).status_code == 200
        except httpx.HTTPError:
            return False

    def recover(self) -> bool:
        """Run the recovery command if the server does not answer; True if it answers now."""
        if self.healthy():
            return False
        if not self.command:
            return False
        print(
            f"[watchdog] LLM server is not answering; running: {shlex.join(self.command)}",
            file=sys.stderr,
            flush=True,
        )
        self.recoveries += 1
        try:
            done = subprocess.run(self.command, cwd=ROOT, timeout=1800, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"[watchdog] recovery failed: {exc}", file=sys.stderr, flush=True)
            return False
        return done.returncode == 0 and self.healthy()


def run_e2e(
    dataset: EvalDatasetV3,
    dataset_path: Path,
    run_dir: Path,
    profile: str,
    limit: int | None,
    retry_errors: bool = False,
) -> dict[str, Any]:
    """Run the pipeline on every case; logged cases are skipped, so an interrupted run resumes.

    With retry_errors, cases logged as failed (an exception, e.g. a timeout while the machine
    slept) are moved to retried_errors.jsonl and run again; the resource measurements of the
    original run are kept and those of the retry are reported separately.

    FACTRAG_MAX_CONSECUTIVE_ERRORS failures in a row stop the stage (summary key "aborted",
    exit code 3) instead of logging the rest of the dataset as errors; a resumed or retry
    run executes the remaining cases.
    """
    settings = build_settings(dataset, PROFILES[profile])
    watchdog = LLMWatchdog(settings.llm_api_base, RECOVER_COMMAND)
    if watchdog.command and not watchdog.healthy() and not watchdog.recover():
        print("[watchdog] LLM server unavailable before the first case", file=sys.stderr)
    seed_everything(settings.llm_seed)
    out = run_dir / f"e2e_{profile}"
    log = CaseLog(out / "cases.jsonl")
    previous = out / "summary.json"
    retried = archive_failed(log, out / "retried_errors.jsonl") if retry_errors else []
    kept_resources = (
        json.loads(previous.read_text(encoding="utf-8")).get("resources")
        if retried and previous.exists()
        else None
    )
    full = CaseLog(run_dir / "full" / f"e2e_{profile}.jsonl")
    cases = dataset.cases[:limit] if limit else dataset.cases
    by_id = {c.id: c for c in dataset.cases}
    consecutive_errors = 0
    aborted: str | None = None
    with ResourceMonitor() as monitor:
        pipeline = open_pipeline(dataset, settings)
        try:
            for i, case in enumerate(cases, 1):
                if case.id in log.done:
                    continue
                print(f"[e2e:{profile}] {case.id} ({i}/{len(cases)})", file=sys.stderr, flush=True)
                started = time.perf_counter()
                row = case_row(case)
                response = None
                error = ""
                for attempt in (1, 2):
                    try:
                        response = pipeline.query(case.question)
                        break
                    except Exception as exc:  # noqa: BLE001 - recorded per case, never fabricated
                        error = f"{type(exc).__name__}: {exc}"
                        print(f"[e2e:{profile}] {case.id} failed: {error}", file=sys.stderr)
                        # Run the case again only if the server was down and is back now.
                        if attempt == 2 or not watchdog.recover():
                            break
                if response is None:
                    row["error"] = error
                    row["wall_ms"] = (time.perf_counter() - started) * 1000
                    log.append(row)
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        aborted = f"{consecutive_errors} consecutive errors, last: {error}"
                        print(f"[e2e:{profile}] stage stopped: {aborted}", file=sys.stderr)
                        break
                    continue
                consecutive_errors = 0
                attempts = response.verification_attempts
                # Vanilla RAG is the first draft: a repair attempt exists only because the
                # NLI filter rejected that draft, so it belongs to FactRAG-NLI.
                first_draft = attempts[0]["raw_text"] if attempts else response.raw_answer
                row.update(
                    outcome(
                        case,
                        answer=response.answer,
                        raw_answer=first_draft,
                        final_draft=response.raw_answer if len(attempts) > 1 else None,
                        attempts=len(attempts),
                        refused=response.refused,
                        refusal_reason=response.refusal_reason,
                        claims=response.claims,
                        sources=[s.chunk_id for s in response.sources],
                        finish_reason=response.generation_metadata.finish_reason
                        if response.generation_metadata
                        else None,
                    ),
                    execution_stats=response.execution_stats,
                    wall_ms=(time.perf_counter() - started) * 1000,
                )
                log.append(row)
                full.append({"id": case.id, "response": response.model_dump(mode="json")})
        finally:
            pipeline.close()
    summary = summarize_e2e(log.rows, by_id)
    if kept_resources is not None:
        summary["resources"] = kept_resources
        summary["resources_retry"] = monitor.summary()
    else:
        summary["resources"] = monitor.summary()
    if watchdog.recoveries:
        summary["llm_server_recoveries"] = watchdog.recoveries
    if aborted:
        summary["aborted"] = aborted
    dump_json(previous, summary)
    record: dict[str, Any] = {
        "cases": len(log.rows),
        "overrides": PROFILES[profile],
        "settings": settings_record(settings),
    }
    if retried:
        record["retried_errors"] = retried
    write_config(run_dir, dataset_path, settings, {f"e2e_{profile}": record})
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


class MemoNLI:
    """NLI backend wrapper that scores each distinct input once per replay.

    Single-pair and batched calls are memoized separately and a batch is re-scored as a
    whole when any pair is new: padding shifts low-precision probabilities slightly, and
    every ablation step must see exactly what the production verifier would compute.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.last_premise: str | None = None
        self._single: dict[tuple[str, str], tuple[Any, str | None]] = {}
        self._batched: dict[tuple[str, str], Any] = {}

    def predict(self, premise: str, hypothesis: str) -> Any:
        key = (premise, hypothesis)
        if key not in self._single:
            result = self.backend.predict(premise, hypothesis)
            self._single[key] = (result, getattr(self.backend, "last_premise", premise))
        result, self.last_premise = self._single[key]
        return result

    def predict_many(self, pairs: Sequence[tuple[str, str]]) -> list[Any]:
        pairs = list(pairs)
        if any(pair not in self._batched for pair in pairs):
            self._batched.update(zip(pairs, self.backend.predict_many(pairs), strict=True))
        return [self._batched[pair] for pair in pairs]


def ablation_steps() -> list[tuple[str, dict[str, Any]]]:
    """Cumulative verifier overrides: step k keeps the baseline switches not yet released."""
    released: set[str] = set()
    steps = []
    for name, keys in ABLATION:
        released.update(keys)
        steps.append((name, {k: v for k, v in BASELINE_VERIFIER.items() if k not in released}))
    return steps


def chunk_texts() -> dict[str, tuple[str, str]]:
    """chunk_id -> (document_id, text) from the dataset build cache (same parser output as the index)."""
    texts: dict[str, tuple[str, str]] = {}
    for path in sorted(PARSE_CACHE.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for chunk in record["chunks"]:
            texts[chunk["chunk_id"]] = (record["document_id"], chunk["text"])
    if not texts:
        raise FileNotFoundError(
            f"No parse cache in {PARSE_CACHE}; run scripts/build_eval_dataset_v3.py"
        )
    return texts


_SECTION_HEADER = re.compile(r"^Документ:[^\n]*\nРаздел: ([^\n]*)\n")


def probe_contexts(dataset: EvalDatasetV3) -> dict[str, Any]:
    """NLI probe premises as SourceContext, with the section and document title of the chunk."""
    from document_parser import document_title
    from generator_verifier import SourceContext

    texts = chunk_texts()
    titles = {
        d.document_id: document_title(ROOT / dataset.corpus_root / d.path)
        for d in dataset.documents
    }
    contexts = {}
    for case in dataset.cases:
        document_id, text = texts[case.nli_probe.premise_chunk_id]
        header = _SECTION_HEADER.match(text)
        contexts[case.id] = SourceContext(
            chunk_id=case.nli_probe.premise_chunk_id,
            document_id=document_id,
            text=text,
            citation="S1",
            section_path=tuple(header.group(1).split(" > ")) if header and header.group(1) else (),
            document_title=titles.get(document_id),
        )
    return contexts


def co_verification_stress(
    verifier: Any, dataset: EvalDatasetV3, contexts: dict[str, Any]
) -> dict[str, Any]:
    """Unsupported claims published next to a supported one from the same source.

    For every Neutral or Contradiction probe the draft is "<longest sentence of its own
    premise> [S1]" followed by "<probe hypothesis> [S1]". The first claim is verbatim
    support, the second must stay blocked. Single-claim probes cannot show this risk,
    which is exactly the case co-verification relaxes. Only drafts whose first claim is
    VERIFIED count.
    """
    from generator_verifier import ClaimStatus, GenerationMetadata, RawGenerationResult
    from grounding import clean_premise, premise_windows

    rows = []
    metadata = GenerationMetadata(backend="stress", finish_reason="stop")
    for case in dataset.cases:
        if case.nli_probe.label == "entailment":
            continue
        context = contexts[case.id]
        sentences = [
            w for w in premise_windows(clean_premise(context.text), 1) if len(w.split()) >= 8
        ]
        if not sentences:
            continue
        support = max(sentences, key=len)
        for lang in ("en", "ru"):
            hypothesis = " ".join(getattr(case.nli_probe, f"hypothesis_{lang}").split())
            result = verifier.verify(
                RawGenerationResult(
                    text=f"{support} [S1]\n{hypothesis} [S1]",
                    contexts={"S1": context},
                    metadata=metadata,
                )
            )
            facts = [claim for claim in result.claims if claim.kind == "fact"]
            if len(facts) != 2 or facts[0].status != ClaimStatus.VERIFIED:
                continue
            rows.append(
                {
                    "id": f"{case.id}:{lang}",
                    "label": case.nli_probe.label,
                    "leaked": facts[1].status == ClaimStatus.VERIFIED,
                    "co_verified": facts[1].co_verified,
                }
            )
    leaked = [row for row in rows if row["leaked"]]
    return {
        "pairs": len(rows),
        "leaked": rate([row["leaked"] for row in rows]),
        "leaked_by_co_verification": sum(row["co_verified"] for row in rows),
        "leaked_ids": [row["id"] for row in leaked],
    }


def replay_case(case: EvalCase, response: dict[str, Any], verifier: Any) -> dict[str, Any]:
    """Re-verify the first stored draft of one logged response with `verifier`."""
    from generator_verifier import RawGenerationResult
    from pipeline import refusal_reason

    row = case_row(case)
    sources = [s["chunk_id"] for s in response["sources"]]
    attempts = response["verification_attempts"]
    if not attempts:  # refused by the relevance gate before generation; nothing to verify
        row.update(
            outcome(
                case,
                answer=response["answer"],
                raw_answer=response["raw_answer"],
                refused=response["refused"],
                refusal_reason=response["refusal_reason"],
                claims=[],
                sources=sources,
                finish_reason=None,
            )
        )
        return row
    first = attempts[0]
    contexts = first["contexts"]
    if verifier.settings.nli_premise_title:
        # Logs written before titles existed: read them from the logged source files.
        from document_parser import document_title

        paths = {s.get("source_id"): s.get("source") for s in response["sources"]}
        contexts = {
            key: value
            if value.get("document_title") or not paths.get(key)
            else {**value, "document_title": document_title(paths[key])}
            for key, value in contexts.items()
        }
    verified = verifier.verify(
        RawGenerationResult.model_validate(
            {
                "text": first["raw_text"],
                "contexts": contexts,
                "metadata": first["generation_metadata"],
            }
        )
    )
    row.update(
        outcome(
            case,
            answer=verified.cleaned_text,
            raw_answer=verified.raw_text,
            refused=verified.refused,
            refusal_reason=refusal_reason(verified),
            claims=verified.claims,
            sources=sources,
            finish_reason=verified.generation_metadata.finish_reason,
        )
    )
    return row


def replay_step_record(
    e2e: dict[str, list[dict[str, Any]]],
    nli: dict[str, Any],
    by_id: dict[str, EvalCase],
) -> dict[str, Any]:
    record: dict[str, Any] = {"e2e": {}, "nli": {}}
    for profile, rows in e2e.items():
        summary = summarize_e2e(rows, by_id)
        record["e2e"][profile] = {
            "vanilla": headline(rows, "vanilla"),
            "factrag": headline(rows, "factrag"),
            "paired_correctness": summary["paired_correctness"],
            "filter_induced_hallucinations": summary["hallucination_ids"]["factrag_only"],
            "by_split": summary["by_split"],
            "claims": {
                key: summary["claims"][key] for key in ("total", "by_status", "faithfulness_micro")
            },
        }
    rule = {k: v for k, v in nli["overall"]["decision_rule"].items() if k != "status_matrix"}
    record["nli"] = {
        "overall": rule,
        "by_hypothesis_language": {
            lang: nli["by_hypothesis_language"][lang]["decision_rule"]["false_rejection_rate"]
            for lang in nli["by_hypothesis_language"]
        },
        "by_split": {
            name: {k: v for k, v in metrics.items() if k != "status_matrix"}
            for name, metrics in nli["by_split"].items()
        },
    }
    return record


def run_replay(
    dataset: EvalDatasetV3,
    dataset_path: Path,
    run_dir: Path,
    source: Path,
    final_only: bool,
) -> dict[str, Any]:
    """Re-verify the logged drafts and the NLI probes of `source` without LLM or retrieval.

    Every stored first-attempt draft (full/e2e_<profile>.jsonl) and every gold probe is
    passed through NLIFactVerifier under each ABLATION step, so the effect of each verifier
    change is measured on identical LLM outputs. Repair attempts are not replayed: they
    need the LLM. Latency is not reported: memoized scores make it meaningless.
    """
    from generator_verifier import NLIFactVerifier

    by_id = {c.id: c for c in dataset.cases}
    split = dataset_split(dataset.cases)
    drafts: dict[str, list[dict[str, Any]]] = {}
    for profile in PROFILES:
        path = source / "full" / f"e2e_{profile}.jsonl"
        if path.exists():
            drafts[profile] = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
    if not drafts:
        raise FileNotFoundError(f"No full/e2e_<profile>.jsonl logs in {source}")
    contexts = probe_contexts(dataset)
    steps = ablation_steps()[-1:] if final_only else ablation_steps()
    last_use = {
        build_settings(dataset, overrides).nli_precision: i
        for i, (_, overrides) in enumerate(steps)
    }
    out = run_dir / "replay"
    records: list[dict[str, Any]] = []
    verifiers: dict[str, Any] = {}
    with ResourceMonitor() as monitor:
        try:
            for index, (name, overrides) in enumerate(steps):
                settings = build_settings(dataset, overrides)
                base = verifiers.get(settings.nli_precision)
                if base is None:
                    base = verifiers[settings.nli_precision] = NLIFactVerifier(settings)
                    base.backend = MemoNLI(base.backend)
                # The verifier reads its settings on every call; the shallow copy shares
                # the loaded classifier and its memo across steps of the same precision.
                verifier = copy.copy(base)
                verifier.settings = settings
                e2e = {
                    profile: [
                        replay_case(by_id[row["id"]], row["response"], verifier) for row in logged
                    ]
                    for profile, logged in drafts.items()
                }
                pairs = [
                    row
                    for case in dataset.cases
                    for row in probe_rows(verifier, case, contexts[case.id])
                ]
                nli = summarize_nli(pairs, settings, split)
                record: dict[str, Any] = {"step": name, "overrides": overrides}
                record.update(replay_step_record(e2e, nli, by_id))
                record["nli"]["co_verification_stress"] = co_verification_stress(
                    verifier, dataset, contexts
                )
                records.append(record)
                brief = {p: m["factrag"]["correct"]["k"] for p, m in record["e2e"].items()}
                brief["nli_frr"] = record["nli"]["overall"]["false_rejection_rate"]
                print(f"[replay] {name}: {brief}", file=sys.stderr, flush=True)
                if last_use[settings.nli_precision] == index:
                    verifiers.pop(settings.nli_precision).close()
        finally:
            for verifier in verifiers.values():
                verifier.close()
    for profile, rows in e2e.items():
        target = out / f"e2e_{profile}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "cases.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        summary = summarize_e2e(rows, by_id)
        summary.pop("latency_ms")
        dump_json(target / "summary.json", summary)
    (out / "nli").mkdir(parents=True, exist_ok=True)
    (out / "nli" / "pairs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs), encoding="utf-8"
    )
    nli.pop("latency_ms_per_pair")
    dump_json(out / "nli" / "summary.json", nli)
    source_name = (
        source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else str(source)
    )
    ablation = {"source_run": source_name, "steps": records, "resources": monitor.summary()}
    dump_json(out / "ablation.json", ablation)
    write_config(
        run_dir,
        dataset_path,
        settings,
        {
            "replay": {
                "source_run": source_name,
                "steps": [r["step"] for r in records],
                "settings": settings_record(settings),
            }
        },
    )
    return ablation


def stage_values(stats: Sequence[dict[str, Any]], stage: str) -> list[float]:
    # TTFT is recorded only with streaming (RAG_LLM_API_STREAM); absence is not 0 ms.
    if stage == "llm_ttft_ms":
        return [s[stage] for s in stats if s.get(stage) is not None]
    return [s.get(stage, 0.0) for s in stats]


def summarize_e2e(rows: Sequence[dict[str, Any]], by_id: dict[str, EvalCase]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("ok")]
    split = dataset_split(by_id.values())
    vanilla, factrag = summarize_variant(rows, "vanilla"), summarize_variant(rows, "factrag")
    lost = [r for r in ok if r["vanilla"]["correct"] and not r["factrag"]["correct"]]
    b = len(lost)
    c = sum(r["factrag"]["correct"] and not r["vanilla"]["correct"] for r in ok)
    hv, hf = vanilla["hallucination"]["rate"], factrag["hallucination"]["rate"]
    claims = [claim for r in ok for claim in r["claims"]]
    facts = [claim for claim in claims if claim.get("kind", "fact") == "fact"]
    statuses = Counter(claim["status"] for claim in claims)
    reasons = Counter(f"{claim['status']}:{claim.get('reason') or 'none'}" for claim in claims)
    stats = [r.get("execution_stats") or {} for r in ok]  # replay rows carry no timings
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
        "by_split": {
            name: {key: headline(subset, key) for key in ("vanilla", "factrag")}
            for name in ("dev", "test")
            if (subset := [r for r in ok if split.get(r["id"]) == name])
        },
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
            # Share of VERIFIED among factual claims; absence statements and verdict markers
            # are hedges, not facts, and are counted under their own reasons only.
            "facts": len(facts),
            "faithfulness_micro": sum(c["status"] == "verified" for c in facts) / len(facts)
            if facts
            else None,
            "inherited_citations": sum(bool(c.get("citation_inherited")) for c in claims),
            "verified_by_support": dict(
                Counter(
                    str(next((s for s in c.get("support", []) if s), None) or "chunk")
                    for c in facts
                    if c["status"] == "verified"
                )
            ),
        },
        "refusal_reasons": dict(Counter(r["refusal_reason"] or "none" for r in ok if r["refused"])),
        "repair": {
            "attempted": len(repaired := [r for r in ok if r.get("attempts", 1) > 1]),
            "published": sum(not r["refused"] for r in repaired),
            "factrag_correct": sum(bool(r["factrag"]["correct"]) for r in repaired),
            "factrag_hallucinations": sum(bool(r["factrag"]["hallucination"]) for r in repaired),
        },
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
    e2e.add_argument(
        "--retry-errors",
        action="store_true",
        help="Run cases logged as failed again (originals go to retried_errors.jsonl)",
    )
    again = sub.add_parser("resummarize", help="Recompute an E2E summary from existing logs")
    again.add_argument("--profile", choices=sorted(PROFILES), default="guard_on")
    replay = sub.add_parser(
        "replay", help="Re-verify logged drafts and NLI probes offline (verifier ablation)"
    )
    replay.add_argument(
        "--source", type=Path, required=True, help="Run directory with full/e2e_<profile>.jsonl"
    )
    replay.add_argument(
        "--final-only", action="store_true", help="Only the current profile, no ablation ladder"
    )
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
    elif args.mode == "replay":
        source = args.source if args.source.is_absolute() else ROOT / args.source
        ablation = run_replay(dataset, dataset_path, run_dir, source, args.final_only)
        print(json.dumps(ablation["steps"][-1]["e2e"], indent=1)[:2000])
    elif args.mode == "resummarize":
        summary = resummarize_e2e(dataset, run_dir, args.profile)
        print(json.dumps(summary["paired_correctness"], indent=1))
    else:
        summary = run_e2e(
            dataset, dataset_path, run_dir, args.profile, args.limit, args.retry_errors
        )
        print(
            json.dumps({k: summary[k] for k in ("cases", "errors", "paired_correctness")}, indent=1)
        )
        if summary.get("aborted"):
            return EXIT_ABORTED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
