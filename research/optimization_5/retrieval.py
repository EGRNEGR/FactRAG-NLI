"""Paired retrieval study of parser v1 and v2 on the 220 v3 questions (local indexes).

Both indexes are built here by one procedure (the parser of research.optimization_5.corpus,
then HybridRetriever.add as in RAGPipeline.ingest_path) with the logged retrieval settings of
run 1929Z, and searched with the same models on the same GPU: local against local.

Evidence is matched independently of the chunking, the same way for both parsers. A quote of
the drafts is located by the smallest run of consecutive chunks of its document whose text
(without the chunk header, overlaps merged, table markup ignored) contains it: one chunk for
every quote under v1, sometimes two under v2 where a quote crosses a chunk boundary. The quote
is found at k when its whole run is in the top-k. Per question: recall@k = found quotes /
quotes, hit@k = every quote found; plus the gate (top rerank score against the threshold).

Usage:  python -m research.optimization_5.retrieval build v1|v2
        python -m research.optimization_5.retrieval compare
"""

from __future__ import annotations

import json
import re
import sys
from functools import cache
from pathlib import Path
from collections.abc import Callable
from typing import Any

import research.optimization_5  # noqa: F401 - selects the baseline run
from eval_dataset import load_drafts
from hybrid_retriever import HybridRetriever
from scripts import build_eval_dataset_v3 as builder
from settings import RAGSettings

from research.opt3 import data
from research.optimization_5 import corpus

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / ".artifacts" / "opt5"
OUT = ROOT / "reports" / "optimization_5" / "retrieval"
KS = (1, 3, 5, 10)
CANDIDATES = 40  # rerank_candidates of the run: ranks beyond the published top 10
_HEADER = re.compile(r"^Документ:[^\n]*\nРаздел:[^\n]*\n")


def settings(version: str) -> RAGSettings:
    logged = dict(data.run_config("judge", "v3")["modes"]["e2e_guard_on"]["settings"])
    for key, value in logged.items():
        if isinstance(value, str) and value.startswith(data.RUN_ROOT):
            logged[key] = str(ROOT / value[len(data.RUN_ROOT) + 1 :])
    logged.update(
        mode="development",
        allow_fallback=True,
        qdrant_path=str(INDEX / f"eval-v3-index-{version}"),
        embedding_device="cuda",
        model_device="cuda",
        parser_version=version,
    )
    # _env_file is a pydantic-settings init option mypy does not see among the fields.
    no_env_file: dict[str, Any] = {"_env_file": None}
    return RAGSettings(**no_env_file, **logged)


@cache
def retriever(version: str) -> HybridRetriever:
    index = HybridRetriever(settings=settings(version))
    if index.fallback_reasons:
        raise RuntimeError(f"model fallback in the study: {index.fallback_reasons}")
    return index


def build(version: str) -> int:
    target = INDEX / f"eval-v3-index-{version}"
    if target.exists():
        raise SystemExit(f"{target} exists; remove it to rebuild")
    root, names = corpus.files(version)
    parser = corpus.parser(version)
    index = retriever(version)
    total = 0
    for name in names:
        total += index.add(parser.parse(root / name).chunks)
        print(f"{name}: {index.size} chunks indexed", file=sys.stderr, flush=True)
    index.close()
    return total


def body(text: str) -> str:
    return builder.without_table_markup(_HEADER.sub("", text, count=1))


def merge(left: str, right: str) -> str:
    """Concatenate two consecutive chunk bodies, dropping the repeated overlap."""
    for size in range(min(len(left), len(right)), 0, -1):
        if left.endswith(right[:size]):
            return left + right[size:]
    return f"{left} {right}"


def ordered_chunks(version: str) -> dict[str, list[tuple[str, str]]]:
    """document_id -> [(chunk_id, body)] in document order."""
    index = retriever(version)
    by_doc: dict[str, list[tuple[int, str, str]]] = {}
    for chunk in index._chunks.values():  # read-only view of the loaded index
        by_doc.setdefault(chunk.document_id, []).append(
            (int(chunk.metadata["ordinal"]), chunk.chunk_id, body(chunk.text))
        )
    return {d: [(cid, text) for _, cid, text in sorted(rows)] for d, rows in by_doc.items()}


def locate(quote: str, chunks: list[tuple[str, str]], widest: int = 3) -> list[list[str]] | None:
    """Every minimal run containing the quote (a passage can repeat, e.g. a synopsis)."""
    needle = builder.without_table_markup(quote)
    for width in range(1, widest + 1):
        runs = []
        for start in range(len(chunks) - width + 1):
            run = chunks[start : start + width]
            text = run[0][1]
            for _, nxt in run[1:]:
                text = merge(text, nxt)
            if needle in text:
                runs.append([cid for cid, _ in run])
        if runs:
            return runs
    return None


def documents(version: str) -> dict[str, str]:
    """Draft document name -> document id of the version's parse."""
    root, names = corpus.files(version)
    return {name: builder.sha256(root / name) for name in names}


def evidence(version: str) -> dict[str, list[list[list[str]] | None]]:
    """Case id -> per quote, its covering chunk runs (None: quote not located)."""
    drafts = load_drafts(builder.DRAFTS)
    if version == "v2":
        drafts = builder.v2_documents(drafts)
    doc_ids = documents(version)
    chunks = ordered_chunks(version)
    return {
        d.id: [locate(q, chunks[doc_ids[d.document]]) for q in d.quotes] for d in drafts
    }


def search_rows(version: str) -> dict[str, dict[str, Any]]:
    index = retriever(version)
    threshold = index.settings.rerank_score_threshold
    rows = {}
    for case in data.dataset("v3").cases:
        hits = index.search(case.question, top_k=CANDIDATES)
        rows[case.id] = {
            "ids": [h.chunk.chunk_id for h in hits],
            "scores": [round(float(h.rerank_score or 0.0), 4) for h in hits],
            "gate": bool(hits) and float(hits[0].rerank_score or 0.0) >= threshold,
            "kinds": [sorted(h.chunk.metadata.get("kinds", ())) for h in hits[:10]],
            "chars": sum(len(h.chunk.text) for h in hits[:10]),
        }
    return rows


def score(quotes: list[list[list[str]] | None], ids: list[str]) -> dict[str, Any]:
    """A quote's rank is that of its best run: the position of the run's last chunk."""
    position = {cid: i + 1 for i, cid in enumerate(ids)}
    ranks = [
        min(
            (max(position[c] for c in run) for run in runs if all(c in position for c in run)),
            default=None,
        )
        if runs
        else None
        for runs in quotes
    ]
    found: dict[str, Any] = {}
    for k in KS:
        hits = [rank is not None and rank <= k for rank in ranks]
        found[f"recall@{k}"] = sum(hits) / len(hits) if hits else None
        found[f"hit@{k}"] = all(hits) if hits else None
    found["rank"] = (
        None if None in ranks or not ranks else max(r for r in ranks if r is not None)
    )
    return found


def compare() -> dict[str, Any]:
    splits = data.split()
    cases = {c.id: c for c in data.dataset("v3").cases}
    table_gold = {}
    v1_chunks = {cid: c for cid, c in retriever("v1")._chunks.items()}
    out: dict[str, Any] = {"questions": {}}
    per_version = {}
    for version in ("v1", "v2"):
        per_version[version] = (evidence(version), search_rows(version))
    for cid, case in cases.items():
        table_gold[cid] = any(
            "table" in v1_chunks[g].metadata.get("kinds", ()) for g in case.gold_chunk_ids
        )
        row: dict[str, Any] = {
            "split": splits[cid],
            "category": case.category,
            "subtype": case.subtype,
            "answerable": case.answerable,
            "table_gold": table_gold[cid],
        }
        for version, (runs, searches) in per_version.items():
            s = searches[cid]
            row[version] = {
                "gate": s["gate"],
                "top_score": s["scores"][0] if s["scores"] else None,
                "quotes": len(runs[cid]),
                "unlocated": sum(r is None for r in runs[cid]),
                "runs": runs[cid],
                **score(runs[cid], s["ids"]),
                "top10_tables": sum("table" in k for k in s["kinds"]),
                "top10_chars": s["chars"],
            }
        out["questions"][cid] = row
    out["summary"] = summarize(out["questions"])
    return out


def summarize(questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Callable[[dict[str, Any]], bool]] = {
        "answerable": lambda r: r["answerable"],
        "answerable, table gold": lambda r: r["answerable"] and r["table_gold"],
        "answerable, text gold": lambda r: r["answerable"] and not r["table_gold"],
        "answerable, dev": lambda r: r["answerable"] and r["split"] == "dev",
        "answerable, test": lambda r: r["answerable"] and r["split"] == "test",
    }
    summary: dict[str, Any] = {}
    for name, keep in groups.items():
        rows = [r for r in questions.values() if keep(r)]
        block: dict[str, Any] = {"n": len(rows)}
        for version in ("v1", "v2"):
            block[version] = {
                metric: round(sum(float(r[version][metric]) for r in rows) / len(rows), 3)
                for metric in ("recall@1", "recall@5", "recall@10", "hit@10")
            }
        block["hit@10 gained"] = sorted(
            cid for cid, r in questions.items() if keep(r) and r["v2"]["hit@10"] and not r["v1"]["hit@10"]
        )
        block["hit@10 lost"] = sorted(
            cid for cid, r in questions.items() if keep(r) and r["v1"]["hit@10"] and not r["v2"]["hit@10"]
        )
        summary[name] = block
    unanswerable = [r for r in questions.values() if not r["answerable"]]
    summary["unanswerable gate passes"] = {
        "n": len(unanswerable),
        "v1": sum(r["v1"]["gate"] for r in unanswerable),
        "v2": sum(r["v2"]["gate"] for r in unanswerable),
    }
    answerable = [r for r in questions.values() if r["answerable"]]
    summary["answerable gate rejects"] = {
        "v1": sum(not r["v1"]["gate"] for r in answerable),
        "v2": sum(not r["v2"]["gate"] for r in answerable),
    }
    every = list(questions.values())
    summary["top10 chars, median"] = {
        v: sorted(r[v]["top10_chars"] for r in every)[len(every) // 2] for v in ("v1", "v2")
    }
    summary["top10 chars, max"] = {v: max(r[v]["top10_chars"] for r in every) for v in ("v1", "v2")}
    summary["unlocated quotes"] = {
        v: sum(r[v]["unlocated"] for r in questions.values()) for v in ("v1", "v2")
    }
    return summary


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "build" and argv[1] in {"v1", "v2"}:
        print(f"indexed {build(argv[1])} chunks")
        return 0
    if argv == ["compare"]:
        result = compare()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "retrieval_v1_v2.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(json.dumps(result["summary"], ensure_ascii=False, indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
