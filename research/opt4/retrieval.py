"""Retrieval experiments of optimization_4 (no LLM): run on the analysis laptop.

The retriever is the run's (settings of 1310Z config.json per dataset, query encoder on the
CPU as in the judge column, reranker on CUDA) over the local copies of the run's indexes.
`reproduce` first checks that the original questions get the logged top-10 again.

  E3 (H3, translation)  EN XQuAD questions retrieved with their professional RU translation
                        (the parallel XQuAD question, an upper bound of an LLM translation):
                        gold rank, gate decision on answerable and unanswerable questions.
  E4 (H4, compound)     v3 compound questions split into parts by a rule (coordinated
                        interrogative clauses); the top-3 chunks of every part are added to
                        the logged context: evidence units covered, context growth.
  E5 (H5, neighbours)   a context chunk that ends inside a sentence gets the next chunk of its
                        document added: evidence units covered, context growth.

Usage:  python -m research.opt4.retrieval reproduce|E3|E4|E5
"""

from __future__ import annotations

import json
import re
import sys
from functools import cache
from pathlib import Path
from collections.abc import Callable
from typing import Any

import research.opt4  # noqa: F401 - selects the baseline run
from hybrid_retriever import HybridRetriever
from settings import RAGSettings

from research.opt3 import data
from research.opt4.diagnose import OUT

INDEX_ROOT = Path(
    __import__("os").environ.get(
        "OPT4_INDEX_ROOT", r"C:\Hakatons\FactRAG-NLI-3090\.claude\worktrees\opt2\.artifacts"
    )
)
INDEX = {"v3": "eval-v3-index", "xquad-full": "xquad-index"}
CONFIG = "judge"
OUT_DIR = OUT / "experiments"


def settings_for(dataset: str) -> RAGSettings:
    logged = dict(data.run_config(CONFIG, dataset)["modes"]["e2e_guard_on"]["settings"])
    for key, value in logged.items():
        if isinstance(value, str) and value.startswith(data.RUN_ROOT):
            logged[key] = str(data.ROOT / value[len(data.RUN_ROOT) + 1 :])
    logged.update(
        mode="development",
        allow_fallback=True,
        qdrant_path=str(INDEX_ROOT / INDEX[dataset]),
        embedding_device="cpu",
    )
    # _env_file is a pydantic-settings init option mypy does not see among the fields.
    no_env_file: dict[str, Any] = {"_env_file": None}
    return RAGSettings(**no_env_file, **logged)


@cache
def retriever(dataset: str) -> HybridRetriever:
    return HybridRetriever(settings=settings_for(dataset))


def search(dataset: str, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    results = retriever(dataset).search(question, top_k=top_k)
    return [
        {"chunk_id": r.chunk.chunk_id, "rerank": float(r.rerank_score or 0.0)} for r in results
    ]


def logged_ids(record: data.CaseRecord) -> list[str]:
    return [s["chunk_id"] for s in record.retrieval[CONFIG]["sources"]]


def units(record: data.CaseRecord) -> list[set[str]]:
    return [set(e.chunk_ids) for e in record.case.evidence]


def covered(record: data.CaseRecord, ids: list[str]) -> int:
    present = set(ids)
    return sum(bool(unit & present) for unit in units(record))


def dump(name: str, result: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=1))


# ----------------------------------------------------------------------------- reproduce


def reproduce(sample: int = 40) -> dict[str, Any]:
    """Original questions against the logged top-10 (same order), per dataset."""
    rows: list[dict[str, Any]] = []
    for dataset in data.DATASETS:
        records = [r for r in data.records().values() if data.dataset_of(r.id) == dataset]
        for record in records[:sample]:
            if not logged_ids(record):
                continue
            got = [h["chunk_id"] for h in search(dataset, record.retrieval[CONFIG]["query"])]
            rows.append({"id": record.id, "same": got == logged_ids(record),
                         "same_set": set(got) == set(logged_ids(record))})
    summary = {
        "checked": len(rows),
        "identical_order": sum(r["same"] for r in rows),
        "identical_set": sum(r["same_set"] for r in rows),
    }
    result = {"summary": summary, "rows": rows}
    dump("retrieval_reproduce", result)
    return result


# ----------------------------------------------------------------------------- E3


@cache
def parallel_ru() -> dict[str, str]:
    """EN question text -> the RU question of the same XQuAD id."""
    import pyarrow.parquet as pq

    raw = data.ROOT / "data" / "xquad" / "raw"
    ru = {r["id"]: r["question"] for r in pq.read_table(raw / "xquad.ru.parquet").to_pylist()}
    en = pq.read_table(raw / "xquad.en.parquet").to_pylist()
    return {r["question"].strip(): ru[r["id"]] for r in en if r["id"] in ru}


def gold_rank(record: data.CaseRecord, ids: list[str]) -> int | None:
    gold = set(record.case.gold_chunk_ids)
    return next((i for i, cid in enumerate(ids, 1) if cid in gold), None)


def e3() -> dict[str, Any]:
    threshold = settings_for("xquad-full").rerank_score_threshold
    rows: list[dict[str, Any]] = []
    for record in data.records().values():
        if data.dataset_of(record.id) != "xquad-full" or record.case.language != "en":
            continue
        translated = parallel_ru().get(record.case.question.strip())
        if translated is None:
            continue
        # Both sides retrieved locally: the laptop reranker differs from the 3090 by ~1e-4
        # (fp16 on another GPU), which reorders near-ties; local vs local cancels that.
        hits = search("xquad-full", translated)
        original = search("xquad-full", record.case.question)
        top_before = max((h["rerank"] for h in original), default=0.0)
        top_after = max((h["rerank"] for h in hits), default=0.0)
        ids = [h["chunk_id"] for h in hits]
        ids_before = [h["chunk_id"] for h in original]
        rows.append({
            "id": record.id,
            "answerable": record.case.expected_behavior == "answer",
            "question": record.case.question,
            "translated": translated,
            "gate_before": top_before >= threshold,
            "gate_after": top_after >= threshold,
            "top_before": round(top_before, 4),
            "top_after": round(top_after, 4),
            "gold_rank_before": gold_rank(record, ids_before),
            "gold_rank_after": gold_rank(record, ids),
            "class": record.rows[CONFIG]["refusal_reason"],
        })
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    summary = {
        "en_answerable": len(ans),
        "en_unanswerable": len(una),
        "gate_passes_answerable": {"before": sum(r["gate_before"] for r in ans),
                                   "after": sum(r["gate_after"] for r in ans)},
        "gate_passes_unanswerable": {"before": sum(r["gate_before"] for r in una),
                                     "after": sum(r["gate_after"] for r in una)},
        "gold_at_1": {"before": sum(r["gold_rank_before"] == 1 for r in ans),
                      "after": sum(r["gold_rank_after"] == 1 for r in ans)},
        "gold_in_top10": {"before": sum(r["gold_rank_before"] is not None for r in ans),
                          "after": sum(r["gold_rank_after"] is not None for r in ans)},
        "gate_refused_answerable_recovered": [
            r["id"] for r in ans if not r["gate_before"] and r["gate_after"]
        ],
        "unanswerable_newly_passing": [r["id"] for r in una if not r["gate_before"] and r["gate_after"]],
        "median_top_rerank_answerable": {
            "before": sorted(r["top_before"] for r in ans)[len(ans) // 2],
            "after": sorted(r["top_after"] for r in ans)[len(ans) // 2],
        },
    }
    result = {"summary": summary, "rows": rows}
    dump("E3_translation_retrieval", result)
    return result


# ----------------------------------------------------------------------------- E4

# Coordinated interrogative clauses: "..., and how ...", "... и что ...", "... и допускает ли ...".
_SPLIT = re.compile(
    r"(?:,\s*|\s+)(?:and|и|а также)\s+(?=(?:what|which|how|whether|does|do|is|are|can|when|why|who|"
    r"что|как|какой|какая|какое|какие|каков|какова|каковы|когда|где|кто|чем|сколько|"
    r"\w+\s+ли\b|ли\b))",
    re.I,
)


def parts(question: str) -> list[str]:
    pieces = [p.strip(" ,?") for p in _SPLIT.split(question) if p.strip(" ,?")]
    return [p + "?" for p in pieces] if len(pieces) > 1 else []


def e4(per_part: int = 3) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in data.records().values():
        if data.dataset_of(record.id) != "v3" or record.case.expected_behavior != "answer":
            continue
        split = parts(record.case.question)
        if not split or len(units(record)) < 2:
            continue
        base = logged_ids(record)
        added: list[str] = []
        for part in split:
            for hit in search("v3", part)[:per_part]:
                if hit["chunk_id"] not in base and hit["chunk_id"] not in added:
                    added.append(hit["chunk_id"])
        rows.append({
            "id": record.id,
            "question": record.case.question,
            "parts": split,
            "units": len(units(record)),
            "covered_before": covered(record, base),
            "covered_after": covered(record, base + added),
            "added_chunks": len(added),
            "correct": bool(record.rows[CONFIG]["factrag"]["correct"]),
        })
    summary = {
        "compound_questions_split": len(rows),
        "all_units_before": sum(r["covered_before"] == r["units"] for r in rows),
        "all_units_after": sum(r["covered_after"] == r["units"] for r in rows),
        "gained_unit": [r["id"] for r in rows if r["covered_after"] > r["covered_before"]],
        "gained_unit_on_wrong": [
            r["id"] for r in rows if r["covered_after"] > r["covered_before"] and not r["correct"]
        ],
        "mean_added_chunks": round(sum(r["added_chunks"] for r in rows) / max(len(rows), 1), 2),
    }
    result = {"summary": summary, "rows": rows}
    dump("E4_subquestion_retrieval", result)
    return result


# ----------------------------------------------------------------------------- E5

_END = re.compile(r"[.!?:;»\")\]|]\s*$")


def ends_inside_sentence(text: str) -> bool:
    return not _END.search(text.rstrip())


@cache
def by_ordinal(dataset: str) -> dict[tuple[str, int], str]:
    index = retriever(dataset)
    table: dict[tuple[str, int], str] = {}
    for chunk in index._chunks.values():  # read-only view of the loaded index
        ordinal = chunk.metadata.get("ordinal")
        if ordinal is not None:
            table[(chunk.document_id, int(ordinal))] = chunk.chunk_id
    return table


def e5() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in data.records().values():
        dataset = data.dataset_of(record.id)
        if record.case.expected_behavior != "answer" or not record.retrieval[CONFIG]["sources"]:
            continue
        sources = record.retrieval[CONFIG]["sources"]
        base = [s["chunk_id"] for s in sources]
        added = []
        for s in sources:
            if not ends_inside_sentence(s["text"]):
                continue
            ordinal = s["metadata"].get("ordinal")
            nxt = by_ordinal(dataset).get((s["document_id"], int(ordinal) + 1)) if ordinal is not None else None
            if nxt and nxt not in base and nxt not in added:
                added.append(nxt)
        if not added:
            continue
        rows.append({
            "id": record.id,
            "dataset": dataset,
            "units": len(units(record)),
            "covered_before": covered(record, base),
            "covered_after": covered(record, base + added),
            "added_chunks": len(added),
            "correct": bool(record.rows[CONFIG]["factrag"]["correct"]),
        })
    summary = {
        "answerable_with_cut_chunks": len(rows),
        "gained_unit": [r["id"] for r in rows if r["covered_after"] > r["covered_before"]],
        "gained_unit_on_wrong": [
            r["id"] for r in rows if r["covered_after"] > r["covered_before"] and not r["correct"]
        ],
        "mean_added_chunks": round(sum(r["added_chunks"] for r in rows) / max(len(rows), 1), 2),
    }
    result = {"summary": summary, "rows": rows}
    dump("E5_neighbour_chunks", result)
    return result


if __name__ == "__main__":
    commands: dict[str, Callable[[], dict[str, Any]]] = {
        "reproduce": reproduce, "E3": e3, "E4": e4, "E5": e5
    }
    commands[sys.argv[1]]()
