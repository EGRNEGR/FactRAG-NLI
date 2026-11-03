"""Per-question dossiers of the problem cases of run 20261103T1310Z (judge, code f083e2d).

Sets:
  think_on_trim    every case where llm_think_on_trim ran (a published, trimmed first answer
                   took a thinking pass): fixed, already correct, still wrong
  xquad_refusals   XQuAD answerable questions where every logged attempt is the canonical LLM
                   refusal (class llm_refusal_only)
  v3_errors        the other v3 errors (context for the hypotheses, not a target set)

For each case: question, gold answer and the scorer criteria (answer_must_contain, absence
terms), the retrieved sources with rerank scores and the gold rank, the gold fragments
(evidence quotes) and whether the scorer pattern matches the gold chunk, every attempt
(role, raw and published text, scorer verdict, claims with status, reason, judge score and
the anchors missing from the cited sources). Causes and fixes are hand-written in
research/opt4/annotations.json after reading the dossiers; report.py merges them.

Usage:  python -m research.opt4.diagnose   (writes reports/optimization_4/diagnostics/)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import research.opt4  # noqa: F401 - selects the baseline run
from evaluate_suite import contains_fact
from grounding import anchors_present, claim_anchors, clean_premise

from research.opt3 import cases, data

OUT = data.ROOT / "reports" / "optimization_4"
DIAG = OUT / "diagnostics"
CONFIG = "judge"


def baseline_rows() -> dict[str, dict[str, Any]]:
    """Per-case table of the baseline (research.opt3.cases on the 1310Z logs)."""
    result = {}
    for record in data.records().values():
        c = cases.case_row(record, CONFIG)
        c["class"] = cases.error_class(record, c)
        result[record.id] = c
    return result


def think_on_trim(row: dict[str, Any]) -> bool:
    """A thinking pass that followed a published (trimmed) non-thinking answer."""
    roles = [a["role"] for a in row["attempts"]]
    if "think" not in roles:
        return False
    before = [a for a in row["attempts"][: roles.index("think")]]
    return bool(before) and before[-1]["published"]


def chunk_texts(record: data.CaseRecord) -> dict[str, str]:
    texts: dict[str, str] = {}
    for source in record.retrieval[CONFIG]["sources"]:
        texts[source["chunk_id"]] = source["text"]
    for attempt in record.attempts[CONFIG]:
        for context in attempt.contexts.values():
            texts.setdefault(context["chunk_id"], context["text"])
    return texts


def sentence_with(text: str, quote: str) -> str:
    """The quote as it stands in the chunk, or the chunk sentence sharing most of its words."""
    flat = " ".join(text.split())
    if quote and quote in flat:
        return quote
    words = {w.casefold() for w in re.findall(r"\w{4,}", quote)}
    best, score = "", 0.0
    for sentence in re.split(r"(?<=[.!?])\s+", flat):
        found = {w.casefold() for w in re.findall(r"\w{4,}", sentence)}
        overlap = len(words & found) / max(len(words), 1)
        if overlap > score:
            best, score = sentence, overlap
    return best


def _anchor_text(context: dict[str, Any]) -> str:
    return " ".join([*context.get("section_path", []), clean_premise(context["text"])])


def claim_view(claim: dict[str, Any], contexts: dict[str, Any]) -> dict[str, Any]:
    cited = [k for k in claim.get("source_ids") or [] if k in contexts]
    hypothesis = claim.get("hypothesis") or claim["claim"]
    anchors = sorted(claim_anchors(hypothesis))
    # The run checks anchors in the cited text and its headings (nli_anchor_headings=true).
    cited_text = " ".join(_anchor_text(contexts[k]) for k in cited)
    all_text = " ".join(_anchor_text(c) for c in contexts.values())
    missing = [a for a in anchors if not anchors_present(frozenset({a}), cited_text)]
    return {
        "claim": claim["claim"],
        "status": claim["status"],
        "reason": claim.get("reason"),
        "grade": claim.get("grade"),
        "judge_score": round(max((e.get("judge_score") or 0.0) for e in claim["evidence"]), 3)
        if claim.get("evidence")
        else None,
        "cited": cited,
        "evidence": [
            {
                "source_id": e["source_id"],
                "p_entailment": round(e["probabilities"]["p_entailment"], 3),
                "p_contradiction": round(e["probabilities"]["p_contradiction"], 3),
                "secondary_entailment": round(e["secondary"]["p_entailment"], 3)
                if e.get("secondary")
                else None,
                "judge_score": round(e["judge_score"], 3) if e.get("judge_score") is not None else None,
                "premise": e.get("premise_text"),
                "section_path": contexts[e["source_id"]].get("section_path", [])
                if e["source_id"] in contexts
                else [],
            }
            for e in claim.get("evidence") or []
        ],
        "anchors": anchors,
        "anchors_missing_in_cited": missing,
        "anchors_missing_everywhere": [
            a for a in missing if not anchors_present(frozenset({a}), all_text)
        ],
    }


def dossier(record: data.CaseRecord, row: dict[str, Any]) -> dict[str, Any]:
    case = record.case
    texts = chunk_texts(record)
    sources = record.retrieval[CONFIG]["sources"]
    gold = set(case.gold_chunk_ids)
    ranked = [
        {
            "rank": i,
            "source_id": s["source_id"],
            "chunk_id": s["chunk_id"],
            "rerank": round(s.get("rerank_score") or 0.0, 4),
            "gold": s["chunk_id"] in gold,
        }
        for i, s in enumerate(sources, 1)
    ]
    llm_chunks = {
        c["chunk_id"] for a in record.attempts[CONFIG] for c in a.contexts.values()
    }
    fragments = []
    for evidence in case.evidence:
        for chunk_id in evidence.chunk_ids:
            text = texts.get(chunk_id, "")
            fragments.append({
                "chunk_id": chunk_id,
                "quote": evidence.quote,
                "in_sources": chunk_id in {s["chunk_id"] for s in sources},
                "given_to_llm": chunk_id in llm_chunks,
                "sentence": sentence_with(text, evidence.quote) if text else None,
                "scorer_matches_chunk": bool(text)
                and all(contains_fact(text, f) for f in case.answer_must_contain),
            })
    attempts = []
    for logged, scored in zip(record.attempts[CONFIG], row["attempts"], strict=True):
        attempts.append({
            "role": logged.role,
            "thinking": logged.thinking,
            "llm_refusal": scored["llm_refusal"],
            "raw_text": logged.raw_text,
            "published_text": None if logged.refused else logged.cleaned_text,
            "draft_correct": scored["draft_correct"],
            "published_correct": scored["published_correct"],
            "contexts": list(logged.contexts),
            "completion_tokens": logged.metadata.get("completion_tokens"),
            "generation_ms": round(logged.metadata.get("generation_ms") or 0.0),
            "claims": [claim_view(c, logged.contexts) for c in logged.claims],
        })
    return {
        "id": case.id,
        "dataset": data.dataset_of(case.id),
        "split": record.split,
        "category": case.category,
        "subtype": case.subtype,
        "language": case.language,
        "expected": case.expected_behavior,
        "question": case.question,
        "ground_truth": case.ground_truth_answer,
        "answer_must_contain": case.answer_must_contain,
        "absence_terms": case.absence_terms,
        "document": case.document,
        "gold_ranks": [r["rank"] for r in ranked if r["gold"]],
        "top_rerank": round(record.top_rerank(CONFIG), 4),
        "refusal_reason": row["reason"],
        "sources": ranked,
        "fragments": fragments,
        "attempts": attempts,
        "path": row["path"],
        "class": row["class"],
        "correct": row["correct"],
        "answer": row["answer"],
        "total_ms": round(row["total_ms"]),
    }


def previous_correct() -> dict[str, bool]:
    """Correctness of the judge column of 0749Z (optimization_3 case table)."""
    path = data.ROOT / "reports" / "optimization_3" / "error_analysis" / "cases.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return {r["id"]: bool(r["judge"]["correct"]) for r in rows}


def build() -> dict[str, list[dict[str, Any]]]:
    records = data.records()
    rows = baseline_rows()
    before = previous_correct()
    sets: dict[str, list[dict[str, Any]]] = {"think_on_trim": [], "xquad_refusals": [], "v3_errors": []}
    for cid, row in rows.items():
        record = records[cid]
        if think_on_trim(row):
            d = dossier(record, row)
            d["outcome_0749Z"] = before[cid]
            d["think_outcome"] = (
                "fixed" if row["correct"] and not before[cid]
                else "already_correct" if row["correct"]
                else "still_wrong"
            )
            sets["think_on_trim"].append(d)
        elif row["class"] == "llm_refusal_only" and data.dataset_of(cid) == "xquad-full":
            sets["xquad_refusals"].append(dossier(record, row))
        elif not row["correct"] and data.dataset_of(cid) == "v3":
            sets["v3_errors"].append(dossier(record, row))
    return sets


def write(sets: dict[str, list[dict[str, Any]]]) -> None:
    DIAG.mkdir(parents=True, exist_ok=True)
    for name, items in sets.items():
        with (DIAG / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        name: {
            "cases": len(items),
            "classes": dict(Counter(i["class"] for i in items)),
            **(
                {"think_outcome": dict(Counter(i["think_outcome"] for i in items))}
                if name == "think_on_trim"
                else {}
            ),
        }
        for name, items in sets.items()
    }
    (DIAG / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def load(name: str) -> list[dict[str, Any]]:
    path = DIAG / f"{name}.jsonl"
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


if __name__ == "__main__":
    write(build())
