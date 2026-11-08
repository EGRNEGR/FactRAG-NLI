"""PHASE 5b: recovery after the thinking pass also ended in an LLM refusal (llm_recovery).

Population: cases whose replayed hybrid (router A_withheld) ends with refusal_reason
llm_refusal and whose best source has a reranker score >= 0.05, under verifier v2 or the
decision layer - answerable cases (possible gains) and unanswerable ones (controls: a
published answer there is a new hallucination).

`generate` drafts, on a local llama-server with the same GGUF, the two recovery modes of
pipeline._recover with the pipeline's exact prompts (generation_system_prompt + the
recovery_note for "evidence_prompt"; per-call temperature and seed for "sampling"), and
stores them in .artifacts/opt2/recovery.jsonl. `apply` adds them to replayed results; the
trigger threshold llm_recovery_min_rerank is chosen on dev.

Usage:  python -m research.opt2.recovery --api-base http://127.0.0.1:8093/v1
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from generator_verifier import (
    NLIFactVerifier,
    RawGenerationResult,
    _HTTPBackend,
    generation_system_prompt,
    recovery_note,
)

from research.opt2 import data
from research.opt2.repairs import question_of

STORE = data.CACHE / "recovery.jsonl"
MODES: dict[str, list[dict[str, Any]]] = {
    "evidence_prompt": [{"note": True}],
    "sampling": [{"temperature": 0.7, "seed": 1000}, {"temperature": 0.7, "seed": 1001}],
}


def top_rerank(record: data.CaseRecord) -> float:
    return max(
        (s.get("rerank_score") or 0.0 for s in record.retrieval.get("sources") or []), default=0.0
    )


def top_sources(record: data.CaseRecord, n: int = 2) -> list[str]:
    sources = record.retrieval.get("sources") or []
    ranked = sorted(sources, key=lambda s: -(s.get("rerank_score") or 0.0))
    return [s["source_id"] for s in ranked[:n]]


def population(min_rerank: float = 0.05) -> list[str]:
    from research.opt2.replay import on_withheld, run
    from research.opt2.routing import judge_verifier

    records = data.records(("base", "think"))
    ids: set[str] = set()
    for name in ("none", "bge-m3-zeroshot"):
        for c in run(records, judge_verifier(name), on_withheld):
            record = records[c.outcome["id"]]
            if c.final.reason == "llm_refusal" and top_rerank(record) >= min_rerank:
                ids.add(record.case.id)
    return sorted(ids)


def load_store() -> dict[tuple[str, str, int], dict[str, Any]]:
    if not STORE.exists():
        return {}
    rows = (json.loads(line) for line in STORE.read_text(encoding="utf-8").splitlines() if line)
    return {(r["id"], r["mode"], r["index"]): r for r in rows}


def prompt(question: str, contexts: dict[str, Any]) -> str:
    from research.opt2.repairs import prompt as with_revision

    payload = json.loads(with_revision(question, {}, contexts))
    payload.pop("revision")
    return json.dumps(payload, ensure_ascii=False)


def generate(api_base: str) -> None:
    from research.opt2.replay import settings

    s = settings(
        llm_api_base=api_base, llm_api_timeout=3600, llm_disable_thinking=True, max_tokens=512
    )
    backend = _HTTPBackend(s)
    system = generation_system_prompt(s)
    store = load_store()
    records = data.records(("base", "think"))
    ids = population()
    print(f"{len(ids)} cases", flush=True)
    with STORE.open("a", encoding="utf-8") as handle:
        for case_id in ids:
            record = records[case_id]
            contexts = record.drafts["base"][0].contexts
            user = prompt(question_of(record, "base"), contexts)
            for mode, plans in MODES.items():
                for index, plan in enumerate(plans):
                    if (case_id, mode, index) in store:
                        continue
                    started = time.perf_counter()
                    if plan.get("note"):
                        reply = backend.complete(
                            system + "\n" + recovery_note(top_sources(record)), user
                        )
                    else:
                        reply = backend.complete(system, user, sampling=plan)
                    row = {
                        "id": case_id,
                        "mode": mode,
                        "index": index,
                        "raw_text": reply.text,
                        "finish_reason": reply.finish_reason,
                        "seconds": round(time.perf_counter() - started, 1),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        case_id,
                        mode,
                        index,
                        row["seconds"],
                        reply.text[:90].replace("\n", " "),
                        flush=True,
                    )


def recovered(
    v: NLIFactVerifier,
    record: data.CaseRecord,
    mode: str,
    store: dict[tuple[str, str, int], dict[str, Any]],
) -> str | None:
    """Published text of the first accepted recovery draft, or None."""
    first = record.drafts["base"][0]
    for index in range(len(MODES[mode])):
        row = store.get((record.case.id, mode, index))
        if row is None:
            return None
        result = v.verify(
            RawGenerationResult.model_validate(
                {
                    "text": row["raw_text"],
                    "contexts": first.contexts,
                    "metadata": {**first.metadata, "finish_reason": row["finish_reason"]},
                }
            )
        )
        if not result.refused:
            return result.cleaned_text
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    generate(parser.parse_args().api_base)
