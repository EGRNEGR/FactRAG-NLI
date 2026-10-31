"""Generate the repair drafts a verifier variant needs but run 0508Z never produced.

When a variant withholds a draft the run had published, the live pipeline would ask the LLM
for one repair (generation_repair_attempts=1) with the rejected claims in the prompt. The
replay cannot invent that draft; this module generates it with a local llama-server running
the same GGUF (temperature 0, seed 42, the pipeline's exact system prompt and JSON prompt,
the logged contexts, thinking as in the configuration). Drafts are stored in
.artifacts/opt2/repairs.jsonl keyed by case, configuration, attempt and the revision, and
`replay.replay_config` uses them instead of counting the case as withheld.

Usage:  python -m research.opt2.repairs --api-base http://127.0.0.1:8093/v1 --judge bge-m3-zeroshot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from typing import Any

from generator_verifier import (
    ClaimStatus,
    SourceContext,
    VerifiedGenerationResult,
    _HTTPBackend,
    generation_system_prompt,
)

from research.opt2 import data

STORE = data.CACHE / "repairs.jsonl"


def revision_of(verified: VerifiedGenerationResult) -> dict[str, Any]:
    """The revision block pipeline._draft sends with a repair request."""
    return {
        "previous_answer": verified.raw_text,
        "rejected_claims": [
            {"text": claim.claim, "status": claim.status.value}
            for claim in verified.claims
            if claim.status != ClaimStatus.VERIFIED
        ],
    }


def key(case_id: str, config: str, attempt: int, revision: dict[str, Any]) -> str:
    blob = json.dumps([case_id, config, attempt, revision], ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def load_store() -> dict[str, dict[str, Any]]:
    if not STORE.exists():
        return {}
    rows = (json.loads(line) for line in STORE.read_text(encoding="utf-8").splitlines() if line)
    return {row["key"]: row for row in rows}


def prompt(question: str, revision: dict[str, Any], contexts: dict[str, Any]) -> str:
    """Byte-identical to LocalLLMGenerator.generate's prompt for the logged contexts."""
    return json.dumps(
        {
            "question": question.strip(),
            "revision": revision,
            "sources": {
                k: SourceContext.model_validate(v).model_dump(
                    mode="json", exclude={"document_title"}
                )
                for k, v in contexts.items()
            },
        },
        ensure_ascii=False,
    )


def generate(api_base: str, judge: str) -> None:
    from research.opt2.replay import replay_config, settings
    from research.opt2.routing import judge_verifier

    v = judge_verifier(judge)
    s = settings(
        llm_api_base=api_base,
        llm_api_timeout=3600,
        llm_disable_thinking=True,
        llm_think_retry="refusal",
        llm_think_max_tokens=1024,
        max_tokens=512,
        temperature=0.0,
        top_p=0.9,
        llm_seed=42,
    )
    backend = _HTTPBackend(s)
    system = generation_system_prompt(s)
    records = data.records(("base", "think"))

    def missing(relevant_only: bool) -> list[tuple[data.CaseRecord, str, Any]]:
        """Missing repairs; relevant = non-thinking ones and thinking ones the router uses."""
        from research.opt2 import replay
        from research.opt2.replay import on_withheld

        replay._STORE = None  # re-read the drafts generated so far
        found = []
        for record in records.values():
            base = replay_config(record, "base", v)
            if base.missing_repair:
                found.append((record, "base", base))
            routed = on_withheld(record, base)
            if relevant_only and (base.missing_repair or not routed):
                continue  # a thinking repair matters once its routing is known
            think = replay_config(record, "think", v)
            if think.missing_repair:
                found.append((record, "think", think))
        return found

    with STORE.open("a", encoding="utf-8") as handle:
        for relevant_only in (True, True, True, False):
            todo = missing(relevant_only)
            print(f"{len(todo)} repair drafts needed (relevant only: {relevant_only})", flush=True)
            if not todo:
                continue
            for record, config, replayed in todo:
                last = replayed.drafts[-1]
                revision = revision_of(last)
                k = key(record.case.id, config, len(replayed.drafts), revision)
                first = record.drafts[config][0]
                started = time.perf_counter()
                reply = backend.complete(
                    system,
                    prompt(question_of(record, config), revision, first.contexts),
                    thinking=config == "think",
                )
                row = {
                    "key": k,
                    "id": record.case.id,
                    "config": config,
                    "attempt": len(replayed.drafts),
                    "revision": revision,
                    "raw_text": reply.text,
                    "finish_reason": reply.finish_reason,
                    "seconds": round(time.perf_counter() - started, 1),
                    "api_base": api_base,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    record.case.id,
                    config,
                    row["seconds"],
                    reply.text[:100].replace("\n", " "),
                    flush=True,
                )


def question_of(record: data.CaseRecord, config: str) -> str:
    """The query string the run sent (logged in the full response)."""
    return data.responses(config)[record.case.id]["query"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--judge", default="bge-m3-zeroshot")
    args = parser.parse_args()
    generate(args.api_base, args.judge)
