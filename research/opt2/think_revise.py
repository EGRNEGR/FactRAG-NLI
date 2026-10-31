"""PHASE 5c: the thinking pass as a targeted revision (llm_think_mode=revise).

In run 0508Z the thinking pass drafted every withheld answer anew. For an answer withheld by
NLI, the "revise" mode instead sends the withheld answer and its rejected claims (the repair
revision block) to the thinking call: the model keeps what was verified and fixes only the
rest. This module generates those thinking revisions with a local llama-server (same GGUF,
the pipeline's exact prompt, thinking on) for every case whose replayed non-thinking result
ends in nli_verification_failed under verifier v2 or the decision layer, and stores them in
.artifacts/opt2/think_revise.jsonl; `revised` replays them.

Usage:  python -m research.opt2.think_revise --api-base http://127.0.0.1:8093/v1
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from generator_verifier import (
    NLIFactVerifier,
    RawGenerationResult,
    VerifiedGenerationResult,
    _HTTPBackend,
    generation_system_prompt,
)

from research.opt2 import data
from research.opt2.repairs import key, prompt, question_of, revision_of

STORE = data.CACHE / "think_revise.jsonl"


def load_store() -> dict[str, dict[str, Any]]:
    if not STORE.exists():
        return {}
    rows = (json.loads(line) for line in STORE.read_text(encoding="utf-8").splitlines() if line)
    return {row["key"]: row for row in rows}


def needed() -> list[tuple[data.CaseRecord, VerifiedGenerationResult]]:
    from research.opt2.replay import replay_config
    from research.opt2.routing import judge_verifier

    records = data.records(("base", "think"))
    seen: dict[str, tuple[data.CaseRecord, VerifiedGenerationResult]] = {}
    for name in ("none", "bge-m3-zeroshot"):
        v = judge_verifier(name)
        for record in records.values():
            base = replay_config(record, "base", v)
            if base.reason == "nli_verification_failed" and base.verified and base.verified.claims:
                k = key(record.case.id, "think_revise", 0, revision_of(base.verified))
                seen.setdefault(k, (record, base.verified))
    return list(seen.values())


def generate(api_base: str) -> None:
    from research.opt2.replay import settings

    s = settings(
        llm_api_base=api_base,
        llm_api_timeout=3600,
        llm_disable_thinking=True,
        llm_think_retry="refusal",
        llm_think_max_tokens=1024,
        max_tokens=512,
    )
    backend = _HTTPBackend(s)
    system = generation_system_prompt(s)
    store = load_store()
    todo = needed()
    print(f"{len(todo)} thinking revisions", flush=True)
    with STORE.open("a", encoding="utf-8") as handle:
        for record, verified in todo:
            revision = revision_of(verified)
            k = key(record.case.id, "think_revise", 0, revision)
            if k in store:
                continue
            started = time.perf_counter()
            first = record.drafts["base"][0]
            reply = backend.complete(
                system, prompt(question_of(record, "base"), revision, first.contexts), thinking=True
            )
            row = {
                "key": k,
                "id": record.case.id,
                "revision": revision,
                "raw_text": reply.text,
                "finish_reason": reply.finish_reason,
                "seconds": round(time.perf_counter() - started, 1),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(record.case.id, row["seconds"], reply.text[:100].replace("\n", " "), flush=True)


def revised(
    v: NLIFactVerifier,
    record: data.CaseRecord,
    base: VerifiedGenerationResult,
    store: dict[str, dict[str, Any]],
) -> VerifiedGenerationResult | None:
    """The verified thinking revision of `base`, or None if it was not generated."""
    row = store.get(key(record.case.id, "think_revise", 0, revision_of(base)))
    if row is None:
        return None
    first = record.drafts["base"][0]
    return v.verify(
        RawGenerationResult.model_validate(
            {
                "text": row["raw_text"],
                "contexts": first.contexts,
                "metadata": {**first.metadata, "finish_reason": row["finish_reason"]},
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    generate(parser.parse_args().api_base)
