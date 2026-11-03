"""Recovery after a persistent LLM refusal: why it publishes almost nothing, and a fix.

Run 0749Z: the recovery pass (2 samples at T = 0.7, same prompt) ran for 47 questions with
the judge and published 3; on XQuAD 44 of the 52 persistent LLM refusals have the gold chunk
ranked first with the answer in it. The refused questions are loose machine translations
("What company developed ..." -> a person; "won the division" -> a divisional round), and the
premise rule of the system prompt ("do not take the premise of the question as a proven
fact ... otherwise return the refusal") makes the model refuse them in every mode.

`generate` drafts, on a local llama-server with the run's GGUF, the recovery variants below
for every question whose replayed result (current code, judge) ends in an LLM refusal - the
answerable ones (possible gains) and the unanswerable/adversarial ones (controls: a
published answer there is a new hallucination). Same contexts, question and JSON prompt as
the run (LocalLLMGenerator.generate), non-thinking. Drafts go to .artifacts/opt3/recovery.jsonl.

  relaxed:          the system prompt without the premise rule (premise_guard=false), T = 0
  relaxed_sampling: the same prompt, T = 0.7, seeds 1000 and 1001 (the run's sampling plan)

`apply` replays the run with a variant's drafts in place of the logged recovery samples.

Usage:  python -m research.opt3.recovery generate --api-base http://127.0.0.1:8093/v1
        python -m research.opt3.recovery evaluate
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from generator_verifier import _HTTPBackend, generation_system_prompt

from research.opt3 import data

STORE = data.CACHE / "recovery.jsonl"
MODES: dict[str, dict[str, Any]] = {
    "relaxed": {"premise_guard": False, "plans": [None]},
    "relaxed_sampling": {
        "premise_guard": False,
        "plans": [{"temperature": 0.7, "seed": 1000}, {"temperature": 0.7, "seed": 1001}],
    },
}


def load_store() -> dict[tuple[str, str, int], dict[str, Any]]:
    if not STORE.exists():
        return {}
    rows = (json.loads(line) for line in STORE.read_text(encoding="utf-8").splitlines() if line)
    return {(r["id"], r["mode"], r["index"]): r for r in rows}


def prompt(question: str, contexts: dict[str, Any]) -> str:
    """Byte-identical to LocalLLMGenerator.generate's prompt for the logged contexts."""
    return json.dumps(
        {
            "question": question.strip(),
            "sources": {
                key: {k: v for k, v in value.items() if k != "document_title"}
                for key, value in contexts.items()
            },
        },
        ensure_ascii=False,
    )


def population() -> list[str]:
    """Questions whose replayed judge result (current code) ends in an LLM refusal."""
    from research.opt3.replay import replay, verifier

    v = verifier("judge")
    return sorted(
        record.id
        for record in data.records().values()
        if replay(record, v, source="judge", recovery=False).reason == "llm_refusal"
    )


def generate(api_base: str, modes: list[str]) -> None:
    records = data.records()
    store = load_store()
    ids = population()
    print(f"{len(ids)} questions", flush=True)
    STORE.parent.mkdir(parents=True, exist_ok=True)
    with STORE.open("a", encoding="utf-8") as handle:
        for mode in modes:
            spec = MODES[mode]
            s = data.settings(
                "judge",
                llm_api_base=api_base,
                llm_api_timeout=3600,
                premise_guard=spec["premise_guard"],
            )
            backend = _HTTPBackend(s)
            system = generation_system_prompt(s)
            for case_id in ids:
                record = records[case_id]
                first = record.attempts["judge"][0]
                user = prompt(record.retrieval["judge"]["query"], first.contexts)
                for index, plan in enumerate(spec["plans"]):
                    if (case_id, mode, index) in store:
                        continue
                    started = time.perf_counter()
                    reply = backend.complete(
                        system, user, thinking=False, **({"sampling": plan} if plan else {})
                    )
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
                    print(case_id, mode, index, row["seconds"], reply.text[:80].replace("\n", " "),
                          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "population"])
    parser.add_argument("--api-base", default="http://127.0.0.1:8093/v1")
    parser.add_argument("--modes", nargs="+", default=["relaxed"])
    args = parser.parse_args()
    if args.action == "generate":
        generate(args.api_base, args.modes)
    else:
        ids = population()
        print(len(ids), ids)
