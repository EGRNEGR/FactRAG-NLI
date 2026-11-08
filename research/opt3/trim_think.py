"""Hybrid reasoning trigger "trimmed": think when the verifier cut a claim from the answer.

Run 0749Z thinks only after a refusal. When the verifier drops an unsupported claim from a
non-thinking answer, the rest is published - in v3 that is mostly the answer to one part of
a multi-part question (10 of the 13 wrong answers with dropped claims, judge column). The
candidate rule: such an answer gets the thinking pass, and the thinking result replaces it
only when it is published with every claim verified; otherwise the trimmed answer stays.

`generate` drafts the thinking pass (pipeline prompt, logged contexts, thinking on, max tokens
llm_think_max_tokens; one repair if withheld with claims, as pipeline._draft does) on a local
llama-server for every question whose replayed non-thinking result is published with a
dropped claim, in either column. `evaluate` replays both columns with the rule.

Usage:  python -m research.opt3.trim_think generate --api-base http://127.0.0.1:8093/v1
        python -m research.opt3.trim_think evaluate
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

from generator_verifier import (
    ClaimStatus,
    VerifiedGenerationResult,
    _HTTPBackend,
    generation_system_prompt,
)
from pipeline import revision_payload

from research.opt3 import data
from research.opt3.recovery import prompt

STORE = data.CACHE / "trim_think.jsonl"


def trimmed(verified: VerifiedGenerationResult | None) -> bool:
    """Published, but at least one claim was withheld."""
    return (
        verified is not None
        and not verified.refused
        and any(c.status != ClaimStatus.VERIFIED for c in verified.claims)
    )


def population() -> list[tuple[str, str]]:
    from research.opt3.replay import replay, verifier

    ids: set[tuple[str, str]] = set()
    for config in data.CONFIGS:
        v = verifier(config)
        for record in data.records().values():
            r = replay(record, v, source=config)
            if (
                r.verified is not None
                and trimmed(r.verified)
                and not r.verified.generation_metadata.thinking
            ):
                ids.add((record.id, column(record, config)))
    return sorted(ids)


def column(record: data.CaseRecord, config: str) -> str:
    """Contexts a draft is generated on: the judge column's unless this column saw others
    (233 questions of 0749Z got a different context order, see the report)."""
    same = record.attempts[config][0].contexts == record.attempts["judge"][0].contexts
    return "judge" if same else config


def load_store() -> dict[tuple[str, str, int], dict[str, Any]]:
    if not STORE.exists():
        return {}
    rows = (json.loads(line) for line in STORE.read_text(encoding="utf-8").splitlines() if line)
    return {(r["id"], r.get("column", "judge"), r["index"]): r for r in rows}


def generate(api_base: str) -> None:
    from research.opt3.replay import verifier, verify

    records = data.records()
    store = load_store()
    s = data.settings("judge", llm_api_base=api_base, llm_api_timeout=3600)
    backend = _HTTPBackend(s)
    system = generation_system_prompt(s)
    v = verifier("judge")
    ids = population()
    print(f"{len(ids)} questions", flush=True)
    STORE.parent.mkdir(parents=True, exist_ok=True)
    with STORE.open("a", encoding="utf-8") as handle:
        for case_id, col in ids:
            record = records[case_id]
            first = record.attempts[col][0]
            question = record.retrieval[col]["query"]
            revision: dict[str, Any] | None = None
            for index in range(1 + s.generation_repair_attempts):
                if (case_id, col, index) in store:
                    row = store[(case_id, col, index)]
                else:
                    user = json.loads(prompt(question, first.contexts))
                    if revision is not None:
                        user = {"question": user["question"], "revision": revision,
                                "sources": user["sources"]}
                    started = time.perf_counter()
                    reply = backend.complete(
                        system, json.dumps(user, ensure_ascii=False), thinking=True
                    )
                    row = {
                        "id": case_id,
                        "column": col,
                        "index": index,
                        "raw_text": reply.text,
                        "finish_reason": reply.finish_reason,
                        "seconds": round(time.perf_counter() - started, 1),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(case_id, col, index, row["seconds"], reply.text[:90].replace("\n", " "),
                          flush=True)
                verified = verify(v, attempt(record, row, col))
                if not verified.refused or not verified.claims:
                    break
                revision = revision_payload(verified)


def attempt(record: data.CaseRecord, row: dict[str, Any], col: str = "judge") -> data.Attempt:
    first = record.attempts[col][0]
    return data.Attempt(
        role="think" if row["index"] == 0 else "think_repair",
        index=-1,
        raw_text=row["raw_text"],
        cleaned_text="",
        refused=False,
        claims=[],
        contexts=first.contexts,
        metadata={**first.metadata, "finish_reason": row["finish_reason"], "thinking": True,
                  "recovery": False},
    )


def drafts_of(
    store: dict[tuple[str, str, int], dict[str, Any]], config: str
) -> Callable[[data.CaseRecord], list[data.Attempt] | None]:
    """trim_drafts callback of research.opt3.replay.replay for the `config` column."""

    def callback(record: data.CaseRecord) -> list[data.Attempt] | None:
        col = column(record, config)
        rows = [store[(record.id, col, i)] for i in range(2) if (record.id, col, i) in store]
        return [attempt(record, row, col) for row in rows] or None

    return callback


def evaluate() -> dict[str, Any]:
    from research.opt2.metrics import paired, summarize
    from research.opt3.experiments import headline
    from research.opt3.replay import logged, replay, scored, verifier

    store = load_store()
    records = data.records()
    report: dict[str, Any] = {}
    for config in data.CONFIGS:
        v = verifier(config)
        base = [logged(r, config) for r in records.values()]
        outcomes = [
            scored(r, replay(r, v, source=config, trim_drafts=drafts_of(store, config)), config)
            for r in records.values()
        ]
        entry: dict[str, Any] = {}
        for part in ("all", "dev", "test"):
            keep = [i for i, o in enumerate(outcomes) if part == "all" or o["split"] == part]
            entry[part] = headline(summarize([outcomes[i] for i in keep]))
            entry[part]["paired"] = paired([base[i] for i in keep], [outcomes[i] for i in keep])
        entry["think_calls"] = sum("think" in o["path"] for o in outcomes) - sum(
            "think" in b["path"] for b in base
        )
        entry["changed"] = [
            {"id": o["id"], "split": o["split"], "before": b["factrag"]["correct"],
             "after": o["factrag"]["correct"], "hallucination": o["factrag"]["hallucination"],
             "path": o["path"]}
            for o, b in zip(outcomes, base, strict=True)
            if o["factrag"]["correct"] != b["factrag"]["correct"] or o["path"] != b["path"]
        ]
        report[config] = entry
        a = entry["all"]
        print(config, "correct", a["correct"], "halluc", a["hallucination"], "FR",
              a["false_refusal"], "+%d -%d p=%.3f" % (a["paired"]["gained"], a["paired"]["lost"],
              a["paired"]["mcnemar_p"]), "dev +%d -%d" % (entry["dev"]["paired"]["gained"],
              entry["dev"]["paired"]["lost"]), "test +%d -%d" % (
              entry["test"]["paired"]["gained"], entry["test"]["paired"]["lost"]),
              "extra think calls", entry["think_calls"])
        for c in entry["changed"]:
            print("   ", c)
    out = data.OUT / "experiments" / "trim_think.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "population", "evaluate"])
    parser.add_argument("--api-base", default="http://127.0.0.1:8093/v1")
    args = parser.parse_args()
    if args.action == "generate":
        generate(args.api_base)
    elif args.action == "evaluate":
        evaluate()
    else:
        ids = population()
        print(len(ids), ids)
