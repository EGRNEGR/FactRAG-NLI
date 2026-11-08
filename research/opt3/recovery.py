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
from collections.abc import Callable
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


def drafts_of(
    mode: str, store: dict[tuple[str, str, int], dict[str, Any]], source: str, keep_logged: bool
) -> Callable[[data.CaseRecord], list[data.Attempt] | None]:
    """recovery_drafts callback of research.opt3.replay.replay for one variant.

    The generated drafts (contexts of the judge column) are used only where the source
    column saw the same contexts; keep_logged appends the run's own recovery samples.
    """

    def callback(record: data.CaseRecord) -> list[data.Attempt] | None:
        rows = [store.get((record.id, mode, i)) for i in range(len(MODES[mode]["plans"]))]
        if any(row is None for row in rows):
            return None
        first = record.attempts[source][0]
        if first.contexts != record.attempts["judge"][0].contexts:
            return None
        generated = [
            data.Attempt(
                role="recovery",
                index=-1,
                raw_text=row["raw_text"],
                cleaned_text="",
                refused=False,
                claims=[],
                contexts=first.contexts,
                metadata={
                    **first.metadata,
                    "finish_reason": row["finish_reason"],
                    "recovery": True,
                    "thinking": False,
                },
            )
            for row in rows
            if row is not None
        ]
        logged = [a for a in record.attempts[source] if a.role == "recovery"]
        return generated + (logged if keep_logged else [])

    return callback


THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def evaluate(modes: list[str]) -> dict[str, Any]:
    """Every variant x trigger threshold, both verifiers; threshold chosen on dev only."""
    from research.opt2.metrics import paired, summarize
    from research.opt3.experiments import headline
    from research.opt3.replay import logged, replay, scored, verifier

    store = load_store()
    records = data.records()
    report: dict[str, Any] = {}
    for config in data.CONFIGS:
        v = verifier(config)
        base = [logged(r, config) for r in records.values()]
        variants: list[tuple[str, Any]] = [("logged_sampling", None)]
        for mode in modes:
            variants.append((mode, drafts_of(mode, store, config, keep_logged=False)))
            variants.append((mode + "+logged_sampling", drafts_of(mode, store, config, True)))
        for name, callback in variants:
            grid = {}
            for threshold in THRESHOLDS:
                outcomes = [
                    scored(
                        r,
                        replay(r, v, source=config, recovery_min_rerank=threshold,
                               recovery_drafts=callback),
                        config,
                    )
                    for r in records.values()
                ]
                grid[threshold] = outcomes

            def dev_key(threshold: float) -> tuple[int, int, float]:
                dev = [o for o in grid[threshold] if o["split"] == "dev"]
                s = summarize(dev)
                return (-s["hallucination"]["k"], s["factrag_correct"]["k"], threshold)

            chosen = max(THRESHOLDS, key=dev_key)
            entry: dict[str, Any] = {"threshold": chosen, "grid": {}}
            for threshold, outcomes in grid.items():
                row: dict[str, Any] = {}
                for part in ("all", "dev", "test"):
                    keep = [i for i, o in enumerate(outcomes) if part == "all" or o["split"] == part]
                    row[part] = headline(summarize([outcomes[i] for i in keep]))
                    row[part]["paired"] = paired([base[i] for i in keep], [outcomes[i] for i in keep])
                row["published_wrong_answerable"] = sum(
                    o["expected"] == "answer" and o["published"] and not o["factrag"]["correct"]
                    for o in outcomes
                )
                row["changed"] = [
                    {"id": o["id"], "split": o["split"], "correct": o["factrag"]["correct"],
                     "hallucination": o["factrag"]["hallucination"], "path": o["path"]}
                    for o, b in zip(outcomes, base, strict=True)
                    if o["published"] != b["published"] or o["factrag"]["correct"] != b["factrag"]["correct"]
                ]
                entry["grid"][str(threshold)] = row
            report.setdefault(config, {})[name] = entry
            r = entry["grid"][str(chosen)]
            print(
                config, name, "threshold", chosen, "| all", r["all"]["correct"],
                "halluc", r["all"]["hallucination"], "FR", r["all"]["false_refusal"],
                "pub_wrong", r["published_wrong_answerable"],
                "| test +%d -%d" % (r["test"]["paired"]["gained"], r["test"]["paired"]["lost"]),
                "| dev +%d -%d" % (r["dev"]["paired"]["gained"], r["dev"]["paired"]["lost"]),
                flush=True,
            )
    out = data.OUT / "experiments" / "recovery.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "population", "evaluate"])
    parser.add_argument("--api-base", default="http://127.0.0.1:8093/v1")
    parser.add_argument("--modes", nargs="+", default=["relaxed"])
    args = parser.parse_args()
    if args.action == "generate":
        generate(args.api_base, args.modes)
    elif args.action == "evaluate":
        evaluate(args.modes)
    else:
        ids = population()
        print(len(ids), ids)
