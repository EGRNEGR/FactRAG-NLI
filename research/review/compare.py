"""Remark 1: FactRAG-NLI against a pipeline without verification and external verifiers.

Two measurements on the logged run 20261104T1222Z, no new LLM calls:

* draft level: the first draft of every one of the 1307 questions (the "Vanilla" answer of
  the run, same retrieval, same generator) is filtered by each verifier of
  research.review.verifiers; adaptive passes (thinking, recovery, translation) are not
  replayed, so the rows differ by the verifier only. Accuracy (all, v3, XQuAD), hallucination
  on the 269 unanswerable and adversarial questions, false refusal on the 1038 answerable,
  exact McNemar against the unverified draft. Thresholds of bge_zs and minicheck: the
  published default (0.5) and the best on the dev split (accuracy, then fewer
  hallucinations), reported on test.
* claim level: a stress set built from the first drafts of 1222Z by research.opt2.stress
  rules (correct claims citing a gold chunk; number swap, negation, answer swap within the
  XQuAD paragraph, transplant from another document). Acceptance per kind on the test split
  (the judge was fitted on the dev split of an earlier run).

Usage:  python -m research.review.compare [draft] [stress] [--only name,name]
Results: reports/review/baselines.json, stress.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

from eval_dataset import normalize
from evaluate_suite import mcnemar_exact
from generator_verifier import NLIFactVerifier, RawGenerationResult, SourceContext

from research.review import OUT  # noqa: I001 - selects the run first
from research.opt2 import data as opt2
from research.opt2.metrics import outcome
from research.opt2.stress import _CITE, _answer_swap, _negate, _number_swap
from research.opt3 import data
from research.review import verifiers as V

GRID = [round(0.05 * i, 2) for i in range(1, 20)]
CONFIG = "judge"


def first_draft(record: data.CaseRecord) -> data.Attempt | None:
    attempts = record.attempts[CONFIG]
    return attempts[0] if attempts and attempts[0].role == "draft" else None


def verify(v: NLIFactVerifier, attempt: data.Attempt, text: str | None = None) -> Any:
    raw = RawGenerationResult.model_validate(
        {
            "text": attempt.raw_text if text is None else text,
            "contexts": attempt.contexts,
            "metadata": attempt.metadata,
        }
    )
    return v.verify(raw)


def outcomes(v: NLIFactVerifier | None, records: dict[str, data.CaseRecord]) -> list[dict[str, Any]]:
    """Draft-level outcome of every case; v None = the unverified draft (Vanilla)."""
    result = []
    for record in records.values():
        draft = first_draft(record)
        vanilla = draft.raw_text if draft else None
        if draft is None:
            published = None
        elif v is None:
            published = vanilla
        else:
            checked = verify(v, draft)
            published = None if checked.refused else checked.cleaned_text
        o = outcome(record, vanilla_text=vanilla, published_text=published)
        o["pair"] = pair(record)
        result.append(o)
    return result


def pair(record: data.CaseRecord) -> str:
    """Language of the question and of the corpus: v3 documents are English, XQuAD Russian."""
    corpus = "ru" if record.id.startswith("V3-XQF") else "en"
    return f"{record.case.language}-{corpus}"


def block(items: list[dict[str, Any]], base: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    answerable = [o for o in items if o["expected"] == "answer"]
    defined = [o for o in items if o["factrag"]["hallucination"] is not None]
    out: dict[str, Any] = {
        "n": len(items),
        "correct": sum(o["factrag"]["correct"] for o in items),
        "hallucination": sum(bool(o["factrag"]["hallucination"]) for o in defined),
        "hallucination_n": len(defined),
        "false_refusal": sum(o["factrag"]["refused"] for o in answerable),
        "false_refusal_n": len(answerable),
    }
    if base is not None:
        b = sum(o["factrag"]["correct"] and not r["factrag"]["correct"] for o, r in zip(items, base))
        c = sum(r["factrag"]["correct"] and not o["factrag"]["correct"] for o, r in zip(items, base))
        out["paired_vs_vanilla"] = {"gained": b, "lost": c, "mcnemar_p": mcnemar_exact(b, c)}
    return out


def table(items: list[dict[str, Any]], base: list[dict[str, Any]]) -> dict[str, Any]:
    def pair_is(lp: str) -> Callable[[dict[str, Any]], bool]:
        return lambda o: bool(o["pair"] == lp)

    parts: dict[str, Callable[[dict[str, Any]], bool]] = {
        "all": lambda o: True,
        "v3": lambda o: o["dataset"] == "v3",
        "xquad-full": lambda o: o["dataset"] == "xquad-full",
        "dev": lambda o: o["split"] == "dev",
        "test": lambda o: o["split"] == "test",
        **{f"pair {lp}": pair_is(lp) for lp in ("en-en", "ru-en", "ru-ru", "en-ru")},
    }
    result = {}
    for name, keep in parts.items():
        idx = [i for i, o in enumerate(items) if keep(o)]
        result[name] = block([items[i] for i in idx], [base[i] for i in idx])
    return result


def _collect_pairs(records: dict[str, data.CaseRecord], texts: list[tuple[str, str]]) -> None:
    """Score every (premise, claim) pair MiniCheck will see in one batched pass."""
    pairs: list[tuple[str, str]] = []

    def collect(v: V.ClaimFilter, hypothesis: str, context: SourceContext) -> tuple[float, bool]:
        pairs.append((v._premise(context), hypothesis))
        return 0.0, False

    collector = V.ClaimFilter("collect", collect)
    for case_id, text in texts:
        draft = first_draft(records[case_id])
        if draft is not None:
            verify(collector, draft, text)
    unique = list(dict.fromkeys(pairs))
    started = time.perf_counter()
    V.minicheck().score_many(unique)
    print(f"minicheck: {len(unique)} pairs, {V.minicheck().misses} new, "
          f"{time.perf_counter() - started:.0f} s", flush=True)


def tune(name: str, records: dict[str, data.CaseRecord]) -> tuple[float, list[dict[str, Any]]]:
    """The threshold with the best dev accuracy (ties: fewer hallucinations, higher t)."""
    dev = {cid: r for cid, r in records.items() if r.split == "dev"}
    best: tuple[tuple[int, int, float], float] | None = None
    for t in GRID:
        items = outcomes(V.build(name, t), dev)
        b = block(items)
        key = (b["correct"], -b["hallucination"], t)
        if best is None or key > best[0]:
            best = (key, t)
    assert best is not None
    t = best[1]
    return t, outcomes(V.build(name, t), records)


def draft_level(names: tuple[str, ...] = V.NAMES) -> dict[str, Any]:
    records = data.records()
    texts = [(cid, d.raw_text) for cid, r in records.items() if (d := first_draft(r))]
    if "minicheck" in names:
        _collect_pairs(records, texts)
    vanilla = outcomes(None, records)
    rows: dict[str, Any] = {"vanilla": {"threshold": None, **table(vanilla, vanilla)}}
    per_case: dict[str, dict[str, bool]] = {o["id"]: {"vanilla": o["factrag"]["correct"]}
                                           for o in vanilla}
    for name in names:
        started = time.perf_counter()
        variants: list[tuple[str, float | None, list[dict[str, Any]]]] = [
            (name, None, outcomes(V.build(name), records))
        ]
        if name in V.DEFAULT_THRESHOLDS:
            t, items = tune(name, records)
            variants.append((f"{name}_dev", t, items))
        for label, tuned, items in variants:
            threshold = tuned if tuned is not None else V.DEFAULT_THRESHOLDS.get(name)
            rows[label] = {"threshold": threshold, **table(items, vanilla)}
            for o in items:
                per_case[o["id"]][label] = o["factrag"]["correct"]
            a = rows[label]["all"]
            print(f"{label:16} t={threshold} correct {a['correct']}/{a['n']} "
                  f"halluc {a['hallucination']}/{a['hallucination_n']} "
                  f"FR {a['false_refusal']}/{a['false_refusal_n']} "
                  f"vs vanilla +{a['paired_vs_vanilla']['gained']}"
                  f"/-{a['paired_vs_vanilla']['lost']} ({time.perf_counter() - started:.0f} s)",
                  flush=True)
    # Paired: FactRAG-NLI (judge) against each baseline on the same drafts.
    against: dict[str, Any] = {}
    for label in rows:
        if label == "judge":
            continue
        b = sum(c["judge"] and not c[label] for c in per_case.values())
        c = sum(c[label] and not c["judge"] for c in per_case.values())
        against[label] = {"judge_gained": b, "judge_lost": c, "mcnemar_p": mcnemar_exact(b, c)}
    return {"run": data.RUN.name, "rows": rows, "judge_vs": against}


# ------------------------------------------------------------------ claim-level stress set


def stress_items(records: dict[str, data.CaseRecord]) -> list[dict[str, Any]]:
    """research.opt2.stress rules on the first drafts of the run (deterministic)."""
    by_group: dict[str, list[data.CaseRecord]] = {}
    for record in records.values():
        by_group.setdefault(opt2.split_group(record.case), []).append(record)
    positives = []
    for cid in sorted(records):
        record = records[cid]
        case = record.case
        draft = first_draft(record)
        if case.expected_behavior != "answer" or draft is None:
            continue
        if not outcome(record, vanilla_text=draft.raw_text, published_text=None)["vanilla"][
            "correct"
        ]:
            continue
        gold = set(case.gold_chunk_ids)
        for claim in draft.claims:
            if claim.get("kind", "fact") != "fact":
                continue
            if {e.get("chunk_id") for e in claim.get("evidence") or []} & gold:
                positives.append((record, claim["claim"]))
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record, text in positives:
        if (record.id, text) in seen:
            continue
        seen.add((record.id, text))
        base = {"id": record.id, "split": record.split}
        items.append({**base, "kind": "positive", "text": text})
        variants = {"number_swap": _number_swap(text), "negation": _negate(text)}
        if record.id.startswith("V3-XQF"):
            answer = record.case.ground_truth_answer.strip(' "«»')
            others = sorted(
                {
                    r.case.ground_truth_answer.strip(' "«»')
                    for r in by_group[opt2.split_group(record.case)]
                    if r.id != record.id and r.case.expected_behavior == "answer"
                }
            )
            others = [
                o
                for o in others
                if normalize(o) != normalize(answer)
                and normalize(answer) not in normalize(o)
                and normalize(o) not in normalize(answer)
            ]
            numeric = bool(re.search(r"\d", answer))
            others = sorted(
                (o for o in others if bool(re.search(r"\d", o)) == numeric),
                key=lambda o: (abs(len(o) - len(answer)), o),
            )
            if len(answer) >= 2 and others:
                variants["answer_swap"] = _answer_swap(text, answer, others[0])
        for kind, changed in variants.items():
            if changed and changed != text:
                items.append({**base, "kind": kind, "text": changed})
    ordered = sorted(positives, key=lambda p: (p[0].id, p[1]))
    for index, (record, text) in enumerate(ordered):
        for other, other_text in ordered[index + 1 :] + ordered[:index]:
            if other.case.document != record.case.document:
                markers = "".join(re.findall(r"\[S\d+\]", text)) or "[S1]"
                moved = _CITE.sub("", other_text).rstrip(". ") + " " + markers + "."
                items.append({"id": record.id, "split": record.split, "kind": "transplant",
                              "text": moved})
                break
    unique = {(i["id"], i["kind"], i["text"]): i for i in items}
    return list(unique.values())


def stress() -> dict[str, Any]:
    records = data.records()
    items = stress_items(records)
    print("stress items:", dict(Counter(i["kind"] for i in items)), flush=True)
    thresholds = json.loads((OUT / "baselines.json").read_text("utf-8"))["rows"]
    if any(label.startswith("minicheck") for label in thresholds):
        _collect_pairs(records, [(i["id"], i["text"]) for i in items])
    result: dict[str, Any] = {"run": data.RUN.name, "kinds": dict(Counter(i["kind"] for i in items))}
    for label, row in thresholds.items():
        if label == "vanilla":
            continue
        name = label.removesuffix("_dev")
        v = V.build(name, row["threshold"] if name in V.DEFAULT_THRESHOLDS else None)
        counts: Counter[tuple[str, str]] = Counter()
        hits: Counter[tuple[str, str]] = Counter()
        for item in items:
            draft = first_draft(records[item["id"]])
            assert draft is not None  # stress items come from first drafts only
            ok = not verify(v, draft, item["text"]).refused
            lp = pair(records[item["id"]])
            for split in ("all", item["split"]) + ((f"test {lp}",) if item["split"] == "test" else ()):
                counts[(item["kind"], split)] += 1
                hits[(item["kind"], split)] += ok
        result[label] = {
            f"{kind}/{split}": {"k": hits[(kind, split)], "n": n, "rate": hits[(kind, split)] / n}
            for (kind, split), n in sorted(counts.items())
        }
        t = result[label]
        print(label, "test:", "; ".join(
            f"{k.split('/')[0]} {100 * t[k]['rate']:.1f}%" for k in t if k.endswith("/test")),
            flush=True)
    return result


def main(args: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names: tuple[str, ...] = V.NAMES
    if "--only" in args:
        i = args.index("--only")
        names = tuple(args[i + 1].split(","))
        args = args[:i] + args[i + 2 :]
    parts = args or ["draft", "stress"]
    if "draft" in parts:
        result = draft_level(names)
        (OUT / "baselines.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    if "stress" in parts:
        result = stress()
        (OUT / "stress.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                        encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
