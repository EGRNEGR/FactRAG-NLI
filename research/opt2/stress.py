"""Claim-level stress set: does a verifier variant still reject false claims?

Qwen3.8 drafts are almost never hallucinated (0 corrections by the verifier in 0508Z), so
the case-level hallucination rate cannot show a weakened filter. This module builds a fixed
set of claims, each verified against the logged contexts of a real draft:

* positive: a fact claim of a correct first draft (answerable question) that cites a gold
  chunk - should be accepted;
* answer_swap (XQuAD): the gold answer inside such a claim replaced by the gold answer of
  another question about the same paragraph, of the same type (number for number) and the
  closest length - a fluent, on-topic, false claim whose new answer occurs in the context
  (the typical RAG error of picking the wrong entity);
* number_swap: the first number of the claim changed (n -> n + 1, years -> n + 3);
* negation: a negation inserted before the main verb (or removed);
* transplant: a correct claim of another document's question, citing this draft's sources.

Generation is deterministic (sorted inputs, no randomness). The set is cached in
.artifacts/opt2/stress_claims.jsonl; `evaluate` reports acceptance per kind and split.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from eval_dataset import normalize
from generator_verifier import NLIFactVerifier, RawGenerationResult

from research.opt2 import data

PATH = data.CACHE / "stress_claims.jsonl"
_NUMBER = re.compile(r"(?<![\w.,])(\d{1,6})(?![\d])")
_CITE = re.compile(r"\s*(\[S\d+\])+\s*\.?\s*$")
_RU_VERBS = re.compile(
    r"\b(является|являются|был|была|было|были|может|могут|должен|должна|должно|должны|"
    r"используется|используются|содержит|содержат|требует|требуют|позволяет|позволяют|"
    r"допускает|допускают|имеет|имеют|называется|называются|называл|называли|составляет|"
    r"составляют|включает|включают|определяет|определяют|означает|проиграли|проиграл|"
    r"победил|победили|стал|стала|стали|получил|получила|получили|находится|находятся)\b",
    re.I,
)


def _claims(draft: data.Draft) -> list[dict[str, Any]]:
    return [c for c in draft.claims if c.get("kind", "fact") == "fact"]


def _gold_cited(claim: dict[str, Any], gold: set[str]) -> bool:
    return bool({e.get("chunk_id") for e in claim.get("evidence") or []} & gold)


def _number_swap(text: str) -> str | None:
    body = _CITE.sub("", text)
    match = _NUMBER.search(body)
    if not match:
        return None
    value = int(match.group(1))
    new = value + 3 if 1000 <= value <= 2100 else value + 1
    return text[: match.start(1)] + str(new) + text[match.end(1) :]


def _negate(text: str) -> str | None:
    if re.search(r"\bне\s+", text, re.I):
        return re.sub(r"\bне\s+", "", text, count=1, flags=re.I)
    match = _RU_VERBS.search(text)
    if not match or match.start() == 0:
        return None
    return text[: match.start()] + "не " + text[match.start() :]


def _answer_swap(text: str, answer: str, other: str) -> str | None:
    pattern = re.compile(re.escape(answer), re.I)
    if not pattern.search(text) or normalize(other) in normalize(text):
        return None
    return pattern.sub(other, text, count=1)


def build() -> list[dict[str, Any]]:
    records = data.records(("base", "think"))
    by_group: dict[str, list[data.CaseRecord]] = {}
    for record in records.values():
        by_group.setdefault(data.split_group(record.case), []).append(record)
    items: list[dict[str, Any]] = []
    positives: list[tuple[data.CaseRecord, str, dict[str, Any]]] = []
    for case_id in sorted(records):
        record = records[case_id]
        case = record.case
        if case.expected_behavior != "answer":
            continue
        gold = set(case.gold_chunk_ids)
        for config in ("base", "think"):
            drafts = record.drafts[config]
            if not drafts or not record.rows[config]["vanilla"]["correct"]:
                continue
            for claim in _claims(drafts[0]):
                if _gold_cited(claim, gold):
                    positives.append((record, config, claim))
    seen: set[tuple[str, str]] = set()
    for record, config, claim in positives:
        key = (record.case.id, claim["claim"])
        if key in seen:
            continue
        seen.add(key)
        base = {"id": record.case.id, "config": config, "split": record.split}
        items.append({**base, "kind": "positive", "text": claim["claim"], "source": claim["claim"]})
        variants = {
            "number_swap": _number_swap(claim["claim"]),
            "negation": _negate(claim["claim"]),
        }
        if record.case.id.startswith("V3-XQF"):
            answer = record.case.ground_truth_answer.strip(' "«»')
            others = sorted(
                {
                    r.case.ground_truth_answer.strip(' "«»')
                    for r in by_group[data.split_group(record.case)]
                    if r.case.id != record.case.id and r.case.expected_behavior == "answer"
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
                variants["answer_swap"] = _answer_swap(claim["claim"], answer, others[0])
        for kind, text in variants.items():
            if text and text != claim["claim"]:
                items.append({**base, "kind": kind, "text": text, "source": claim["claim"]})
    # transplant: pair each positive with the next positive of a different document
    ordered = sorted(positives, key=lambda p: (p[0].case.id, p[2]["claim"]))
    for index, (record, config, claim) in enumerate(ordered):
        for other, _, other_claim in ordered[index + 1 :] + ordered[:index]:
            if other.case.document != record.case.document:
                markers = "".join(re.findall(r"\[S\d+\]", claim["claim"])) or "[S1]"
                text = _CITE.sub("", other_claim["claim"]).rstrip(". ") + " " + markers + "."
                items.append(
                    {
                        "id": record.case.id,
                        "config": config,
                        "split": record.split,
                        "kind": "transplant",
                        "text": text,
                        "source": claim["claim"],
                    }
                )
                break
    unique = {(i["id"], i["config"], i["kind"], i["text"]): i for i in items}
    return list(unique.values())


def load() -> list[dict[str, Any]]:
    if not PATH.exists():
        PATH.parent.mkdir(parents=True, exist_ok=True)
        with PATH.open("w", encoding="utf-8") as handle:
            for item in build():
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines()]


def accepted(v: NLIFactVerifier, record: data.CaseRecord, item: dict[str, Any]) -> bool:
    draft = record.drafts[item["config"]][0]
    raw = RawGenerationResult.model_validate(
        {"text": item["text"], "contexts": draft.contexts, "metadata": draft.metadata}
    )
    return not v.verify(raw).refused


def evaluate(
    v: NLIFactVerifier,
    items: Iterable[dict[str, Any]] | None = None,
    records: dict[str, data.CaseRecord] | None = None,
) -> dict[str, dict[str, Any]]:
    """Acceptance rate per kind, overall and per split (dev/test)."""
    records = records or data.records(("base", "think"))
    counts: Counter[tuple[str, str]] = Counter()
    hits: Counter[tuple[str, str]] = Counter()
    for item in items if items is not None else load():
        ok = accepted(v, records[item["id"]], item)
        for split in ("all", item["split"]):
            counts[(item["kind"], split)] += 1
            hits[(item["kind"], split)] += ok
    result: dict[str, dict[str, Any]] = {}
    for (kind, split), n in sorted(counts.items()):
        result.setdefault(kind, {})[split] = {
            "k": hits[(kind, split)],
            "n": n,
            "rate": hits[(kind, split)] / n,
        }
    return result


def summary_line(result: dict[str, dict[str, Any]], split: str = "all") -> str:
    parts = []
    for kind in ("positive", "answer_swap", "number_swap", "negation", "transplant"):
        if kind in result and split in result[kind]:
            r = result[kind][split]
            parts.append(f"{kind} {100 * r['rate']:.1f}% ({r['k']}/{r['n']})")
    return "; ".join(parts)


if __name__ == "__main__":
    from research.opt2.nli_models import CachedNLI
    from research.opt2.replay import verifier

    items = load()
    print(Counter(i["kind"] for i in items))
    for item in items[:12]:
        print(item["kind"], "|", item["text"][:150])
    res = evaluate(verifier(CachedNLI("mdeberta")), items)
    print("all :", summary_line(res))
    print("dev :", summary_line(res, "dev"))
    print("test:", summary_line(res, "test"))
