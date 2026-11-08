"""Offline replay of run 0749Z under a verifier variant (pipeline order, logged drafts only).

The pipeline of the run, per question (pipeline.RAGPipeline.query):
  draft -> repair (if withheld with claims) -> think + think_repair (llm_think_retry=refusal:
  any refusal except context budget) -> recovery samples (an LLM refusal with best rerank
  >= 0.5; the first published sample wins).
The replay re-verifies the logged texts of one configuration ("source") with a verifier
variant and walks the same decisions. A step the variant needs but the run never generated
(the run published earlier, or the variant asks for a repair the run did not need) is
"missing": the case keeps the last available result, withheld, and is counted in `missing`
(pessimistic for the variant). Every NLI prediction goes through the persistent caches of
research.opt2.nli_models, so variants differ only by decision logic.

Usage:  python -m research.opt3.replay      (agreement with the logged verification)
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from generator_verifier import (
    ClaimJudge,
    ClaimStatus,
    NLIFactVerifier,
    RawGenerationResult,
    VerifiedGenerationResult,
)
from pipeline import refusal_reason, think_retry

from research.opt2.metrics import Outcome, outcome
from research.opt2.nli_models import CachedNLI
from research.opt3 import data


def verifier(
    config: str, judge: dict[str, Any] | None = None, **overrides: Any
) -> NLIFactVerifier:
    """The verifier of `config` ("rules" | "judge") with the run's settings and `overrides`;
    `judge` overrides fields of the decision layer spec (threshold, partial_threshold, ...)."""
    s = data.settings(config, **overrides)
    if s.nli_decision != "judge":
        return NLIFactVerifier(s, backend=CachedNLI("mdeberta"))
    assert s.nli_judge_path is not None  # required by settings when nli_decision == "judge"
    spec = {**json.loads(Path(s.nli_judge_path).read_text("utf-8")), **(judge or {})}
    claim_judge = ClaimJudge(spec)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(spec, handle)
    try:
        return NLIFactVerifier(
            s.model_copy(update={"nli_judge_path": handle.name}),
            backend=CachedNLI("mdeberta"),
            secondary=CachedNLI("bge-m3-zeroshot"),
            judge=claim_judge,
        )
    finally:
        Path(handle.name).unlink()


def verify(v: NLIFactVerifier, attempt: data.Attempt) -> VerifiedGenerationResult:
    raw = RawGenerationResult.model_validate(
        {"text": attempt.raw_text, "contexts": attempt.contexts, "metadata": attempt.metadata}
    )
    return v.verify(raw)


def trimmed(verified: VerifiedGenerationResult) -> bool:
    """Published, but at least one claim was withheld."""
    return not verified.refused and any(c.status != ClaimStatus.VERIFIED for c in verified.claims)


@dataclass
class Replayed:
    verified: VerifiedGenerationResult | None  # None: refused by the relevance gate
    path: list[str] = field(default_factory=list)
    missing: str | None = None  # role the variant needed but the run never generated
    texts: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str | None:
        if self.verified is None:
            return "relevance_threshold_not_met"
        return refusal_reason(self.verified)

    @property
    def published(self) -> str | None:
        if self.verified is None or self.verified.refused:
            return None
        return self.verified.cleaned_text


class _Walk:
    def __init__(self, record: data.CaseRecord, source: str, v: NLIFactVerifier) -> None:
        self.pool: dict[str, list[data.Attempt]] = {}
        for attempt in record.attempts[source]:
            self.pool.setdefault(attempt.role, []).append(attempt)
        self.v = v
        self.result = Replayed(None)

    def take(self, role: str) -> VerifiedGenerationResult | None:
        queue = self.pool.get(role) or []
        if not queue:
            self.result.missing = self.result.missing or role
            return None
        attempt = queue.pop(0)
        verified = verify(self.v, attempt)
        self.result.path.append(role)
        self.result.texts.append(attempt.raw_text)
        return verified

    def draft(self, first: str, repair: str, repairs: int) -> VerifiedGenerationResult | None:
        verified = self.take(first)
        for _ in range(repairs):
            if verified is None or not verified.refused or not verified.claims:
                break
            repaired = self.take(repair)
            if repaired is None:
                break
            verified = repaired
        return verified


def replay(
    record: data.CaseRecord,
    v: NLIFactVerifier,
    *,
    source: str = "judge",
    think_mode: str = "refusal",
    recovery: bool = True,
    recovery_min_rerank: float = 0.5,
    recovery_drafts: Callable[[data.CaseRecord], list[data.Attempt] | None] | None = None,
    trim_drafts: Callable[[data.CaseRecord], list[data.Attempt] | None] | None = None,
) -> Replayed:
    """Replay one case; `recovery_drafts` (a recovery variant) returns the drafts that replace
    the logged recovery samples (None: the case has none, keep the logged ones).
    `trim_drafts` (trigger "trimmed"): thinking drafts for a published non-thinking answer
    with a withheld claim; they replace it only if published with every claim verified."""
    if not record.attempts[source]:
        return Replayed(None, missing=None)
    s = v.settings
    walk = _Walk(record, source, v)
    verified = walk.draft("draft", "repair", s.generation_repair_attempts)
    if verified is not None and think_retry(think_mode, refusal_reason(verified)):
        thought = walk.draft("think", "think_repair", s.generation_repair_attempts)
        if thought is not None:
            verified = thought
    elif trim_drafts is not None and verified is not None and trimmed(verified):
        replacement = trim_drafts(record)
        if replacement:
            walk.pool["think"] = [a for a in replacement if a.role == "think"]
            walk.pool["think_repair"] = [a for a in replacement if a.role == "think_repair"]
            thought = walk.draft("think", "think_repair", s.generation_repair_attempts)
            if thought is not None and not thought.refused and not trimmed(thought):
                verified = thought
    if (
        verified is not None
        and recovery
        and refusal_reason(verified) == "llm_refusal"
        and record.top_rerank(source) >= recovery_min_rerank
    ):
        replacement = recovery_drafts(record) if recovery_drafts else None
        if replacement is not None:
            walk.pool["recovery"] = list(replacement)
        for _ in range(len(walk.pool.get("recovery") or []) or s.llm_recovery_samples):
            sample = walk.take("recovery")
            if sample is None:
                break
            if not sample.refused:
                verified = sample
                break
    walk.result.verified = verified
    return walk.result


def scored(record: data.CaseRecord, r: Replayed, source: str = "judge") -> Outcome:
    first = record.attempts[source][0].raw_text if record.attempts[source] else None
    o = outcome(record, vanilla_text=first, published_text=r.published)
    o.update(path="+".join(r.path) or "gate", missing=r.missing, reason=r.reason)
    return o


def logged(record: data.CaseRecord, source: str) -> Outcome:
    """The run's own outcome of a case, in the replay's Outcome format."""
    row = record.rows[source]
    attempts = record.attempts[source]
    o = outcome(
        record,
        vanilla_text=attempts[0].raw_text if attempts else None,
        published_text=None if row["refused"] else row["answer"],
    )
    o.update(path="+".join(a.role for a in attempts) or "gate", missing=None,
             reason=row["refusal_reason"])
    return o


def main() -> None:
    from research.opt2.metrics import pct, summarize

    records = data.records()
    for config in data.CONFIGS:
        v = verifier(config)
        same = total = 0
        diffs = []
        outcomes = []
        for record in records.values():
            r = replay(record, v, source=config)
            o = scored(record, r, config)
            outcomes.append(o)
            ref = logged(record, config)
            total += 1
            if o["factrag"]["correct"] == ref["factrag"]["correct"] and o["published"] == ref[
                "published"
            ]:
                same += 1
            else:
                diffs.append((record.id, ref["path"], o["path"], r.missing))
        s = summarize(outcomes)
        print(config, "replay", pct(s["factrag_correct"]), "halluc", pct(s["hallucination"]),
              "agreement with run", f"{same}/{total}")
        for d in diffs:
            print("  diff", *d)


if __name__ == "__main__":
    main()
