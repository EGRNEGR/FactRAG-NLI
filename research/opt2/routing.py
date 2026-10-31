"""PHASE 5: selective thinking - which signal should trigger the thinking pass?

Both 0508Z configurations (qwen38-27b without and with thinking) answered every question with
the same retrieval, prompt and seed, so a router is simulated exactly: the non-thinking result
(replayed under a verifier) decides whether the case takes the thinking result. Only signals
available at run time are used (refusal and its reason, claim grades and judge scores,
reranker scores, the form of the question); dataset and category are never inputs.

Cost uses the logged RTX 3090 generation times: a routed case costs its non-thinking time
plus its thinking time. Thresholds of parametrised routers are chosen on dev; test is
reported once per router.

Usage:  python -m research.opt2.routing [judge]   (judge: bge-m3-zeroshot | none)
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from generator_verifier import ClaimJudge, NLIFactVerifier

from research.opt2 import data
from research.opt2.metrics import outcome, paired, summarize
from research.opt2.nli_models import CachedNLI
from research.opt2.replay import Replayed, replay_config, verifier

EXEMPT = (None, "relevance_threshold_not_met", "context_budget_exceeded")
_MULTI = re.compile(
    r"\b(и|and)\b.*\?|,\s*(и|а также|and)\s+(что|как|какой|какие|what|which|how)\b|\?.*\?",
    re.I,
)


@dataclass
class Case:
    record: data.CaseRecord
    base: Replayed
    think: Replayed
    base_ms: float
    think_ms: float

    @property
    def top_rerank(self) -> float:
        sources = self.record.retrieval.get("sources") or []
        return max((s.get("rerank_score") or 0.0 for s in sources), default=0.0)

    def claims(self) -> list[Any]:
        if self.base.verified is None:
            return []
        return [c for c in self.base.verified.claims if c.kind == "fact"]

    @property
    def min_confidence(self) -> float:
        claims = self.claims()
        return min((c.confidence for c in claims), default=0.0)

    @property
    def grades(self) -> set[str]:
        return {c.grade or "" for c in self.claims()}

    @property
    def multi_part(self) -> bool:
        return bool(_MULTI.search(self.record.case.question))


def generation_ms(record: data.CaseRecord, config: str) -> float:
    return sum(d.metadata.get("generation_ms") or 0.0 for d in record.drafts[config])


def prepare(v: NLIFactVerifier) -> list[Case]:
    records = data.records(("base", "think"))
    return [
        Case(
            record,
            replay_config(record, "base", v),
            replay_config(record, "think", v),
            generation_ms(record, "base"),
            generation_ms(record, "think"),
        )
        for record in records.values()
    ]


Router = Callable[[Case], bool]


def withheld(c: Case) -> bool:
    return c.base.refused and c.base.reason not in EXEMPT


ROUTERS: dict[str, tuple[str, Callable[[float], Router] | Router, list[float] | None]] = {
    "A0_never": ("no thinking pass", lambda c: False, None),
    "A1_first_llm_refusal": (
        "first draft was an LLM refusal (0508Z baseline)",
        lambda c: c.base.reason == "llm_refusal" and c.base.attempts == 1,
        None,
    ),
    "A_withheld": ("any withheld answer except gate/budget (f888cdf default)", withheld, None),
    "B_withheld_or_low_retrieval": (
        "withheld, or published with top rerank score < t",
        lambda t: lambda c: withheld(c) or (not c.base.refused and c.top_rerank < t),
        [0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
    ),
    "C_withheld_or_conflict": (
        "withheld, or a claim graded contradicted/ambiguous/partial (dropped from the answer)",
        lambda c: withheld(c)
        or (not c.base.refused and bool(c.grades & {"contradicted", "ambiguous", "partially_supported"})),
        None,
    ),
    "D_withheld_or_low_confidence": (
        "withheld, or published with the lowest claim confidence < t",
        lambda t: lambda c: withheld(c) or (not c.base.refused and c.min_confidence < t),
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ),
    "F_withheld_or_multi_part": (
        "withheld, or the question has several parts (form of the question only)",
        lambda c: withheld(c) or (not c.base.refused and c.multi_part),
        None,
    ),
    "E_adaptive": (
        "withheld, or published with (conflict or confidence < t or rerank < 0.05)",
        lambda t: lambda c: withheld(c)
        or (
            not c.base.refused
            and (
                bool(c.grades & {"contradicted", "ambiguous", "partially_supported"})
                or c.min_confidence < t
                or c.top_rerank < 0.05
            )
        ),
        [0.4, 0.5, 0.6, 0.7, 0.8],
    ),
    "Z_always": ("thinking for every question (qwen38-27b-think)", lambda c: True, None),
}


def choose(c: Case, routed: bool, policy: str) -> Replayed:
    """Final result: 'replace' takes the thinking result when routed, except that a
    published non-thinking answer is kept if the thinking pass withheld its answer;
    'confidence' keeps whichever published answer has the higher lowest claim score."""
    if not routed:
        return c.base
    if c.base.refused or c.think.published is None:
        return c.think if not c.think.refused or c.base.refused else c.base
    if policy == "replace":
        return c.think

    def score(r: Replayed) -> float:
        claims = [x for x in (r.verified.claims if r.verified else []) if x.kind == "fact"]
        return min((x.confidence for x in claims), default=0.0)

    return c.think if score(c.think) >= score(c.base) else c.base


def evaluate(cases: list[Case], router: Router, policy: str) -> dict[str, Any]:
    outs, calls, ms = [], 0, 0.0
    for c in cases:
        routed = router(c)
        final = choose(c, routed, policy)
        config = "think" if final is c.think and routed else "base"
        drafts = c.record.drafts[config]
        outs.append(
            outcome(
                c.record,
                vanilla_text=drafts[0].raw_text if drafts else None,
                published_text=final.published,
            )
        )
        calls += routed
        ms += c.base_ms + (c.think_ms if routed else 0.0)
    return {
        "outcomes": outs,
        "thinking_calls": calls,
        "generation_hours": ms / 3.6e6,
        "all": summarize(outs),
        "dev": summarize([o for o in outs if o["split"] == "dev"]),
        "test": summarize([o for o in outs if o["split"] == "test"]),
        **{ds: summarize([o for o in outs if o["dataset"] == ds]) for ds in data.DATASETS},
    }


def judge_verifier(name: str) -> NLIFactVerifier:
    if name == "none":
        return verifier(CachedNLI("mdeberta"))
    from research.opt2.judge import judge_verifier as build

    spec = json.loads((data.ROOT / "data" / "nli_judge" / f"{name}.json").read_text("utf-8"))
    return build(name, ClaimJudge(spec))


def main(judge_name: str) -> dict[str, Any]:
    cases = prepare(judge_verifier(judge_name))
    reference = evaluate(cases, ROUTERS["A1_first_llm_refusal"][1], "replace")  # type: ignore[arg-type]
    report: dict[str, Any] = {"verifier": judge_name, "routers": {}}
    for name, (text, factory, grid) in ROUTERS.items():
        for policy in ("replace", "confidence"):
            if grid is None:
                chosen, router = None, factory
            else:
                # threshold with the most correct dev answers; ties: fewer thinking calls
                best = None
                for t in grid:
                    res = evaluate(cases, factory(t), policy)  # type: ignore[operator]
                    key = (res["dev"]["factrag_correct"]["k"], -res["thinking_calls"])
                    if best is None or key > best[0]:
                        best = (key, t)
                chosen = best[1] if best else None
                router = factory(chosen)  # type: ignore[operator]
            res = evaluate(cases, router, policy)  # type: ignore[arg-type]
            res["vs_A1"] = paired(reference["outcomes"], res["outcomes"])
            res["vs_A1_test"] = paired(
                [o for o in reference["outcomes"] if o["split"] == "test"],
                [o for o in res["outcomes"] if o["split"] == "test"],
            )
            res.pop("outcomes")
            res.update(description=text, threshold=chosen, policy=policy)
            report["routers"][f"{name}/{policy}"] = res
            s = res["all"]
            print(
                f"{name:30s} {policy:10s} t={chosen} all {s['factrag_correct']['k']} "
                f"dev {res['dev']['factrag_correct']['k']} test {res['test']['factrag_correct']['k']} "
                f"halluc {s['hallucination']['k']} FR {s['false_refusal']['k']} "
                f"calls {res['thinking_calls']} hours {res['generation_hours']:.2f}",
                flush=True,
            )
    oracle = sum(
        bool(
            outcome(c.record, vanilla_text=None, published_text=c.base.published)["factrag"]["correct"]
            or outcome(c.record, vanilla_text=None, published_text=c.think.published)["factrag"]["correct"]
        )
        for c in cases
    )
    report["oracle_best_of_two"] = oracle
    print("oracle best of two:", oracle)
    out = data.OUT / "routing"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"routing_{judge_name}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), "utf-8")
    return report


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bge-m3-zeroshot")
