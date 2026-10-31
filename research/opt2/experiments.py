"""Experiment registry of the verifier/decision research (PHASES 3-5, 7).

Each variant = an NLI backend (single model or ensemble) + verifier setting overrides. A run
replays the 0508Z pipeline on all 1307 cases (both thinking routers), evaluates the
claim-level stress set, and records configuration, timings, VRAM and errors in
reports/optimization_2/experiments/<id>.json; EXPERIMENTS.csv is rebuilt from those files.

Usage:  python -m research.opt2.experiments V00_baseline V10_bge_m3 ...
        python -m research.opt2.experiments --list
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from generator_verifier import NLIProbabilities

from research.opt2 import data, stress
from research.opt2.metrics import summarize
from research.opt2.nli_models import CachedNLI
from research.opt2.replay import on_first_llm_refusal, on_withheld, run, verifier

EXPERIMENTS = data.OUT / "experiments"


class EnsembleNLI:
    """Pair-level combination of several cached NLI models.

    mode "min": entailment = min over models, contradiction = max (both must agree);
    mode "mean": arithmetic means; mode "first_veto": the first model's probabilities unless
    another model's contradiction is higher (adds contradiction evidence only).
    """

    def __init__(self, names: Sequence[str], mode: str) -> None:
        self.members = [CachedNLI(name) for name in names]
        self.mode = mode
        self.last_premise: str | None = None

    def predict_many(self, pairs: Sequence[tuple[str, str]]) -> list[NLIProbabilities | None]:
        columns = [m.predict_many(pairs) for m in self.members]
        result: list[NLIProbabilities | None] = []
        for row in zip(*columns, strict=True):
            present = [p for p in row if p is not None]
            if len(present) < len(row):
                result.append(row[0] if self.mode == "first_veto" else None)
                continue
            if self.mode == "min":
                e = min(p.p_entailment for p in present)
                c = max(p.p_contradiction for p in present)
            elif self.mode == "mean":
                e = sum(p.p_entailment for p in present) / len(present)
                c = sum(p.p_contradiction for p in present) / len(present)
            else:
                e = present[0].p_entailment
                c = max(p.p_contradiction for p in present)
            result.append(
                NLIProbabilities(p_entailment=e, p_neutral=max(0.0, 1 - e - c), p_contradiction=c)
            )
        return result

    def predict(self, premise: str, hypothesis: str) -> NLIProbabilities | None:
        return self.predict_many([(premise, hypothesis)])[0]

    @property
    def misses(self) -> int:
        return sum(m.misses for m in self.members)


@dataclass(frozen=True)
class Variant:
    id: str
    description: str
    backend: Callable[[], Any]
    overrides: dict[str, Any] = field(default_factory=dict)
    models: tuple[str, ...] = ("mdeberta",)


def single(name: str) -> Callable[[], Any]:
    return lambda: CachedNLI(name)


def ensemble(names: Sequence[str], mode: str) -> Callable[[], Any]:
    return lambda: EnsembleNLI(names, mode)


VARIANTS: dict[str, Variant] = {}


def register(variant: Variant) -> None:
    VARIANTS[variant.id] = variant


register(Variant("V00_baseline", "verifier v2 as run in 0508Z (mDeBERTa-v3-base)", single("mdeberta")))


def gpu_info() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"gpu": None}
    return {
        "gpu": torch.cuda.get_device_name(0),
        "max_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
    }


def evaluate(variant: Variant, records: dict[str, data.CaseRecord]) -> dict[str, Any]:
    started = time.perf_counter()
    backend = variant.backend()
    v = verifier(backend, **variant.overrides)
    result: dict[str, Any] = {
        "experiment_id": variant.id,
        "description": variant.description,
        "models": list(variant.models),
        "overrides": variant.overrides,
        "dataset_versions": {k: v[1] for k, v in data.FROZEN.items()},
        "source_run": data.RUN.name,
        "seed": 42,
        "llm": "Qwen3.8-27B UD-Q4_K_XL drafts of run 0508Z (temperature 0, seed 42)",
        "reranker": "bge-reranker-v2-m3 (contexts of run 0508Z)",
        "hardware": platform.node() + " / " + platform.processor(),
        "errors": [],
    }
    try:
        for router_name, router in (("withheld", on_withheld), ("first_llm_refusal", on_first_llm_refusal)):
            t0 = time.perf_counter()
            cases = run(records, v, router)
            outcomes = [c.outcome for c in cases]
            result[f"replay_{router_name}"] = {
                "all": summarize(outcomes),
                "dev": summarize([o for o in outcomes if o["split"] == "dev"]),
                "test": summarize([o for o in outcomes if o["split"] == "test"]),
                **{ds: summarize([o for o in outcomes if o["dataset"] == ds]) for ds in data.DATASETS},
                "thinking_calls": sum(c.thinking for c in cases),
                "missing_repair": sum(c.missing_repair for c in cases),
                "seconds": time.perf_counter() - t0,
            }
            result.setdefault("case_outcomes", {})[router_name] = {
                o["id"]: [o["factrag"]["correct"], bool(o["factrag"]["hallucination"]), c.thinking]
                for o, c in zip(outcomes, cases, strict=True)
            }
        t0 = time.perf_counter()
        result["stress"] = stress.evaluate(v, records=records)
        result["stress_seconds"] = time.perf_counter() - t0
    except Exception as exc:  # recorded, never hidden
        result["errors"].append(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    result["nli_pairs_computed"] = getattr(backend, "misses", None)
    result["seconds"] = time.perf_counter() - started
    result.update(gpu_info())
    return result


def rate(block: dict[str, Any] | None) -> float | None:
    return None if not block or block.get("rate") is None else round(100 * block["rate"], 2)


def csv_row(res: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment_id": res["experiment_id"],
        "description": res["description"],
        "models": "+".join(res["models"]),
        "overrides": json.dumps(res["overrides"], ensure_ascii=False),
        "seed": res["seed"],
        "dataset_versions": "v3:" + res["dataset_versions"]["v3"][:12] + " xquad-full:" + res["dataset_versions"]["xquad-full"][:12],
        "hardware": res.get("gpu"),
        "seconds": round(res["seconds"], 1),
        "max_vram_mb": res.get("max_vram_mb"),
        "errors": len(res["errors"]),
    }
    for router in ("withheld", "first_llm_refusal"):
        rep = res.get(f"replay_{router}")
        if not rep:
            continue
        tag = "w" if router == "withheld" else "f"
        for part in ("all", "dev", "test", "v3", "xquad-full"):
            s = rep[part]
            row[f"{tag}_{part}_factrag"] = rate(s["factrag_correct"])
            row[f"{tag}_{part}_halluc"] = rate(s["hallucination"])
            row[f"{tag}_{part}_false_refusal"] = rate(s["false_refusal"])
        row[f"{tag}_all_macro_f1"] = round(rep["all"]["filter"]["macro_f1"], 4)
        row[f"{tag}_all_mcc"] = round(rep["all"]["filter"]["mcc"], 4)
        row[f"{tag}_thinking_calls"] = rep["thinking_calls"]
        row[f"{tag}_missing_repair"] = rep["missing_repair"]
    for kind, block in (res.get("stress") or {}).items():
        for part in ("dev", "test"):
            if part in block:
                row[f"stress_{kind}_{part}"] = round(100 * block[part]["rate"], 2)
    return row


def rebuild_csv() -> None:
    rows = [
        csv_row(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(EXPERIMENTS.glob("*.json"))
    ]
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (data.OUT / "EXPERIMENTS.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str]) -> None:
    # Run as __main__, this module is a second copy; the registry lives in the imported one.
    from research.opt2 import experiments as registry
    from research.opt2 import variants  # noqa: F401  (registers the research variants)

    if "--list" in argv:
        for variant in registry.VARIANTS.values():
            print(f"{variant.id}: {variant.description}")
        return
    records = data.records(("base", "think"))
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    for name in argv:
        variant = registry.VARIANTS[name]
        res = evaluate(variant, records)
        (EXPERIMENTS / f"{variant.id}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        w = res.get("replay_withheld", {})
        print(
            variant.id,
            "| test",
            rate(w.get("test", {}).get("factrag_correct")),
            "dev",
            rate(w.get("dev", {}).get("factrag_correct")),
            "all",
            rate(w.get("all", {}).get("factrag_correct")),
            "halluc",
            rate(w.get("all", {}).get("hallucination")),
            "| stress dev",
            stress.summary_line(res.get("stress", {}), "dev"),
            "| errors",
            len(res["errors"]),
            f"| {res['seconds']:.0f}s",
            flush=True,
        )
    rebuild_csv()


if __name__ == "__main__":
    main(sys.argv[1:])
