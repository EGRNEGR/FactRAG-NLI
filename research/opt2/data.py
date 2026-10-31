"""Frozen inputs of the optimization_2 research: datasets, run logs and the dev/test split.

The source run (20261031T0508Z, PLAN=full on the RTX 3090) is read in place and never
written. Answers are scored against the dataset versions the run used (sha256 recorded in
its config.json, git commit b058977), not against the later scorer fix, so every number here
is comparable with the run's COMPARISON.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from eval_dataset import EvalCase, EvalDatasetV3, load_dataset

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(
    os.environ.get(
        "OPT2_RUN",
        str(
            Path(r"C:\Hakatons\FactRAG-NLI-3090")
            / "reports"
            / "runs"
            / "Logs_Full"
            / "20261031T0508Z_rtx3090_full"
        ),
    )
)
OUT = ROOT / "reports" / "optimization_2"
CACHE = ROOT / ".artifacts" / "opt2"
DATASETS = ("v3", "xquad-full")
FROZEN = {
    "v3": ("data/eval_dataset_v3.json", "33cb533ee7cc46adfd3edf81fd9c270ed1fe7399f1c0c0b93f321275b71a44d3"),
    "xquad-full": (
        "data/xquad_dataset_full.json",
        "53b3db2531977f9232e450b484b5e8b93b5c61a6621e01b1ee72e5f029997edb",
    ),
}
FROZEN_COMMIT = "b058977"
CONFIGS = {
    "base": "qwen38-27b+bge-reranker-v2-m3",
    "think": "qwen38-27b-think+bge-reranker-v2-m3",
    "qwen25": "qwen25-14b+bge-reranker-v2-m3",
    "large": "qwen38-27b+bge-reranker-large",
}


def frozen_dataset_path(name: str) -> Path:
    """The dataset file of the source run; restored from git once and checked by sha256."""
    relative, sha = FROZEN[name]
    path = CACHE / "datasets" / Path(relative).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = subprocess.run(
            ["git", "show", f"{FROZEN_COMMIT}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        path.write_bytes(blob)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha:
        raise RuntimeError(f"{path}: sha256 {digest} differs from the run's {sha}")
    return path


@cache
def dataset(name: str) -> EvalDatasetV3:
    return load_dataset(frozen_dataset_path(name))


@cache
def cases() -> dict[str, EvalCase]:
    return {case.id: case for name in DATASETS for case in dataset(name).cases}


def dataset_of(case_id: str) -> str:
    return "xquad-full" if case_id.startswith("V3-XQF") else "v3"


def split_group(case: EvalCase) -> str:
    """Leakage unit: XQuAD questions about one paragraph (either language) form one group."""
    if case.id.startswith("V3-XQF") and case.gold_chunk_ids:
        return "xq:" + case.gold_chunk_ids[0]
    if case.id.startswith("V3-XQF"):
        return "xq:" + case.document + ":" + case.id
    return "v3:" + case.id


@cache
def split() -> dict[str, str]:
    """50/50 dev/test split, stratified by dataset and category, grouped by `split_group`.

    Groups are ordered by SHA-1 inside each (dataset, category) stratum and alternate between
    dev and test. Every threshold and routing rule is fitted on dev only; test is touched
    once per candidate for the final numbers.
    """
    strata: dict[tuple[str, str], dict[str, list[str]]] = {}
    for case in cases().values():
        key = (dataset_of(case.id), case.category)
        strata.setdefault(key, {}).setdefault(split_group(case), []).append(case.id)
    result: dict[str, str] = {}
    for groups in strata.values():
        ordered = sorted(groups, key=lambda g: hashlib.sha1(g.encode()).hexdigest())
        for index, group in enumerate(ordered):
            for case_id in groups[group]:
                result[case_id] = "dev" if index % 2 == 0 else "test"
    return result


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@cache
def case_rows(config: str) -> dict[str, dict[str, Any]]:
    """Scored case rows (e2e_guard_on/cases.jsonl) of one configuration, both datasets."""
    rows: dict[str, dict[str, Any]] = {}
    for name in DATASETS:
        rows.update(
            (row["id"], row)
            for row in _jsonl(RUN / CONFIGS[config] / name / "e2e_guard_on" / "cases.jsonl")
        )
    return rows


@cache
def responses(config: str) -> dict[str, dict[str, Any]]:
    """Full logged responses (contexts, drafts, every verification attempt) of one config."""
    result: dict[str, dict[str, Any]] = {}
    for name in DATASETS:
        for row in _jsonl(RUN / CONFIGS[config] / name / "full" / "e2e_guard_on.jsonl"):
            result[row["id"]] = row["response"]
    return result


@dataclass(frozen=True)
class Draft:
    """One generated draft with its verification, as logged."""

    config: str
    attempt: int
    thinking: bool
    raw_text: str
    refused: bool
    claims: list[dict[str, Any]]
    contexts: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class CaseRecord:
    """Everything the offline experiments know about one question."""

    case: EvalCase
    split: str
    drafts: dict[str, list[Draft]] = field(default_factory=dict)
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)


def drafts_of(config: str, response: dict[str, Any]) -> list[Draft]:
    return [
        Draft(
            config=config,
            attempt=index,
            thinking=config == "think",
            raw_text=attempt["raw_text"],
            refused=attempt["refused"],
            claims=attempt["claims"],
            contexts=attempt["contexts"],
            metadata=attempt["generation_metadata"],
        )
        for index, attempt in enumerate(response.get("verification_attempts") or [])
    ]


def records(configs: tuple[str, ...] = ("base", "think")) -> dict[str, CaseRecord]:
    """Case records with drafts and scored rows of `configs` (full logs are large: ~70 MB each)."""
    labels = split()
    result = {
        case_id: CaseRecord(case=case, split=labels[case_id]) for case_id, case in cases().items()
    }
    for config in configs:
        rows = case_rows(config)
        full = responses(config)
        for case_id, record in result.items():
            record.rows[config] = rows[case_id]
            response = full[case_id]
            record.drafts[config] = drafts_of(config, response)
            if config == "base":
                record.retrieval = {
                    "sources": response.get("sources") or [],
                    "diagnostics": response.get("retrieval_diagnostics") or [],
                    "refusal_reason": response.get("refusal_reason"),
                }
    return result
