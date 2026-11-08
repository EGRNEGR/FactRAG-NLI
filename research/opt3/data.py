"""Frozen inputs of the optimization_3 research: the live PLAN=compare run 20261102T0749Z.

The run (RTX 3090, code of commit 816c1de, byte-identical by the code_sha256 of its
config.json) answered all 1307 questions twice: hybrid thinking + recovery with verifier v2
rules ("rules") and with the decision layer ("judge"). Its logs are read in place and never
written. The datasets are the run's own versions (sha256 of its config.json, restored from
commit 816c1de: the XQuAD scorer stems of 545c51f), so every number here matches the run's
COMPARISON.md. The dev/test split is that of optimization_2 (same case ids and groups).

Every logged verification attempt is tagged with its role in pipeline order:
draft -> repair -> think -> think_repair -> recovery (sampling drafts).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from eval_dataset import EvalCase, EvalDatasetV3, load_dataset
from settings import RAGSettings

from research.opt2 import data as opt2

ROOT = opt2.ROOT
RUN = Path(
    os.environ.get(
        "OPT3_RUN",
        str(
            Path(r"C:\Hakatons\FactRAG-NLI-3090")
            / "reports"
            / "runs"
            / "20261102T0749Z_rtx3090_compare"
        ),
    )
)
OUT = ROOT / "reports" / "optimization_3"
CACHE = ROOT / ".artifacts" / "opt3"
DATASETS = opt2.DATASETS
_COLUMNS = {
    "rules": "qwen38-27b-hybrid+bge-reranker-v2-m3",
    "judge": "qwen38-27b-hybrid+bge-reranker-v2-m3+judge",
}
# The columns the run has (PLAN=compare: both; PLAN=main, e.g. 20261103T1310Z: judge only).
CONFIGS = {name: folder for name, folder in _COLUMNS.items() if (RUN / folder).is_dir()}
RUN_ROOT = "/home/artem/Downloads/FactRAG-NLI-3090/FactRAG-NLI-rtx3090-linux"

RUN_COMMIT = "816c1de"
dataset_of = opt2.dataset_of


@cache
def run_config(config: str, dataset: str = "v3") -> dict[str, Any]:
    config_json: dict[str, Any] = json.loads(
        (RUN / CONFIGS[config] / dataset / "config.json").read_text("utf-8")
    )
    return config_json


@cache
def dataset(name: str) -> EvalDatasetV3:
    """The dataset file of the run, checked by its sha256: the checked-out file when it is
    that version (the RTX 3090 copy has no git), else restored from RUN_COMMIT."""
    logged = run_config("judge", name)["dataset"]
    current = ROOT / logged["path"]
    if current.exists() and hashlib.sha256(current.read_bytes()).hexdigest() == logged["sha256"]:
        return load_dataset(current)
    path = CACHE / "datasets" / Path(logged["path"]).name
    if current.exists() and not path.exists():
        # A Windows checkout may hold the run's version with CRLF line endings.
        lf = current.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(lf).hexdigest() == logged["sha256"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(lf)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = subprocess.run(
            ["git", "show", f"{RUN_COMMIT}:{logged['path']}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        path.write_bytes(blob)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != logged["sha256"]:
        raise RuntimeError(f"{path}: sha256 {digest} differs from the run's {logged['sha256']}")
    return load_dataset(path)


@cache
def cases() -> dict[str, EvalCase]:
    return {case.id: case for name in DATASETS for case in dataset(name).cases}


@cache
def split() -> dict[str, str]:
    """The dev/test split of optimization_2 (research.opt2.data.split: stratified by dataset
    and category, grouped by XQuAD paragraph, SHA-1 order), computed on the run's datasets.

    Ids, categories and gold chunks of these files are those of the frozen opt2 files, so the
    split is the same (tests/test_research_split.py); computing it here needs no git.
    """
    strata: dict[tuple[str, str], dict[str, list[str]]] = {}
    for case in cases().values():
        key = (dataset_of(case.id), case.category)
        strata.setdefault(key, {}).setdefault(opt2.split_group(case), []).append(case.id)
    result: dict[str, str] = {}
    for groups in strata.values():
        ordered = sorted(groups, key=lambda g: hashlib.sha1(g.encode()).hexdigest())
        for index, group in enumerate(ordered):
            for case_id in groups[group]:
                result[case_id] = "dev" if index % 2 == 0 else "test"
    return result


def settings(config: str, **overrides: Any) -> RAGSettings:
    """The exact settings of the run (config.json) with local paths and `overrides`.

    Development mode allows the injected cached NLI backends of the replay; the decision
    logic and every threshold are the run's.
    """
    logged = dict(run_config(config)["modes"]["e2e_guard_on"]["settings"])
    for key, value in logged.items():
        if isinstance(value, str) and value.startswith(RUN_ROOT):
            logged[key] = str(ROOT / value[len(RUN_ROOT) + 1 :])
    logged.update(mode="development", allow_fallback=True)
    logged.update(overrides)
    # _env_file is a pydantic-settings init option mypy does not see among the fields.
    no_env_file: dict[str, Any] = {"_env_file": None}
    return RAGSettings(**no_env_file, **logged)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@cache
def case_rows(config: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name in DATASETS:
        path = RUN / CONFIGS[config] / name / "e2e_guard_on" / "cases.jsonl"
        rows.update((row["id"], row) for row in _jsonl(path))
    return rows


@cache
def responses(config: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in DATASETS:
        for row in _jsonl(RUN / CONFIGS[config] / name / "full" / "e2e_guard_on.jsonl"):
            result[row["id"]] = row["response"]
    return result


@dataclass(frozen=True)
class Attempt:
    """One logged generation with the run's verification of it."""

    role: str  # draft | repair | think | think_repair | recovery
    index: int
    raw_text: str
    cleaned_text: str
    refused: bool
    claims: list[dict[str, Any]]
    contexts: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def thinking(self) -> bool:
        return bool(self.metadata.get("thinking"))


def roles(attempts: list[dict[str, Any]]) -> list[str]:
    """Pipeline roles of logged attempts (repairs follow their draft in the same mode)."""
    result: list[str] = []
    previous: str | None = None
    for attempt in attempts:
        meta = attempt["generation_metadata"]
        if meta.get("recovery"):
            role = "recovery"
        elif meta.get("thinking"):
            role = "think_repair" if previous == "think" else "think"
        else:
            role = "repair" if previous == "draft" else "draft"
        result.append(role)
        previous = role
    return result


def attempts_of(response: dict[str, Any]) -> list[Attempt]:
    logged = response.get("verification_attempts") or []
    return [
        Attempt(
            role=role,
            index=index,
            raw_text=a["raw_text"],
            cleaned_text=a["cleaned_text"],
            refused=a["refused"],
            claims=a["claims"],
            contexts=a["contexts"],
            metadata=a["generation_metadata"],
        )
        for index, (a, role) in enumerate(zip(logged, roles(logged), strict=True))
    ]


@dataclass
class CaseRecord:
    case: EvalCase
    split: str
    attempts: dict[str, list[Attempt]] = field(default_factory=dict)
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    retrieval: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.case.id

    def top_rerank(self, config: str = "judge") -> float:
        sources = self.retrieval[config].get("sources") or []
        return max((s.get("rerank_score") or 0.0 for s in sources), default=0.0)


@cache
def records() -> dict[str, CaseRecord]:
    labels = split()
    result = {cid: CaseRecord(case=case, split=labels[cid]) for cid, case in cases().items()}
    for config in CONFIGS:
        rows, full = case_rows(config), responses(config)
        for cid, record in result.items():
            record.rows[config] = rows[cid]
            response = full[cid]
            record.attempts[config] = attempts_of(response)
            record.retrieval[config] = {
                "sources": response.get("sources") or [],
                "diagnostics": response.get("retrieval_diagnostics") or [],
                "refusal_reason": response.get("refusal_reason"),
                "query": response.get("query"),
            }
    return result
