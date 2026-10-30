"""LLM server watchdog of evaluate_suite.run_e2e: recovery, case retry and the error stop."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import evaluate_suite
from eval_dataset import load_dataset
from evaluate_suite import LLMWatchdog, run_e2e

ROOT = Path(__file__).resolve().parents[1]


class FakePipeline:
    """Raises for the first `failures` queries, then refuses every question."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def query(self, question: str) -> Any:
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("All connection attempts failed")
        return SimpleNamespace(
            verification_attempts=[],
            answer="refused",
            raw_answer="refused",
            refused=True,
            refusal_reason="no_evidence",
            claims=[],
            sources=[],
            generation_metadata=None,
            execution_stats={},
            model_dump=lambda mode: {},
        )

    def close(self) -> None:
        pass


class FakeWatchdog:
    """Stands in for LLMWatchdog; `recovers` says whether recover() brings the server back."""

    instances: list[FakeWatchdog] = []

    def __init__(self, api_base: str | None, command: str, *, recovers: bool) -> None:
        self.command = ["true"]
        self.recovers = recovers
        self.recoveries = 0
        FakeWatchdog.instances.append(self)

    def healthy(self) -> bool:
        return True

    def recover(self) -> bool:
        if self.recovers:
            self.recoveries += 1
        return self.recovers


class NoMonitor:
    def __enter__(self) -> NoMonitor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        return {}


def run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failures: int, recovers: bool
) -> tuple[dict[str, Any], FakePipeline, list[dict[str, Any]]]:
    pipeline = FakePipeline(failures)
    FakeWatchdog.instances.clear()
    monkeypatch.setattr(
        evaluate_suite,
        "build_settings",
        lambda dataset, overrides: SimpleNamespace(
            llm_api_base="http://127.0.0.1:8080/v1", llm_seed=1
        ),
    )
    monkeypatch.setattr(evaluate_suite, "open_pipeline", lambda dataset, settings: pipeline)
    monkeypatch.setattr(evaluate_suite, "ResourceMonitor", NoMonitor)
    monkeypatch.setattr(evaluate_suite, "write_config", lambda *args: None)
    monkeypatch.setattr(evaluate_suite, "settings_record", lambda settings: {})
    monkeypatch.setattr(
        evaluate_suite,
        "LLMWatchdog",
        lambda api_base, command: FakeWatchdog(api_base, command, recovers=recovers),
    )
    monkeypatch.setattr(evaluate_suite, "MAX_CONSECUTIVE_ERRORS", 3)
    dataset = load_dataset(ROOT / "data" / "eval_dataset_v3.json")
    summary = run_e2e(dataset, ROOT / "data" / "eval_dataset_v3.json", tmp_path, "guard_on", 6)
    log = tmp_path / "e2e_guard_on" / "cases.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return summary, pipeline, rows


def test_recovered_server_reruns_the_failed_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summary, pipeline, rows = run(monkeypatch, tmp_path, failures=1, recovers=True)
    assert summary["errors"] == 0 and len(rows) == 6
    assert pipeline.calls == 7  # the failed first case ran once more
    assert summary["llm_server_recoveries"] == 1
    assert "aborted" not in summary


def test_consecutive_errors_stop_the_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    summary, pipeline, rows = run(monkeypatch, tmp_path, failures=100, recovers=False)
    # Three failures in a row end the stage; the other cases stay unlogged for the retry stage.
    assert len(rows) == 3 and all("error" in row for row in rows)
    assert pipeline.calls == 3
    assert summary["aborted"].startswith("3 consecutive errors")


def test_isolated_errors_do_not_stop_the_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summary, _, rows = run(monkeypatch, tmp_path, failures=2, recovers=False)
    assert len(rows) == 6 and summary["errors"] == 2
    assert "aborted" not in summary


def test_health_url_and_missing_command() -> None:
    watchdog = LLMWatchdog("http://127.0.0.1:8080/v1/", "")
    assert watchdog.health_url == "http://127.0.0.1:8080/health"
    assert watchdog.command == []
    # Nothing answers on this port and there is no recovery command.
    assert LLMWatchdog("http://127.0.0.1:9/v1", "").recover() is False
    assert LLMWatchdog(None, "bash x.sh ensure").healthy() is True
