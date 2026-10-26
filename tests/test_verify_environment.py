"""Audit behavior tests; simulated checks do not certify a real deployment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import verify_environment as audit_module


def test_actual_write_probe_and_disk_limit(tmp_path: Path) -> None:
    assert audit_module.writable_directory(tmp_path, 0)["writable"]
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="Free disk"):
        audit_module.writable_directory(tmp_path, 1e12)


def test_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="directory does not exist"):
        audit_module.inspect_transformer(tmp_path / "missing", "embedding")


def test_truncated_safetensors_rejected(tmp_path: Path) -> None:
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / filename).write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"broken")
    with pytest.raises(Exception, match="header"):
        audit_module.inspect_transformer(tmp_path, "embedding")


def test_production_report_is_json_and_fails(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "RAG_MODE": "production",
        "RAG_ALLOW_FALLBACK": "false",
        "RAG_API_TOKENS": "[]",
        "RAG_DOCUMENT_ROOT": str(tmp_path / "documents"),
        "RAG_QDRANT_PATH": str(tmp_path / "qdrant"),
        "RAG_NETWORK_PROBE_TIMEOUT": "0.1",
    }
    process = subprocess.run(
        [sys.executable, "scripts/verify_environment.py"],
        env=environment,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=15,
        check=False,
    )
    report = json.loads(process.stdout)
    assert process.returncode == 1 and report["ok"] is False
    failures = {item["name"] for item in report["checks"] if not item["ok"]}
    assert {"settings", "embedding", "reranker", "nli", "llm"} <= failures


@pytest.mark.parametrize(
    "policy,connected,expected",
    [("local_only", True, 0), ("air_gap", True, 1), ("air_gap", False, 0)],
)
def test_network_policy_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, policy: str, connected: bool, expected: int
) -> None:
    monkeypatch.setenv("RAG_NETWORK_POLICY", policy)
    monkeypatch.setenv("RAG_DOCUMENT_ROOT", str(tmp_path / "documents"))
    monkeypatch.setenv("RAG_MINIMUM_FREE_DISK_GB", "0")

    def model_check(*args: Any) -> dict[str, bool]:
        return {"test_check": True}

    monkeypatch.setattr(audit_module, "inspect_transformer", model_check)
    monkeypatch.setattr(audit_module, "inspect_llm", model_check)
    monkeypatch.setattr(audit_module, "hardware", model_check)
    monkeypatch.setattr(
        audit_module,
        "probe_network",
        lambda timeout: [
            {
                "dns_resolved": connected,
                "tcp_connected": connected,
                "dns_failed": not connected,
                "probe_exit_code": 0,
            }
        ],
    )
    report, code = audit_module.audit()
    assert code == expected
    network = next(item for item in report["checks"] if item["name"] == "network")
    assert network["details"]["air_gap_certified"] is False


def test_timeout_does_not_prove_air_gap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAG_NETWORK_POLICY", "air_gap")
    monkeypatch.setenv("RAG_DOCUMENT_ROOT", str(tmp_path / "documents"))
    monkeypatch.setenv("RAG_MINIMUM_FREE_DISK_GB", "0")
    monkeypatch.setattr(
        audit_module,
        "probe_network",
        lambda timeout: [{"dns_resolved": False, "tcp_connected": False, "timeout": True}],
    )
    report, code = audit_module.audit()
    assert code == 1
    assert next(item for item in report["checks"] if item["name"] == "network")["ok"] is False
