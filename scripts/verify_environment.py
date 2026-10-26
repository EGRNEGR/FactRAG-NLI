"""Audit local RAG artifacts/resources. Print one JSON object; exit 0 or 1.

Run from the deployment directory: python scripts/verify_environment.py.
DNS/TCP probes are explicit diagnostics, never model or document uploads.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Self
from functools import partial
from pydantic import model_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import RAGSettings, validate_transformer_directory  # noqa: E402


class _DiagnosticSettings(RAGSettings):
    """Read typed paths after a failed deployment check, solely to report all failures.

    Never passed to a runtime loader. The original validation error remains fatal.
    """

    @model_validator(mode="after")
    def validate_deployment(self) -> Self:
        return self


def inspect_transformer(path: Path, role: str) -> dict[str, Any]:
    """Parse tensor containers on CPU; verify classifier/tokenizer structure."""
    validate_transformer_directory(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if role == "embedding" and not (path / "tokenizer.json").is_file():
        raise ValueError("BGE-M3 requires tokenizer.json")
    files: set[Path] = set()
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path / index_name
        if index.exists():
            files.update(
                path / name for name in json.loads(index.read_text())["weight_map"].values()
            )
            break
    if not files:
        files = {
            next(p for p in (path / "model.safetensors", path / "pytorch_model.bin") if p.is_file())
        }
    keys: set[str] = set()
    hashes: dict[str, str] = {}
    for artifact in sorted(files):
        with artifact.open("rb") as stream:
            hashes[artifact.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        if artifact.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(artifact), framework="numpy") as tensors:
                for name in tensors.keys():
                    tensors.get_slice(name).get_shape()
                    keys.add(name)
        else:
            torch = importlib.import_module("torch")
            state = torch.load(str(artifact), map_location="meta", weights_only=True)
            if not isinstance(state, dict):
                raise ValueError(f"Invalid state dict: {artifact.name}")
            keys.update(state)
    if not keys:
        raise ValueError("Empty checkpoint")
    if role in {"reranker", "nli"} and not any(
        "classifier" in name or "classification_head" in name or name.endswith("score.weight")
        for name in keys
    ):
        raise ValueError("Classification head absent from checkpoint")
    if role == "nli":
        labels = config.get("id2label", {})
        if len(labels) != 3:
            raise ValueError("NLI must declare three labels in id2label")
    transformers = importlib.import_module("transformers")
    transformers.AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )
    return {
        "path": str(path),
        "tensors": len(keys),
        "sha256": hashes,
        "integrity": "container parsed; hashes recorded, not matched to a trusted manifest",
    }


def inspect_llm(settings: RAGSettings) -> dict[str, Any]:
    """Parse GGUF or a complete Transformers/AWQ directory."""
    path = settings.llm_model_path
    if path.is_dir():
        return inspect_transformer(path, "llm")
    from gguf import GGUFReader

    reader = GGUFReader(str(path), mode="r")
    if not reader.tensors:
        raise ValueError("GGUF contains no tensors")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "tensors": len(reader.tensors), "sha256": digest}


def writable_directory(path: Path, minimum_gb: float) -> dict[str, Any]:
    """Test an actual write/fsync/read/delete; do not rely on os.access."""
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path) as stream:
        stream.write(b"rag-write-probe")
        stream.flush()
        os.fsync(stream.fileno())
        stream.seek(0)
        if stream.read() != b"rag-write-probe":
            raise OSError("Read-after-write probe failed")
    free = shutil.disk_usage(path).free / 1024**3
    if free < minimum_gb:
        raise ValueError(f"Free disk {free:.2f} GiB is below {minimum_gb} GiB")
    return {"path": str(path), "writable": True, "free_disk_gib": round(free, 3)}


def hardware(minimum_ram: float) -> dict[str, Any]:
    """Report free host memory and CUDA/ROCm properties when torch is installed."""
    import psutil

    memory = psutil.virtual_memory()
    result: dict[str, Any] = {
        "ram_total_gib": memory.total / 1024**3,
        "ram_available_gib": memory.available / 1024**3,
        "gpus": [],
    }
    if result["ram_available_gib"] < minimum_ram:
        raise ValueError(f"Available RAM {result['ram_available_gib']:.2f} GiB < {minimum_ram} GiB")
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        result["accelerator_detection"] = "unavailable: torch is not installed"
        return result
    result["accelerator_detection"] = (
        "rocm" if torch.version.hip else "cuda" if torch.cuda.is_available() else "cpu"
    )
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        result["gpus"].append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "vram_free_gib": free / 1024**3,
                "vram_total_gib": total / 1024**3,
            }
        )
    return result


def probe_network(timeout: float) -> list[dict[str, Any]]:
    """Bound DNS resolution in a subprocess (socket timeouts do not bound DNS)."""
    code = """import socket, sys, json
host, timeout = sys.argv[1], float(sys.argv[2])
try:
    addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
except socket.gaierror:
    print(json.dumps({'dns_failed': True}), flush=True)
    sys.exit(0)
print(json.dumps({'dns_resolved': bool(addresses)}), flush=True)
try:
    connection = socket.create_connection((host, 443), timeout=timeout)
except OSError:
    print(json.dumps({'tcp_failed': True}), flush=True)
else:
    connection.close()
    print(json.dumps({'tcp_connected': True}), flush=True)
"""
    observations: list[dict[str, Any]] = []
    for host in ("huggingface.co", "openai.com"):
        entry: dict[str, Any] = {"host": host, "dns_resolved": False, "tcp_connected": False}
        try:
            process = subprocess.run(
                [sys.executable, "-c", code, host, str(timeout)],
                capture_output=True,
                text=True,
                timeout=timeout * 2,
                check=False,
            )
            for line in process.stdout.splitlines():
                entry.update(json.loads(line))
            entry["probe_exit_code"] = process.returncode
        except subprocess.TimeoutExpired as exc:
            raw_output = exc.stdout or b""
            decoded = (
                raw_output.decode("utf-8", errors="replace")
                if isinstance(raw_output, bytes)
                else raw_output
            )
            for line in decoded.splitlines():
                entry.update(json.loads(line))
            entry["timeout"] = True
        observations.append(entry)
    return observations


def audit() -> tuple[dict[str, Any], int]:
    """Collect independent checks; missing artifacts never disappear in fallback mode."""
    checks: list[dict[str, Any]] = []

    def check(name: str, action: Callable[[], Any]) -> None:
        try:
            checks.append({"name": name, "ok": True, "details": action()})
        except Exception as exc:
            checks.append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    settings: RAGSettings | None = None
    try:
        settings = RAGSettings()
        checks.append({"name": "settings", "ok": True})
    except Exception as exc:
        checks.append({"name": "settings", "ok": False, "error": str(exc)})
        try:
            settings = _DiagnosticSettings()
        except Exception as diagnostic_error:
            checks.append(
                {"name": "diagnostic_settings", "ok": False, "error": str(diagnostic_error)}
            )
    RAGSettings.activate_offline_mode()
    check("hardware", lambda: hardware(settings.minimum_free_ram_gb if settings else 16.0))
    if settings is not None:
        for role, path in (
            ("embedding", settings.embedding_model_path),
            ("reranker", settings.reranker_model_path),
            ("nli", settings.nli_model_path),
        ):
            check(role, partial(inspect_transformer, path, role))
        check("llm", lambda: inspect_llm(settings))
        for name, path in (
            ("qdrant_storage", settings.qdrant_storage_path),
            ("documents", settings.document_root),
        ):
            check(name, partial(writable_directory, path, settings.minimum_free_disk_gb))

        def qdrant_probe() -> dict[str, Any]:
            from qdrant_client import QdrantClient

            client = QdrantClient(path=str(settings.qdrant_storage_path))
            try:
                return {"collections": [c.name for c in client.get_collections().collections]}
            finally:
                client.close()

        check("qdrant", qdrant_probe)
        observations = probe_network(settings.network_probe_timeout)
        observed_blocked = all(
            e.get("dns_failed") is True and e.get("probe_exit_code") == 0 for e in observations
        )
        checks.append(
            {
                "name": "network",
                "ok": settings.network_policy == "local_only" or observed_blocked,
                "details": {
                    "policy": settings.network_policy,
                    "observations": observations,
                    "local_files_only": True,
                    "offline_environment_enabled": True,
                    "air_gap_certified": False,
                    "scope": "Local model loaders prohibit downloads. DNS/TCP probes are observations, not a firewall audit.",
                },
            }
        )
    ok = all(item["ok"] for item in checks)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "environment": settings.environment if settings else "invalid",
        "checks": checks,
    }, 0 if ok else 1


def main() -> int:
    """Keep stdout machine-readable even when third-party imports emit diagnostics."""
    with contextlib.redirect_stdout(sys.stderr):
        try:
            report, code = audit()
        except Exception as exc:
            report, code = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
