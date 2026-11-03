"""Live check of llm_think_on_trim on the development laptop (not the 3090).

Runs evaluate_suite.py e2e on a few v3 questions with the full pipeline: the run's judge
settings (config.json of 0749Z, local paths), llm_think_on_trim on, the laptop llama-server
with the same GGUF. PyTorch models run on the CPU (the 8 GB GPU holds part of the LLM), so
NLI scores may differ from the 3090 in the last digits; the check is about the pipeline
logic: which questions take the extra thinking pass and what gets published.

Usage:  python -m research.opt3.live --api-base http://127.0.0.1:8093/v1 V3-MH-024 ...
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

from research.opt3 import data

INDEX = Path(r"C:\Hakatons\FactRAG-NLI-3090\.claude\worktrees\opt2\.artifacts")


def subset(ids: list[str]) -> Path:
    source = json.loads((data.ROOT / "data" / "eval_dataset_v3.json").read_text("utf-8"))
    source["cases"] = [c for c in source["cases"] if c["id"] in set(ids)]
    if len(source["cases"]) != len(set(ids)):
        raise SystemExit("unknown case id")
    path = data.CACHE / "live_v3.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(source, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def environment(api_base: str) -> dict[str, str]:
    s = data.settings("judge")
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_")}
    for key, value in s.model_dump(mode="json").items():
        if value is None or key in {"mode", "allow_fallback"}:
            continue
        env[f"RAG_{key.upper()}"] = (
            json.dumps(value) if isinstance(value, (list, dict, bool)) else str(value)
        )
    env.update(
        RAG_LLM_API_BASE=api_base,
        RAG_LLM_API_TIMEOUT="3600",
        RAG_LLM_THINK_ON_TRIM="true",
        RAG_MODEL_DEVICE="cpu",
        RAG_EMBEDDING_DEVICE="cpu",
        RAG_NLI_PRECISION="float32",
        RAG_MODEL_PRECISION="float32",
        RAG_EMBEDDING_BATCH_SIZE="2",
        RAG_RERANK_BATCH_SIZE="2",
        FACTRAG_INDEX=str(INDEX / "eval-v3-index"),
        FACTRAG_PARSE_CACHE=str(INDEX / "eval_v3_parse_cache"),
        PYTHONIOENCODING="utf-8",
        # Production mode needs an API token; the check calls no API, a throwaway one serves.
        RAG_API_TOKENS=json.dumps([secrets.token_hex(24)]),
    )
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8093/v1")
    parser.add_argument("ids", nargs="+")
    args = parser.parse_args()
    dataset = subset(args.ids)
    run_dir = data.OUT / "live_check"
    command = [
        sys.executable, "-X", "utf8", "evaluate_suite.py", "--dataset", str(dataset),
        "--run-dir", str(run_dir), "e2e", "--profile", "guard_on",
    ]
    raise SystemExit(subprocess.run(command, cwd=data.ROOT, env=environment(args.api_base)).returncode)


if __name__ == "__main__":
    main()
