"""Reproduce the whole optimization_2 study from the logs of run 20261031T0508Z.

Inputs: the run folder (OPT2_RUN or --run), the models in models/ (mdeberta-nli,
nli-bge-m3-zeroshot; nli-xlmr-large-xnli and nli-xlmr-large-xnli-anli for the model comparison)
and git history for the frozen datasets. Every stage writes into reports/optimization_2; NLI
predictions are cached in .artifacts/opt2/nli_cache, so a second run takes minutes.

LLM drafts the run never produced (repairs after a stricter verdict, recovery attempts) come
from .artifacts/opt2/{repairs,recovery,think_revise}.jsonl; with --api-base (a llama-server running
Qwen3.8-27B-UD-Q4_K_XL, e.g. scripts/linux/llama_server.sh with LLM_PROFILE=qwen38-27b-hybrid)
missing ones are generated first.

Usage:  python -m research.opt2.reproduce [--run DIR] [--api-base http://127.0.0.1:8080/v1]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

STAGES = [
    ["research.opt2.baseline"],
    ["research.opt2.paired"],
    [
        "research.opt2.experiments",
        "V00_baseline",
        "V01_no_title_route",
        "V01_no_anchor_windows",
        "V01_no_code_span",
        "V01_no_anchor_guard",
        "V01_no_co_verification",
        "V01_window_095",
        "V02_keyword_normalization",
        "V10_bge-m3-zeroshot",
        "V10_xlmr-large-xnli",
        "V10_xlmr-large-xnli-anli",
        "V20_min_mdeberta+bge-m3-zeroshot",
        "V20_mean_mdeberta+bge-m3-zeroshot",
        "V20_min_mdeberta+xlmr-large-xnli",
        "V20_mean_mdeberta+xlmr-large-xnli",
    ],
    ["research.opt2.judge", "bge-m3-zeroshot"],
    ["research.opt2.judge", "bge-m3-zeroshot", "stacked"],
    ["research.opt2.experiments", "V30_judge_bge-m3-zeroshot", "V31_judge_stacked"],
    ["research.opt2.routing", "none"],
    ["research.opt2.routing", "bge-m3-zeroshot"],
    ["research.opt2.retrieval"],
    ["research.opt2.calibration"],
    ["research.opt2.final"],
    ["research.opt2.speed"],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="folder of run 20261031T0508Z (default: OPT2_RUN)")
    parser.add_argument("--api-base", help="llama-server for the missing LLM drafts")
    args = parser.parse_args()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if args.run:
        env["OPT2_RUN"] = args.run
    stages = list(STAGES)
    if args.api_base:
        judge = stages.index(["research.opt2.judge", "bge-m3-zeroshot"])
        stages[judge + 1 : judge + 1] = [
            ["research.opt2.repairs", "--api-base", args.api_base],
            ["research.opt2.recovery", "--api-base", args.api_base],
            ["research.opt2.think_revise", "--api-base", args.api_base],
        ]
    for stage in stages:
        print("==", " ".join(stage), flush=True)
        subprocess.run([sys.executable, "-X", "utf8", "-m", *stage], env=env, check=True)


if __name__ == "__main__":
    main()
