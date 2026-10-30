#!/usr/bin/env bash
# One-time installation of FactRAG-NLI on a Linux PC with an RTX 3090.
#   bash scripts/linux/setup.sh            # everything: venv, llama.cpp, models, .env, checks
#   SKIP_MODELS=1 bash scripts/linux/setup.sh
#   LLM_PROFILES="qwen38-27b" bash scripts/linux/setup.sh    # without the Qwen2.5 baseline
# Needs: NVIDIA driver (nvidia-smi), python3 >= 3.11 with venv, curl, tar; ~45 GB of disk.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.rtx3090.sh"
cd "$PROJECT_ROOT"
LLM_PROFILES="${LLM_PROFILES:-qwen38-27b qwen25-14b}"

step() { printf '\n=== %s\n' "$*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

step "1/7 System check"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found: install the NVIDIA driver"
nvidia-smi --query-gpu=name,memory.total,driver_version,temperature.gpu --format=csv,noheader
command -v curl >/dev/null || fail "curl not found: sudo apt install curl"
PY_BOOT="${PY_BOOT:-}"
if [[ -z "$PY_BOOT" ]]; then
  for candidate in python3.12 python3.13 python3.11 python3; do
    if command -v "$candidate" >/dev/null &&
      "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
      PY_BOOT="$candidate"; break
    fi
  done
fi
[[ -n "$PY_BOOT" ]] || fail "Python >= 3.11 not found: sudo apt install python3.12 python3.12-venv"
echo "Python: $($PY_BOOT --version) ($PY_BOOT)"
df -h "$PROJECT_ROOT" | tail -1 | awk '{print "Free disk: " $4}'

step "2/7 Python environment (.venv)"
if [[ ! -x "$PYTHON" ]]; then
  "$PY_BOOT" -m venv .venv || fail "venv failed: sudo apt install ${PY_BOOT}-venv"
fi
"$PYTHON" -m pip install --upgrade pip wheel
# PyTorch wheels bundle the CUDA 12.8 runtime; driver 595 (CUDA 13.2) runs them.
"$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
"$PYTHON" -m pip install -e ".[web,documents,retrieval,audit,dev]" pyarrow huggingface_hub
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch does not see the GPU"
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {name}, sm_{major}{minor}, "
      f"bf16={torch.cuda.is_bf16_supported()}")
PY

step "3/7 llama.cpp $LLAMA_BUILD ($LLAMA_ASSET)"
if [[ -x "$LLAMA_DIR/llama-server" ]] && [[ "$(cat "$LLAMA_DIR/BUILD" 2>/dev/null)" == "$LLAMA_BUILD-$LLAMA_ASSET" ]]; then
  echo "already installed: $LLAMA_DIR"
else
  tmp="$(mktemp -d)"
  base="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD"
  curl -fL --retry 3 -o "$tmp/llama.tar.gz" "$base/llama-$LLAMA_BUILD-bin-$LLAMA_ASSET.tar.gz"
  curl -fL --retry 3 -o "$tmp/cudart.tar.gz" "$base/cudart-llama-$LLAMA_BUILD-bin-$LLAMA_ASSET.tar.gz"
  mkdir -p "$tmp/x"
  tar -xzf "$tmp/llama.tar.gz" -C "$tmp/x"
  tar -xzf "$tmp/cudart.tar.gz" -C "$tmp/x"
  server="$(find "$tmp/x" -type f -name llama-server | head -1)"
  [[ -n "$server" ]] || fail "llama-server not found in the release archive"
  rm -rf "$LLAMA_DIR"; mkdir -p "$LLAMA_DIR"
  cp -a "$(dirname "$server")"/. "$LLAMA_DIR"/
  # CUDA runtime libraries may sit in another folder of the cudart archive.
  find "$tmp/x" -name 'lib*.so*' -exec cp -an {} "$LLAMA_DIR"/ \;
  echo "$LLAMA_BUILD-$LLAMA_ASSET" >"$LLAMA_DIR/BUILD"
  rm -rf "$tmp"
fi
LD_LIBRARY_PATH="$LLAMA_DIR" "$LLAMA_DIR/llama-server" --version 2>&1 | tail -n 3
LD_LIBRARY_PATH="$LLAMA_DIR" "$LLAMA_DIR/llama-server" --list-devices 2>&1 | grep -i -m1 "cuda\|3090" ||
  fail "llama-server does not see a CUDA device (check $LLAMA_DIR and the driver)"

step "4/7 Models (Hugging Face)"
if [[ "${SKIP_MODELS:-0}" == "1" ]]; then
  echo "SKIP_MODELS=1"
else
  # Encoders: huggingface_hub without the Xet transfer (it stalled on a 16 GB file in testing).
  HF_HUB_DISABLE_XET=1 "$PYTHON" - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

ENCODERS = {
    # name: (repo, ignored patterns); ONNX exports and duplicate .bin weights are not used.
    "bge-m3": ("BAAI/bge-m3", ["onnx/*", "imgs/*", "*.jpg"]),
    "bge-reranker-large": ("BAAI/bge-reranker-large", ["onnx/*", "pytorch_model.bin"]),
    "bge-reranker-v2-m3": ("BAAI/bge-reranker-v2-m3", ["onnx/*", "pytorch_model.bin"]),
    "mdeberta-nli": (
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        ["onnx/*", "pytorch_model.bin"],
    ),
}
for name, (repo, ignore) in ENCODERS.items():
    print(f"- {repo} -> models/{name}", flush=True)
    snapshot_download(repo, local_dir=Path("models") / name, ignore_patterns=ignore)
PY
  # GGUF: resumable curl (an interrupted 16 GB download continues) and a SHA-256 check.
  mkdir -p models/llm
  for profile in $LLM_PROFILES; do
    eval "$(LLM_PROFILE="$profile" bash -c 'source scripts/linux/env.rtx3090.sh && printf "repo=%q file=%q sha=%q" "$LLM_REPO" "$LLM_FILE" "$LLM_SHA256"')"
    target="models/llm/$file"
    if [[ -f "$target" ]]; then
      echo "- $file present"
    else
      echo "- $repo/$file -> models/llm"
      curl -fL --retry 20 --retry-delay 5 -C - -o "$target.part" \
        "https://huggingface.co/$repo/resolve/main/$file"
      mv "$target.part" "$target"
    fi
    if [[ ! -f "$target.sha256ok" ]]; then
      echo "  checking SHA-256 ..."
      echo "$sha  $target" | sha256sum -c - || fail "$target is corrupt: delete it and rerun setup"
      touch "$target.sha256ok"
    fi
  done
fi

step "5/7 Configuration (.env)"
if [[ ! -f .env ]]; then
  token="$("$PYTHON" -c 'import json, secrets; print(json.dumps([secrets.token_urlsafe(48)]))')"
  sed "s|^RAG_API_TOKENS=.*|RAG_API_TOKENS=$token|" .env.rtx3090.example >.env
  echo "created .env from .env.rtx3090.example (API token generated)"
else
  echo ".env exists, left unchanged (template: .env.rtx3090.example)"
fi

step "6/7 Datasets"
# Both builds re-parse the corpus with this machine's parser and fail if any evidence quote no
# longer binds to a chunk; they also leave the parse caches that the replay mode reads.
"$PYTHON" -X utf8 scripts/build_eval_dataset_v3.py --check >/dev/null
"$PYTHON" -X utf8 scripts/build_xquad_dataset.py --check

step "7/7 Tests"
"$PYTHON" -m pytest -q -x tests/test_llm_http.py tests/test_settings.py tests/test_evaluate_suite.py tests/test_replay.py \
  tests/test_gpu_duty_cycle.py

printf '\nSetup complete. Start the benchmark:\n  bash scripts/linux/run_benchmark.sh\n'
