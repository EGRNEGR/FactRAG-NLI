# FactRAG-NLI on one RTX 3090 (24 GB, Ampere sm_86) under Linux.
# Sourced by setup.sh, llama_server.sh and run_benchmark.sh; every value can be overridden
# from the environment, e.g.  LLM_PROFILE=qwen25-14b bash scripts/linux/run_benchmark.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT

# ---------------------------------------------------------------- LLM profiles
# qwen38-27b  Qwen3.8-27B (Qwen, 2026-08), dense hybrid Gated DeltaNet/attention; only 16 of 64
#             layers keep a KV cache (4 KV heads x 256), so 8K context costs ~0.5 GB.
#             UD-Q4_K_XL 16.4 GB + KV/compute ~1.5 GB + BGE-M3, reranker, mDeBERTa ~3.5 GB
#             fits under the ~23.4 GB left by the desktop session.
# qwen38-27b-think  the same weights with thinking on: llama-server caps the hidden reasoning
#             at LLAMA_REASONING_BUDGET tokens and RAG_MAX_TOKENS rises to 1024 (the settings
#             ceiling). In a laptop test it answered an inverted question ("who lost to X" from
#             "X beat Y") that the non-thinking mode refused; it costs ~2-3x generation time.
# qwen25-14b  the previous model (Qwen2.5-14B-Instruct Q4_K_M, 8.4 GB), fully offloaded:
#             the same-hardware baseline for the comparison.
LLM_PROFILE="${LLM_PROFILE:-qwen38-27b}"
case "$LLM_PROFILE" in
  qwen38-27b)
    LLM_REPO="unsloth/Qwen3.8-27B-GGUF"
    LLM_FILE="Qwen3.8-27B-UD-Q4_K_XL.gguf"
    LLM_SHA256="3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e"
    LLM_THINKING_OFF=true
    LLM_REASONING_BUDGET=-1
    LLM_MAX_TOKENS=512
    ;;
  qwen38-27b-think)
    LLM_REPO="unsloth/Qwen3.8-27B-GGUF"
    LLM_FILE="Qwen3.8-27B-UD-Q4_K_XL.gguf"
    LLM_SHA256="3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e"
    LLM_THINKING_OFF=false
    LLM_REASONING_BUDGET="${LLAMA_REASONING_BUDGET:-640}"
    LLM_MAX_TOKENS=1024
    ;;
  qwen38-27b-q5)
    # Only on a headless machine (sudo systemctl isolate multi-user.target): ~1 GB more VRAM.
    LLM_REPO="unsloth/Qwen3.8-27B-GGUF"
    LLM_FILE="Qwen3.8-27B-UD-Q5_K_M.gguf"
    LLM_SHA256="2de73110cb254cbf09b54b717578dadff12ef1194e7271527e68202f39ba4bfd"
    LLM_THINKING_OFF=true
    LLM_REASONING_BUDGET=-1
    LLM_MAX_TOKENS=512
    ;;
  qwen25-14b)
    LLM_REPO="bartowski/Qwen2.5-14B-Instruct-GGUF"
    LLM_FILE="Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    LLM_SHA256="e47ad95dad6ff848b431053b375adb5d39321290ea2c638682577dafca87c008"
    LLM_THINKING_OFF=false
    LLM_REASONING_BUDGET=-1
    LLM_MAX_TOKENS=512
    ;;
  *)
    echo "Unknown LLM_PROFILE=$LLM_PROFILE (qwen38-27b | qwen38-27b-think | qwen38-27b-q5 | qwen25-14b)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
# LLM_REASONING_BUDGET -1 = no --reasoning-budget flag (non-thinking profiles)
export LLM_PROFILE LLM_REPO LLM_FILE LLM_SHA256 LLM_THINKING_OFF LLM_REASONING_BUDGET
export RAG_MAX_TOKENS="$LLM_MAX_TOKENS"
# Derived values are always recomputed: with_profile.sh re-sources this file under another
# profile inside an environment that already holds the parent's exports.
export LLM_PATH="models/llm/$LLM_FILE"

# ---------------------------------------------------------------- llama.cpp
# CUDA 12.8 build: driver 595 (CUDA 13.2) runs it natively; the 13.4 build would rely on
# minor-version compatibility.
export LLAMA_BUILD="${LLAMA_BUILD:-b12002}"
export LLAMA_ASSET="${LLAMA_ASSET:-ubuntu-cuda-12.8-x64}"
export LLAMA_DIR="${LLAMA_DIR:-$PROJECT_ROOT/tools/llama}"
export LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
export LLAMA_PORT="${LLAMA_PORT:-8080}"
export LLAMA_CTX="${LLAMA_CTX:-8192}"
export LLAMA_GPU_LAYERS="${LLAMA_GPU_LAYERS:-999}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---------------------------------------------------------------- retrieval runtime
# The 8 GB laptop needed batch 2 next to llama-server; 24 GB takes 16/8 (code defaults).
export FACTRAG_EMBEDDING_BATCH="${FACTRAG_EMBEDDING_BATCH:-16}"
export FACTRAG_RERANK_BATCH="${FACTRAG_RERANK_BATCH:-8}"
# llama-server holds ~18 GB next to the encoders: growable segments keep PyTorch's cache from
# fragmenting the remaining few GB over a multi-hour run.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Reranker: bge-reranker-v2-m3 (multilingual) is the default since run 20261030T1420Z: with the
# top-scope gate below, Recall@5 of the LLM context is 0.837 on eval_dataset_v3 and 0.987 on
# XQuAD, against 0.820 / 0.671 for bge-reranker-large with the per-chunk threshold of that run.
export RERANKER="${RERANKER:-bge-reranker-v2-m3}"
# Gate scope "top": the threshold only decides refusal (the best substantive chunk must reach
# it) and the top_k ranked chunks reach the LLM. With "chunk" every source had to pass the
# threshold, which dropped low-scored gold chunks behind confident distractors.
export RAG_RERANK_GATE_SCOPE="${GATE_SCOPE:-top}"
# The threshold depends on the reranker's score scale. v2-m3: 0.02, chosen on the dev halves of
# eval_dataset_v3 and XQuAD (an answerable question rejected costs a likely correct answer, an
# unanswerable one passed costs a refusal the LLM gives anyway in ~95% of cases): dev rejects
# 2 of 164 answerable and 21 of 46 unanswerable, test 0 of 166 and 19 of 42.
# bge-reranker-large: 0.05, the production value.
case "$RERANKER" in
  bge-reranker-large) default_gate=0.05 ;;
  bge-reranker-v2-m3) default_gate=0.02 ;;
  *) default_gate=0.05 ;;
esac
# GATE_THRESHOLD overrides it for every reranker.
export RAG_RERANK_SCORE_THRESHOLD="${GATE_THRESHOLD:-$default_gate}"
unset default_gate
# Ten ranked chunks reach the LLM. Local A/B with Qwen3.8 on eval_dataset_v3 (40 cases: 16 whose
# evidence ranks 6-10, 12 unanswerable, 12 adversarial): top_k 10 answered 37 correctly against
# 28 with top_k 5, no new hallucination; generation ~1.7x longer. TOP_K=5 restores the old value.
export RAG_RERANK_TOP_K="${TOP_K:-10}"
# NLI hypothesis without a leading "Нет," / "No,": Qwen3.8 opens corrections this way and
# mDeBERTa read the verdict as a contradiction (replay of its drafts: 3 correct corrections
# verified instead of rejected; Qwen2.5 drafts of run 20261026T2139Z unchanged).
export RAG_NLI_STRIP_VERDICT_PREFIX="${STRIP_VERDICT_PREFIX:-true}"

# ---------------------------------------------------------------- GPU duty cycle
export WORK_MINUTES="${WORK_MINUTES:-45}"
export REST_MINUTES="${REST_MINUTES:-5}"
# An early rest starts when the GPU reaches this temperature (Celsius).
export GPU_TEMP_LIMIT="${GPU_TEMP_LIMIT:-83}"

# A ROS 2 setup (source /opt/ros/*/setup.bash) exports PYTHONPATH, whose entries precede the
# venv's site-packages; the project needs none of them.
unset PYTHONPATH
export PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
export PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_TELEMETRY=1

# ---------------------------------------------------------------- pipeline settings
# Environment variables take precedence over .env in RAGSettings, so these follow the profile.
export RAG_LLM_MODEL_PATH="$LLM_PATH"
export RAG_LLM_API_BASE="http://$LLAMA_HOST:$LLAMA_PORT/v1"
export RAG_LLM_DISABLE_THINKING="$LLM_THINKING_OFF"
# 8192 lets all five reranked sources into the prompt (with 4096, S5 was omitted in 4 of 423
# generations of run 20261026T2139Z_v3).
export RAG_LLM_CONTEXT_WINDOW="$LLAMA_CTX"
export RAG_RERANKER_MODEL_PATH="models/$RERANKER"
export RAG_EMBEDDING_BATCH_SIZE="$FACTRAG_EMBEDDING_BATCH"
export RAG_RERANK_BATCH_SIZE="$FACTRAG_RERANK_BATCH"
