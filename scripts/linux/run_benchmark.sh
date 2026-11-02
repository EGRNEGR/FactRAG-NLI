#!/usr/bin/env bash
# FactRAG-NLI benchmark on the RTX 3090 with GPU rest periods (45 min work / 5 min rest).
#
#   bash scripts/linux/run_benchmark.sh              # PLAN=full, XQUAD=full, detached, shows the log
#   PLAN=main bash scripts/linux/run_benchmark.sh    # best configuration only, ~3.5 h
#   PLAN=compare bash scripts/linux/run_benchmark.sh # best configuration, judge vs rules, ~7 h
#   PLAN=quick bash scripts/linux/run_benchmark.sh   # 10 cases per dataset, ~15 min smoke test
#   bash scripts/linux/run_benchmark.sh --resume     # continue the last run after a stop/reboot
#   bash scripts/linux/run_benchmark.sh --foreground # stay attached to the terminal
#
# Plans (configuration = LLM profile + reranker + verifier); every e2e stage runs
# eval_dataset_v3 (220) and XQuAD:
#   full   qwen38-27b-hybrid+bge-reranker-v2-m3+judge (main: decision layer, thinking pass,
#          recovery), qwen38-27b-hybrid+bge-reranker-v2-m3 (the same with verifier v2: the
#          decision-layer ablation), qwen38-27b+bge-reranker-v2-m3 (no thinking, the 0508Z
#          reference), qwen25-14b+bge-reranker-v2-m3+judge (the previous LLM, where the verifier
#          matters most); ~10.5 h. After run 20261031T0508Z (reports/optimization_2) the plan
#          dropped bge-reranker-large (75.2% against 84.2%) and qwen38-27b-think (always thinking:
#          87.8% against 89.8% for the hybrid at 2.6x the generation time).
#   compare  qwen38-27b-hybrid+bge-reranker-v2-m3 with the decision layer (+judge) and with
#          verifier v2: the first two rows of full, no Qwen2.5 download; ~7 h
#   main   qwen38-27b-hybrid+bge-reranker-v2-m3+$VERIFIER only (default judge;
#          VERIFIER=rules runs verifier v2 without the second NLI model)
#   quick  main with 10 cases per dataset
#   XQUAD  full (all 1087 eligible questions, default: 1307 questions with v3)
#          | standard (198 sampled questions, the set of run 20261030T1420Z)
# PLAN and XQUAD given together with --resume switch the last run over (finished cases stay).
# The run is detached (setsid + nohup): closing the AnyDesk session or the terminal does not
# stop it. Ctrl+C only stops watching the log. Stop the run: bash scripts/linux/stop_benchmark.sh
set -euo pipefail
verifier_now="${VERIFIER:-}"  # env.rtx3090.sh fills in the default (judge)
source "$(dirname "${BASH_SOURCE[0]}")/env.rtx3090.sh"
cd "$PROJECT_ROOT"

FOREGROUND=0
RESUME=0
for arg in "$@"; do
  case "$arg" in
    --foreground) FOREGROUND=1 ;;
    --resume) RESUME=1 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

STATE=.artifacts/benchmark
mkdir -p "$STATE"
if [[ "$RESUME" == "1" ]]; then
  [[ -f "$STATE/last_run" ]] || { echo "Nothing to resume" >&2; exit 1; }
  # RUN_ID, PLAN, XQUAD and VERIFIER of the interrupted run; values given now take precedence
  # (finished cases are kept, the remaining stages follow the new choice).
  plan_now="${PLAN:-}" xquad_now="${XQUAD:-}"
  VERIFIER=judge  # runs before VERIFIER was recorded
  source "$STATE/last_run"
  PLAN="${plan_now:-$PLAN}"
  XQUAD="${xquad_now:-${XQUAD:-standard}}"  # runs before XQUAD existed used the standard set
  VERIFIER="${verifier_now:-$VERIFIER}"
else
  PLAN="${PLAN:-full}"
  XQUAD="${XQUAD:-full}"
  RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%MZ)_rtx3090_$PLAN}"
fi
printf 'RUN_ID=%q\nPLAN=%q\nXQUAD=%q\nVERIFIER=%q\n' "$RUN_ID" "$PLAN" "$XQUAD" "$VERIFIER" >"$STATE/last_run"
RUN_ROOT="reports/runs/$RUN_ID"
LOG="$RUN_ROOT/benchmark.log"
mkdir -p "$RUN_ROOT"

if [[ "$FOREGROUND" == "0" && -z "${FACTRAG_DETACHED:-}" ]]; then
  if [[ -f "$STATE/pid" ]] && kill -0 "$(cat "$STATE/pid")" 2>/dev/null; then
    echo "A benchmark is already running (PID $(cat "$STATE/pid")); log: $(cat "$STATE/log")" >&2
    exit 1
  fi
  args=(--foreground)
  [[ "$RESUME" == "1" ]] && args+=(--resume)
  FACTRAG_DETACHED=1 RUN_ID="$RUN_ID" PLAN="$PLAN" XQUAD="$XQUAD" VERIFIER="$VERIFIER" setsid nohup bash "$0" "${args[@]}" >>"$LOG" 2>&1 </dev/null &
  echo $! >"$STATE/pid"
  echo "$LOG" >"$STATE/log"
  echo "Benchmark $RUN_ID started in the background (PID $!)."
  echo "Log: $PROJECT_ROOT/$LOG   (Ctrl+C stops watching, not the run)"
  echo "Status: cat $PROJECT_ROOT/.artifacts/duty_cycle/status.json"
  sleep 1
  exec tail -n +1 --pid="$(cat "$STATE/pid")" -f "$LOG"
fi

case "$PLAN" in
  full) CONFIGS="qwen38-27b-hybrid:bge-reranker-v2-m3:judge qwen38-27b-hybrid:bge-reranker-v2-m3:rules qwen38-27b:bge-reranker-v2-m3:rules qwen25-14b:bge-reranker-v2-m3:judge" ;;
  compare) CONFIGS="qwen38-27b-hybrid:bge-reranker-v2-m3:judge qwen38-27b-hybrid:bge-reranker-v2-m3:rules" ;;
  main | quick) CONFIGS="qwen38-27b-hybrid:bge-reranker-v2-m3:$VERIFIER" ;;
  *) echo "Unknown PLAN=$PLAN (full | compare | main | quick)" >&2; exit 2 ;;
esac
LIMIT=()
[[ "$PLAN" == "quick" ]] && LIMIT=(--limit 10)

echo "=== FactRAG-NLI benchmark $RUN_ID (plan $PLAN, XQuAD $XQUAD) on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version,temperature.gpu --format=csv,noheader

XQ=(env FACTRAG_INDEX=.artifacts/xquad-index FACTRAG_PARSE_CACHE=.artifacts/xquad_parse_cache)
case "$XQUAD" in
  # The full set logs to its own folders: case logs of the two sets never mix.
  standard) XQ_DATA=(--dataset data/xquad_dataset.json); XQ_BUILD=(); XQ_DIR=xquad ;;
  full) XQ_DATA=(--dataset data/xquad_dataset_full.json); XQ_BUILD=(--full); XQ_DIR=xquad-full ;;
  *) echo "Unknown XQUAD=$XQUAD (standard | full)" >&2; exit 2 ;;
esac
stages=()
add() { stages+=(--stage "$1=$2"); }
q() { printf '%q ' "$@"; }

add "xquad-dataset" "$(q "$PYTHON" -X utf8 scripts/build_xquad_dataset.py "${XQ_BUILD[@]}" --check)"
# Index-building stages run before any llama-server and keep the encoder on the GPU; the
# decision layer (VERIFIER=judge) moves it to the CPU only for the e2e stages.
IDX=(env EMBEDDING_DEVICE="$RAG_MODEL_DEVICE" RAG_EMBEDDING_DEVICE="$RAG_MODEL_DEVICE")
add "xquad-nli" "$(q "${IDX[@]}" "${XQ[@]}" "$PYTHON" -X utf8 evaluate_suite.py "${XQ_DATA[@]}" --run-dir "$RUN_ROOT/nli/$XQ_DIR" "${LIMIT[@]}" nli)"
done_retrieval=" "
for config in $CONFIGS; do
  # profile:reranker[:verifier]; verifier v2 ("rules") keeps the folder names of earlier runs.
  IFS=: read -r profile reranker verifier <<<"$config"
  verifier="${verifier:-rules}"
  dir="$RUN_ROOT/$profile+$reranker"
  [[ "$verifier" != "rules" ]] && dir+="+$verifier"
  with=(env RERANKER="$reranker" VERIFIER="$verifier" bash scripts/linux/with_profile.sh "$profile")
  if [[ "$done_retrieval" != *" $reranker "* ]]; then
    add "v3-retrieval[$reranker]" "$(q "${IDX[@]}" "${with[@]}" "$PYTHON" -X utf8 evaluate_suite.py --run-dir "$RUN_ROOT/retrieval/v3+$reranker" "${LIMIT[@]}" retrieval)"
    add "xquad-retrieval[$reranker]" "$(q "${IDX[@]}" "${with[@]}" "${XQ[@]}" "$PYTHON" -X utf8 evaluate_suite.py "${XQ_DATA[@]}" --run-dir "$RUN_ROOT/retrieval/$XQ_DIR+$reranker" "${LIMIT[@]}" retrieval)"
    done_retrieval+="$reranker "
  fi
  # ensure: a server already answering with this profile (e.g. on --resume) is kept.
  add "llama-server[$profile]" "$(q "${with[@]}" bash scripts/linux/llama_server.sh ensure)"
  add "v3-e2e[$config]" "$(q "${with[@]}" "$PYTHON" -X utf8 evaluate_suite.py --run-dir "$dir/v3" "${LIMIT[@]}" e2e --profile guard_on)"
  add "xquad-e2e[$config]" "$(q "${with[@]}" "${XQ[@]}" "$PYTHON" -X utf8 evaluate_suite.py "${XQ_DATA[@]}" --run-dir "$dir/$XQ_DIR" "${LIMIT[@]}" e2e --profile guard_on)"
  # Cases that failed on infrastructure (timeouts) run once more; logged answers are kept.
  add "v3-retry[$config]" "$(q "${with[@]}" "$PYTHON" -X utf8 evaluate_suite.py --run-dir "$dir/v3" "${LIMIT[@]}" e2e --profile guard_on --retry-errors)"
  add "xquad-retry[$config]" "$(q "${with[@]}" "${XQ[@]}" "$PYTHON" -X utf8 evaluate_suite.py "${XQ_DATA[@]}" --run-dir "$dir/$XQ_DIR" "${LIMIT[@]}" e2e --profile guard_on --retry-errors)"
done
if [[ "${KEEP_SERVER:-0}" != "1" ]]; then
  add "llama-server-stop" "$(q bash scripts/linux/llama_server.sh stop)"
fi
add "comparison" "$(q "$PYTHON" -X utf8 scripts/compare_runs.py "$RUN_ROOT")"

# Watchdog: an e2e stage whose llama-server stops answering restarts it (same profile, the
# stage runs under with_profile.sh) and runs the failed case again; 5 failures in a row end
# the stage, and the retry stage continues with a fresh process.
export FACTRAG_LLM_RECOVER_CMD="bash scripts/linux/llama_server.sh ensure"

set +e
"$PYTHON" scripts/linux/gpu_duty_cycle.py --keep-going \
  --work-minutes "$WORK_MINUTES" --rest-minutes "$REST_MINUTES" --temp-limit "$GPU_TEMP_LIMIT" \
  "${stages[@]}"
code=$?
set -e
# Server log (appended across restarts) next to the results, for crash diagnostics.
cp -f .artifacts/llama/server.log "$RUN_ROOT/llama-server.log" 2>/dev/null || true
echo "=== finished $(date) with exit code $code"
echo "Results: $PROJECT_ROOT/$RUN_ROOT/COMPARISON.md"
rm -f "$STATE/pid"
exit "$code"
