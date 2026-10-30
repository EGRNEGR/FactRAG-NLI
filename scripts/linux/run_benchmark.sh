#!/usr/bin/env bash
# FactRAG-NLI benchmark on the RTX 3090 with GPU rest periods (45 min work / 5 min rest).
#
#   bash scripts/linux/run_benchmark.sh              # PLAN=full, detached, shows the log
#   bash scripts/linux/run_benchmark.sh --resume     # continue the last run after a stop/reboot
#   PLAN=quick bash scripts/linux/run_benchmark.sh   # 10 cases per dataset, ~15 min smoke test
#   bash scripts/linux/run_benchmark.sh --foreground # stay attached to the terminal
#
# Plans (configuration = LLM profile + reranker):
#   full   qwen38-27b+bge-reranker-large, qwen25-14b+bge-reranker-large (same-GPU baseline),
#          qwen38-27b+bge-reranker-v2-m3; each on eval_dataset_v3 (220) and XQuAD (198)
#   main   qwen38-27b+bge-reranker-large only
#   quick  main with 10 cases per dataset
# The run is detached (setsid + nohup): closing the AnyDesk session or the terminal does not
# stop it. Ctrl+C only stops watching the log. Stop the run: bash scripts/linux/stop_benchmark.sh
set -euo pipefail
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
  source "$STATE/last_run"   # RUN_ID and PLAN of the interrupted run
else
  PLAN="${PLAN:-full}"
  RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%MZ)_rtx3090_$PLAN}"
  printf 'RUN_ID=%q\nPLAN=%q\n' "$RUN_ID" "$PLAN" >"$STATE/last_run"
fi
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
  FACTRAG_DETACHED=1 RUN_ID="$RUN_ID" PLAN="$PLAN" setsid nohup bash "$0" "${args[@]}" >>"$LOG" 2>&1 </dev/null &
  echo $! >"$STATE/pid"
  echo "$LOG" >"$STATE/log"
  echo "Benchmark $RUN_ID started in the background (PID $!)."
  echo "Log: $PROJECT_ROOT/$LOG   (Ctrl+C stops watching, not the run)"
  echo "Status: cat $PROJECT_ROOT/.artifacts/duty_cycle/status.json"
  sleep 1
  exec tail -n +1 --pid="$(cat "$STATE/pid")" -f "$LOG"
fi

case "$PLAN" in
  full) CONFIGS="qwen38-27b:bge-reranker-large qwen25-14b:bge-reranker-large qwen38-27b:bge-reranker-v2-m3" ;;
  main | quick) CONFIGS="qwen38-27b:bge-reranker-large" ;;
  *) echo "Unknown PLAN=$PLAN (full | main | quick)" >&2; exit 2 ;;
esac
LIMIT=()
[[ "$PLAN" == "quick" ]] && LIMIT=(--limit 10)

echo "=== FactRAG-NLI benchmark $RUN_ID (plan $PLAN) on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version,temperature.gpu --format=csv,noheader

XQ=(env FACTRAG_INDEX=.artifacts/xquad-index FACTRAG_PARSE_CACHE=.artifacts/xquad_parse_cache)
XQ_DATA=(--dataset data/xquad_dataset.json)
stages=()
add() { stages+=(--stage "$1=$2"); }
q() { printf '%q ' "$@"; }

add "xquad-dataset" "$(q "$PYTHON" -X utf8 scripts/build_xquad_dataset.py --check)"
add "xquad-nli" "$(q "${XQ[@]}" "$PYTHON" -X utf8 evaluate_suite.py "${XQ_DATA[@]}" --run-dir "$RUN_ROOT/nli/xquad" "${LIMIT[@]}" nli)"
done_retrieval=" "
for config in $CONFIGS; do
  profile="${config%%:*}"
  reranker="${config#*:}"
  dir="$RUN_ROOT/$profile+$reranker"
  with=(env RERANKER="$reranker" bash scripts/linux/with_profile.sh "$profile")
  if [[ "$done_retrieval" != *" $reranker "* ]]; then
    add "v3-retrieval[$reranker]" "$(q "${with[@]}" "$PYTHON" -X utf8 evaluate_suite.py --run-dir "$RUN_ROOT/retrieval/v3+$reranker" "${LIMIT[@]}" retrieval)"
    add "xquad-retrieval[$reranker]" "$(q "${with[@]}" "${XQ[@]}" "$PYTHON" -X utf8 evaluate_suite.py "${XQ_DATA[@]}" --run-dir "$RUN_ROOT/retrieval/xquad+$reranker" "${LIMIT[@]}" retrieval)"
    done_retrieval+="$reranker "
  fi
  add "llama-server[$profile]" "$(q "${with[@]}" bash scripts/linux/llama_server.sh restart)"
  add "v3-e2e[$config]" "$(q "${with[@]}" "$PYTHON" -X utf8 evaluate_suite.py --run-dir "$dir/v3" "${LIMIT[@]}" e2e --profile guard_on)"
  add "xquad-e2e[$config]" "$(q "${with[@]}" "${XQ[@]}" "$PYTHON" -X utf8 evaluate_suite.py "${XQ_DATA[@]}" --run-dir "$dir/xquad" "${LIMIT[@]}" e2e --profile guard_on)"
  # Cases that failed on infrastructure (timeouts) run once more; logged answers are kept.
  add "v3-retry[$config]" "$(q "${with[@]}" "$PYTHON" -X utf8 evaluate_suite.py --run-dir "$dir/v3" "${LIMIT[@]}" e2e --profile guard_on --retry-errors)"
  add "xquad-retry[$config]" "$(q "${with[@]}" "${XQ[@]}" "$PYTHON" -X utf8 evaluate_suite.py "${XQ_DATA[@]}" --run-dir "$dir/xquad" "${LIMIT[@]}" e2e --profile guard_on --retry-errors)"
done
if [[ "${KEEP_SERVER:-0}" != "1" ]]; then
  add "llama-server-stop" "$(q bash scripts/linux/llama_server.sh stop)"
fi
add "comparison" "$(q "$PYTHON" -X utf8 scripts/compare_runs.py "$RUN_ROOT")"

set +e
"$PYTHON" scripts/linux/gpu_duty_cycle.py --keep-going \
  --work-minutes "$WORK_MINUTES" --rest-minutes "$REST_MINUTES" --temp-limit "$GPU_TEMP_LIMIT" \
  "${stages[@]}"
code=$?
set -e
echo "=== finished $(date) with exit code $code"
echo "Results: $PROJECT_ROOT/$RUN_ROOT/COMPARISON.md"
rm -f "$STATE/pid"
exit "$code"
