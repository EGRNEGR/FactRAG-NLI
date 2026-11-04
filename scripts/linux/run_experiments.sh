#!/usr/bin/env bash
# optimization_4: fast experiment loop on the RTX 3090 against the baseline run 1310Z.
#
#   bash scripts/linux/run_experiments.sh              # PLAN=quick: D0, G3, G4 (13 min measured)
#   PLAN=all bash scripts/linux/run_experiments.sh     # + e2e of E1 with 20 controls, G2 (~30 min)
#   PLAN=g3 bash scripts/linux/run_experiments.sh      # e2e of query_translation on the G3
#                                                      # population + 20 controls (31 min)
#   PLAN=e2e IDS=V3-MH-024,V3-TC-020 FLAGS="RAG_NLI_ANCHOR_SCOPE=document" \
#     bash scripts/linux/run_experiments.sh            # real pipeline on these questions only
#   IDS=V3-MH-024,V3-XQFEN-130 bash scripts/linux/run_experiments.sh   # steps on these only
#   bash scripts/linux/run_experiments.sh --foreground # stay attached to the terminal
#
# optimization_5 (parser v2): the same loop against a PARSER=v2 PLAN=main run,
#   PLAN=opt5 BASELINE=<run id> bash scripts/linux/run_experiments.sh   # E1, G2, G2 + E1
#   E1 e2e on its population of that run + 20 controls, steps D0 and G2, then G2 again with
#   the verifier of E1 (--set nli_anchor_scope=document). The baseline keeps its own settings
#   (question translation on); results go to reports/optimization_5/live/<RUN_ID>/.
#
# Nothing of the baseline is written: reports/runs/20261103T1310Z_rtx3090_main is read only,
# the results go to reports/optimization_4/live/<RUN_ID>/ (run.log, every intermediate result,
# SUMMARY.md). Stages:
#   e2e     the real pipeline (evaluate_suite.py --ids) on selected questions, paired with the
#           baseline (research.opt4.compare). PLAN=all: RAG_NLI_ANCHOR_SCOPE=document (E1) on the
#           9 questions where a claim was withheld for a missing anchor + 20 untouched controls
#           (E1 itself is decided exactly by the offline replay; this confirms it live).
#           PLAN=g3: QUERY_TRANSLATION=refusal (G3 in the pipeline) on the 151 questions its
#           trigger selects + the 20 controls (run 20261103T1839Z: 31 min measured).
#           PLAN=e2e: the questions of IDS with the settings of FLAGS.
#           The baseline 1310Z ran without question translation, so every e2e stage runs
#           with QUERY_TRANSLATION=off unless FLAGS set it (QUERY_TRANSLATION, not
#           RAG_QUERY_TRANSLATION: env.rtx3090.sh derives the RAG_ variable from it).
#   steps   research.opt4.steps from the logged state, in priority order (a run stopped early
#           keeps the finished experiments): D0 determinism, G3 translation and re-retrieval
#           after a refusal, G4 question normalization after an LLM refusal (PLAN=all: + G2
#           neighbour chunk), paired with the baseline
# The time estimate printed first comes from the per-call medians of run 1310Z
# (python -m research.opt4.steps plan); the log reports the measured time of every stage.
# Exit code 1 if any stage finds a regression (a question correct in the baseline and wrong
# now, or a new hallucination).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.rtx3090.sh"
cd "$PROJECT_ROOT"

FOREGROUND=0
for arg in "$@"; do
  case "$arg" in
    --foreground) FOREGROUND=1 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

PLAN="${PLAN:-quick}"
# The baseline 1310Z of optimization_4 ran without question translation, so its e2e stages
# run with QUERY_TRANSLATION=off unless FLAGS set it; an optimization_5 baseline keeps its own.
BASE_FLAGS="QUERY_TRANSLATION=off"
STAGE_DIR=reports/optimization_4/live
TAG=opt4
COMBINED=""
# The parser of the baseline: 1310Z (optimization_4) was indexed with v1.
BASE_PARSER=v1
case "$PLAN" in
  opt5)
    [[ -n "${BASELINE:-}" ]] ||
      { echo "PLAN=opt5 needs BASELINE=<run id of a PARSER=v2 PLAN=main run>" >&2; exit 2; }
    export OPT4_RUN="$PROJECT_ROOT/reports/runs/$BASELINE"
    BASE_PARSER=v2
    EXPERIMENTS="${EXPERIMENTS:-D0,G2}"; POPULATIONS="E1 controls"
    FLAGS="RAG_NLI_ANCHOR_SCOPE=document"; BASE_FLAGS=""; COMBINED="nli_anchor_scope=document"
    STAGE_DIR=reports/optimization_5/live; TAG=opt5 ;;
  quick)
    EXPERIMENTS="${EXPERIMENTS:-D0,G3,G4}"; POPULATIONS=""; FLAGS="" ;;
  all)
    EXPERIMENTS="${EXPERIMENTS:-D0,G3,G4,G2}"; POPULATIONS="E1 controls"
    FLAGS="RAG_NLI_ANCHOR_SCOPE=document" ;;
  g3)
    EXPERIMENTS=""; POPULATIONS="G3 controls"; FLAGS="QUERY_TRANSLATION=refusal" ;;
  e2e)
    [[ -n "${IDS:-}" ]] || { echo "PLAN=e2e needs IDS=..." >&2; exit 2; }
    EXPERIMENTS=""; POPULATIONS="ids"; FLAGS="${FLAGS:-}" ;;
  *) echo "Unknown PLAN=$PLAN (quick | all | g3 | e2e | opt5)" >&2; exit 2 ;;
esac
export PARSER="$BASE_PARSER" RAG_PARSER_VERSION="$BASE_PARSER"
export OPT4_RUN="${OPT4_RUN:-$PROJECT_ROOT/reports/runs/20261103T1310Z_rtx3090_main}"
BASE_CONFIG="$OPT4_RUN/qwen38-27b-hybrid+bge-reranker-v2-m3+judge"
[[ -d "$BASE_CONFIG/v3" && -d "$BASE_CONFIG/xquad-full" ]] ||
  { echo "Baseline run not found: $BASE_CONFIG" >&2; exit 1; }
# The v3 questions of a parser v2 baseline live in its own index and dataset binding.
V3=()
V3_DATA=()
if [[ "$RAG_PARSER_VERSION" == "v2" ]]; then
  V3=(FACTRAG_INDEX=.artifacts/eval-v3-index-parser-v2 FACTRAG_PARSE_CACHE=.artifacts/eval_v3_parse_cache_parser_v2)
  V3_DATA=(--dataset data/eval_dataset_v3_parser_v2.json)
fi
if [[ -f .artifacts/benchmark/pid ]] && kill -0 "$(cat .artifacts/benchmark/pid)" 2>/dev/null; then
  echo "A benchmark is running (PID $(cat .artifacts/benchmark/pid)); wait for it or stop it." >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%MZ)_${TAG}_$PLAN}"
OUT="$STAGE_DIR/$RUN_ID"
LOG="$OUT/run.log"
mkdir -p "$OUT"

if [[ "$FOREGROUND" == "0" && -z "${FACTRAG_DETACHED:-}" ]]; then
  FACTRAG_DETACHED=1 RUN_ID="$RUN_ID" PLAN="$PLAN" EXPERIMENTS="$EXPERIMENTS" \
    IDS="${IDS:-}" FLAGS="$FLAGS" BASELINE="${BASELINE:-}" setsid nohup bash "$0" --foreground >>"$LOG" 2>&1 </dev/null &
  pid=$!
  echo "Experiments $RUN_ID started in the background (PID $pid)."
  echo "Log: $PROJECT_ROOT/$LOG   (Ctrl+C stops watching, not the run)"
  sleep 1
  exec tail -n +1 --pid="$pid" -f "$LOG"
fi

started=$(date +%s)
stage() { echo; echo "=== $(date '+%H:%M:%S') $*"; }
elapsed() { echo "    stage time: $(( $(date +%s) - $1 )) s"; }
failed=0

stage "FactRAG-NLI optimization_4 experiments $RUN_ID (plan $PLAN) on $(hostname)"
echo "baseline: $OPT4_RUN"
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader
STEPS_IDS=()
[[ -n "${IDS:-}" ]] && STEPS_IDS=(--ids "$IDS")
if [[ -n "$EXPERIMENTS" ]]; then
  "$PYTHON" -X utf8 -m research.opt4.steps plan --experiments "$EXPERIMENTS" "${STEPS_IDS[@]}" |
    tee "$OUT/plan.json"
fi

WITH=(env RERANKER=bge-reranker-v2-m3 VERIFIER=judge bash scripts/linux/with_profile.sh qwen38-27b-hybrid)
t=$(date +%s); stage "llama-server (qwen38-27b-hybrid)"
"${WITH[@]}" bash scripts/linux/llama_server.sh ensure
elapsed "$t"

# ---------------------------------------------------------------- e2e: real pipeline, by ids
if [[ -n "$POPULATIONS" ]]; then
  t=$(date +%s); stage "e2e on selected questions, settings: ${FLAGS:-baseline}"
  mkdir -p "$OUT/e2e"
  : >"$OUT/e2e/ids.txt"
  for population in $POPULATIONS; do
    if [[ "$population" == "ids" ]]; then
      tr ', ' '\n\n' <<<"$IDS" | grep -v '^$' >>"$OUT/e2e/ids.txt"
    else
      "$PYTHON" -X utf8 -m research.opt4.compare ids "$population" >>"$OUT/e2e/ids.txt"
    fi
  done
  echo "$FLAGS" >"$OUT/e2e/flags.txt"
  echo "$(wc -l <"$OUT/e2e/ids.txt") questions"
  # FLAGS are the settings of the variant (empty: the baseline configuration itself);
  # the last assignment wins, so FLAGS override BASE_FLAGS (opt4: QUERY_TRANSLATION=off).
  read -r -a base <<<"$BASE_FLAGS"
  read -r -a flags <<<"$FLAGS"
  E2E=(env "${base[@]}" "${flags[@]}" FACTRAG_LLM_RECOVER_CMD="bash scripts/linux/llama_server.sh ensure")
  if grep -qv '^V3-XQF' "$OUT/e2e/ids.txt"; then
    "${E2E[@]}" "${V3[@]}" "${WITH[@]}" "$PYTHON" -X utf8 evaluate_suite.py "${V3_DATA[@]}" \
      --run-dir "$OUT/e2e/v3" --ids "@$OUT/e2e/ids.txt" e2e --profile guard_on || failed=1
  fi
  if grep -q '^V3-XQF' "$OUT/e2e/ids.txt"; then
    "${E2E[@]}" FACTRAG_INDEX=.artifacts/xquad-index FACTRAG_PARSE_CACHE=.artifacts/xquad_parse_cache \
      "${WITH[@]}" "$PYTHON" -X utf8 evaluate_suite.py --dataset data/xquad_dataset_full.json \
      --run-dir "$OUT/e2e/xquad-full" --ids "@$OUT/e2e/ids.txt" e2e --profile guard_on || failed=1
  fi
  "$PYTHON" -X utf8 -m research.opt4.compare run "$OUT/e2e" || failed=1
  elapsed "$t"
fi

# ---------------------------------------------------------------- steps from the logged state
if [[ -n "$EXPERIMENTS" ]]; then
  t=$(date +%s); stage "steps: $EXPERIMENTS"
  "${WITH[@]}" "$PYTHON" -X utf8 -m research.opt4.steps run --experiments "$EXPERIMENTS" \
    --out "$OUT/steps" "${STEPS_IDS[@]}" || failed=1
  elapsed "$t"
  if [[ -n "$COMBINED" && "$EXPERIMENTS" == *G2* ]]; then
    # G2 + E1: the G2 steps once more with the verifier of E1 (opt5).
    t=$(date +%s); stage "steps: G2 with $COMBINED"
    "${WITH[@]}" "$PYTHON" -X utf8 -m research.opt4.steps run --experiments G2 \
      --set "$COMBINED" --out "$OUT/steps_g2_e1" "${STEPS_IDS[@]}" || failed=1
    elapsed "$t"
  fi
fi

if [[ "${KEEP_SERVER:-0}" != "1" ]]; then
  stage "llama-server stop"
  bash scripts/linux/llama_server.sh stop || true
fi
"$PYTHON" -X utf8 -m research.opt4.live_report "$OUT" || failed=1
stage "finished in $(( ($(date +%s) - started) / 60 )) min, exit code $failed"
echo "Results: $PROJECT_ROOT/$OUT/SUMMARY.md"
exit "$failed"
