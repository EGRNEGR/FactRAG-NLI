#!/usr/bin/env bash
# llama-server for FactRAG-NLI on the RTX 3090:  llama_server.sh start|stop|status|restart|ensure
#   ensure  start or restart only if the server of this profile is not answering (watchdog of
#           evaluate_suite.py); prints the server log tail and memory state first.
# The log is appended across restarts (.artifacts/llama/server.log), so a crash stays visible.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.rtx3090.sh"

STATE_DIR="$PROJECT_ROOT/.artifacts/llama"
PID_FILE="$STATE_DIR/server.pid"
LOG_FILE="$STATE_DIR/server.log"
SERVER="$LLAMA_DIR/llama-server"
mkdir -p "$STATE_DIR"

ready() {
  curl -fsS -m 3 "http://$LLAMA_HOST:$LLAMA_PORT/health" 2>/dev/null | grep -q '"ok"' &&
    curl -fsS -m 3 "http://$LLAMA_HOST:$LLAMA_PORT/v1/models" >/dev/null 2>&1
}

running_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  kill -0 "$pid" 2>/dev/null && echo "$pid"
}

start() {
  if pid="$(running_pid)"; then
    echo "llama-server already running (PID $pid, $(cat "$STATE_DIR/model"))"
    if [[ "$(cat "$STATE_DIR/model")" != "$LLM_PROFILE" ]]; then
      echo "It serves another profile; run: LLM_PROFILE=$LLM_PROFILE $0 restart" >&2
      exit 1
    fi
    return 0
  fi
  [[ -x "$SERVER" ]] || { echo "Missing $SERVER; run scripts/linux/setup.sh" >&2; exit 1; }
  [[ -f "$PROJECT_ROOT/$LLM_PATH" ]] || { echo "Missing $LLM_PATH; run setup.sh" >&2; exit 1; }
  if curl -fsS -m 2 "http://$LLAMA_HOST:$LLAMA_PORT/health" >/dev/null 2>&1; then
    echo "Port $LLAMA_PORT is served by a process we did not start; stop it first." >&2
    exit 1
  fi
  local args=(
    -m "$PROJECT_ROOT/$LLM_PATH"
    -ngl "$LLAMA_GPU_LAYERS" -c "$LLAMA_CTX" -np 1
    -fa on --jinja --cache-ram 0
    --host "$LLAMA_HOST" --port "$LLAMA_PORT"
  )
  if [[ "$LLM_THINKING_OFF" == "true" ]]; then
    # Server-side default (b12002 deprecates --chat-template-kwargs for this); the pipeline
    # also sends chat_template_kwargs {"enable_thinking": false} with every request.
    args+=(--reasoning off)
  elif [[ "$LLM_REASONING_BUDGET" != "-1" ]]; then
    args+=(--reasoning-budget "$LLM_REASONING_BUDGET")
  fi
  echo "Starting llama-server $LLAMA_BUILD: $LLM_FILE, ctx $LLAMA_CTX, ngl $LLAMA_GPU_LAYERS"
  printf '\n===== %s start %s (%s)\n' "$(date '+%F %T')" "$LLM_PROFILE" "$LLM_FILE" >>"$LOG_FILE"
  # Own session: the duty cycle freezes the process group of the running stage, and a server
  # started by the watchdog of an e2e stage must not be frozen or terminated with that stage.
  LD_LIBRARY_PATH="$LLAMA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    setsid nohup "$SERVER" "${args[@]}" >>"$LOG_FILE" 2>&1 </dev/null &
  echo $! >"$PID_FILE"
  echo "$LLM_PROFILE" >"$STATE_DIR/model"
  for _ in $(seq 1 300); do
    if ready; then
      echo "llama-server ready: http://$LLAMA_HOST:$LLAMA_PORT/v1 (PID $(cat "$PID_FILE"))"
      nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/VRAM used, total: /'
      return 0
    fi
    if ! running_pid >/dev/null; then
      echo "llama-server exited; last log lines:" >&2
      tail -n 30 "$LOG_FILE" >&2
      exit 1
    fi
    sleep 1
  done
  echo "llama-server not ready after 300 s; see $LOG_FILE" >&2
  exit 1
}

stop() {
  if pid="$(running_pid)"; then
    kill "$pid"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
    echo "llama-server stopped (PID $pid)"
  else
    echo "llama-server is not running"
  fi
  rm -f "$PID_FILE" "$STATE_DIR/model"
}

ensure() {
  if pid="$(running_pid)" && [[ "$(cat "$STATE_DIR/model" 2>/dev/null)" == "$LLM_PROFILE" ]] && ready; then
    echo "llama-server is answering (PID $pid, $LLM_PROFILE)"
    return 0
  fi
  echo "llama-server ($LLM_PROFILE) is down or not answering; diagnostics:" >&2
  if pid="$(running_pid)"; then echo "  process $pid still exists" >&2; else echo "  no process" >&2; fi
  [[ -f "$LOG_FILE" ]] && tail -n 40 "$LOG_FILE" | sed 's/^/  log: /' >&2
  free -m 2>/dev/null | sed 's/^/  ram: /' >&2 || true
  nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null |
    sed 's/^/  gpu: /' >&2 || true
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null |
    sed 's/^/  gpu app: /' >&2 || true
  stop
  start
}

status() {
  if pid="$(running_pid)"; then
    echo "running: PID $pid, model $(cat "$STATE_DIR/model"), ready=$(ready && echo yes || echo no)"
  else
    echo "stopped"
  fi
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  ensure) ensure ;;
  status) status ;;
  *) echo "usage: $0 start|stop|restart|status|ensure" >&2; exit 2 ;;
esac
