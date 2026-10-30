#!/usr/bin/env bash
# llama-server for FactRAG-NLI on the RTX 3090:  llama_server.sh start|stop|status|restart
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
    if [[ "$(cat "$STATE_DIR/model")" != "$LLM_FILE" ]]; then
      echo "It serves another model; run: LLM_PROFILE=$LLM_PROFILE $0 restart" >&2
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
  fi
  echo "Starting llama-server $LLAMA_BUILD: $LLM_FILE, ctx $LLAMA_CTX, ngl $LLAMA_GPU_LAYERS"
  LD_LIBRARY_PATH="$LLAMA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    nohup "$SERVER" "${args[@]}" >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "$LLM_FILE" >"$STATE_DIR/model"
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
  status) status ;;
  *) echo "usage: $0 start|stop|restart|status" >&2; exit 2 ;;
esac
