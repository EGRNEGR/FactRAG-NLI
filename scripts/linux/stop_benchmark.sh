#!/usr/bin/env bash
# Stop a detached benchmark and llama-server. Resume later: run_benchmark.sh --resume
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.rtx3090.sh"
cd "$PROJECT_ROOT"
PID_FILE=.artifacts/benchmark/pid
if [[ -f "$PID_FILE" ]] && pid="$(cat "$PID_FILE")" && kill -0 "$pid" 2>/dev/null; then
  # The run is a session leader (setsid): the whole session gets SIGTERM; the duty cycle
  # resumes a frozen stage before terminating it, so the case log stays consistent.
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  echo "Benchmark stopped (PID $pid)"
else
  echo "No benchmark is running"
fi
rm -f "$PID_FILE"
bash scripts/linux/llama_server.sh stop
