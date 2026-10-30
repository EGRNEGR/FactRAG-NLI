#!/usr/bin/env bash
# Run one command under an LLM profile:  with_profile.sh <profile> <command> [args...]
# Used by run_benchmark.sh so that stages of different models carry their own RAG_* settings.
set -euo pipefail
export LLM_PROFILE="$1"
shift
source "$(dirname "${BASH_SOURCE[0]}")/env.rtx3090.sh"
cd "$PROJECT_ROOT"
exec "$@"
