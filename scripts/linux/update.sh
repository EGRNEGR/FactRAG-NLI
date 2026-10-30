#!/usr/bin/env bash
# Update the project files from GitHub without git (the AnyDesk machine has an unpacked copy):
#   bash scripts/linux/update.sh
# Downloads the branch archive and overwrites the files it contains, listing the changed ones.
# models/, .venv/, tools/, .artifacts/, .env and run results in reports/runs/ are not part of
# the archive and stay untouched; files deleted upstream are left in place.
# UPDATE_REPO / UPDATE_BRANCH select another source (default EGRNEGR/FactRAG-NLI, rtx3090-linux).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.rtx3090.sh"
cd "$PROJECT_ROOT"
REPO="${UPDATE_REPO:-EGRNEGR/FactRAG-NLI}"
BRANCH="${UPDATE_BRANCH:-rtx3090-linux}"

PID_FILE=.artifacts/benchmark/pid
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "A benchmark is running (PID $(cat "$PID_FILE")): stop it first with" >&2
  echo "  bash scripts/linux/stop_benchmark.sh" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "Downloading $REPO@$BRANCH ..."
curl -4 -fL --connect-timeout 20 --retry 5 --retry-delay 3 --retry-all-errors \
  -o "$tmp/src.tar.gz" "https://codeload.github.com/$REPO/tar.gz/refs/heads/$BRANCH"
mkdir "$tmp/src"
tar -xzf "$tmp/src.tar.gz" -C "$tmp/src" --strip-components=1

changed=0
while IFS= read -r -d '' file; do
  rel="${file#"$tmp/src/"}"
  if ! cmp -s "$file" "$rel" 2>/dev/null; then
    echo "  updated: $rel"
    changed=$((changed + 1))
  fi
done < <(find "$tmp/src" -type f -print0 | sort -z)
cp -a "$tmp/src/." ./

commit="$(curl -4 -fsS -m 20 -H 'Accept: application/vnd.github.sha' \
  "https://api.github.com/repos/$REPO/commits/$BRANCH" 2>/dev/null || echo unknown)"
mkdir -p .artifacts/update
printf '%s %s %s\n' "$(date '+%F %T')" "$BRANCH" "$commit" >>.artifacts/update/history
echo "$changed file(s) updated to $REPO@$BRANCH (commit ${commit:0:7})"
