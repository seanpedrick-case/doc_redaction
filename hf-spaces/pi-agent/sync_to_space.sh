#!/usr/bin/env bash
# Flatten monorepo paths into a temp directory for the Pi agent HF Space repo.
# Usage (from repo root):
#   hf-spaces/pi-agent/sync_to_space.sh /path/to/output-dir
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:?Output directory required}"
MANIFEST="$(dirname "$0")/sync-manifest.txt"

rm -rf "$OUT"
mkdir -p "$OUT"

cp "$(dirname "$0")/Dockerfile" "$OUT/Dockerfile"
cp "$(dirname "$0")/README.md" "$OUT/README.md"
cp "$(dirname "$0")/.dockerignore" "$OUT/.dockerignore"

while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"
  line="$(echo "$line" | xargs)"
  [[ -z "$line" ]] && continue
  src="$ROOT/$line"
  if [[ ! -e "$src" ]]; then
    echo "Missing: $src" >&2
    exit 1
  fi
  dest="$OUT/$line"
  mkdir -p "$(dirname "$dest")"
  cp -a "$src" "$dest"
done < "$MANIFEST"

echo "Flattened Pi agent Space tree: $OUT"
