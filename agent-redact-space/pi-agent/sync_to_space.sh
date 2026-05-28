#!/usr/bin/env bash
# Flatten monorepo paths into a temp directory for the Pi agent HF Space repo.
# Usage (from repo root):
#   agent-redact-space/pi-agent/sync_to_space.sh /path/to/output-dir
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:?Output directory required}"
MANIFEST="$(dirname "$0")/sync-manifest.txt"

_is_lfs_pointer() {
  [[ -f "$1" ]] && head -1 "$1" 2>/dev/null | grep -q "^version https://git-lfs.github.com/spec/v1"
}

rm -rf "$OUT"
mkdir -p "$OUT"

cp "$(dirname "$0")/Dockerfile" "$OUT/Dockerfile"
cp "$(dirname "$0")/README.md" "$OUT/README.md"
cp "$(dirname "$0")/.dockerignore" "$OUT/.dockerignore"
cp "$(dirname "$0")/.gitattributes" "$OUT/.gitattributes"

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
  if [[ "$line" == *.pdf ]] && _is_lfs_pointer "$dest"; then
    echo "Copied file is a Git LFS pointer, not a PDF: $line" >&2
    echo "Run 'git lfs pull' in the monorepo before syncing." >&2
    exit 1
  fi
done < "$MANIFEST"

echo "Flattened Pi agent Space tree: $OUT"
