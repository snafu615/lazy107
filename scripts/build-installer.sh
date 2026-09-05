#!/usr/bin/env bash
# Rebuild the single-file installer dist/lazy107-<ver>-install.sh from
# scripts/installer-head.sh + the newest wheel in dist/.
# Build a wheel first with:  python -m pip wheel . --no-deps -w dist
set -euo pipefail
cd "$(dirname "$0")/.."

head="scripts/installer-head.sh"
whl="$(ls -t dist/lazy107-*.whl 2>/dev/null | head -n1 || true)"
if [[ -z "$whl" ]]; then
    echo "no wheel in dist/ — run: python -m pip wheel . --no-deps -w dist" >&2
    exit 1
fi
base="$(basename "$whl")"
ver="${base#lazy107-}"
ver="${ver%-py3-none-any.whl}"
out="dist/lazy107-${ver}-install.sh"
cat "$head" "$whl" > "$out"
chmod +x "$out"
echo "built $out ($(wc -c < "$out") bytes)"
