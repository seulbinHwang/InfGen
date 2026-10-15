#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
devices="${1:-1}"
if (($#)); then shift; fi
exec "${PYTHON:-python3}" run.py --test --devices "$devices" "$@"
