#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
devices="${1:-8}"
if (($#)); then shift; fi
exec "${PYTHON:-python3}" run.py --train --devices "$devices" "$@"
