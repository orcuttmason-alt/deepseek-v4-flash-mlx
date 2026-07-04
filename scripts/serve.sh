#!/usr/bin/env bash
# DeepSeek V4-Flash — stdin JSON REPL daemon. One JSON per line:
#   {"prompt": "...", "max_tokens": 200} | {"cmd": "reset"} | {"cmd": "stats"}
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u oracle/v4_serve_fast.py "$@"
