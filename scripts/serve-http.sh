#!/usr/bin/env bash
# DeepSeek V4-Flash — OpenAI-compatible HTTP server on :18091 (/v1/chat/completions, /v1/models).
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u oracle/v4_http.py "$@"
