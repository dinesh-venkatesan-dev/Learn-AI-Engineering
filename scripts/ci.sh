#!/usr/bin/env bash
# What CI runs on every push, run here: lint, the fast tests, the Dockerfile lint. No API keys, no model calls.
# Tests marked db need the course Postgres (docker compose up -d); without it they're skipped, and this says so.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --frozen --quiet
uv run ruff check src tests
if uv run python -c "import socket; socket.create_connection(('127.0.0.1', 5443), timeout=1).close()" 2>/dev/null; then
  uv run pytest -q -p no:cacheprovider
else
  echo "ci: Postgres isn't running on 5443: skipping tests marked db (CI runs them)"
  uv run pytest -q -p no:cacheprovider -m "not integration and not db"
fi
if command -v hadolint >/dev/null; then hadolint Dockerfile; fi
docker compose config --quiet
echo "ci: all steps passed"
