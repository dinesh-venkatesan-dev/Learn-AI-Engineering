#!/usr/bin/env bash
# Start the Nimbus academy with one command: Python packages, .env, Postgres + Redis (Docker), the UI build, and the
# course app on http://localhost:8300. Safe to re-run: each step only does work when something changed.
#   ./start.sh          start (Ctrl-C stops the app; the database keeps running)
#   ./start.sh stop     stop Postgres and Redis (your data stays; `docker compose down -v` deletes it)
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\033[1m▸ %s\033[0m\n' "$*"; }
warn() { printf '\033[33m! %s\033[0m\n' "$*"; }
fail() { printf '\033[31m✗ %s\033[0m\n' "$*"; exit 1; }

if [ "${1:-}" = "stop" ]; then
  docker compose stop
  exit 0
fi

# 1. Tools
command -v uv >/dev/null || fail "uv is missing: https://docs.astral.sh/uv/getting-started/installation/"
command -v npm >/dev/null || fail "Node.js 22+ is missing: https://nodejs.org (or: brew install node)"
node_major=$(node -p 'process.versions.node.split(".")[0]')
[ "$node_major" -ge 22 ] || fail "Node.js 22+ is needed (you have $(node -v))"

# 2. The port
if lsof -iTCP:8300 -sTCP:LISTEN >/dev/null 2>&1; then
  fail "port 8300 is already in use: the academy may already be running at http://localhost:8300"
fi

# 3. Python packages
say "Python packages"
uv sync --frozen --quiet

# 4. API keys
if [ ! -f .env ]; then
  cp .env.example .env
  warn "created .env from .env.example: add your MISTRAL_API_KEY (free: https://console.mistral.ai)"
fi
grep -qE '^MISTRAL_API_KEY=.+' .env || warn "MISTRAL_API_KEY is empty in .env: labs that call a model will fail until you add it"

# 5. Postgres + Redis
if docker info >/dev/null 2>&1; then
  say "Postgres + Redis (Docker)"
  docker compose up -d --wait postgres redis
else
  warn "Docker isn't running: Postgres and Redis are skipped. Labs that need them say so; start Docker and re-run."
fi

# 6. The UI: install and build only when something changed
if [ ! -d frontend/node_modules ] || [ frontend/package-lock.json -nt frontend/node_modules/.package-lock.json ]; then
  say "UI packages"
  (cd frontend && npm ci --no-audit --no-fund --loglevel=error >/dev/null)
fi
if [ ! -f frontend/dist/index.html ] \
  || [ -n "$(find frontend/src frontend/index.html frontend/package.json -newer frontend/dist/index.html -print -quit)" ]; then
  say "Building the UI"
  mkdir -p .cache
  (cd frontend && npm run build >../.cache/ui-build.log 2>&1) || { cat .cache/ui-build.log; fail "the UI build failed"; }
fi

# 7. The course app
say "Nimbus academy → http://localhost:8300  (Ctrl-C to stop)"
exec uv run academy
