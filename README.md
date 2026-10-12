# Nimbus academy

One clean course for AI engineering: **build one product, learn every alternative.**

The product is **Nimbus**, an assistant for a café chain's documents (`src/nimbus`): a real Python package with typed
configuration, one module per responsibility, incremental indexing, metrics, a CLI and tests. The course app
(`src/academy` + `frontend/`) teaches it module by module:

- 🧪 **Labs** run each alternative on the Nimbus documents (every loader, every splitter, every search method) and
  show the results side by side.
- ⚖️ **Scorecards** compare them on the same 45 evaluation questions: hit, precision, recall, nDCG and MAP @1/3/5, MRR, coverage, answerability, latency p50/p95,
  per question type and per source, plus tokens and cost where a model is called.
- 🏭 **The production path** upgrades Nimbus only when the scorecard proves it, and every evaluation is kept in a
  history, so each change gets a before/after (for example: chunk size 500 → 150 took hit@5 from 0.56 to 0.60, but split two answers apart).
- 📄 Each page shows **the real code** from the package, read from the file, with notes on the key lines.

## Nothing from the old courses is lost

`content/syllabus.json` lists every learning point from the earlier CLI courses (1,362 points from 307 lessons, extracted once by `scripts/build_syllabus.py`), plus
points that are new here (`content/additions.json`, e.g. 14 on semantic caching). `content/curriculum.json` gives each
point exactly one home module: a module takes whole parts or lessons (`from`), can hand single lessons to a module where
they fit better (`except`), and the old per-course capstones are built by a module's production path (`capstones`),
because Nimbus itself is the capstone. A module counts as built only when its labs and production path cover **all** of
its points, and `tests/test_curriculum.py` fails if one is missed. The app shows the coverage per track and module, and
the full checklist on each track page.

| Track | Modules |
|---|---|
| 🦜 LangChain | 5 (all built) |
| 📚 RAG | 9 (all built) |
| 🕸️ Agents with LangGraph | 10 (all built) |
| 🔌 MCP | 4 (all built) |
| 🗄️ Data agents | 3 (all built) |
| ⚡ FastAPI | 4 (all built) |
| 🚀 Shipping to production | 7 (01–05 built; 06 Kubernetes and 07 Traffic & the cloud need Docker for kind) |

## Start

You need [uv](https://docs.astral.sh/uv/), Docker (Docker Desktop or OrbStack) and Node 22+. Then, in the repo folder:

```bash
./start.sh          # everything below, then the course app → http://localhost:8300  (Ctrl-C stops it)
./start.sh stop     # stop Postgres and Redis when you are done (data is kept)
```

It installs the Python packages, creates `.env` from `.env.example` (add your keys, below), starts Postgres and Redis
in Docker, builds the UI, and starts the app. Re-running only redoes what changed. Without Docker it still starts:
labs that need Postgres or Redis say so.

## Setup by hand

The same steps one by one, in the repo folder:

```bash
uv sync                                      # Python packages, in .venv
cp .env.example .env                         # then add your API keys (below)
docker compose up -d                         # Postgres 17 + pgvector (container learn-ai-course-postgres, port 5443)
cd frontend && npm install && npm run build  # the course app (once, and after UI changes)
```

API keys go in `.env` (git-ignored). All free plans:

| Key | Used for | Free plan |
|---|---|---|
| `MISTRAL_API_KEY` ([console.mistral.ai](https://console.mistral.ai)) | the default chat model, `ministral-14b-latest` | about 30 requests a minute |
| `GROQ_API_KEY` ([console.groq.com](https://console.groq.com)) | the reasoning model, `openai/gpt-oss-20b` (LangChain 01 reasoning lab) | 30 requests a minute, 200K tokens a day |

Labs that call a model say so before they run, with how many calls they make. Mistral's own reasoning model
(`magistral-small`) answers 429 on the free plan, which is why reasoning runs on Groq. Local models need Ollama:
`brew install ollama`, then `ollama serve` and `ollama pull qwen3:1.7b` (about 1.4 GB); the local-model lab tells you
if it isn't running. Embeddings, search and retrieval evaluation run locally with no key.

## Use it

```bash
./start.sh                      # or just the course app: uv run academy → http://localhost:8300
uv run nimbus rag ingest        # ingest data/nimbus (incremental: re-runs skip unchanged chunks; --force re-embeds)
uv run nimbus rag eval          # score retrieval on data/eval/retrieval.json
uv run nimbus rag search "How many vacation days do employees have?"
uv run nimbus rag stats         # what's in the index
uv run nimbus support triage "My order NC-12093 came with the wrong beans"   # triage + streamed reply draft
uv run nimbus support eval --prompt v2 --method json_schema                   # triage scorecard (20 model calls)
uv run pytest                   # unit tests + end-to-end tests against Postgres
./scripts/ci.sh                 # what CI runs on every push: lint, fast tests, Dockerfile lint, compose check
uv run python scripts/eval_gate.py support-reply:<commit>   # the quality gate CI runs on pull requests
docker compose --profile app up -d --build                 # the whole stack: API, RQ worker, Postgres, Redis
uv run python -m nimbus.api.loadtest http://127.0.0.1:8000 --users 8   # a load test
```

## Configuration

Every setting comes from an environment variable (prefix `NIMBUS_`) or `.env`:

| Variable | Default | |
|---|---|---|
| `NIMBUS_DATABASE_URL` | `postgresql+psycopg://nimbus:nimbus@127.0.0.1:5443/nimbus` | the Postgres database |
| `NIMBUS_DATA_DIR` | `data/nimbus` | the documents to ingest |
| `NIMBUS_CHUNK_SIZE` / `NIMBUS_CHUNK_OVERLAP` | `500` / `80` | characters |
| `NIMBUS_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | must be the same at ingestion and search |
| `NIMBUS_CHAT_MODEL` | `ministral-14b-latest` | provider set by `NIMBUS_CHAT_PROVIDER` (`mistralai`) |
| `NIMBUS_CHAT_REQUESTS_PER_SECOND` | `0.4` | client-side rate limit, under the free plan's limit (shared per provider) |
| `NIMBUS_REASONING_PROVIDER` / `NIMBUS_REASONING_MODEL` | `groq` / `openai/gpt-oss-20b` | the reasoning role |
| `NIMBUS_LOCAL_MODEL` / `NIMBUS_OLLAMA_URL` | `qwen3:1.7b` / `http://localhost:11434` | the local role |
| `NIMBUS_TRIAGE_PROMPT_VERSION` / `NIMBUS_TRIAGE_METHOD` | `v2` / `json_schema` | chosen by the triage scorecard |
| `NIMBUS_FALLBACK_PROVIDER` / `NIMBUS_FALLBACK_MODEL` | `groq` / `openai/gpt-oss-20b` | the backup when the default provider is down |
| `NIMBUS_AGENT_OUTPUT` | `tool` | how the support agent returns its outcome (ToolStrategy), chosen by the agent scorecard |
| `NIMBUS_LOG_LEVEL` | `INFO` | |

## Layout

```
├── start.sh                 one command: packages, .env, Postgres + Redis, UI build, the app
├── src/nimbus/              the product
│   ├── config.py            settings
│   ├── llm/                 configured models per role, error classification, usage/cost, conversations
│   ├── support/             triage, the LCEL pipeline, tools, the support agent (v1) and the guarded agent (v2),
│   │                        each with its scorecard (triage, outage drill, tool use, agent scenarios, guardrails)
│   ├── rag/ingestion/       loaders, cleaning, chunking, embeddings, store, pipeline
│   ├── rag/evaluation.py    the retrieval scorecard
│   └── cli.py               `nimbus` command
├── src/academy/             the course app
│   ├── catalog.py           curriculum, syllabus, coverage, source code lookup
│   ├── labs/                @lab experiments per track and module (labs/langchain/m01_models.py, …)
│   └── web/app.py           HTTP API + serves the app
├── content/                 curriculum.json, syllabus.json, and one page per module (content/rag/01_foundations.json)
├── data/nimbus/             the Nimbus documents (including distractors and formats later modules learn to load)
├── data/support/            the support order system: 12 orders, subscriptions, customers (a fixed "today")
├── data/eval/               labelled sets: retrieval questions, tickets, tool requests, agent scenarios, guardrail cases
├── frontend/                React + TypeScript + Tailwind
├── tests/
└── docker-compose.yml
```

`.state/` keeps the scorecard histories and saved conversations (git-ignored).

## Clean up

```bash
docker compose down -v    # delete the database container and its volume
```
