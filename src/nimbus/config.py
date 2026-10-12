"""Settings, read from environment variables (prefix NIMBUS_) or a .env file, with safe defaults for local use.

Every tunable lives here, so behaviour changes through configuration, not code edits:
  NIMBUS_DATABASE_URL=postgresql+psycopg://...   NIMBUS_CHUNK_SIZE=800   NIMBUS_LOG_LEVEL=DEBUG
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parents[2]  # ai-course/


class Settings(BaseSettings):
  model_config = SettingsConfigDict(env_prefix="NIMBUS_", env_file=PROJECT_DIR / ".env", extra="ignore")

  # Where documents come from, and where the index lives
  data_dir: Path = PROJECT_DIR / "data" / "nimbus"
  database_url: str = "postgresql+psycopg://nimbus:nimbus@127.0.0.1:5443/nimbus"
  table_name: str = "rag_chunks"

  # Loading
  loader_version: Literal["v1", "v2", "v3"] = Field(
    "v3", description="v1: text, HTML, PDF. v2: + CSV, JSON, code, Office, email. v3: + images and scanned PDFs (OCR)"
  )
  ocr_engine: Literal["tesseract", "vision"] = Field("tesseract", description="Who reads images in loaders v3")
  # Chunking
  chunk_size: int = Field(300, ge=100, le=4000, description="Maximum characters per chunk")
  chunk_overlap: int = Field(50, ge=0, description="Characters repeated between neighbouring chunks")
  chunk_strategy: Literal["recursive", "sections"] = Field("recursive", description="sections: split at headings first")
  chunk_context: Literal["none", "headers", "llm"] = Field(
    "headers", description="What each chunk gets in front before embedding: its source and section, or a model sentence"
  )
  min_chunk_chars: int = Field(40, ge=0, description="Drop fragments shorter than this (a lone heading, a page number)")

  # Embeddings (a local model: no API key, no per-token cost)
  embedding_provider: Literal["local", "mistral"] = "local"
  embedding_model: str = "BAAI/bge-small-en-v1.5"  # mistral: mistral-embed, with embedding_dims=1024 and a new table
  embedding_dims: int = 384
  search_mode: Literal["dense", "hybrid"] = Field("hybrid", description="hybrid: full-text + vector search, RRF")
  search_candidates: int = Field(10, ge=1, le=200, description="Candidates fetched before reranking")
  rerank: Literal["off", "flashrank"] = Field(
    "flashrank", description="flashrank: a local cross-encoder reranks candidates"
  )
  reranker_model: str = "ms-marco-MiniLM-L-12-v2"
  query_translation: Literal["off", "non_ascii", "always"] = Field(
    "non_ascii",
    description="Translate questions to English before searching: never, when they contain non-ASCII "
    "letters (Hindi, Tamil, accented Latin), or always",
  )
  embedding_batch_size: int = Field(64, ge=1)
  model_cache_dir: Path = PROJECT_DIR / ".cache" / "models"

  # Indexing
  index_batch_size: int = Field(100, ge=1, description="Chunks written per database round trip")

  # Chat model (Mistral's free plan by default: 30 requests/min). The API key is read from MISTRAL_API_KEY.
  chat_provider: str = "mistralai"
  chat_model: str = "ministral-14b-latest"
  chat_requests_per_second: float = Field(0.4, gt=0, description="Client-side rate limit, below the provider's")
  chat_timeout_s: int = Field(60, ge=1)
  chat_max_retries: int = Field(6, ge=0)
  small_model: str = Field("ministral-8b-latest", description="Routing: simple questions go here")
  # Example prices in US$ per million tokens (input, output). Providers change them: keep them in config.
  model_prices: dict[str, tuple[float, float]] = Field(
    default_factory=lambda: {
      "ministral-3b-latest": (0.04, 0.04),
      "ministral-8b-latest": (0.10, 0.10),
      "ministral-14b-latest": (0.20, 0.20),
      "openai/gpt-oss-20b": (0.10, 0.50),
      "openai/gpt-oss-120b": (0.15, 0.75),
    }
  )

  # A reasoning model (thinks before answering; returns its reasoning). Groq's free plan: 30 requests/min.
  reasoning_provider: str = "groq"
  reasoning_model: str = "openai/gpt-oss-20b"
  reasoning_requests_per_second: float = Field(0.4, gt=0)
  # Judges for evaluations, best first; each Groq model has its own daily token quota (200K on the free plan)
  judge_models: list[str] = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

  # Support desk memory: checkpoints encrypted at rest when a key is set (32 bytes as hex: `openssl rand -hex 32`)
  checkpoint_key: SecretStr | None = None
  desk_history_tokens: int = Field(3000, ge=500, description="The most recent history the desk's model sees")
  # The Nimbus assistant's architecture (Agents Module 7 measured all three)
  assistant_architecture: Literal["single", "router", "supervisor"] = "router"

  # A backup model on a different provider, used when the default one is down (support pipeline fallbacks)
  fallback_provider: str = "groq"
  fallback_model: str = "openai/gpt-oss-20b"
  fallback_requests_per_second: float = Field(0.4, gt=0)

  # A local model through Ollama: no key, no rate limit, works offline (`ollama serve`, `ollama pull qwen3:1.7b`)
  local_model: str = "qwen3:1.7b"
  ollama_url: str = "http://localhost:11434"

  # Support triage
  triage_prompt_version: str = "v2"
  # How the schema reaches the model. json_schema measured on the 20 labelled tickets (prompt v2): all fields right
  # 0.85 vs 0.80 with function_calling, 534 vs 1,031 input tokens per ticket, so 36% cheaper.
  triage_method: Literal["function_calling", "json_schema"] = "json_schema"
  agent_output: Literal["tool", "provider"] = "tool"  # how the support agent returns SupportOutcome
  triage_concurrency: int = Field(4, ge=1, description="Tickets triaged in parallel (the rate limiter still applies)")
  eval_tickets: Path = PROJECT_DIR / "data" / "eval" / "tickets.json"
  eval_tool_requests: Path = PROJECT_DIR / "data" / "eval" / "tool_requests.json"
  eval_agent_scenarios: Path = PROJECT_DIR / "data" / "eval" / "agent_scenarios.json"
  eval_guardrail_cases: Path = PROJECT_DIR / "data" / "eval" / "guardrail_cases.json"
  conversations_dir: Path = PROJECT_DIR / ".state" / "conversations"

  # Evaluation
  eval_questions: Path = PROJECT_DIR / "data" / "eval" / "retrieval.json"

  # The HTTP API (FastAPI track). No keys configured = open, for local development only.
  api_keys: list[SecretStr] = Field(default_factory=list, description="Accepted X-API-Key values")
  cors_origins: list[str] = Field(
    default_factory=lambda: ["http://localhost:5173"], description="Browser apps allowed to call the API (exact)"
  )
  api_database_url: str = Field(
    f"sqlite:///{PROJECT_DIR / '.cache' / 'api' / 'notes.db'}", description="Users and notes (SQLModel)"
  )
  jwt_secret: SecretStr = Field(
    SecretStr("dev-only-secret-change-me-in-production-0123456789"), description="Signs access tokens (32+ bytes)"
  )
  jwt_minutes: int = Field(60, ge=1, description="How long an access token is valid")
  upload_max_bytes: int = Field(2_000_000, ge=1_000, description="Largest accepted upload")
  # The support agent as a service (FastAPI track, Module 4)
  agent_concurrency: int = Field(4, ge=1, description="Agent runs at the same time; more requests wait their turn")
  agent_timeout_s: float = Field(60, gt=0, description="A request's budget before it answers 504")
  agent_checkpointer: Literal["memory", "postgres"] = Field(
    "memory", description="Where conversations live: memory (lost on restart) or the course Postgres"
  )
  redis_url: str = Field("redis://127.0.0.1:6390/0", description="Background jobs (RQ)")
  # Keeping abuse out (Production track, Module 1)
  api_key_hashes: dict[str, str] = Field(
    default_factory=dict, description="sha256(key) → plan: the database holds hashes, never keys"
  )
  plan_limits: dict[str, int] = Field(default_factory=lambda: {"free": 3, "pro": 20}, description="Requests/minute")
  daily_token_budget: int = Field(20_000, ge=1, description="Model tokens per caller per day")
  circuit_threshold: int = Field(2, ge=1, description="Failures in a row before the primary model is skipped")
  circuit_cooldown_s: float = Field(30, gt=0, description="How long the primary is skipped once the circuit opens")
  # Caching (Production track, Module 2). The threshold was chosen on data/eval/cache_pairs.json with the entity guard
  semantic_cache_threshold: float = Field(0.75, ge=0.5, le=1.0, description="Cosine similarity to reuse an answer")
  cache_ttl_s: int = Field(3600, ge=1, description="How long a cached answer lives")
  # Tracing (Production track, Module 3). Nothing is uploaded unless a LangSmith key is set HERE (ai-course/.env:
  # NIMBUS_LANGSMITH_API_KEY) and tracing is switched on.
  langsmith_api_key: SecretStr | None = None
  langsmith_tracing: bool = False
  langsmith_project: str = "nimbus"
  langsmith_endpoint: str = "https://api.smith.langchain.com"
  langsmith_sampling_rate: float = Field(1.0, ge=0, le=1, description="Share of traces kept")
  support_prompt: str = Field("support-reply:production", description="Registry ref the support reply runs")
  # Shipping (Production track, Module 5)
  canary_prompt: str = Field("support-reply:canary", description="What the canary share of callers runs")
  canary_percent: int = Field(0, ge=0, le=100, description="Share of callers on the canary; 0 = rollback")
  shadow_prompt: str | None = Field(None, description="Also answer every request with this prompt, unseen")
  fake_model_delay: float | None = Field(None, ge=0, description="Load tests: a fake model taking this long")
  log_format: Literal["text", "json"] = "text"

  log_level: str = "INFO"

  @model_validator(mode="after")
  def overlap_smaller_than_chunk(self) -> "Settings":
    if self.chunk_overlap >= self.chunk_size:
      raise ValueError("chunk_overlap must be smaller than chunk_size")
    return self


@lru_cache
def get_settings() -> Settings:
  return Settings()
