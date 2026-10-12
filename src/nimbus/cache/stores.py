"""Where semantic cache entries live. Every store answers one question: the nearest cached entry in a scope.

MemoryStore        a list in this process: fastest, lost on restart, not shared between processes
SQLiteStore        a file: survives restarts, one machine, brute-force similarity in Python
RedisVectorStore   Redis 8 Vector Sets (VADD / VSIM): shared by every process, HNSW search in Redis
PgVectorStore      PostgreSQL + pgvector: in the database you already run, SQL, HNSW index
"""

import json
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Entry:
  question: str
  answer: str
  vector: list[float]
  scope: str = "public:en"  # tenant / audience / language: answers are only reused within it
  model: str = ""
  prompt_version: str = ""
  index_version: str = ""
  sources: list[str] = field(default_factory=list)
  created_at: float = field(default_factory=time.time)
  ttl_s: float = 3600
  hits: int = 0
  id: str = field(default_factory=lambda: uuid.uuid4().hex)

  def expired(self, now: float | None = None) -> bool:
    return (now or time.time()) - self.created_at > self.ttl_s

  def payload(self) -> dict:
    return {k: v for k, v in asdict(self).items() if k != "vector"}


def unit(vector) -> np.ndarray:
  v = np.asarray(vector, dtype=np.float32)
  return v / (np.linalg.norm(v) or 1.0)


class MemoryStore:
  name = "memory"

  def __init__(self) -> None:
    self.entries: list[Entry] = []
    self._matrices: dict[str, tuple[list[Entry], np.ndarray]] = {}  # per scope, normalized once, rebuilt on change

  def add(self, entry: Entry) -> None:
    self.entries.append(entry)
    self._matrices.pop(entry.scope, None)

  def nearest(self, vector, scope: str) -> tuple[Entry, float] | None:
    if scope not in self._matrices:
      candidates = [e for e in self.entries if e.scope == scope]
      if not candidates:
        return None
      self._matrices[scope] = (candidates, np.stack([unit(e.vector) for e in candidates]))
    candidates, matrix = self._matrices[scope]
    scores = matrix @ unit(vector)
    best = int(np.argmax(scores))
    return candidates[best], float(scores[best])

  def remove(self, entry_id: str) -> None:
    self.entries = [e for e in self.entries if e.id != entry_id]
    self._matrices.clear()

  def hit(self, entry: Entry) -> None:
    entry.hits += 1

  def __len__(self) -> int:
    return len(self.entries)


class SQLiteStore:
  name = "sqlite"

  def __init__(self, path: Path) -> None:
    self.path = path
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
      db.execute("CREATE TABLE IF NOT EXISTS cache (id TEXT PRIMARY KEY, scope TEXT, vector BLOB, payload TEXT)")
      db.execute("CREATE INDEX IF NOT EXISTS cache_scope ON cache(scope)")

  def add(self, entry: Entry) -> None:
    with closing(sqlite3.connect(self.path)) as db, db:  # sqlite3's `with` commits; closing() closes
      db.execute(
        "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?)",
        (entry.id, entry.scope, unit(entry.vector).tobytes(), json.dumps(entry.payload())),
      )

  def nearest(self, vector, scope: str) -> tuple[Entry, float] | None:
    with closing(sqlite3.connect(self.path)) as db:
      rows = db.execute("SELECT vector, payload FROM cache WHERE scope = ?", (scope,)).fetchall()
    if not rows:
      return None
    matrix = np.stack([np.frombuffer(v, dtype=np.float32) for v, _ in rows])
    scores = matrix @ unit(vector)
    best = int(np.argmax(scores))
    payload = json.loads(rows[best][1])
    return Entry(vector=matrix[best].tolist(), **payload), float(scores[best])

  def remove(self, entry_id: str) -> None:
    with closing(sqlite3.connect(self.path)) as db, db:
      db.execute("DELETE FROM cache WHERE id = ?", (entry_id,))

  def hit(self, entry: Entry) -> None:
    entry.hits += 1
    with closing(sqlite3.connect(self.path)) as db, db:
      db.execute("UPDATE cache SET payload = ? WHERE id = ?", (json.dumps(entry.payload()), entry.id))

  def __len__(self) -> int:
    with closing(sqlite3.connect(self.path)) as db:
      return db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]


class RedisVectorStore:
  """Redis 8 Vector Sets: VADD stores a vector with JSON attributes (scope, expiry), VSIM finds the nearest within a
  FILTER. The answer and metadata live in a hash next to it, with its own TTL."""

  name = "redis"

  def __init__(self, redis, key: str = "semcache") -> None:
    self.redis, self.key = redis, key

  def add(self, entry: Entry) -> None:
    values = [str(x) for x in unit(entry.vector).tolist()]
    attrs = json.dumps({"scope": entry.scope})
    self.redis.execute_command("VADD", self.key, "VALUES", len(values), *values, entry.id, "SETATTR", attrs)
    self.redis.set(f"{self.key}:entry:{entry.id}", json.dumps(entry.payload()), ex=int(entry.ttl_s))

  def nearest(self, vector, scope: str) -> tuple[Entry, float] | None:
    values = [str(x) for x in unit(vector).tolist()]
    found = self.redis.execute_command(
      "VSIM", self.key, "VALUES", len(values), *values, "WITHSCORES", "COUNT", 1, "FILTER", f'.scope == "{scope}"'
    )
    if not found:
      return None
    entry_id, score = next(iter(found.items())) if isinstance(found, dict) else (found[0], found[1])  # redis-py 8: dict
    entry_id, score = entry_id.decode() if isinstance(entry_id, bytes) else entry_id, float(score)
    raw = self.redis.get(f"{self.key}:entry:{entry_id}")
    if raw is None:  # the payload expired: drop the vector too
      self.redis.execute_command("VREM", self.key, entry_id)
      return None
    return Entry(vector=[], **json.loads(raw)), 2 * score - 1  # VSIM scores are (1 + cosine) / 2

  def remove(self, entry_id: str) -> None:
    self.redis.execute_command("VREM", self.key, entry_id)
    self.redis.delete(f"{self.key}:entry:{entry_id}")

  def hit(self, entry: Entry) -> None:
    entry.hits += 1
    key = f"{self.key}:entry:{entry.id}"
    self.redis.set(key, json.dumps(entry.payload()), keepttl=True)

  def clear(self) -> None:
    keys = [self.key, *self.redis.scan_iter(f"{self.key}:entry:*")]
    self.redis.delete(*keys)

  def __len__(self) -> int:
    return int(self.redis.execute_command("VCARD", self.key) or 0)


class PgVectorStore:
  """PostgreSQL + pgvector: an HNSW index on cosine distance, filtered by scope in SQL."""

  name = "pgvector"

  def __init__(self, conninfo: str, dims: int, table: str = "semantic_cache") -> None:
    import psycopg

    self.conninfo, self.table = conninfo, table
    with psycopg.connect(conninfo, autocommit=True) as db:
      db.execute("CREATE EXTENSION IF NOT EXISTS vector")
      db.execute(
        f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, scope TEXT, embedding vector({dims}), "
        "payload JSONB, expires_at TIMESTAMPTZ)"
      )
      db.execute(f"CREATE INDEX IF NOT EXISTS {table}_hnsw ON {table} USING hnsw (embedding vector_cosine_ops)")

  def _vector(self, vector) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in unit(vector).tolist()) + "]"

  def add(self, entry: Entry) -> None:
    import psycopg

    with psycopg.connect(self.conninfo, autocommit=True) as db:
      db.execute(
        f"INSERT INTO {self.table} VALUES (%s, %s, %s::vector, %s, now() + make_interval(secs => %s)) "
        "ON CONFLICT (id) DO NOTHING",
        (entry.id, entry.scope, self._vector(entry.vector), json.dumps(entry.payload()), entry.ttl_s),
      )

  def nearest(self, vector, scope: str) -> tuple[Entry, float] | None:
    import psycopg

    with psycopg.connect(self.conninfo) as db:
      row = db.execute(
        f"SELECT payload, 1 - (embedding <=> %s::vector) FROM {self.table} WHERE scope = %s AND expires_at > now() "
        "ORDER BY embedding <=> %s::vector LIMIT 1",
        (self._vector(vector), scope, self._vector(vector)),
      ).fetchone()
    if row is None:
      return None
    return Entry(vector=[], **row[0]), float(row[1])

  def remove(self, entry_id: str) -> None:
    import psycopg

    with psycopg.connect(self.conninfo, autocommit=True) as db:
      db.execute(f"DELETE FROM {self.table} WHERE id = %s", (entry_id,))

  def hit(self, entry: Entry) -> None:
    entry.hits += 1

  def clear(self) -> None:
    import psycopg

    with psycopg.connect(self.conninfo, autocommit=True) as db:
      db.execute(f"TRUNCATE {self.table}")

  def __len__(self) -> int:
    import psycopg

    with psycopg.connect(self.conninfo) as db:
      return db.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
