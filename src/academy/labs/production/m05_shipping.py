"""Production · Module 5 labs: shipping the service. Configuration from the environment, the Docker image, the whole
stack with compose, logs and metrics, CI, load tests, and safe releases (canary, shadow, rollback).

Steps that need Docker's daemon run when it's up. The same service also runs as plain processes (nimbus.api.stack):
the API under uvicorn and an RQ worker, configured by the same environment variables compose passes."""

import asyncio
import io
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
import structlog
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from academy.labs.runtime import Metric, callout, code, example, lab, scorecard, table
from nimbus.api.app import create_app
from nimbus.api.loadtest import load
from nimbus.api.release import variant
from nimbus.api.stack import Stack
from nimbus.config import PROJECT_DIR, Settings
from nimbus.observability.metrics import ALERTS, setup_json_logs
from nimbus.prompts.registry import PromptRegistry
from nimbus.support import reply_eval as rv

DOCKERFILE = PROJECT_DIR / "Dockerfile"
SECRET = re.compile(r"KEY|SECRET|PASSWORD|TOKEN", re.IGNORECASE)
CUSTOMER = {"X-Customer-ID": "c-1042"}
BIN = Path(sys.executable).parent


def docker_up() -> bool:
  if shutil.which("docker") is None:
    return False
  return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0


def run(command: list[str], timeout: float = 600, env: dict | None = None) -> tuple[int, str, float]:
  """(exit code, the output's last lines, seconds)."""
  started = time.perf_counter()
  done = subprocess.run(
    command, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=timeout, env=os.environ | (env or {})
  )
  tail = "\n".join((done.stdout.strip() or done.stderr.strip()).splitlines()[-4:])  # the result, not progress bars
  return done.returncode, tail, round(time.perf_counter() - started, 1)


def fake_app(**settings) -> TestClient:
  return TestClient(
    create_app(settings=Settings(**settings), chat_model=FakeListChatModel(responses=["Happy to help!"]))
  )


def versions() -> dict[str, str]:
  """The support reply prompt's commits: v1 (shipped), v2, v3 (released by Module 4's improvement loop)."""
  history = PromptRegistry().history("support-reply")
  return {f"v{i + 1}": h["commit"] for i, h in enumerate(history)}


# ── Lab 1: configuration and secrets ───────────────────────────────────────────
def from_environment() -> dict:
  """One image, every environment: only the variables differ."""
  with patch.dict(os.environ, {"NIMBUS_REDIS_URL": "redis://redis:6379/0", "NIMBUS_CANARY_PERCENT": "5"}):
    settings = Settings(_env_file=None)
  return {"redis_url": settings.redis_url, "canary_percent": settings.canary_percent}


def precedence(env_file: Path) -> list[list]:
  """One setting from every source: argument > environment variable > .env file > default."""
  with patch.dict(os.environ):
    os.environ.pop("NIMBUS_CANARY_PERCENT", None)
    rows = [["default", Settings(_env_file=None).canary_percent]]
    rows.append([".env file (NIMBUS_CANARY_PERCENT=10)", Settings(_env_file=env_file).canary_percent])
    os.environ["NIMBUS_CANARY_PERCENT"] = "20"
    rows.append(["+ environment variable (20)", Settings(_env_file=env_file).canary_percent])
    rows.append(["+ argument (30)", Settings(_env_file=env_file, canary_percent=30).canary_percent])
  return rows


def wrong_value() -> str:
  """A wrong value is refused when Settings is created, before anything is served."""
  with patch.dict(os.environ, {"NIMBUS_CANARY_PERCENT": "150"}):
    try:
      Settings(_env_file=None)
    except ValidationError as error:
      return "\n".join(str(error).splitlines()[:3])
  return "accepted"


def service_with_wrong_value() -> tuple[int, str, float]:
  """The real service started with canary_percent=150: it exits at once, with the error."""
  return run([str(BIN / "uvicorn"), "nimbus.api.app:app", "--port", "0"], 60, {"NIMBUS_CANARY_PERCENT": "150"})


def secrets() -> dict:
  settings = Settings(_env_file=None, jwt_secret="a-real-secret-from-the-environment")
  return {
    "repr(settings.jwt_secret)": repr(settings.jwt_secret),
    "secret in str(settings)": "a-real-secret" in str(settings),
    "secret in settings.model_dump_json()": "a-real-secret" in settings.model_dump_json(),
    "read where it's used": len(settings.jwt_secret.get_secret_value()),  # security.py signs tokens with it
  }


def kept_out() -> list[list[str]]:
  """.env never reaches git or the image."""
  return [[name, ".env" in (PROJECT_DIR / name).read_text().split()] for name in (".gitignore", ".dockerignore")]


def json_values() -> dict:
  with patch.dict(os.environ, {"NIMBUS_PLAN_LIMITS": '{"free": 5, "pro": 100}'}):
    return Settings(_env_file=None).plan_limits


@lab("prod.config", "Configuration and secrets", calls="No model calls")
def config_lab():
  """The Settings class every Nimbus process reads: values from environment variables, in a fixed order of
  precedence, validated at start-up, secrets hidden, complex values as JSON."""
  env_file = PROJECT_DIR / ".cache" / "lab-settings.env"
  env_file.parent.mkdir(parents=True, exist_ok=True)
  env_file.write_text("NIMBUS_CANARY_PERCENT=10\n")
  order = precedence(env_file)
  env_file.unlink()
  refused = wrong_value()
  exit_code, tail, seconds = service_with_wrong_value()
  hidden = secrets()
  ignored = kept_out()
  view = table(["source", "canary_percent"], order, title="One setting, every source")
  return [
    view,
    code(f"exit code {exit_code} after {seconds} s\n{tail}", "text", "uvicorn with NIMBUS_CANARY_PERCENT=150"),
    table(["", "value"], [[k, str(v)] for k, v in hidden.items()], title="A SecretStr"),
    callout(
      f"The last source wins: {' → '.join(str(r[1]) for r in order)}. A wrong value stopped the real service in "
      f"{seconds} s with exit code {exit_code}, before it served a request. The secret appeared in neither str() nor "
      f"the JSON dump; .env is ignored by {', '.join(n for n, ok in ignored if ok)}.",
      "info",
      "Configuration is data, checked at start-up",
    ),
    example(
      "prod.l10.1",
      "Typed settings from the environment",
      from_environment,
      "nimbus/config.py",
      [code(json.dumps(from_environment()))],
    ),
    example("prod.l10.2", "Precedence", precedence, "", [view]),
    example(
      "prod.l10.3",
      "Validation at start-up",
      service_with_wrong_value,
      "Settings refuses the value; the service never starts.",
      [code(refused, "text"), code(f"exit code {exit_code} after {seconds} s", "text")],
    ),
    example("prod.l10.4", "Secrets as SecretStr", secrets, "", [code(json.dumps(hidden), "json")]),
    example("prod.l10.4", "Kept out of git and the image", kept_out, "", [table(["file", "ignores .env"], ignored)]),
    example("prod.l10.5", "Complex values as JSON", json_values, "", [code(json.dumps(json_values()))]),
  ]


# ── Lab 2: Docker ──────────────────────────────────────────────────────────────
def instructions(path: Path = DOCKERFILE) -> list[str]:
  """The Dockerfile's instructions, continuation lines joined, comments dropped."""
  joined = path.read_text().replace("\\\n", " ")
  return [" ".join(line.split()) for line in joined.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def stages() -> list[list[str]]:
  return [[i.split()[-1], i.split()[1]] for i in instructions() if i.startswith("FROM")]


def rebuilt_on_code_change() -> list[list[str]]:
  """A step re-runs when its input changed: the code, or the output of a step before it in the same stage."""
  rows, changed = [], False
  for step in instructions():
    if step.startswith("FROM"):
      changed = False
    changed = changed or step.startswith(("COPY src", "COPY --from=build"))
    rows.append([step[:80], "re-runs" if changed else "cached"])
  return rows


def runtime_config() -> dict:
  """What the image sets, and what it must not contain: configuration and keys come at `docker run` time."""
  steps = instructions()
  return {
    "ENV in the image": [s for s in steps if s.startswith("ENV")],
    "secrets in the Dockerfile": [s for s in steps if s.startswith(("ENV", "ARG")) and SECRET.search(s)],
    "run": "docker run -p 8000:8000 -e MISTRAL_API_KEY -e NIMBUS_CANARY_PERCENT=5 learn-ai-course-api",
  }


def healthcheck(url: str) -> int:
  """The image's HEALTHCHECK command, run against a server: exit 0 = healthy."""
  command = next(s for s in instructions() if s.startswith("HEALTHCHECK")).split(" CMD ", 1)[1]
  args = shlex.split(command.replace("http://127.0.0.1:8000", url))
  return subprocess.run([sys.executable, *args[1:]], capture_output=True, timeout=10).returncode


def check_image(tag: str = "learn-ai-course-api") -> dict:
  """Build, then check what shipped: size, user, health, a real request."""
  subprocess.run(["docker", "build", "-t", tag, "."], cwd=PROJECT_DIR, check=True, capture_output=True, timeout=1800)
  info = json.loads(subprocess.run(["docker", "image", "inspect", tag], capture_output=True, check=True).stdout)[0]
  name = f"learn-ai-course-check-{uuid.uuid4().hex[:6]}"
  subprocess.run(["docker", "run", "-d", "--name", name, "-p", "127.0.0.1::8000", tag], check=True, capture_output=True)
  try:
    port = (
      subprocess.run(["docker", "port", name, "8000"], capture_output=True, text=True).stdout.split(":")[-1].strip()
    )
    deadline, health = time.monotonic() + 60, None
    while time.monotonic() < deadline and health != "healthy":
      time.sleep(2)
      health = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Health.Status}}", name], capture_output=True, text=True
      ).stdout.strip()
    menu = httpx.get(f"http://127.0.0.1:{port}/menu", timeout=10).status_code
  finally:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
  return {"size_mb": round(info["Size"] / 1e6), "user": info["Config"]["User"], "health": health, "GET /menu": menu}


@lab("prod.docker", "Docker: the service as an image", calls="No model calls")
def docker_lab():
  """The Dockerfile in two stages, which steps a code change re-runs, configuration at run time, the HEALTHCHECK run
  against a real server, and, when Docker is running, the built image checked."""
  lint = run(["hadolint", str(DOCKERFILE)], 60) if shutil.which("hadolint") else (None, "hadolint isn't installed", 0)
  layers = rebuilt_on_code_change()
  config = runtime_config()
  with Stack({"NIMBUS_FAKE_MODEL_DELAY": "0"}, job_worker=False) as stack:
    healthy = healthcheck(stack.url)
  stopped = healthcheck(stack.url)
  if docker_up():
    image = check_image()
    built = table(["", "value"], [[k, str(v)] for k, v in image.items()], title="The built image")
  else:
    image = None
    built = callout(
      "Docker's daemon isn't running, so the image wasn't built. Start Docker (OrbStack or Docker Desktop) and run "
      "this lab again: it builds the image and checks its size, user, health and a real request.",
      "warning",
      "Image not built",
    )
  cached = sum(r[1] == "cached" for r in layers)
  view = table(["instruction", "after a code change"], layers)
  return [
    code(DOCKERFILE.read_text(), "dockerfile", "Dockerfile"),
    view,
    built,
    callout(
      f"{cached} of {len(layers)} steps stay cached when only the code changes: the dependencies are installed before "
      f"the code is copied. The HEALTHCHECK command exited {healthy} against the running server and {stopped} once "
      f"it stopped. hadolint: {'no findings' if lint[0] == 0 else lint[1]}. Secrets in the Dockerfile: "
      f"{len(config['secrets in the Dockerfile'])}.",
      "info",
      "An image that runs the same everywhere",
    ),
    example("prod.l11.1", "Two stages", stages, "", [table(["stage", "base image"], stages())]),
    example("prod.l11.2", "Layer caching", rebuilt_on_code_change, "", [view]),
    example("prod.l11.3", "Configuration at run time", runtime_config, "", [code(json.dumps(config, indent=2))]),
    example(
      "prod.l11.4",
      "HEALTHCHECK",
      healthcheck,
      "Docker runs this every 10 s; 3 failures in a row mark the container unhealthy.",
      [table(["server", "exit code"], [["running", healthy], ["stopped", stopped]])],
    ),
    example(
      "prod.l11.5",
      "Checking the result",
      check_image,
      "Needs Docker's daemon.",
      [built if image else code("Docker isn't running here: not built", "text")],
    ),
  ]


# ── Lab 3: docker compose ──────────────────────────────────────────────────────
def compose_config() -> dict:
  """The whole stack as compose resolves it (no daemon needed)."""
  command = ["docker", "compose", "--profile", "app", "config", "--format", "json"]
  return json.loads(subprocess.run(command, cwd=PROJECT_DIR, capture_output=True, text=True, check=True).stdout)


def redacted(env: dict | None) -> dict:
  return {k: ("•••" if SECRET.search(k) and v else v) for k, v in (env or {}).items()}


def services(config: dict) -> list[list[str]]:
  rows = []
  for name, s in config["services"].items():
    published = [f"{p.get('host_ip', '0.0.0.0')}:{p['published']}→{p['target']}" for p in s.get("ports", [])]
    rows.append(
      [
        name,
        s.get("image", "") + (" (built here)" if "build" in s else ""),
        " ".join(s.get("command") or []) or "(image default)",
        ", ".join(published) or "not published",
        ", ".join(s.get("depends_on", {})),
      ]
    )
  return rows


def by_name(config: dict) -> dict:
  """How the API finds Postgres and Redis: by service name, on the stack's private network."""
  env = redacted(config["services"]["api"].get("environment"))
  return {k: v for k, v in env.items() if k.startswith("NIMBUS_") and ("://" in str(v))}


def one_image(config: dict) -> list[list[str]]:
  return [
    [n, config["services"][n]["image"], " ".join(config["services"][n].get("command") or ["uvicorn (CMD)"])]
    for n in ("api", "worker")
  ]


def survives_restart(checkpointer: str) -> int:
  """Chat, restart the API, read the thread back: the messages that survived."""
  thread = f"restart-{checkpointer}-{uuid.uuid4().hex[:6]}"
  with Stack({"NIMBUS_AGENT_CHECKPOINTER": checkpointer, "NIMBUS_FAKE_MODEL_DELAY": "0"}, job_worker=False) as stack:
    body = {"message": "Where is my order?", "thread_id": thread}
    httpx.post(f"{stack.url}/agent/chat", json=body, headers=CUSTOMER, timeout=60).raise_for_status()
    stack.restart()
    return len(httpx.get(f"{stack.url}/agent/threads/{thread}", headers=CUSTOMER, timeout=30).json())


def drain(workers: int, jobs: int = 6) -> float:
  """`--scale worker=N` without Docker: N worker processes on one queue. Seconds to finish `jobs` one-second jobs."""
  import redis
  from rq import Queue

  from nimbus.config import get_settings

  with redis.Redis.from_url(get_settings().redis_url) as connection:
    queue = Queue("learn-scale", connection=connection)
    queue.empty()
    for _ in range(jobs):
      queue.enqueue("time.sleep", 1)
    started = time.perf_counter()
    command = [str(BIN / "rq"), "worker", "--burst", "--url", get_settings().redis_url, "learn-scale"]
    processes = [
      subprocess.Popen(
        [*command, "--worker-class", "rq.worker.SimpleWorker"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
      )
      for _ in range(workers)
    ]
    for p in processes:
      p.wait(timeout=60)
    return round(time.perf_counter() - started, 1)


def removed_by_down(config: dict) -> dict:
  return {
    "containers": list(config["services"]),
    "network": [n["name"] for n in config.get("networks", {}).values()],
    "volumes (-v)": [v.get("name", k) for k, v in config.get("volumes", {}).items()],
  }


@lab("prod.compose", "docker compose: the whole stack", calls="No model calls")
def compose_lab():
  """compose.yaml as compose resolves it: services, how they find each other, one image in two roles. Then, with the
  same environment variables, the stack as processes: a restart that keeps conversations, and more workers."""
  config = compose_config()
  rows = services(config)
  names = by_name(config)
  kept = {c: survives_restart(c) for c in ("memory", "postgres")}
  scaled = {n: drain(n) for n in (1, 3)}
  down = removed_by_down(config)
  view = table(["service", "image", "command", "published", "depends on"], rows)
  return [
    view,
    scorecard(
      [{"variant": f"checkpointer={c}", "kept": n} for c, n in kept.items()],
      [Metric("kept", "messages after a restart", "higher", "int")],
      title="Restart the API",
    ),
    table(["workers", "6 one-second jobs took (s)"], [[n, s] for n, s in scaled.items()], title="Scale the workers"),
    callout(
      f"Conversations kept after a restart: {kept['postgres']} messages with Postgres, {kept['memory']} in memory. "
      f"3 workers drained 6 one-second jobs in {scaled[3]} s, 1 worker in {scaled[1]} s, with no code change. "
      "Postgres and Redis publish ports on 127.0.0.1 only, for the course's own tools; in production, only the API "
      "would be reachable.",
      "decision",
      "One file describes the stack",
    ),
    example("prod.l12.1", "compose.yaml", compose_config, "docker-compose.yml", [view]),
    example("prod.l12.2", "Services find each other by name", by_name, "", [code(json.dumps(names, indent=2))]),
    example(
      "prod.l12.3", "One image, two roles", one_image, "", [table(["service", "image", "command"], one_image(config))]
    ),
    example(
      "prod.l12.4",
      "Restart: conversations survive",
      survives_restart,
      "",
      [table(["checkpointer", "messages after restart"], [[c, n] for c, n in kept.items()])],
    ),
    example(
      "prod.l12.5", "Scale the workers", drain, "", [table(["workers", "seconds"], [[n, s] for n, s in scaled.items()])]
    ),
    example(
      "prod.l12.6",
      "down -v",
      removed_by_down,
      "docker compose --profile app down -v",
      [code(json.dumps(down, indent=2))],
    ),
  ]


# ── Lab 4: observability ───────────────────────────────────────────────────────
@contextmanager
def json_logs():
  """JSON log lines into a buffer for the lab; the server's own logging is restored afterwards."""
  root = logging.getLogger()
  saved = root.handlers[:], root.level
  buffer = io.StringIO()
  setup_json_logs("INFO", buffer)
  try:
    yield buffer
  finally:
    root.handlers = saved[0]
    root.setLevel(saved[1])
    structlog.reset_defaults()


class ProviderDown(FakeListChatModel):
  """A model provider that can't be reached: the API answers 503."""

  async def _agenerate(self, *args, **kwargs):
    raise httpx.ConnectError("provider unreachable")


def request_ids(client: TestClient) -> list[list[str]]:
  mine = client.get("/menu", headers={"X-Request-ID": "checkout-7f3a"}).headers["x-request-id"]
  new = client.get("/menu").headers["x-request-id"]
  return [["caller sent checkout-7f3a", mine], ["caller sent nothing", new]]


def samples() -> dict[tuple, float]:
  from prometheus_client import generate_latest
  from prometheus_client.parser import text_string_to_metric_families

  from nimbus.observability.metrics import REGISTRY

  found = {}
  for family in text_string_to_metric_families(generate_latest(REGISTRY).decode()):
    for s in family.samples:
      found[(s.name, tuple(sorted(s.labels.items())))] = s.value
  return found


def p95_seconds(delta: dict) -> float:
  """p95 from the latency histogram's buckets, as Prometheus's histogram_quantile does (the bucket's upper bound)."""
  buckets: dict[float, float] = {}
  for (name, labels), value in delta.items():
    if name == "nimbus_request_seconds_bucket":
      le = float(dict(labels)["le"])
      buckets[le] = buckets.get(le, 0) + value  # summed over paths
  total = buckets.get(float("inf"), 0)
  return next((le for le, count in sorted(buckets.items()) if count >= 0.95 * total), 0.0) if total else 0.0


def alerts(delta: dict) -> list[list]:
  requests: dict[str, float] = {}
  for (name, labels), value in delta.items():
    if name == "nimbus_requests_total":
      status = dict(labels)["status"]
      requests[status] = requests.get(status, 0) + value  # summed over paths
  total = sum(requests.values()) or 1
  errors = sum(v for s, v in requests.items() if s.startswith("5")) / total
  limited = requests.get("429", 0) / total
  tokens = sum(v for (n, _), v in delta.items() if n == "nimbus_model_tokens_total")
  p95 = p95_seconds(delta)
  return [
    ["error rate > 2%", f"{errors:.0%}", errors > 0.02],
    ["p95 latency > 10 s", f"≤ {p95} s", p95 > 10],
    ["429s > 5%", f"{limited:.0%}", limited > 0.05],
    ["tokens per hour", int(tokens), tokens > 2e6],
  ]


@lab("prod.observability", "Observability: request ids, JSON logs, metrics", calls="No model calls")
def observability_lab():
  """Requests through the API with a fake model, then two during a provider outage: the request id each response and
  log line carries, the JSON log lines, the Prometheus metrics they produced, and the alert rules evaluated on them."""
  before = samples()
  with json_logs() as buffer, fake_app() as client:
    ids = request_ids(client)
    for _ in range(3):
      client.get("/menu")
    client.get("/orders/NC-9999")  # 404
    client.post("/support/reply", json={"message": "How much is a cold brew?", "tone": "warm"})
    client.post("/support/reply", json={"message": ""})  # 422
    with TestClient(create_app(settings=Settings(), chat_model=ProviderDown(responses=["-"]))) as down:
      for _ in range(2):
        down.post("/support/reply", json={"message": "How much is a cold brew?", "tone": "warm"})  # 503: an outage
  after = samples()
  delta = {k: v - before.get(k, 0) for k, v in after.items() if v - before.get(k, 0)}
  lines = [json.loads(line) for line in buffer.getvalue().splitlines() if line.startswith("{")]
  access = [line for line in lines if line.get("logger") == "nimbus.api.access"]
  counts = sorted(
    [[dict(lb)["path"], dict(lb)["status"], int(v)] for (n, lb), v in delta.items() if n == "nimbus_requests_total"]
  )
  rules = alerts(delta)
  view = table(["path", "status", "requests"], counts, title="nimbus_requests_total (this run)")
  return [
    table(["", "X-Request-ID returned"], ids),
    code("\n".join(json.dumps(line) for line in access[:4]), "json", "JSON log lines"),
    view,
    table(["alert", "now", "fires"], rules, title="Alert rules on these numbers"),
    callout(
      f"{len(access)} access log lines, each one JSON object with the request id; {sum(r[2] for r in counts)} requests "
      f"counted by path template and status. Firing: {', '.join(r[0] for r in rules if r[2]) or 'none'}.",
      "info",
      "Fields, not sentences",
    ),
    example("prod.l13.1", "Request ids", request_ids, "nimbus/api/middleware.py", [table(["", "returned"], ids)]),
    example("prod.l13.2", "JSON logs with structlog", setup_json_logs, "", [code(json.dumps(access[0]), "json")]),
    example("prod.l13.3", "Prometheus metrics", samples, "GET /metrics, parsed", [view]),
    example(
      "prod.l13.4",
      "What to alert on",
      alerts,
      "\n".join(f"{k}: {v}" for k, v in ALERTS.items()),
      [table(["alert", "now", "fires"], rules)],
    ),
  ]


# ── Lab 5: testing and CI ──────────────────────────────────────────────────────
def pyramid() -> list[list]:
  fast = run([str(BIN / "pytest"), "-q", "-p", "no:cacheprovider", "tests/test_api.py"], 300)
  slow = run([str(BIN / "pytest"), "-q", "-p", "no:cacheprovider", "-m", "integration", "--collect-only"], 120)
  return [
    ["API contract (fake model)", fast[1].splitlines()[-1], f"{fast[2]} s", "0"],
    ["quality with a real model", slow[1].splitlines()[-1], "minutes", "1+ per test"],
  ]


def injected() -> list[list]:
  """create_app(settings=...): a test passes its own keys and a fake model; nothing is patched."""
  settings = Settings(api_keys=["test-key"])
  with TestClient(create_app(settings=settings, chat_model=FakeListChatModel(responses=["Hi!"]))) as client:
    body = {"message": "Hello", "tone": "warm"}
    no_key = client.post("/support/reply", json=body)
    with_key = client.post("/support/reply", json=body, headers={"X-API-Key": "test-key"})
  return [["no key", no_key.status_code, no_key.json()], ["test-key", with_key.status_code, with_key.json()]]


def ci_jobs() -> list[list[str]]:
  import yaml

  workflow = yaml.safe_load((PROJECT_DIR / ".github" / "workflows" / "ci.yml").read_text())
  return [
    [name, job.get("if", "every push"), "; ".join(s.get("run", s.get("uses", "")) for s in job["steps"])]
    for name, job in workflow["jobs"].items()
  ]


def ci_locally() -> tuple[int, str, float]:
  return run(["bash", "scripts/ci.sh"], 600)


@lab("prod.ci", "Testing and CI", calls="No model calls")
def ci_lab():
  """The fast tests that run on every change, a test that injects its own settings and model, the GitHub Actions
  workflow, and the same steps run here."""
  layers = pyramid()
  calls = injected()
  jobs = ci_jobs()
  exit_code, tail, seconds = ci_locally()
  view = table(["layer", "result", "time", "model calls"], layers)
  return [
    view,
    table(["job", "runs on", "steps"], jobs, title=".github/workflows/ci.yml"),
    code(f"exit code {exit_code} in {seconds} s\n{tail}", "text", "scripts/ci.sh"),
    callout(
      f"The fast suite needs no keys and no network. scripts/ci.sh ran CI's steps here in {seconds} s "
      f"({'passed' if exit_code == 0 else 'FAILED'}); the real-model checks run on pull requests only.",
      "info" if exit_code == 0 else "warning",
      "Every change, tested before users see it",
    ),
    example("prod.l14.1", "The test pyramid", pyramid, "tests/test_api.py", [view]),
    example("prod.l14.2", "Dependency injection", injected, "", [table(["key", "status", "body"], calls)]),
    example("prod.l14.3", "CI on GitHub Actions", ci_jobs, "", [table(["job", "runs on", "steps"], jobs)]),
    example("prod.l14.4", "The same steps locally", ci_locally, "scripts/ci.sh", [code(tail, "text")]),
  ]


# ── Lab 6: load testing ────────────────────────────────────────────────────────
def connections_per_process(port: int) -> list[int]:
  """Open connections each API process is serving right now (from lsof), busiest first."""
  if shutil.which("lsof") is None:
    return []
  out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:ESTABLISHED"], capture_output=True, text=True).stdout
  pids = Counter(line.split()[1] for line in out.splitlines()[1:] if f"127.0.0.1:{port}->" in line)
  return sorted(pids.values(), reverse=True)


async def load_and_look(stack: Stack, users: int, seconds: float) -> dict:
  async def look() -> list[int]:
    await asyncio.sleep(seconds / 2)
    return await asyncio.to_thread(connections_per_process, stack.port)

  result, spread = await asyncio.gather(load(stack.url, users, seconds), look())
  return result | {"connections_per_process": spread}


def load_curve(api_workers: int, users: list[int], seconds: float = 6) -> list[dict]:
  """The agent endpoint with a fake 1-second model and 4 agent slots per process, under more and more users."""
  env = {"NIMBUS_FAKE_MODEL_DELAY": "1", "NIMBUS_AGENT_CONCURRENCY": "4"}
  with Stack(env, api_workers=api_workers, job_worker=False) as stack:
    return [asyncio.run(load_and_look(stack, n, seconds)) for n in users]


def littles_law(concurrency: int = 4, model_seconds: float = 1.0, users: int = 16) -> dict:
  """Ceiling = slots / time per request; beyond it requests queue: latency ≈ users / ceiling."""
  ceiling = concurrency / model_seconds
  return {"ceiling_rps": ceiling, "latency_at_users_s": max(model_seconds, users / ceiling)}


@lab("prod.load", "Load testing: what breaks first", calls="No model calls (a fake 1-second model)")
def load_lab():
  """N virtual users for 6 s each against the agent endpoint, one API process and then two (uvicorn --workers 2).
  The model is a fake that takes 1 s, so this measures the service, not the provider's rate limit."""
  one = load_curve(1, [2, 4, 8, 16])
  two = load_curve(2, [8, 16])
  predicted = littles_law()
  rows = [{"variant": f"1 process · {r['users']} users", **r} for r in one]
  rows += [{"variant": f"2 processes · {r['users']} users", **r} for r in two]
  metrics = [
    Metric("throughput_rps", "requests/s", "higher", "float"),
    Metric("p50_s", "p50 s", "lower", "float"),
    Metric("p95_s", "p95 s", "lower", "float"),
    Metric("error_rate", "errors", "lower", "ratio"),
  ]
  card = scorecard(rows, metrics, title="The agent endpoint under load")
  top_one = max(r["throughput_rps"] for r in one)
  top_two = max(r["throughput_rps"] for r in two)
  at16 = one[-1]
  return [
    card,
    callout(
      f"One process levelled off at {top_one} requests/s (Little's law: 4 slots ÷ 1 s = "
      f"{predicted['ceiling_rps']:.0f}). Past it, requests queued: at 16 users p50 was {at16['p50_s']} s "
      f"(predicted ≈ {predicted['latency_at_users_s']:.0f} s), "
      f"with no errors. Two processes reached {top_two} requests/s, not {2 * predicted['ceiling_rps']:.0f}: the "
      f"connections weren't shared evenly ({' vs '.join(map(str, two[-1]['connections_per_process']))} at "
      f"{two[-1]['users']} users), so one process queued while the other had free slots. A load balancer in front of "
      "separate instances spreads requests instead (Module 6). With a real model, the provider's rate limit "
      "(0.4 requests/s on Mistral's free plan) is the ceiling long before this.",
      "decision",
      "Find the ceiling before users do",
    ),
    example("prod.l15.1", "A load generator", load, "nimbus/api/loadtest.py", [card]),
    example(
      "prod.l15.2",
      "Throughput, p50, p95, errors",
      load_curve,
      "",
      [
        table(
          ["users", "requests/s", "p50 s", "p95 s", "errors"],
          [[r["users"], r["throughput_rps"], r["p50_s"], r["p95_s"], r["error_rate"]] for r in one],
        )
      ],
    ),
    example("prod.l15.3", "Little's law", littles_law, "", [code(json.dumps(predicted))]),
    example(
      "prod.l15.4",
      "Scale out: two processes",
      load_curve,
      "uvicorn nimbus.api.app:app --workers 2",
      [
        table(
          ["users", "requests/s", "p50 s", "p95 s", "connections per process"],
          [[r["users"], r["throughput_rps"], r["p50_s"], r["p95_s"], r["connections_per_process"]] for r in two],
        )
      ],
    ),
  ]


# ── Lab 7: safe releases ───────────────────────────────────────────────────────
def canary_share(percent: int, users: int = 1000) -> dict:
  """Who gets the canary: a hash of the user id, so the same user always lands on the same side."""
  ids = [f"user-{i}" for i in range(users)]
  first = [variant(u, percent) for u in ids]
  again = [variant(u, percent) for u in ids]
  return {
    "percent": percent,
    "on canary": first.count("canary"),
    "changed sides": sum(a != b for a, b in zip(first, again, strict=True)),
  }


def shadow(questions: list[dict], shadow_ref: str) -> list[dict]:
  """Every request also answered by the shadow prompt, after the response; users only see the stable answer."""
  app = create_app(settings=Settings(shadow_prompt=shadow_ref))
  with TestClient(app) as client:
    for q in questions:
      client.post("/support/reply", json=q["inputs"]).raise_for_status()
    return list(app.state.shadow)


def watch(url: str, users: int = 60) -> dict[str, dict]:
  """Per-variant numbers from real requests: how many, errors, median latency."""
  seen: dict[str, list] = {"stable": [], "canary": []}
  for i in range(users):
    started = time.perf_counter()
    r = httpx.post(f"{url}/support/reply", json={"message": "Hi", "tone": "warm"}, headers={"X-User-ID": f"u{i}"})
    seen[r.headers.get("x-variant", "stable")].append((r.status_code, time.perf_counter() - started))
  return {
    v: {
      "requests": len(s),
      "errors": sum(c >= 500 for c, _ in s),
      "p50_ms": round(sorted(t for _, t in s)[len(s) // 2] * 1000, 1) if s else 0,
    }
    for v, s in seen.items()
  }


def rollout_and_rollback(steps=(5, 25, 100)) -> list[list]:
  """Raise the canary step by step, then roll back: each step is a restart with one variable changed."""
  rows = []
  with Stack({"NIMBUS_FAKE_MODEL_DELAY": "0"}, job_worker=False) as stack:
    for percent in (*steps, 0):
      stack.restart(NIMBUS_CANARY_PERCENT=str(percent))
      numbers = watch(stack.url)
      rows.append(
        [
          f"{percent}%" + (" (rollback)" if percent == 0 else ""),
          numbers["canary"]["requests"],
          numbers["stable"]["requests"],
          numbers["canary"]["errors"] + numbers["stable"]["errors"],
        ]
      )
  return rows


@lab("prod.releases", "Safe releases: shadow, canary, rollback", uses_model=True, calls="~6 calls to Mistral")
def releases_lab():
  """The v3 support prompt released safely: shadow answers on three questions (real model), canary assignment by a
  hash of the user id, a rollout 5% → 25% → 100% watched per variant, and a rollback that's one variable."""
  v = versions()
  PromptRegistry().tag("support-reply", "canary", v["v3"])  # what NIMBUS_CANARY_PROMPT (support-reply:canary) runs
  questions = [e for e in rv.load_examples() if e["id"] in ("h1", "h3", "h4")]
  compared = shadow(questions, f"support-reply:{v['v3']}")
  facts = [
    [
      q["inputs"]["message"],
      rv.facts({"answer": c["shown"]}, q["reference_outputs"])["score"],
      rv.facts({"answer": c["shadow"]}, q["reference_outputs"])["score"],
    ]
    for q, c in zip(questions, compared, strict=True)
  ]
  shares = [canary_share(p) for p in (5, 25, 100)]
  steps = rollout_and_rollback()
  stable_ok, shadow_ok = sum(r[1] for r in facts), sum(r[2] for r in facts)
  view = table(["question", "stable (seen) facts", "shadow v3 facts"], facts)
  return [
    view,
    table(
      ["percent", "on canary of 1,000 users", "changed sides when asked again"], [list(s.values()) for s in shares]
    ),
    table(
      ["step", "canary requests", "stable requests", "5xx"],
      steps,
      title="Rollout, watched, then rolled back (60 users each)",
    ),
    callout(
      f"Shadow: v3 had the facts in {shadow_ok} of 3 answers, the stable prompt users saw in {stable_ok} of 3; no user "
      f"saw v3. Canary assignment moved {sum(s['changed sides'] for s in shares)} users between sides. Rollback was "
      f"a restart with NIMBUS_CANARY_PERCENT=0: {steps[-1][1]} canary requests afterwards.",
      "decision",
      "Release to a few, watch, then everyone",
    ),
    example("prod.l16.1", "Shadow testing", shadow, "routers/support.py: shadow_reply", [view]),
    example("prod.l16.2", "Canary by a hash", canary_share, "nimbus/api/release.py", [code(json.dumps(shares))]),
    example("prod.l16.3", "Watch, then raise", watch, "", [table(["step", "canary", "stable", "5xx"], steps[:-1])]),
    example(
      "prod.l16.4",
      "Rollback is configuration",
      rollout_and_rollback,
      "",
      [table(["step", "canary", "stable", "5xx"], steps[-1:])],
    ),
  ]


# ── Production path: deploy, then the release checklist ───────────────────────
def release_checklist() -> list[list[str]]:
  """Deploy the whole service (API ×2, a job worker, Postgres, Redis), then release the v3 prompt to a 10% canary
  the way a team would: every step has a check, and a failed check stops the release."""
  v3 = versions()["v3"]
  key = "checklist-key"
  env = {"NIMBUS_API_KEYS": json.dumps([key]), "NIMBUS_AGENT_CHECKPOINTER": "postgres", "NIMBUS_LOG_FORMAT": "json"}
  rows: list[list[str]] = []

  def check(step: str, ok: bool | None, evidence: str) -> bool:
    rows.append([step, "✅" if ok else ("⏭️ skipped" if ok is None else "❌"), evidence])
    return ok is not False

  code_, tail, s = run([str(BIN / "ruff"), "check", "src", "tests"], 120)
  fast = run([str(BIN / "pytest"), "-q", "-p", "no:cacheprovider", "tests/test_api.py"], 300)
  if not check("CI: lint and fast tests", code_ == 0 and fast[0] == 0, fast[1].splitlines()[-1]):
    return rows
  lint = run(["hadolint", str(DOCKERFILE)], 60) if shutil.which("hadolint") else (1, "hadolint missing", 0)
  check("Image: Dockerfile lint", lint[0] == 0, "hadolint: no findings" if lint[0] == 0 else lint[1])
  if docker_up():
    built = run(["docker", "build", "-t", "learn-ai-course-api", "."], 1800)
    check("Image: build", built[0] == 0, f"{built[2]} s")
  else:
    check("Image: build", None, "Docker's daemon isn't running; the stack runs as processes instead")
  gate = run([sys.executable, "scripts/eval_gate.py", f"support-reply:{v3}"], 900)
  verdict = next((line for line in gate[1].splitlines() if "floors" in line), gate[1].splitlines()[-1])
  if not check("Candidate passes the eval gate", gate[0] == 0, verdict[:200]):
    return rows
  headers = {"X-API-Key": key}
  with Stack(env, api_workers=2) as stack, httpx.Client(base_url=stack.url, timeout=120) as client:
    check("Deployed: health", client.get("/health").status_code == 200, f"{stack.url}, API ×2 + RQ worker")
    refused = client.post("/support/reply", json={"message": "Hi", "tone": "warm"}).status_code
    check("Contract: no key is refused", refused == 401, f"status {refused}")
    thread = f"checklist-{uuid.uuid4().hex[:6]}"
    body = {"message": "Where is my order NC-12093?", "thread_id": thread}
    asked = client.post("/agent/chat", json=body, headers=headers | CUSTOMER)
    before = client.get(f"/agent/threads/{thread}", headers=headers | CUSTOMER).json()
    stack.restart()
    after = client.get(f"/agent/threads/{thread}", headers=headers | CUSTOMER).json()
    check(
      "Conversations survive a restart",
      asked.status_code == 200 and len(before) >= 2 and after == before,
      f"chat answered {asked.status_code}; {len(after)} of {len(before)} messages after the restart",
    )
    stack.restart(NIMBUS_CANARY_PERCENT="10", NIMBUS_CANARY_PROMPT=f"support-reply:{v3}")
    users = {variant(f"user-{i}", 10): f"user-{i}" for i in range(50)}  # one user on each side
    example_ = next(e for e in rv.load_examples() if e["id"] == "h1")
    sides, statuses = {}, []
    for side, user in sorted(users.items()):
      r = client.post("/support/reply", json=example_["inputs"], headers=headers | {"X-User-ID": user})
      statuses.append(r.status_code)
      facts = rv.facts({"answer": r.json().get("answer", "")}, example_["reference_outputs"])["score"]
      sides[side] = (r.headers.get("x-variant"), facts)
    canary_ok = sides["canary"][0] == "canary" and sides["stable"][0] == "stable"
    check("Canary at 10%", canary_ok, f"X-Variant per user: {sides['canary'][0]} / {sides['stable'][0]}")
    check(
      "Canary answer has the facts",
      sides["canary"][1] == 1,
      f"canary facts {sides['canary'][1]}, stable {sides['stable'][1]}",
    )
    errors = sum(s >= 500 for s in statuses)  # each API process keeps its own /metrics: Prometheus scrapes them all
    check("Watch: no 5xx", errors == 0, f"5xx in the canary's responses: {errors} of {len(statuses)}")
    stack.restart(NIMBUS_CANARY_PERCENT="0")
    back = client.post(
      "/support/reply", json={"message": "Hi", "tone": "warm"}, headers=headers | {"X-User-ID": users["canary"]}
    )
    check(
      "Rollback drill: canary to 0",
      back.headers.get("x-variant") == "stable",
      f"the canary user now gets {back.headers.get('x-variant')}",
    )
  return rows


@lab(
  "prod.release-checklist",
  "Ship it: deploy, then run the release checklist",
  uses_model=True,
  calls="~17 calls to Mistral (12 for the eval gate)",
)
def checklist_lab():
  """The capstone: the service deployed (API ×2, RQ worker, Postgres, Redis), then the v3 support prompt released to a
  10% canary through a checklist: CI, the image, the eval gate, health, the API contract, restarts, the canary, its
  numbers, and a rollback drill."""
  rows = release_checklist()
  failed = [r[0] for r in rows if r[1] == "❌"]
  skipped = [r[0] for r in rows if r[1].startswith("⏭")]
  return [
    table(["step", "result", "evidence"], rows),
    callout(
      (
        f"Stopped at: {', '.join(failed)}. Fix it and run the checklist again."
        if failed
        else f"All {len(rows) - len(skipped)} checks passed: the v3 prompt can go to 10% of users, then 25% and "
        "100% while its numbers hold."
      )
      + (f" Skipped: {', '.join(skipped)}." if skipped else ""),
      "warning" if failed else "decision",
      "Release decision",
    ),
  ]
