"""The whole service as processes on this machine: what `docker compose --profile app up` starts, without Docker.
The API (uvicorn, one or more worker processes) and an RQ job worker, configured only by environment variables, the
same ones compose passes to the containers.

  with Stack({"NIMBUS_AGENT_CHECKPOINTER": "postgres"}, api_workers=2) as stack:
    httpx.get(f"{stack.url}/health")
    stack.restart(NIMBUS_CANARY_PERCENT="10")   # a configuration change: same code, new settings
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from nimbus.config import PROJECT_DIR

BIN = Path(sys.executable).parent
LOGS = PROJECT_DIR / ".cache" / "stack"


def free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


class Stack:
  def __init__(self, env: dict[str, str], api_workers: int = 1, job_worker: bool = True, redis_url: str = ""):
    self.env, self.api_workers, self.job_worker = dict(env), api_workers, job_worker
    self.redis_url = redis_url or env.get("NIMBUS_REDIS_URL", "redis://127.0.0.1:6390/0")
    self.api = self.worker = None
    self.port = free_port()  # kept across restarts: clients keep their base URL
    self.url = f"http://127.0.0.1:{self.port}"
    LOGS.mkdir(parents=True, exist_ok=True)

  def _spawn(self, name: str, command: list[str]) -> subprocess.Popen:
    with open(LOGS / f"{name}.log", "ab") as log:  # the child keeps its own copy of the file handle
      return subprocess.Popen(command, env=os.environ | self.env, stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_DIR)

  def start_api(self) -> None:
    command = [str(BIN / "uvicorn"), "nimbus.api.app:app", "--port", str(self.port), "--workers", str(self.api_workers)]
    self.api = self._spawn("api", [*command, "--no-access-log", "--log-level", "warning"])
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
      if self.api.poll() is not None:
        raise RuntimeError(f"the API exited ({self.api.returncode}): see {LOGS / 'api.log'}")
      try:
        if httpx.get(f"{self.url}/health", timeout=1).status_code == 200:
          return
      except httpx.HTTPError:
        time.sleep(0.2)
    raise TimeoutError("the API didn't become healthy in 60 s")

  def start_worker(self) -> None:
    command = [str(BIN / "rq"), "worker", "--url", self.redis_url, "--worker-class", "rq.worker.SimpleWorker"]
    self.worker = self._spawn("worker", [*command, "reports"])

  @staticmethod
  def _stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
      return
    process.terminate()  # SIGTERM: uvicorn finishes in-flight requests, rq finishes the current job
    try:
      process.wait(timeout=20)
    except subprocess.TimeoutExpired:
      process.kill()
      process.wait()

  def restart(self, **env: str) -> None:
    """Stop the API, change its environment, start it again (a rolling restart has one process at a time)."""
    self._stop(self.api)
    self.env |= env
    self.start_api()

  def __enter__(self) -> "Stack":
    self.start_api()
    if self.job_worker:
      self.start_worker()
    return self

  def __exit__(self, *exc) -> None:
    self._stop(self.api)
    self._stop(self.worker)
