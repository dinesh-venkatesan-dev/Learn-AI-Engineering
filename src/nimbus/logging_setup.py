"""One logging configuration for the CLI, the API and the tests."""

import logging


def setup_logging(level: str = "INFO") -> None:
  logging.basicConfig(
    level=level.upper(), datefmt="%H:%M:%S", format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
  )
  for noisy in ("httpx", "httpx2", "urllib3", "sqlalchemy.engine"):  # libraries that are chatty at INFO
    logging.getLogger(noisy).setLevel(logging.WARNING)
