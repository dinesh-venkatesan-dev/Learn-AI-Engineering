"""Analysis with pandas (Data Module 3): SQL fetches the data, Python does the statistics.

An AnalysisSession owns a temporary folder. load_dataframe runs SQL through the guard and keeps the result as a
named DataFrame (a pickle file); run_pandas runs the model's code in a SEPARATE process: fresh, with a timeout, without
your environment variables (no API keys), and only what it prints comes back.

Running model-written code is running untrusted code. A subprocess limits accidents, not attacks: for untrusted users
use a real sandbox (Deep Agents, Module 6).

  with AnalysisSession() as session:
    agent = create_agent(model, [describe_table, *session.tools()], system_prompt=ANALYST)
"""

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import tool

from nimbus.data.context import GLOSSARY
from nimbus.data.guard import UnsafeQuery, run_safe

TIMEOUT_SECONDS = 20
ANALYST = (
  "You are Nimbus Coffee's data analyst. Check columns with describe_table, load the rows you need with "
  "load_dataframe (aggregate in SQL where it's simple), then compute statistics with run_pandas and print them. Every "
  f"number in your answer must come from tool output.\n\n{GLOSSARY}\nAnswer in at most four sentences."
)


class AnalysisSession:
  def __init__(self) -> None:
    self.workdir = Path(tempfile.mkdtemp(prefix="nimbus-analysis-"))
    self.frames: dict[str, list[str]] = {}

  def __enter__(self) -> "AnalysisSession":
    return self

  def __exit__(self, *exc) -> None:
    self.close()

  def close(self) -> None:
    shutil.rmtree(self.workdir, ignore_errors=True)  # the DataFrames go with the session

  def load(self, name: str, sql: str) -> str:
    import pandas as pd

    if not name.isidentifier():
      return f"ERROR: {name!r} isn't a valid Python name."
    try:
      columns, rows = run_safe(sql)
    except (UnsafeQuery, sqlite3.Error) as error:
      return f"ERROR: {error}"
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_pickle(self.workdir / f"{name}.pkl")
    self.frames[name] = columns
    return f"DataFrame {name!r}: {len(frame)} rows, columns {columns}. First rows:\n{frame.head(3).to_string()}"

  def run(self, code: str) -> str:
    loader = "import pandas as pd\n" + "".join(
      f"{n} = pd.read_pickle({str(self.workdir / (n + '.pkl'))!r})\n" for n in self.frames
    )
    try:
      result = subprocess.run(  # noqa: S603 - the point: model code runs in its own process
        [sys.executable, "-I", "-c", loader + code],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        cwd=self.workdir,
        env={"PATH": os.environ.get("PATH", "")},  # no API keys in this process
        check=False,
      )
    except subprocess.TimeoutExpired:
      return f"ERROR: the code ran longer than {TIMEOUT_SECONDS} s and was stopped."
    return (result.stdout + result.stderr).strip()[-3000:] or "(nothing printed)"

  def tools(self) -> list:
    session = self

    @tool
    def load_dataframe(name: str, sql: str) -> str:
      """Run a SQLite SELECT and keep the result as a pandas DataFrame called `name` (for run_pandas)."""
      return session.load(name, sql)

    @tool
    def run_pandas(code: str) -> str:
      """Run Python/pandas code. Loaded DataFrames are available by name (e.g. `daily`), as is `pd`. Only what you
      print() comes back: print the final numbers."""
      return session.run(code)

    return [load_dataframe, run_pandas]
