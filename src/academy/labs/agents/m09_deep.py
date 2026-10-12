"""Agents with LangGraph · Module 9 labs: deep agents (create_deep_agent), the virtual filesystem, planning, backends
(real files, long-term files, running code), permissions, subagents, streaming and async subagents. Then: the weekly
ops review written by a plain agent and by a deep agent, checked against the warehouse."""

import csv
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from academy.labs.runtime import Metric, answer, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR
from nimbus.llm import get_chat_model

WORK = PROJECT_DIR / ".cache" / "labs" / "agents" / "deep"
LIMITS = {"recursion_limit": 80}  # ministral can loop on `ls` over an empty filesystem: always cap a run


def _thread(name: str) -> dict:
  return {"configurable": {"thread_id": name}, **LIMITS}


def _calls(messages) -> list[str]:
  out = []
  for m in messages:
    for c in getattr(m, "tool_calls", None) or []:
      target = (
        c["args"].get("subagent_type")
        or c["args"].get("file_path")
        or c["args"].get("path")
        or c["args"].get("cafe")
        or ""
      )
      out.append(f"{c['name']}({target})" if target else c["name"])
  return out


def _file(result, path):
  item = (result.get("files") or {}).get(path)
  if not item:
    return ""
  content = item.get("content") if isinstance(item, dict) else item
  return "\n".join(content) if isinstance(content, list) else str(content)


def _cafe_notes() -> dict:
  """Source documents for the agent: this week's low reviews and a supplier note, as files."""
  from deepagents.backends.utils import create_file_data

  from nimbus.ops.data import cafe_names, low_reviews

  lines = ["# Reviews rated 2 or lower, 24–30 September 2026", ""]
  for cafe in cafe_names():
    for r in low_reviews(cafe):
      lines.append(f"- #{r['review_id']} {cafe} ({r['rating']}★): {r['comment']}")
  supplier = (
    "# Supplier notes\n\n- Green Valley: oat milk deliveries to Bengaluru were two days late on 25 and 26 September.\n"
    "- Hill Estates: the courier for delivery orders changed on 24 September; packaging is thinner."
  )
  return {"/reviews.md": create_file_data("\n".join(lines)), "/supplier_notes.md": create_file_data(supplier)}


# ── Lab 1: what is a deep agent ────────────────────────────────────────────────
JOB = (
  "Compare Nimbus Bandra and Nimbus Indiranagar this week: revenue, change and rating. Write your comparison to "
  "/comparison.md."
)


def deep_vs_plain(model):
  from deepagents import create_deep_agent
  from langchain.agents import create_agent

  from nimbus.ops.agents import weekly_numbers

  deep = create_deep_agent(
    model=model, tools=[weekly_numbers], system_prompt="You are Nimbus's ops analyst. Use the tools for every number."
  )
  plain = create_agent(
    model, [weekly_numbers], system_prompt="You are Nimbus's ops analyst. Use the tools for every number."
  )
  return (
    deep,
    plain,
    deep.invoke({"messages": [HumanMessage(JOB)]}, LIMITS),
    plain.invoke({"messages": [HumanMessage(JOB)]}, LIMITS),
  )


def built_in_tools(deep):
  return sorted(deep.nodes["tools"].bound.tools_by_name)  # filesystem tools and `task`, next to yours


@lab(
  "agents.deep-basics",
  "What is a deep agent? create_agent plus a harness",
  uses_model=True,
  calls="~10 calls to Mistral",
)
def deep_basics():
  """The same two-café comparison by create_deep_agent and create_agent: the built-in tools, the files in state, and
  what each one leaves behind."""
  deep, plain, d, p = deep_vs_plain(get_chat_model())
  return [
    table(
      ["", "create_deep_agent", "create_agent"],
      [
        ["tools", ", ".join(built_in_tools(deep)), ", ".join(sorted(plain.nodes["tools"].bound.tools_by_name))],
        ["tool calls", ", ".join(_calls(d["messages"])), ", ".join(_calls(p["messages"]))],
        ["state keys", ", ".join(sorted(d)), ", ".join(sorted(p))],
        ["files", ", ".join(d.get("files", {})) or "—", "— (no filesystem)"],
      ],
    ),
    code(
      _file(d, "/comparison.md") or d["messages"][-1].text,
      "markdown",
      title="/comparison.md, read back from the deep agent's state",
    ),
    answer(p["messages"][-1].text, "create_agent's answer (no file: it has no filesystem)"),
    callout(
      "create_deep_agent is create_agent with a harness: a virtual filesystem (ls, read_file, write_file, edit_file, "
      "glob, grep), `task` for subagents, and `files` in state next to `messages`, so the work is left where you can "
      "read it afterwards. What it returns is an ordinary LangGraph graph: invoke, stream and checkpointers work as "
      "before. Always pass the model (without one, deepagents falls back to an Anthropic model).",
      "decision",
      "Deep agents",
    ),
  ] + [
    example(
      "deep.l01.1",
      "create_deep_agent and its built-in tools",
      built_in_tools,
      "",
      [table(["tools"], [[t] for t in built_in_tools(deep)])],
    ),
    example(
      "deep.l01.2", "`files` next to `messages`", deep_vs_plain, "", [table(["state keys"], [[", ".join(sorted(d))]])]
    ),
    example(
      "deep.l01.3",
      "The same job, both ways",
      deep_vs_plain,
      "",
      [
        table(
          ["agent", "tool calls"],
          [["deep", ", ".join(_calls(d["messages"]))], ["plain", ", ".join(_calls(p["messages"]))]],
        )
      ],
    ),
    example("deep.l01.4", "An ordinary LangGraph graph", deep_vs_plain, "", [table(["type"], [[type(deep).__name__]])]),
  ]


# ── Lab 2: the virtual filesystem ──────────────────────────────────────────────
def seeded_run(model):
  from deepagents import create_deep_agent
  from langgraph.checkpoint.memory import InMemorySaver

  agent = create_deep_agent(
    model=model,
    checkpointer=InMemorySaver(),
    system_prompt=(
      "You are Nimbus's ops analyst. The documents are in the filesystem: look before you answer, quote review "
      "numbers. Keep notes in /notes/. Be concise."
    ),
  )
  first = agent.invoke(
    {
      "messages": [
        HumanMessage(
          "Which complaints do the supplier notes explain? Write your findings to /notes/issues.md, citing review "
          "numbers."
        )
      ],
      "files": _cafe_notes(),
    },
    _thread("week-39"),
  )
  second = agent.invoke(
    {"messages": [HumanMessage("Add a 'Recommendation' line to your notes file.")]}, _thread("week-39")
  )  # same thread: the files are still there
  other = agent.get_state({"configurable": {"thread_id": "another-week"}}).values  # a new thread: empty filesystem
  return first, second, other, agent.get_state({"configurable": {"thread_id": "week-39"}}).values


@lab(
  "agents.deep-files",
  "The virtual filesystem: documents in, notes out, read back",
  uses_model=True,
  calls="~12 calls to Mistral",
)
def deep_files():
  """This week's low reviews and supplier notes seeded as files; the agent explores them, writes notes, and edits
  them on the next turn of the same thread."""
  first, second, other, saved = seeded_run(get_chat_model())
  notes = _file(saved, "/notes/issues.md")
  return [
    table(
      ["Turn", "tool calls"],
      [
        ["1: findings", ", ".join(_calls(first["messages"]))],
        ["2: add a recommendation", ", ".join(_calls(second["messages"][len(first["messages"]) :]))],
      ],
    ),
    code(notes or "(no /notes/issues.md)", "markdown", title="/notes/issues.md after turn 2"),
    table(
      ["Thread", "files"],
      [
        ["week-39", ", ".join(saved.get("files", {}))],
        ["another-week", ", ".join(other.get("files", {})) or "— (empty)"],
      ],
    ),
    callout(
      "Files go in with the input ({'files': {path: file_data}}) and, with the default StateBackend, live in the "
      "graph state, not on your disk. The agent looks around with ls, glob and grep, reads in pages with read_file, "
      "and keeps notes with write_file and edit_file. With a checkpointer the files persist across the turns of a "
      "thread; a new thread starts empty. Your code reads them from result['files'] or get_state().",
      "decision",
      "Virtual filesystem",
    ),
  ] + [
    example(
      "deep.l02.1", "Seeding files (StateBackend)", _cafe_notes, "", [table(["file"], [[f] for f in _cafe_notes()])]
    ),
    example(
      "deep.l02.2",
      "ls, glob, grep, read_file, write_file, edit_file",
      seeded_run,
      "",
      [table(["tool calls"], [[", ".join(_calls(first["messages"]))]])],
    ),
    example("deep.l02.3", "Files persist across turns of a thread", seeded_run, "", [code(notes, "markdown")]),
    example(
      "deep.l02.4",
      "A new thread starts empty",
      seeded_run,
      "",
      [table(["another-week files"], [[", ".join(other.get("files", {})) or "none"]])],
    ),
    example(
      "deep.l02.5",
      "Reading them from your code",
      seeded_run,
      "",
      [table(["files in week-39"], [[f] for f in saved.get("files", {})])],
    ),
  ]


# ── Lab 3: planning ────────────────────────────────────────────────────────────
def planned_run(model):
  from deepagents import create_deep_agent
  from langchain.agents.middleware import TodoListMiddleware

  from nimbus.ops.agents import list_cafes, weekly_numbers

  agent = create_deep_agent(
    model=model,
    tools=[list_cafes, weekly_numbers],
    middleware=[TodoListMiddleware()],
    system_prompt=(
      "You are Nimbus's ops analyst. Plan with write_todos first, then work through the plan, marking each step."
    ),
  )
  plans = []
  for update in agent.stream(
    {"messages": [HumanMessage("Get every café's revenue this week and name the top three.")]},
    LIMITS,
    stream_mode="updates",
  ):
    for change in update.values():
      if isinstance(change, dict) and change.get("todos"):
        plans.append([(t["content"][:50], t["status"]) for t in change["todos"]])  # the plan, each time it changes
  return plans, agent


@lab(
  "agents.deep-planning",
  "Planning: a todo list the agent writes, follows and updates",
  uses_model=True,
  calls="~15 calls to Mistral",
)
def deep_planning():
  """Every café's revenue and the top three, with TodoListMiddleware: the plan as it changes while the agent works."""
  plans, _ = planned_run(get_chat_model())
  last = plans[-1] if plans else []
  return [
    table(
      ["Update", "todos (status)"],
      [[i + 1, "; ".join(f"{c} [{s}]" for c, s in p)] for i, p in enumerate(plans)],
      title=f"{len(plans)} plan updates",
    ),
    table(["Final todo", "status"], [list(t) for t in last]),
    callout(
      "Planning is opt-in since deepagents 0.7: middleware=[TodoListMiddleware()] adds write_todos, a `todos` key in "
      "state and instructions on when to plan. Each todo moves pending → in_progress → completed; stream updates show "
      f"the plan changing ({len(plans)} updates here) and the final state holds the finished list, ready for a "
      "progress bar."
      + (
        f" Here {sum(st == 'completed' for _, st in last)} of the final {len(last)} todo(s) were completed: the agent "
        "rewrote the list as it went and stopped without ticking it off, so treat a plan as intent, not proof of work."
        if not all(st == "completed" for _, st in last)
        else " Here every todo ended completed."
      ),
      "decision",
      "Planning",
    ),
  ] + [
    example("deep.l03.1", "TodoListMiddleware", planned_run, "", [table(["updates"], [[len(plans)]])]),
    example("deep.l03.2", "{content, status}", planned_run, "", [table(["todo", "status"], [list(t) for t in last])]),
    example(
      "deep.l03.3",
      "Watching the plan change",
      planned_run,
      "",
      [table(["update", "statuses"], [[i + 1, ", ".join(s for _, s in p)] for i, p in enumerate(plans)])],
    ),
    example("deep.l03.4", "The finished list", planned_run, "", [table(["todo", "status"], [list(t) for t in last])]),
  ]


# ── Lab 4: backends ────────────────────────────────────────────────────────────
@dataclass
class Manager:
  user_id: str


def real_files(model):
  from deepagents import create_deep_agent
  from deepagents.backends import FilesystemBackend

  workspace = WORK / "workspace"
  shutil.rmtree(workspace, ignore_errors=True)
  workspace.mkdir(parents=True)
  for path, data in _cafe_notes().items():
    (workspace / path.lstrip("/")).write_text(
      "\n".join(data["content"]) if isinstance(data["content"], list) else data["content"]
    )
  backend = FilesystemBackend(root_dir=workspace)  # virtual_mode=True by default: "/" is the workspace
  refused = {}
  for path in ("/reviews.md", "/../../pyproject.toml"):
    try:
      found = backend.read(path, limit=1)
      refused[path] = "read" if found.file_data else found.error
    except ValueError as error:
      refused[path] = f"refused: {error}"[:120]
  agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt="You are Nimbus's ops analyst. Read files before you change them. Be concise.",
  )
  agent.invoke(
    {"messages": [HumanMessage("Create /actions.md with one checklist item per supplier in /supplier_notes.md.")]},
    LIMITS,
  )
  actions = workspace / "actions.md"
  return refused, sorted(p.name for p in workspace.iterdir()), actions.read_text() if actions.exists() else ""


def long_term_files(model):
  from deepagents import create_deep_agent
  from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.store.memory import InMemoryStore

  from nimbus.ops.agents import weekly_numbers

  store = InMemoryStore()
  backend = CompositeBackend(
    default=StateBackend(), routes={"/memories/": StoreBackend(namespace=lambda rt: ("managers", rt.context.user_id))}
  )
  agent = create_deep_agent(
    model=model,
    tools=[weekly_numbers],
    backend=backend,
    store=store,
    checkpointer=InMemorySaver(),
    context_schema=Manager,
    system_prompt=(
      "You assist Nimbus café managers. Files under /memories/ last between conversations: read "
      "/memories/preferences.md (if it exists) first, and save what a manager asks you to remember there. Use the "
      "tools for every number. Be concise."
    ),
  )

  def ask(user, thread, text):
    return agent.invoke({"messages": [HumanMessage(text)]}, _thread(thread), context=Manager(user))["messages"][-1].text

  replies = {
    "priya, monday": ask(
      "priya",
      "mon",
      "Remember: I manage Nimbus Bandra; my weekly update is 3 short bullets about Bandra only, revenue first.",
    ),
    "priya, friday": ask("priya", "fri", "Check what you remember about me, then give me my weekly update."),
    "arjun, friday": ask("arjun", "fri-arjun", "What do you remember about me?"),
  }
  stored = {
    "/".join(item.namespace) + item.key: str(item.value.get("content"))[:120] for item in store.search(("managers",))
  }
  return replies, stored


@lab(
  "agents.deep-backends",
  "Backends: real files on disk, and long-term files per user",
  uses_model=True,
  calls="~15 calls to Mistral",
)
def deep_backends():
  """A FilesystemBackend workspace (paths stay inside it; the agent edits real files), then a CompositeBackend that
  keeps /memories/ in a store per manager and everything else in the thread."""
  model = get_chat_model()
  refused, on_disk, actions = real_files(model)
  replies, stored = long_term_files(model)
  return [
    table(
      ["backend.read(...)", "Result"],
      list(refused.items()),
      title="FilesystemBackend: virtual_mode keeps paths inside root_dir",
    ),
    table(["On disk after the run"], [[f] for f in on_disk]),
    code(actions or "(no actions.md)", "markdown", title="actions.md, read with plain Python"),
    table(["Turn", "reply"], list(replies.items()), title="CompositeBackend: /memories/ in the store, per manager"),
    table(["Store key", "content"], list(stored.items())),
    callout(
      "FilesystemBackend(root_dir=...) makes a folder the agent's '/'; with virtual_mode (the default) '../' and "
      "host paths are refused, and the agent edits real files, so never point it at a folder you can't afford to "
      "lose. StoreBackend keeps files in a LangGraph store that every thread can read; a CompositeBackend routes "
      "/memories/ there and everything else to the thread. namespace=lambda rt: ... makes them per user, from the "
      "runtime context. The store key drops the route prefix.",
      "decision",
      "Backends",
    ),
  ] + [
    example(
      "deep.l04.1", "FilesystemBackend(root_dir=...)", real_files, "", [table(["on disk"], [[f] for f in on_disk])]
    ),
    example(
      "deep.l04.2",
      "virtual_mode keeps paths inside",
      real_files,
      "",
      [table(["path", "result"], list(refused.items()))],
    ),
    example("deep.l04.3", "The agent edits real files", real_files, "", [code(actions, "markdown")]),
    example(
      "deep.l04.4",
      "Never a folder you can't lose",
      real_files,
      "The workspace here is .cache/labs/agents/deep/workspace, recreated every run.",
      [table(["root_dir"], [[str(WORK / "workspace")]])],
    ),
    example(
      "deep.l05.1",
      "StoreBackend: files every thread can read",
      long_term_files,
      "",
      [table(["turn", "reply"], list(replies.items())[:2])],
    ),
    example(
      "deep.l05.2", "CompositeBackend routes by path", long_term_files, "", [table(["key"], [[k] for k in stored])]
    ),
    example(
      "deep.l05.3",
      "Per-user namespace from the context",
      long_term_files,
      "",
      [answer(replies["arjun, friday"], "arjun")],
    ),
    example(
      "deep.l05.4", "Reading the store directly", long_term_files, "", [table(["key", "content"], list(stored.items()))]
    ),
  ]


# ── Lab 5: running code ────────────────────────────────────────────────────────
def export_sales(workspace):
  from nimbus.ops.data import cafe_names, cafe_week

  with (workspace / "sales.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["cafe", "revenue", "completed_orders"])
    for cafe in cafe_names():
      week = cafe_week(cafe)
      writer.writerow([cafe, week["revenue_rupees"], week["completed_orders"]])
  return {cafe: cafe_week(cafe)["revenue_rupees"] for cafe in cafe_names()}


def shell_agent(model):
  from deepagents import create_deep_agent
  from deepagents.backends import LocalShellBackend

  workspace = WORK / "shell"
  shutil.rmtree(workspace, ignore_errors=True)
  workspace.mkdir(parents=True)
  truth = export_sales(workspace)
  backend = LocalShellBackend(root_dir=workspace, timeout=30)  # inherit_env=False: the shell sees no API keys
  agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt=(
      "You are a data analyst. For any calculation, write a Python script with write_file and run it with execute. "
      "The shell starts inside your workspace: use RELATIVE paths in commands and scripts (python3 report.py, "
      "open('sales.csv')), never paths starting with '/'. Standard library only (csv, not pandas)."
    ),
  )
  result = agent.invoke(
    {
      "messages": [
        HumanMessage(
          "Write report.py: read sales.csv and print the average revenue per completed order for each café, sorted "
          "from highest. Run it and reply with what it printed."
        )
      ]
    },
    LIMITS,
  )
  outputs = [m.text for m in result["messages"] if m.type == "tool" and m.name == "execute"]
  return truth, outputs, result["messages"][-1].text, workspace


@lab(
  "agents.deep-execute",
  "Running code: the execute tool, and why it belongs in a sandbox",
  uses_model=True,
  calls="~10 calls to Mistral",
)
def deep_execute():
  """The week's sales exported to a CSV in a workspace; the agent writes a script, runs it with execute, and the
  result is checked against the warehouse."""
  from nimbus.ops.data import cafe_week

  truth, outputs, reply, workspace = shell_agent(get_chat_model())
  expected = {cafe: round(rev / cafe_week(cafe)["completed_orders"], 2) for cafe, rev in truth.items()}
  printed = "\n".join(outputs)
  failed = sum("exit code" in o.lower() and "exit code 0" not in o.lower() for o in outputs)
  found = {
    cafe: (f"{v:.2f}" in printed or f"{v:.1f}" in printed or str(round(v)) in printed) for cafe, v in expected.items()
  }
  return [
    table(
      ["execute output"], [[o[:400]] for o in outputs] or [["(none)"]], title=f"{len(outputs)} runs, {failed} failed"
    ),
    table(
      ["Café", "average per order (plain Python)", "in the script's output"],
      [[c, f"{v:.2f}", "✓" if found[c] else "✗"] for c, v in expected.items()],
    ),
    answer(reply, "Reply"),
    callout(
      f"LocalShellBackend gives the agent files on disk and an `execute` tool that runs shell commands in root_dir, "
      f"on this machine, with your permissions: not isolated. File tools use virtual paths ('/report.py') but the "
      f"shell starts in root_dir, so commands need relative paths: the prompt says so, and still {failed} of "
      f"{len(outputs)} runs here failed on paths like /workspace/report.py before one worked. The commands don't "
      "see your environment variables (inherit_env=False). Checked against plain Python: "
      f"{sum(found.values())}/{len(found)} "
      "cafés right. For anything you don't fully trust, a sandbox backend (LangSmith, Daytona, E2B, Modal, Runloop, "
      "Vercel) runs the same tools in a throwaway container with the same agent code.",
      "decision",
      "Running code",
    ),
  ] + [
    example(
      "deep.l06.1", "LocalShellBackend: not isolated", shell_agent, "", [table(["workspace"], [[str(workspace)]])]
    ),
    example(
      "deep.l06.2",
      "Write, run, read, fix: relative paths",
      shell_agent,
      "",
      [table(["execute output"], [[o] for o in outputs])],
    ),
    example(
      "deep.l06.3",
      "No environment variables",
      shell_agent,
      "inherit_env=False is the default.",
      [table(["inherit_env"], [["False"]])],
    ),
    example(
      "deep.l06.4",
      "Check what you can check",
      export_sales,
      "",
      [table(["café", "expected"], [[c, f"{v:.2f}"] for c, v in expected.items()])],
    ),
    example(
      "deep.l06.5",
      "Real isolation: sandbox backends",
      shell_agent,
      "create_deep_agent(backend=<sandbox>): the agent code stays the same.",
      [table(["sandboxes"], [["LangSmith Sandboxes, Daytona, E2B, Modal, Runloop, Vercel"]])],
    ),
  ]


# ── Lab 6: permissions ─────────────────────────────────────────────────────────
def permissions_run(model):
  from deepagents import create_deep_agent
  from deepagents.backends.utils import create_file_data
  from deepagents.middleware.permissions import FilesystemPermission
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.types import Command

  rules = [
    FilesystemPermission(operations=["read", "write"], paths=["/secrets/**"], mode="deny"),  # unreadable
    FilesystemPermission(operations=["write"], paths=["/data/**"], mode="deny"),  # source data stays read-only
    FilesystemPermission(operations=["write"], paths=["/reports/**"], mode="interrupt"),  # a person approves
  ]
  agent = create_deep_agent(
    model=model,
    permissions=rules,
    checkpointer=InMemorySaver(),
    system_prompt="You are Nimbus's ops analyst. Be concise.",
  )
  files = {
    "/data/reviews.md": _cafe_notes()["/reviews.md"],
    "/secrets/payroll.md": create_file_data("Bandra payroll: ₹4,10,000"),
  }
  out = {}
  r = agent.invoke(
    {"messages": [HumanMessage("Read /secrets/payroll.md and tell me the Bandra payroll.")], "files": files},
    _thread("secret"),
  )
  out["read /secrets/payroll.md"] = [m.text[:160] for m in r["messages"] if m.type == "tool"][:2]
  r = agent.invoke(
    {"messages": [HumanMessage("Fix the typo 'Wrong item' → 'Incorrect item' in /data/reviews.md.")], "files": files},
    _thread("data"),
  )
  out["edit /data/reviews.md"] = [m.text[:160] for m in r["messages"] if m.type == "tool"][:2]
  r = agent.invoke(
    {"messages": [HumanMessage("Write a two-line summary of /data/reviews.md to /reports/week39.md.")], "files": files},
    _thread("report"),
  )
  paused = r.get("__interrupt__")
  out["write /reports/week39.md"] = [str(paused[0].value)[:200]] if paused else ["(no pause)"]
  if paused:
    r = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), _thread("report"))
    out["after approval"] = [", ".join(r.get("files", {}))]
  return out


@lab(
  "agents.deep-permissions",
  "Permissions: which paths the agent may read, write, or only with approval",
  uses_model=True,
  calls="~10 calls to Mistral",
)
def deep_permissions():
  """Three rules (secrets unreadable, source data read-only, reports need approval) and three requests that hit
  them."""
  out = permissions_run(get_chat_model())
  return [
    table(["Request", "what the agent got"], [[k, "\n".join(v)] for k, v in out.items()]),
    callout(
      "permissions=[FilesystemPermission(operations, paths, mode)] guards the file tools: 'read' and/or 'write' "
      "(write covers write_file, edit_file and delete) on glob paths. mode='deny' fails the call with a message the "
      "model reads; mode='interrupt' pauses before the call (a checkpointer is needed) for a person to approve. Rules "
      "are checked in order, the first match wins, and anything unmatched is allowed.",
      "decision",
      "Permissions",
    ),
  ] + [
    example(
      "deep.l07.1",
      "FilesystemPermission(operations, paths, mode)",
      permissions_run,
      "",
      [table(["request"], [[k] for k in out])],
    ),
    example(
      "deep.l07.2",
      "mode='deny'",
      permissions_run,
      "",
      [table(["request", "result"], [[k, "\n".join(v)] for k, v in list(out.items())[:2]])],
    ),
    example(
      "deep.l07.3",
      "mode='interrupt'",
      permissions_run,
      "",
      [table(["request", "result"], [[k, "\n".join(v)] for k, v in list(out.items())[2:]])],
    ),
    example(
      "deep.l07.4",
      "Order: the first match wins",
      permissions_run,
      "The secrets rule comes first, so even reads are denied there.",
      [table(["rules"], [["/secrets/** deny · /data/** write deny · /reports/** write interrupt"]])],
    ),
  ]


# ── Lab 7: subagents ───────────────────────────────────────────────────────────
def subagent_run(model):
  from deepagents import create_deep_agent

  from nimbus.ops.agents import SALES_ANALYST, VOICE_OF_CUSTOMER

  agent = create_deep_agent(
    model=model,
    subagents=[SALES_ANALYST, VOICE_OF_CUSTOMER],
    system_prompt=(
      "You are Nimbus's chief of staff. Delegate numbers to sales-analyst and complaints to voice-of-customer (in "
      "parallel when you can), then write a 3-bullet brief."
    ),
  )
  task = agent.nodes["tools"].bound.tools_by_name["task"]
  result = agent.invoke(
    {"messages": [HumanMessage("Brief me on Nimbus Indiranagar this week: numbers and complaints.")]}, LIMITS
  )
  return task.description, result


def structured_subagent(model):
  from deepagents import create_deep_agent
  from langchain.agents import create_agent
  from langchain.agents.structured_output import ToolStrategy
  from pydantic import BaseModel, Field

  from nimbus.ops.agents import weekly_numbers

  class CafeNumbers(BaseModel):
    cafe: str
    revenue_rupees: int = Field(description="Exactly as weekly_numbers reports it")
    change_percent: float
    best_seller: str

  reporter = create_agent(
    model,
    [weekly_numbers],
    response_format=ToolStrategy(CafeNumbers),
    system_prompt="Report ONE café with weekly_numbers. Copy numbers exactly.",
  )
  agent = create_deep_agent(
    model=model,
    subagents=[
      {
        "name": "cafe-reporter",
        "description": "Reports one café's numbers as structured data. Call it once per café.",
        "runnable": reporter,  # an existing agent, plugged in as is
      }
    ],
    system_prompt=(
      "Get Nimbus Bandra's and Nimbus Colaba's numbers with cafe-reporter, both in the same turn so they run in "
      "parallel, then name the café with the higher revenue in one sentence."
    ),
  )
  result = agent.invoke({"messages": [HumanMessage("Which café did better this week?")]}, LIMITS)
  turns = [
    [c["args"].get("subagent_type") for c in m.tool_calls if c["name"] == "task"]
    for m in result["messages"]
    if getattr(m, "tool_calls", None)
  ]
  structured = [
    m.text for m in result["messages"] if m.type == "tool" and m.name == "task" and m.text.strip().startswith("{")
  ]
  # A simple job on a smaller model. (Kept in its own agent: in deepagents 0.7.23, adding a second subagent next to
  # the parallel structured reporters made a reporter's model call hang until ReadTimeout, every time we tried.)
  copywriter = {
    "name": "copywriter",
    "model": get_chat_model(model="ministral-3b-latest"),
    "description": "Turns facts into one friendly sentence for the staff newsletter.",
    "system_prompt": "Write exactly one friendly sentence from the facts. Keep numbers as given.",
    "tools": [],
  }
  writer = create_deep_agent(model=model, subagents=[copywriter], system_prompt="Ask copywriter for the sentence.")
  facts = "; ".join(structured)
  line = writer.invoke({"messages": [HumanMessage(f"A newsletter line about the better café: {facts}")]}, LIMITS)
  result = {"messages": [*result["messages"], *line["messages"][-1:]]}
  return turns, structured, result["messages"][-1].text


@lab(
  "agents.deep-subagents",
  "Subagents: delegating with a clean context",
  uses_model=True,
  calls="~25 calls (Mistral, ministral-3b for the copywriter)",
)
def deep_subagents():
  """A chief of staff delegating to the ops agent's two analysts, then an existing create_agent with structured
  output plugged in as a subagent, called in parallel, plus a smaller model for a simple job."""
  model = get_chat_model()
  description, result = subagent_run(model)
  turns, structured, line = structured_subagent(model)
  main_messages = len(result["messages"])
  tasks = [m for m in result["messages"] if m.type == "tool" and m.name == "task"]
  return [
    table(
      ["Delegated", "came back"],
      [
        [c, t.text[:200]]
        for c, t in zip([x for x in _calls(result["messages"]) if x.startswith("task")], tasks, strict=False)
      ],
    ),
    answer(result["messages"][-1].text, "The brief"),
    table(
      ["Main agent's turn", "task calls"],
      [[i + 1, ", ".join(map(str, t))] for i, t in enumerate(turns) if t],
      title="An existing agent as a subagent, in parallel",
    ),
    table(["Structured result (JSON)"], [[s[:200]] for s in structured] or [["(none)"]]),
    answer(line, "The copywriter's sentence (ministral-3b)"),
    callout(
      "Each subagent is an option of the `task` tool, chosen by its description, so write the description like a "
      "job ad. A subagent starts fresh with only the task description (it shares the files, not the conversation) "
      f"and only its final message comes back: the main agent held {main_messages} messages while the analysts did "
      "their own tool calls out of its sight. Any existing agent can be a subagent (CompiledSubAgent), a "
      "response_format makes its result structured JSON, several task calls in one turn run in parallel, and a "
      "simple job can use a smaller model.",
      "decision",
      "Subagents",
    ),
  ] + [
    example(
      "deep.l08.1",
      "subagents=[{name, description, system_prompt, tools}]",
      subagent_run,
      "",
      [code(description[:1500], "text", "the task tool's description")],
    ),
    example(
      "deep.l08.2",
      "A fresh context with only the task",
      subagent_run,
      "",
      [table(["task calls"], [[c] for c in _calls(result["messages"]) if c.startswith("task")])],
    ),
    example(
      "deep.l08.3",
      "Only the final message comes back",
      subagent_run,
      "",
      [table(["result"], [[t.text[:200]] for t in tasks])],
    ),
    example(
      "deep.l08.4",
      "general-purpose is always there",
      subagent_run,
      "",
      [table(["in the task tool"], [["general-purpose" in description]])],
    ),
    example(
      "deep.l08.5",
      "Isolation, measured",
      subagent_run,
      "",
      [table(["main agent's messages", "subagent results"], [[main_messages, len(tasks)]])],
    ),
    example(
      "deep.l09.1",
      "CompiledSubAgent: any agent",
      structured_subagent,
      "",
      [table(["turn", "task calls"], [[i + 1, ", ".join(map(str, t))] for i, t in enumerate(turns) if t])],
    ),
    example(
      "deep.l09.2",
      "Structured results",
      structured_subagent,
      "",
      [table(["JSON"], [[s[:200]] for s in structured] or [["(none)"]])],
    ),
    example(
      "deep.l09.3",
      "Parallel delegation",
      structured_subagent,
      "",
      [table(["calls per turn"], [[len(t)] for t in turns if t])],
    ),
    example("deep.l09.4", "A cheaper model per subagent", structured_subagent, "", [answer(line, "ministral-3b")]),
  ]


# ── Lab 8: streaming a deep agent ──────────────────────────────────────────────
def streamed(model):
  from deepagents import create_deep_agent
  from langchain_core.tools import tool
  from langgraph.config import get_stream_writer

  from nimbus.ops.agents import weekly_numbers

  @tool
  def stock_check(cafe: str) -> str:
    """Check one café's fridges (slow: one at a time)."""
    write = get_stream_writer()
    for n in range(1, 4):
      time.sleep(0.1)
      write({"progress": f"{cafe}: fridge {n}/3 checked"})
    return f"{cafe}: all stocked except oat milk (2 cartons)."

  analyst = {
    "name": "store-analyst",
    "tools": [weekly_numbers, stock_check],
    "description": "Checks one café's revenue and stock. Call once per café.",
    "system_prompt": "Use both tools. Reply in two sentences: revenue exactly as given, and the stock.",
  }
  return create_deep_agent(
    model=model,
    subagents=[analyst],
    system_prompt="Use store-analyst once per café, in parallel, then summarise in two bullets.",
  )


def who(namespace: tuple) -> str:
  return "main" if not namespace else f"subagent {namespace[0].split(':')[1][:6]}"


@lab(
  "agents.deep-streaming",
  "Streaming a deep agent: the main agent and its subagents, live",
  uses_model=True,
  calls="~25 calls to Mistral",
)
def deep_streaming():
  """Two cafés checked by subagents in parallel, streamed three ways: updates with namespaces and custom progress,
  the main agent's tokens only, and stream_events v3 with one handle per subagent."""
  model = get_chat_model()
  question = {"messages": [HumanMessage("How are Nimbus Bandra and Nimbus Colaba doing: revenue and stock?")]}
  events = []
  for namespace, mode, data in streamed(model).stream(
    question, LIMITS, stream_mode=["updates", "custom"], subgraphs=True
  ):
    if mode == "custom":
      events.append((who(namespace), "progress", data["progress"]))
      continue
    for change in data.values():
      for message in (change or {}).get("messages", []) if isinstance(change, dict) else []:
        for call in getattr(message, "tool_calls", None) or []:
          events.append(
            (
              who(namespace),
              "tool call",
              f"{call['name']}({call['args'].get('subagent_type') or call['args'].get('cafe', '')})",
            )
          )
  tokens = [
    token.text
    for namespace, (token, meta) in streamed(model).stream(question, LIMITS, stream_mode="messages", subgraphs=True)
    if not namespace and meta.get("langgraph_node") == "model" and token.text
  ]
  stream = streamed(model).stream_events(question, LIMITS, version="v3")
  handles = [(s.name, s.status, (s.output["messages"][-1].text if s.output else "")[:150]) for s in stream.subagents]
  return [
    table(["Who", "event", "detail"], [list(e) for e in events], title="updates + custom, subgraphs=True"),
    answer("".join(tokens), "Main agent's tokens only (namespace ())"),
    table(
      ["Subagent", "status", "final output"],
      [list(h) for h in handles],
      title="stream_events(version='v3') · stream.subagents",
    ),
    callout(
      "With subgraphs=True every event carries a namespace: () for the main agent, ('tools:<id>',) for a subagent "
      "inside a task call. Filter stream_mode='messages' on the namespace to show only the main agent's words; tools "
      "can send progress with get_stream_writer(); stream_events v3 gives one handle per subagent call with its name, "
      "status and output (experimental: each projection can be read once).",
      "decision",
      "Streaming",
    ),
  ] + [
    example(
      "deep.l10.1",
      "subgraphs=True: namespaces",
      who,
      "",
      [table(["who", "event"], [[e[0], e[2]] for e in events if e[1] == "tool call"])],
    ),
    example("deep.l10.2", "The main agent's tokens", streamed, "", [answer("".join(tokens)[:500], "main agent")]),
    example(
      "deep.l10.3",
      "Custom progress from a tool",
      streamed,
      "",
      [table(["who", "progress"], [[e[0], e[2]] for e in events if e[1] == "progress"])],
    ),
    example(
      "deep.l10.4",
      "stream_events v3: subagent handles",
      streamed,
      "",
      [table(["name", "status"], [[h[0], h[1]] for h in handles])],
    ),
  ]


# ── Lab 9: async subagents ─────────────────────────────────────────────────────
@contextmanager
def research_server():
  from academy.labs.agents.m08_production import PORT, URL, start_server

  process = start_server()
  try:
    yield URL
  finally:
    process.terminate()
    process.wait()
  del PORT


def async_subagent_run(model, url):
  from deepagents import create_deep_agent
  from langgraph.checkpoint.memory import InMemorySaver

  from nimbus.support.tools import SupportStore, build_tools

  menu = next(t for t in build_tools(SupportStore()) if t.name == "cafe_menu_item")
  agent = create_deep_agent(
    model=model,
    tools=[menu],
    checkpointer=InMemorySaver(),
    subagents=[
      {
        "name": "researcher",
        "graph_id": "researcher",
        "url": url,
        "description": (
          "Slow, thorough research: every café's revenue for the whole quarter, top and bottom café. Takes a while."
        ),
      }
    ],
    system_prompt=(
      "You are Nimbus's assistant. Start long research with start_async_task and do NOT wait for it: say it's "
      "running and answer other questions. When asked about it, use check_async_task. Be concise."
    ),
  )
  tools = sorted(t for t in agent.nodes["tools"].bound.tools_by_name if "async" in t)
  turns = []
  result = agent.invoke(
    {"messages": [HumanMessage("Kick off quarter research on every café. Meanwhile: how much is a cold brew?")]},
    _thread("ops"),
  )
  turns.append(("start + a question", _calls(result["messages"]), result["messages"][-1].text))
  for _ in range(8):
    time.sleep(10)  # the user comes back later
    seen = len(result["messages"])
    result = agent.invoke(
      {"messages": [HumanMessage("Is the research done? If so, give me the findings.")]}, _thread("ops")
    )
    turns.append(("check", _calls(result["messages"][seen:]), result["messages"][-1].text))
    statuses = [t.get("status") for t in (result.get("async_tasks") or {}).values()]
    if statuses and all(s not in ("pending", "running") for s in statuses):
      break
  return tools, turns, {k[:8]: v.get("status") for k, v in (result.get("async_tasks") or {}).items()}


@lab(
  "agents.deep-async",
  "Async subagents: long jobs in the background",
  uses_model=True,
  calls="~20 calls to Mistral; starts and stops `langgraph dev`",
)
def deep_async():
  """A quarter's research on a LangGraph server, started in the background while the agent answers a menu question,
  then checked until it's done."""
  with research_server() as url:
    tools, turns, tasks = async_subagent_run(get_chat_model(), url)
  return [
    table(["Async tools"], [[t] for t in tools]),
    table(["Turn", "tool calls", "reply"], [[t, ", ".join(c), r[:300]] for t, c, r in turns]),
    table(["Task", "status"], list(tasks.items())),
    callout(
      "An async subagent is a graph served by an Agent Protocol server (here `langgraph dev`, the same API as a "
      "LangSmith deployment). AsyncSubAgent {name, description, graph_id, url} adds start, check, update, cancel and "
      "list tools; task ids live in the agent's state (`async_tasks`), so with a checkpointer a later turn can check "
      "on them. The flow: start, keep talking, check, read the result.",
      "decision",
      "Async subagents",
    ),
  ] + [
    example(
      "deep.l11.1",
      "A subagent on a server",
      research_server,
      "server/ops_research.py, hosted from server/langgraph.json.",
      [table(["graph"], [["researcher"]])],
    ),
    example("deep.l11.2", "Five async tools", async_subagent_run, "", [table(["tools"], [[t] for t in tools])]),
    example(
      "deep.l11.3", "Task ids in state", async_subagent_run, "", [table(["task", "status"], list(tasks.items()))]
    ),
    example(
      "deep.l11.4",
      "Start, keep talking, check",
      async_subagent_run,
      "",
      [table(["turn", "calls"], [[t, ", ".join(c)] for t, c, _ in turns])],
    ),
  ]


# ── Production path: the weekly ops review ─────────────────────────────────────
def evaluate_review(repeats: int = 2) -> list[dict]:
  from nimbus.ops.agents import REQUEST, deep_agent, plain_agent
  from nimbus.ops.data import WEEK
  from nimbus.ops.evaluation import run, score

  model, rows = get_chat_model(), []
  for _ in range(repeats):
    for name, agent in (("create_agent", plain_agent(model)), ("deep agent", deep_agent(model))):
      out = run(agent, REQUEST.format(start=WEEK[0], end=WEEK[1]))
      rows.append({"variant": name, **out, **score(out["report"])})
  return rows


@lab(
  "agents.ops-review-eval",
  "Evaluate: the weekly ops review, plain agent vs deep agent",
  uses_model=True,
  calls="~90 calls to Mistral: two runs of each, about 6 minutes",
)
def ops_review_eval():
  """This week's review for all six cafés, written twice by a create_agent and twice by the deep agent (planning, a
  file, two analysts), checked fact by fact against the warehouse."""
  started = time.perf_counter()
  rows = evaluate_review()
  seconds = time.perf_counter() - started
  out = []
  for name in ("create_agent", "deep agent"):
    mine = [r for r in rows if r["variant"] == name]
    out.append(
      {
        "variant": name,
        "facts": f"{sum(r['right'] for r in mine)}/{sum(r['total'] for r in mine)}",
        "accuracy": sum(r["right"] for r in mine) / max(sum(r["total"] for r in mine), 1),
        "calls": round(sum(r["calls"] for r in mine) / len(mine), 1),
        "tokens": round(sum(r["tokens"] for r in mine) / len(mine)),
        "largest_prompt": round(sum(r["largest_prompt"] for r in mine) / len(mine)),
        "seconds": round(sum(r["seconds"] for r in mine) / len(mine), 1),
      }
    )
  plain, deep = out
  missed = [
    [r["variant"], cafe, ", ".join(k for k, v in facts.items() if not v)]
    for r in rows
    for cafe, facts in r["checks"].items()
    if not all(facts.values())
  ]
  return [
    scorecard(
      out,
      [
        Metric("facts", "facts right", "none", "text"),
        Metric("accuracy", "accuracy"),
        Metric("calls", "model calls", "lower", "text"),
        Metric("tokens", "tokens per review", "lower", "int"),
        Metric("largest_prompt", "largest prompt (tokens)", "lower", "int"),
        Metric("seconds", "seconds", "lower", "text"),
      ],
      title=f"Two reviews each · {seconds:.0f} s",
    ),
    table(["Agent", "café", "facts missed"], missed or [["—", "—", "none"]]),
    code(rows[-1]["report"][:3000], "markdown", title="The deep agent's /review.md (last run)"),
    callout(
      f"Both wrote correct reviews ({plain['facts']} vs {deep['facts']} facts), but the deep agent took "
      f"{deep['tokens'] / max(plain['tokens'], 1):.0f}× the tokens and "
      f"{deep['seconds'] / max(plain['seconds'], 1):.1f}× the "
      "time: planning, delegating twelve subagent tasks and writing a file are overhead for a job one agent can do "
      "in a few parallel tool calls. The weekly numbers ship on the plain agent; the deep agent's harness earns its "
      "place in Module 10, where the review needs memory, skills, approvals and files a manager can read."
      if plain["accuracy"] >= deep["accuracy"]
      else (
        f"The deep agent was more accurate ({deep['facts']} vs {plain['facts']}), at "
        f"{deep['tokens'] / max(plain['tokens'], 1):.0f}× the tokens."
      ),
      "decision",
      "The review on a plain agent" if plain["accuracy"] >= deep["accuracy"] else "The review on the deep agent",
    ),
  ]
