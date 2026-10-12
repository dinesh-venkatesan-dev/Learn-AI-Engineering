"""MCP · Module 1 labs: why MCP, a server and a client, transports and protocol eras, the JSON-RPC underneath, and
the Nimbus support tools served over MCP, measured against the same tools in-process."""

import asyncio
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from langchain_core.messages import HumanMessage

import nimbus.mcp.server as nimbus_server
from academy.labs.runtime import Input, Metric, answer, callout, code, columns, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR
from nimbus.llm import get_chat_model

SERVER_FILE = Path(nimbus_server.__file__)
CAFE_PRICES = {"filter coffee": 120, "cappuccino": 180, "oat latte": 220, "cold brew": 200}


def _ms(started: float) -> float:
  return (time.perf_counter() - started) * 1000


def _quiet_stdio(path: Path) -> PythonStdioTransport:
  """A stdio transport for a server file, without FastMCP's start-up banner in our logs."""
  return PythonStdioTransport(
    path, env={**os.environ, "FASTMCP_SHOW_SERVER_BANNER": "false", "FASTMCP_LOG_LEVEL": "WARNING"}
  )


# ── Lab 1: your first server and client ─────────────────────────────────────────
# Each learning point is a small function: the page shows its body, and the lab runs it.


def write_a_server():
  from fastmcp import FastMCP
  from fastmcp.exceptions import ToolError

  mcp = FastMCP("Nimbus Café", instructions="Prices of Nimbus café drinks, in rupees.")

  @mcp.tool  # the function's name, type hints and docstring become the tool's public description
  def price(item: str, cups: int = 1) -> int:
    """Total price of some cups of a café drink, in rupees.

    Args:
      item: The drink, e.g. "Oat Latte".
      cups: How many cups.
    """
    if item.casefold() not in CAFE_PRICES:
      raise ToolError(f"{item!r} isn't on the café menu")  # a message for the client (and its model)
    return CAFE_PRICES[item.casefold()] * cups

  return mcp


async def what_a_client_sees(mcp):
  from fastmcp import Client

  async with Client(mcp) as client:  # in memory: the server object lives in this process
    server_name = client.server_info.name
    instructions = client.instructions  # hosts often add these to the system prompt
    tools = await client.list_tools()
  return server_name, instructions, tools


async def call_a_tool(mcp):
  from fastmcp import Client

  async with Client(mcp) as client:
    result = await client.call_tool("price", {"item": "Oat Latte", "cups": 2})
  content = result.content  # a list of content blocks: what a MODEL reads
  structured = result.structured_content  # JSON: what a program reads
  data = result.data  # the JSON turned back into a Python value
  return content, structured, data


async def when_a_tool_fails(mcp):
  from fastmcp import Client
  from fastmcp.exceptions import ToolError

  async with Client(mcp) as client:
    try:
      await client.call_tool("price", {"item": "Unicorn Frappe"})  # by default an error result raises
      raised = None
    except ToolError as error:
      raised = error
    inspected = await client.call_tool("price", {"item": "Unicorn Frappe"}, raise_on_error=False)
    bad_type = await client.call_tool("price", {"item": "Cappuccino", "cups": "three"}, raise_on_error=False)
  return raised, inspected, bad_type  # bad_type: validated against the schema before your function runs


def from_the_terminal(server_file):
  import subprocess

  # The CLI starts the server file, connects to it as a client, and prints what it got
  listed = subprocess.run(["fastmcp", "list", server_file], capture_output=True, text=True, timeout=60)
  called = subprocess.run(
    ["fastmcp", "call", server_file, "cafe_menu_item", "name=Cold Brew"], capture_output=True, text=True, timeout=60
  )
  return listed.stdout.strip(), called.stdout.strip()


TOOLS = {
  "get_order": ("order_id", "NC-12093"),
  "cafe_menu_item": ("name", "Oat Latte"),
}


async def _time_transport(target, tool: str, args: dict, calls: int) -> dict:
  started = time.perf_counter()
  async with Client(target) as client:
    connect = _ms(started)
    timings, result = [], None
    for _ in range(calls):
      t = time.perf_counter()
      result = await client.call_tool(tool, args, raise_on_error=False)
      timings.append(_ms(t))
  return {"connect": connect, "timings": timings, "result": result}


@lab(
  "mcp.first-server",
  "Your first MCP server and client: the same tool, called three ways",
  inputs=[
    Input("tool", "Tool", "select", default="get_order", options=list(TOOLS)),
    Input("argument", "Argument", "text", default="NC-12093", help="An order id (NC-12093, NC-20417) or a menu item."),
    Input("calls", "Calls per transport", "number", default=5, min=1, max=20),
  ],
)
def first_server(tool: str, argument: str, calls: int):
  """The Nimbus support server called as a plain function, over MCP in memory, and over MCP stdio."""
  key = TOOLS[tool][0]
  args = {key: argument}

  # 1. a plain Python call: no protocol at all
  fn = getattr(nimbus_server, tool)
  direct_timings, direct_value, direct_error = [], None, None
  for _ in range(calls):
    t = time.perf_counter()
    try:
      direct_value = fn(argument).model_dump()
    except Exception as error:  # noqa: BLE001
      direct_error = f"{type(error).__name__}: {error}"
    direct_timings.append(_ms(t))

  # 2. MCP in memory, 3. MCP over stdio (a subprocess)
  memory = asyncio.run(_time_transport(nimbus_server.mcp, tool, args, calls))
  stdio = asyncio.run(_time_transport(_quiet_stdio(SERVER_FILE), tool, args, calls))

  def row(label: str, connect: float, timings: list[float], ok: bool, note: str) -> dict:
    return {
      "variant": label,
      "connect_ms": round(connect, 1),
      "call_p50_ms": round(statistics.median(timings), 3),
      "same_answer": "✓" if ok else "✗",
      "note": note,
    }

  reference = memory["result"].structured_content
  rows = [
    row("plain Python call", 0, direct_timings, direct_value == reference or bool(direct_error), "no protocol"),
    row("MCP · in memory", memory["connect"], memory["timings"], True, "same process: tests"),
    row(
      "MCP · stdio",
      stdio["connect"],
      stdio["timings"],
      stdio["result"].structured_content == reference,
      "a subprocess: local hosts",
    ),
  ]

  async def listing():
    async with Client(nimbus_server.mcp) as client:
      return await client.list_tools()

  tools = asyncio.run(listing())
  result = memory["result"]
  sample = write_a_server()
  server_name, instructions, sample_tools = asyncio.run(what_a_client_sees(sample))
  content, structured, data = asyncio.run(call_a_tool(sample))
  raised, inspected, bad_type = asyncio.run(when_a_tool_fails(sample))
  listed, called = from_the_terminal(str(SERVER_FILE))  # `uv run` puts the fastmcp CLI on PATH
  price_tool = sample_tools[0]
  return [
    table(
      ["Tool", "Description", "Arguments (JSON schema)"],
      [[t.name, t.description, json.dumps(t.input_schema["properties"])] for t in tools],
      title="What any MCP client sees: the Nimbus support server's tools",
    ),
    columns(
      [
        code(json.dumps([c.model_dump(exclude_none=True) for c in result.content], indent=2), "json", "content"),
        code(json.dumps(result.structured_content, indent=2), "json", "structured_content"),
        code(repr(result.data), "text", "data"),
      ],
      title=f"{tool}({key}={argument!r}) → is_error={result.is_error}",
    ),
    scorecard(
      rows,
      [
        Metric("connect_ms", "connect", "lower", "ms", "Opening the connection (stdio starts a Python process)"),
        Metric("call_p50_ms", "per call (p50)", "lower", "ms"),
        Metric("same_answer", "same answer", "none", "text"),
        Metric("note", "use it for", "none", "text"),
      ],
      title=f"The same tool, {calls} calls per transport",
    ),
    callout(
      f"Connecting over stdio took {stdio['connect']:,.0f} ms once (a new Python process imports the server); "
      f"after that a call took {rows[1]['call_p50_ms']:g} ms in memory and {rows[2]['call_p50_ms']:g} ms over stdio, "
      f"against {rows[0]['call_p50_ms']:g} ms as a plain function. MCP isn't for speed: it lets any host (an agent, an "
      "IDE, a desktop app) use your tools without glue code. Use the in-memory transport in tests, stdio when a local "
      "host starts your server, and HTTP (later in this track) when many clients share one server. Keep the "
      "connection open: connecting per call pays the start-up every time.",
      "decision",
      "Which transport",
    ),
  ] + [
    example(
      "mcp.l02.1",
      "FastMCP + @mcp.tool: a function becomes an MCP tool",
      write_a_server,
      "`FastMCP(name, instructions)` is the server. `@mcp.tool` registers a function: its name, type hints and "
      "docstring become the tool's public description. Nothing else to write.",
      [table(["Expression", "Value"], [["type(mcp).__name__", type(sample).__name__], ["mcp.name", sample.name]])],
    ),
    example(
      "mcp.l02.2",
      "What a client sees",
      what_a_client_sees,
      "A client connects and asks for the tool list. Each tool arrives as a name, a description and a **JSON "
      "schema** of its arguments, built from your type hints and docstring (`Args:` became the descriptions).",
      [
        table(
          ["Field", "Value"],
          [
            ["server name", server_name],
            ["instructions", instructions],
            ["tool.name", price_tool.name],
            ["tool.description", price_tool.description],
          ],
        ),
        code(json.dumps(price_tool.input_schema, indent=2), "json", "tool.input_schema"),
      ],
    ),
    example(
      "mcp.l02.3",
      "call_tool: three views of one result",
      call_a_tool,
      "`content` is text blocks for a model, `structured_content` is JSON for a program (it matches the tool's "
      "output schema), and `data` is that JSON as a Python value.",
      [
        table(
          ["Variable", "Value"],
          [
            ["content", repr(content)],
            ["structured", json.dumps(structured)],
            ["data", f"{data!r} ({type(data).__name__})"],
          ],
        )
      ],
    ),
    example(
      "mcp.l02.4",
      "Errors: ToolError and invalid arguments",
      when_a_tool_fails,
      "`raise ToolError(message)` sends the client an **error result** with your message. The client raises it by "
      "default; with `raise_on_error=False` you inspect `is_error` instead. Arguments that break the schema never "
      "reach your function.",
      [
        table(
          ["Case", "What the client got"],
          [
            ["raised", f"{type(raised).__name__}: {raised}"],
            ["inspected", f"is_error={inspected.is_error}, text={inspected.content[0].text!r}"],
            ["cups='three'", " ".join(bad_type.content[0].text.split())[:160]],
          ],
        )
      ],
    ),
    example(
      "mcp.l02.5",
      "A server is also a program: the fastmcp CLI",
      from_the_terminal,
      "`fastmcp list` starts the server file, connects as a client and prints its tools. `fastmcp call` calls one. "
      "No client code needed to try a server.",
      [code(listed, "text", "$ fastmcp list src/nimbus/mcp/server.py"), code(called, "json", "$ fastmcp call …")],
    ),
  ]


# ── shared: the Nimbus server as a web service ─────────────────────────────────
@contextmanager
def nimbus_http(port: int = 8765):
  """Start src/nimbus/mcp/server.py --http PORT, wait for /health, yield the /mcp URL, stop it afterwards."""
  import subprocess
  import sys

  import httpx

  env = {**os.environ, "FASTMCP_SHOW_SERVER_BANNER": "false", "FASTMCP_LOG_LEVEL": "WARNING"}
  process = subprocess.Popen(
    [sys.executable, str(SERVER_FILE), "--http", str(port)],
    env=env,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
  )
  try:
    for _ in range(100):
      try:
        if httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5).status_code == 200:
          break
      except httpx.TransportError:
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/mcp"
  finally:
    process.terminate()
    process.wait(timeout=10)


# ── Lab 2: why MCP ─────────────────────────────────────────────────────────────
def integrations(apps: int, tools: int) -> tuple[int, int]:
  return (
    apps * tools,
    apps + tools,
  )  # wired by hand: every app × every tool; with MCP: one client each + one server each


def three_kinds_of_things():
  from fastmcp import FastMCP

  mcp = FastMCP("Nimbus Café")

  @mcp.tool  # the MODEL decides to call it
  def price(item: str) -> int:
    """Price of a café drink, in rupees."""
    return CAFE_PRICES[item.casefold()]

  @mcp.resource("nimbus://menu")  # the APP decides to read it
  def menu() -> dict:
    """The café menu."""
    return CAFE_PRICES

  @mcp.prompt  # the USER picks it
  def recommend(mood: str) -> str:
    """Ask for a drink recommendation."""
    return f"Recommend one Nimbus café drink for someone who feels {mood}. Prices: {CAFE_PRICES}"

  return mcp


async def offered(mcp):
  async with Client(mcp) as client:
    return (
      [t.name for t in await client.list_tools()],
      [str(r.uri) for r in await client.list_resources()],
      [p.name for p in await client.list_prompts()],
    )


async def no_glue_agent(question: str):
  from langchain.agents import create_agent
  from langchain.mcp import MCPAdapter

  async with MCPAdapter(nimbus_server.mcp) as adapter:  # the same server, in memory
    tools = await adapter.list_tools()  # MCP tools → LangChain tools: no code per tool
    agent = create_agent(get_chat_model(), tools, system_prompt="You are Nimbus support. Use the tools. Be brief.")
    result = await agent.ainvoke({"messages": [HumanMessage(question)]})  # MCP tools are async
  from nimbus.llm.models import aclose_clients

  await aclose_clients()  # async model clients belong to this event loop: close them before it ends
  return [t.name for t in tools], result


@lab(
  "mcp.why",
  "Why MCP: one protocol between AI apps and the things they use",
  uses_model=True,
  calls="~3 calls to Mistral",
)
def why_mcp():
  """The integration count with and without MCP, a server offering a tool, a resource and a prompt, and the Nimbus
  support server plugged into an agent with no glue code."""
  rows = [[a, t, *integrations(a, t)] for a, t in ((3, 5), (5, 20), (10, 50))]
  sample = three_kinds_of_things()
  tools, resources, prompts = asyncio.run(offered(sample))
  names, result = asyncio.run(no_glue_agent("Is order NC-78123 shipped, and how many points will it earn?"))
  calls = [c["name"] for m in result["messages"] for c in getattr(m, "tool_calls", None) or []]
  return [
    table(["AI apps", "tools / data sources", "integrations by hand", "with MCP (clients + servers)"], rows),
    table(
      ["A server offers", "who decides", "here"],
      [
        ["tools", "the model calls them", ", ".join(tools)],
        ["resources", "the app reads them", ", ".join(resources)],
        ["prompts", "the user picks them", ", ".join(prompts)],
      ],
    ),
    table(
      ["The Nimbus server in an agent", "Result"],
      [["tools loaded", ", ".join(names)], ["tool calls", ", ".join(calls)], ["answer", result["messages"][-1].text]],
    ),
    callout(
      "Without a standard, every AI app wires every tool by hand: apps × tools integrations. MCP puts one protocol "
      "between a HOST (your AI app), a CLIENT (its connection to one server) and a SERVER (which offers tools the "
      "model calls, resources the app reads and prompts the user picks): apps + tools. The payoff is in the last "
      "table: the Nimbus support server became an agent's tools with no code per tool.",
      "decision",
      "Why MCP",
    ),
  ] + [
    example(
      "mcp.l01.1",
      "Every app wires every tool by hand",
      integrations,
      "",
      [table(["apps", "tools", "by hand", "with MCP"], rows)],
    ),
    example(
      "mcp.l01.2",
      "Host, client, server",
      no_glue_agent,
      "The agent is the host; MCPAdapter holds the client; nimbus/mcp/server.py is the server.",
      [table(["server", "tools"], [["Nimbus Support", ", ".join(names)]])],
    ),
    example(
      "mcp.l01.3",
      "Tools, resources, prompts",
      three_kinds_of_things,
      "",
      [
        table(
          ["kind", "names"],
          [["tools", ", ".join(tools)], ["resources", ", ".join(resources)], ["prompts", ", ".join(prompts)]],
        )
      ],
    ),
    example(
      "mcp.l01.4",
      "A complete server and a client",
      offered,
      "",
      [table(["what the client listed"], [[f"{tools} · {resources} · {prompts}"]])],
    ),
    example(
      "mcp.l01.5",
      "The same server in an agent, no glue",
      no_glue_agent,
      "",
      [answer(result["messages"][-1].text, "agent")],
    ),
  ]


# ── Lab 3: transports and protocol eras ────────────────────────────────────────
INFO_SERVER = '''
import os
from fastmcp import FastMCP
mcp = FastMCP("Info")
served = {"calls": 0}

@mcp.tool
def where_am_i() -> dict:
  """This server's process id, a call counter, and one environment variable."""
  served["calls"] += 1
  region = os.environ.get("NIMBUS_REGION", "(not set)")
  return {"pid": os.getpid(), "calls_served": served["calls"], "NIMBUS_REGION": region}

if __name__ == "__main__":
  mcp.run()
'''


def in_memory():
  from fastmcp import FastMCP

  server = FastMCP("In-process")

  @server.tool
  def where_am_i() -> dict:
    """This server's process id."""
    return {"pid": os.getpid()}

  async def run():
    async with Client(server) as client:  # the server object lives in this process
      return (await client.call_tool("where_am_i", {})).data

  return asyncio.run(run())


def stdio(info_file: Path, env: dict | None = None):
  async def run():
    transport = PythonStdioTransport(info_file, env=env)  # the client starts the server as a subprocess
    async with Client(transport) as client:
      return (await client.call_tool("where_am_i", {})).data

  return asyncio.run(run())


def shared_http(url: str):
  async def run():
    async with Client(url) as first, Client(url) as second:  # many clients, one server
      a = (await first.call_tool("get_order", {"order_id": "NC-12093"})).data
      b = (await second.call_tool("get_order", {"order_id": "NC-78123"})).data
      return a.status if hasattr(a, "status") else a, b.status if hasattr(b, "status") else b

  return asyncio.run(run())


def protocol_eras(url: str):
  async def run():
    out = {}
    for mode in ("auto", "legacy"):
      async with Client(url, mode=mode) as client:
        out[mode] = client.protocol_version  # auto tries the modern, sessionless era first
    return out

  return asyncio.run(run())


@lab("mcp.transports", "Transports and protocol eras: how a client reaches a server", calls="No model calls")
def transports():
  """The same question asked in memory, over stdio (with and without an environment variable), and over HTTP by two
  clients at once; then the protocol era each client negotiates."""
  work = PROJECT_DIR / ".cache" / "labs" / "mcp"
  work.mkdir(parents=True, exist_ok=True)
  info_file = work / "info_server.py"
  info_file.write_text(INFO_SERVER)
  os.environ["NIMBUS_REGION"] = "south-india"  # set in THIS process
  mine = os.getpid()
  memory = in_memory()
  bare = stdio(info_file)
  passed = stdio(info_file, env={"NIMBUS_REGION": os.environ["NIMBUS_REGION"]})
  with nimbus_http() as url:
    shared = shared_http(url)
    eras = protocol_eras(url)
  return [
    table(
      ["Transport", "Result"],
      [
        ["this lab's process", mine],
        ["in memory", f"pid {memory['pid']} (the same process)"],
        ["stdio", f"pid {bare['pid']} (a child process), NIMBUS_REGION {bare['NIMBUS_REGION']}"],
        ["stdio with env=", f"NIMBUS_REGION {passed['NIMBUS_REGION']}"],
        ["HTTP, two clients, one server", f"{shared}"],
        *[[f"HTTP, mode={k}", f"protocol {v}"] for k, v in eras.items()],
      ],
    ),
    callout(
      "In memory, the server object lives in your process: right for tests. Over stdio, the client starts the server "
      "as a subprocess and talks over its stdin/stdout: one user, one machine (an IDE, a desktop app), and the "
      "server only sees the environment you pass it, never your API keys by accident. Over Streamable HTTP the server "
      "is a web service at /mcp that many clients share. Clients negotiate the protocol era: the modern, sessionless "
      "one first, the legacy handshake for older servers.",
      "decision",
      "Transports",
    ),
  ] + [
    example("mcp.l03.1", "In memory", in_memory, "", [table(["pid"], [[memory["pid"]]])]),
    example("mcp.l03.2", "stdio: a subprocess", stdio, "", [table(["pid"], [[bare["pid"]]])]),
    example(
      "mcp.l03.3",
      "Streamable HTTP: one server, many clients",
      shared_http,
      "",
      [table(["client 1", "client 2"], [list(shared)])],
    ),
    example(
      "mcp.l03.4",
      "stdio gets a minimal environment",
      stdio,
      "",
      [table(["without env", "with env"], [[bare["NIMBUS_REGION"], passed["NIMBUS_REGION"]]])],
    ),
    example("mcp.l03.5", "Protocol eras", protocol_eras, "", [table(["mode", "protocol"], list(eras.items()))]),
  ]


# ── Lab 4: the protocol on the wire ────────────────────────────────────────────
MODERN, LEGACY = "2026-07-28", "2025-11-25"


def body_of(response) -> dict:
  if response.headers.get("content-type", "").startswith("text/event-stream"):
    return json.loads(next(line[5:] for line in response.text.splitlines() if line.startswith("data:")))
  return response.json()


def modern_call(url: str, request_id: int, method: str, params: dict, name: str | None = None) -> tuple[dict, dict]:
  import httpx

  envelope = {"io.modelcontextprotocol/protocolVersion": MODERN, "io.modelcontextprotocol/clientCapabilities": {}}
  headers = {"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": MODERN, "Mcp-Method": method}
  if name:
    headers["Mcp-Name"] = name  # a routing header: a gateway can route without parsing the body
  message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": {**params, "_meta": envelope}}
  return message, body_of(httpx.post(url, json=message, headers=headers))


def legacy_session(url: str):
  import httpx

  headers = {"Accept": "application/json, text/event-stream"}
  init = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": LEGACY, "capabilities": {}, "clientInfo": {"name": "by-hand", "version": "1"}},
  }
  response = httpx.post(url, json=init, headers=headers)
  session = response.headers.get("mcp-session-id")
  httpx.post(
    url, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers={**headers, "Mcp-Session-Id": session}
  )
  listing = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
  with_session = body_of(httpx.post(url, json=listing, headers={**headers, "Mcp-Session-Id": session}))
  without = body_of(httpx.post(url, json=listing, headers=headers))
  return session, with_session, without


@lab("mcp.wire", "The protocol on the wire: JSON-RPC messages, sent by hand", calls="No model calls")
def wire():
  """The Nimbus server over HTTP, spoken to with plain httpx: discover, list, call, the two kinds of failure, and
  a legacy session."""
  with nimbus_http() as url:
    discover = modern_call(url, 1, "server/discover", {})
    listing = modern_call(url, 2, "tools/list", {})
    called = modern_call(
      url, 3, "tools/call", {"name": "get_order", "arguments": {"order_id": "NC-78123"}}, name="get_order"
    )
    tool_failed = modern_call(
      url, 4, "tools/call", {"name": "get_order", "arguments": {"order_id": "NC-99999"}}, name="get_order"
    )
    protocol_failed = modern_call(url, 5, "menu/teleport", {})
    session, with_session, without = legacy_session(url)
  trim = lambda r, keep: {k: v for k, v in r.get("result", r).items() if k in keep} if "result" in r else r  # noqa: E731
  return [
    table(
      ["Request", "sent", "received"],
      [
        [
          "server/discover",
          json.dumps(discover[0])[:200],
          json.dumps(trim(discover[1], ("supportedVersions", "capabilities")))[:300],
        ],
        ["tools/list", json.dumps(listing[0])[:120], ", ".join(t["name"] for t in listing[1]["result"]["tools"])],
        [
          "tools/call",
          json.dumps(called[0])[:200],
          json.dumps(trim(called[1], ("structuredContent", "isError")))[:300],
        ],
        ["a TOOL failure", "get_order NC-99999", json.dumps(trim(tool_failed[1], ("content", "isError")))[:300]],
        ["a PROTOCOL failure", "menu/teleport", json.dumps(protocol_failed[1])[:300]],
      ],
    ),
    table(
      ["Legacy era", "Result"],
      [
        ["session id from initialize", (session or "")[:16] + "…"],
        ["tools/list with it", ", ".join(t["name"] for t in with_session.get("result", {}).get("tools", []))],
        ["tools/list without it", json.dumps(without)[:200]],
      ],
    ),
    callout(
      "MCP is JSON-RPC 2.0: a request {jsonrpc, id, method, params} gets a response with the same id, a result or an "
      "error. In the modern era every request carries the protocol version and the client's capabilities, so any "
      "replica can answer: no sticky sessions, and routing headers let a gateway route without parsing. A tool that "
      "fails returns a normal result with isError (the model reads it and recovers); a bad request gets a JSON-RPC "
      "error. The legacy era needs an initialize handshake and a session id on every later request.",
      "decision",
      "On the wire",
    ),
  ] + [
    example(
      "mcp.l04.1",
      "JSON-RPC 2.0",
      modern_call,
      "",
      [
        code(json.dumps(called[0], indent=1), "json", "request"),
        code(json.dumps(called[1], indent=1)[:1200], "json", "response"),
      ],
    ),
    example(
      "mcp.l04.2",
      "Modern: discover, list, call",
      modern_call,
      "",
      [table(["tools"], [[t["name"]] for t in listing[1]["result"]["tools"]])],
    ),
    example(
      "mcp.l04.3",
      "Tool failure vs protocol failure",
      modern_call,
      "",
      [
        table(
          ["case", "response"],
          [["tool", json.dumps(tool_failed[1])[:200]], ["protocol", json.dumps(protocol_failed[1])[:200]]],
        )
      ],
    ),
    example(
      "mcp.l04.4",
      "Legacy: a session",
      legacy_session,
      "",
      [table(["with session", "without"], [[str(bool(with_session.get("result"))), json.dumps(without)[:120]]])],
    ),
    example(
      "mcp.l04.5",
      "Why the modern era matters",
      modern_call,
      "Any replica can answer any request; the FastAPI and Production tracks deploy it that way.",
      [table(["header"], [["MCP-Protocol-Version, Mcp-Method, Mcp-Name"]])],
    ),
  ]


# ── Production path: the support tools over MCP, measured ──────────────────────
def evaluate_over_mcp() -> list:
  from fastmcp.client.transports import PythonStdioTransport as Stdio  # noqa: N814
  from langchain.mcp import MCPAdapter

  from nimbus.config import get_settings
  from nimbus.support.tool_eval import evaluate_tools_async, load_requests
  from nimbus.support.tools import SupportStore, build_tools

  model = get_chat_model()
  requests = load_requests(get_settings().eval_tool_requests)

  async def run():
    reports = [await evaluate_tools_async("native LangChain tools", model, build_tools(SupportStore()), requests)]
    stdio_target = Stdio(
      SERVER_FILE, env={**os.environ, "FASTMCP_SHOW_SERVER_BANNER": "false", "FASTMCP_LOG_LEVEL": "WARNING"}
    )
    async with MCPAdapter(stdio_target) as adapter:
      reports.append(await evaluate_tools_async("MCP over stdio", model, await adapter.list_tools(), requests))
    with nimbus_http() as url:
      async with MCPAdapter(url) as adapter:
        reports.append(await evaluate_tools_async("MCP over HTTP", model, await adapter.list_tools(), requests))
    from nimbus.llm.models import aclose_clients

    await aclose_clients()
    return reports

  return asyncio.run(run())


@lab(
  "mcp.support-server-eval",
  "Evaluate: the support tools served over MCP",
  uses_model=True,
  calls="~90 calls to Mistral: 12 requests × 3, about 4 minutes",
)
def support_server_eval():
  """LangChain Module 3's 12 tool requests answered by the same model with the native tools, and with the same tools
  served by the Nimbus MCP server over stdio and over HTTP."""
  started = time.perf_counter()
  reports = evaluate_over_mcp()
  seconds = time.perf_counter() - started
  rows = [r.summary() for r in reports]
  native = rows[0]
  misses = [
    [r.variant, req["id"], req["request"], ", ".join(c["name"] for c in run.tool_calls), run.answer[:200]]
    for r in reports
    for req, run, s in r.rows
    if not s["success"]
  ]
  return [
    scorecard(
      rows,
      [
        Metric("success", "success"),
        Metric("tools_ok", "right tools"),
        Metric("args_ok", "right arguments"),
        Metric("answer_ok", "right facts"),
        Metric("rounds", "model calls", "lower", "text"),
        Metric("tokens", "tokens", "lower", "int"),
        Metric("latency_p50_ms", "latency p50", "lower", "ms"),
      ],
      title=f"12 tool requests · {seconds:.0f} s",
    ),
    table(["Tools", "id", "request", "calls", "answer"], misses or [["—", "—", "every request right", "", ""]]),
    callout(
      "; ".join(f"{r['variant']}: success {r['success']:.0%}, p50 {r['latency_p50_ms'] / 1000:.1f} s" for r in rows)
      + ". Latency is the model's, not the protocol's: the MCP layer adds milliseconds to calls that take seconds. "
      + (
        "The misses are listed above: over MCP the model sometimes made a different choice with the same tools (a "
        "stray check_refund before escalating, an invented answer instead of 'we don't sell that'). Two things "
        "differ: the MCP tools' descriptions and schemas are rebuilt from the server's signatures, and MCP has no "
        "return_direct, so after escalate_to_human the model keeps going. With 12 requests, one or two misses is "
        "also within run-to-run variation: run it again before reading much into it. "
        if any(r["success"] < native["success"] for r in rows[1:])
        else "Every variant answered the same. "
      )
      + "The support tools ship as an MCP server for hosts (stdio locally, HTTP shared); the agents keep the native "
      "tools in-process.",
      "decision",
      "Ship the MCP server",
    ),
  ]
