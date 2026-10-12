"""MCP · Module 3 labs: MCP with LangChain and LangGraph. MCPAdapter, MCP prompts and resources in apps, many servers in
one agent, human-in-the-loop from annotations and elicitation, connections and caching, and agents served over MCP.
Then: the support agent as an MCP client of the Nimbus gateway, measured."""

import asyncio
import json
import os
import sys
import time
import warnings
from pathlib import Path

from fastmcp import Client, FastMCP
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from academy.labs.runtime import Metric, answer, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR
from nimbus.llm import get_chat_model

warnings.filterwarnings("ignore", message=".*langchain.mcp.*beta.*")
WORK = PROJECT_DIR / ".cache" / "labs" / "mcp"


def run(coro):
  return asyncio.run(coro)


async def _close_models():
  from nimbus.llm.models import aclose_clients

  await aclose_clients()


def _support_file() -> Path:
  import nimbus.mcp.server as support

  return Path(support.__file__)


def _quiet(path: Path):
  from fastmcp.client.transports import PythonStdioTransport

  return PythonStdioTransport(
    path, env={**os.environ, "FASTMCP_SHOW_SERVER_BANNER": "false", "FASTMCP_LOG_LEVEL": "WARNING"}
  )


def _calls(messages) -> list:
  return [(c["name"], c["args"]) for m in messages for c in getattr(m, "tool_calls", None) or []]


# ── Lab 1: MCPAdapter ──────────────────────────────────────────────────────────
async def adapter_tour():
  from langchain.agents import create_agent
  from langchain.mcp import MCPAdapter
  from langgraph.graph import START, MessagesState, StateGraph
  from langgraph.prebuilt import ToolNode, tools_condition

  out = {}
  async with MCPAdapter(_quiet(_support_file())) as adapter:  # a local server: started over stdio
    tools = await adapter.list_tools()  # every MCP tool → a LangChain StructuredTool
    by_name = {t.name: t for t in tools}
    out["tools"] = [
      f"{type(t).__name__} {t.name}({', '.join(t.args)}) {t.metadata['mcp']['tool'].get('annotations', {})}"
      for t in tools
    ]
    out["server"] = by_name["get_order"].metadata["mcp"]["server"]
    out["ainvoke(args) → content blocks"] = await by_name["cafe_menu_item"].ainvoke({"name": "Cold Brew"})
    message = await by_name["get_order"].ainvoke(
      {"type": "tool_call", "name": "get_order", "id": "call-1", "args": {"order_id": "NC-78123"}}
    )
    out["ainvoke(ToolCall) → ToolMessage"] = (
      f"status={message.status}, artifact={json.dumps(message.artifact, default=str)[:160]}"
    )
    failed = await by_name["get_order"].ainvoke(
      {"type": "tool_call", "name": "get_order", "id": "call-2", "args": {"order_id": "NC-99999"}}
    )
    out["a server-side error"] = f"status={failed.status}: {failed.text[:80]}"
    try:
      by_name["cafe_menu_item"].invoke({"name": "Cold Brew"})
    except NotImplementedError as error:
      out["invoke() (sync)"] = f"NotImplementedError: {str(error)[:90]}"
    model = get_chat_model()
    agent = create_agent(
      model, tools, system_prompt="You help Nimbus customers. Report numbers exactly as the tools give them. Be brief."
    )
    result = await agent.ainvoke(
      {"messages": [HumanMessage("Where is NC-78123, and how many loyalty points will it earn?")]}
    )
    out["create_agent: tools used"] = [n for n, _ in _calls(result["messages"])]
    out["create_agent: answer"] = result["messages"][-1].text
    bound = model.bind_tools(tools)

    async def call_model(state: MessagesState) -> dict:
      return {
        "messages": [await bound.ainvoke([SystemMessage("You help Nimbus staff. Be brief."), *state["messages"]])]
      }

    graph = StateGraph(MessagesState).add_node("model", call_model).add_node("tools", ToolNode(tools))
    graph = (
      graph.add_edge(START, "model")
      .add_conditional_edges("model", tools_condition)
      .add_edge("tools", "model")
      .compile()
    )
    result = await graph.ainvoke(
      {"messages": [HumanMessage("How much are an Oat Latte and a Cold Brew, and are they vegan?")]}
    )
    out["LangGraph ToolNode: tools used"] = [n for n, _ in _calls(result["messages"])]
    out["LangGraph ToolNode: answer"] = result["messages"][-1].text
  await _close_models()
  return out


@lab("mcp.adapter", "MCPAdapter: MCP server tools as LangChain tools", uses_model=True, calls="~5 calls to Mistral")
def adapter():
  """The Nimbus support server started over stdio: its tools as LangChain tools (metadata, annotations), called
  directly, then used by create_agent and by a hand-built LangGraph graph."""
  out = run(adapter_tour())
  return [
    table(
      ["What", "Result"],
      [[k, json.dumps(v, ensure_ascii=False, default=str) if not isinstance(v, str) else v] for k, v in out.items()],
    ),
    callout(
      "MCP is a standard way to serve tools from any process, in any language. MCPAdapter(target) connects (a URL, a "
      "script path over stdio, an in-process server, a config) and list_tools() turns every MCP tool into a LangChain "
      "StructuredTool with its name, description, JSON schema and the server's annotations in metadata['mcp']. Called "
      "with plain arguments it returns content blocks; with a ToolCall, a ToolMessage with status and artifact. MCP "
      "tools are async: use ainvoke and agent.ainvoke. Agents and graphs use them like any other tools.",
      "decision",
      "MCPAdapter",
    ),
  ] + [
    example(
      "lc.l57.1",
      "MCP: tools served by another process",
      adapter_tour,
      "",
      [table(["server"], [[json.dumps(out["server"])]])],
    ),
    example("lc.l57.2", "MCPAdapter over stdio", adapter_tour, "", [table(["tools"], [[t] for t in out["tools"]])]),
    example(
      "lc.l57.3",
      "list_tools(): normal LangChain tools",
      adapter_tour,
      "",
      [table(["tools"], [[t] for t in out["tools"]])],
    ),
    example(
      "lc.l57.4", "Give them to an agent (ainvoke)", adapter_tour, "", [answer(out["create_agent: answer"], "agent")]
    ),
    example(
      "mcp.l14.1", "MCPAdapter + list_tools()", adapter_tour, "", [table(["tools"], [[t] for t in out["tools"]])]
    ),
    example(
      "mcp.l14.2", "Name, description, schema, annotations", adapter_tour, "", [code(json.dumps(out["server"]), "json")]
    ),
    example(
      "mcp.l14.3",
      "Calling one directly",
      adapter_tour,
      "",
      [
        table(
          ["call", "result"],
          [
            [k, str(out[k])[:200]]
            for k in ("ainvoke(args) → content blocks", "ainvoke(ToolCall) → ToolMessage", "a server-side error")
          ],
        )
      ],
    ),
    example("mcp.l14.4", "Async only", adapter_tour, "", [code(out["invoke() (sync)"], "text")]),
    example(
      "mcp.l14.5",
      "An agent and a hand-built graph",
      adapter_tour,
      "",
      [
        table(
          ["", "tools", "answer"],
          [
            ["create_agent", str(out["create_agent: tools used"]), out["create_agent: answer"]],
            ["ToolNode graph", str(out["LangGraph ToolNode: tools used"]), out["LangGraph ToolNode: answer"]],
          ],
        )
      ],
    ),
  ]


# ── Lab 2: prompts and resources in apps ───────────────────────────────────────
def to_langchain(messages) -> list:
  return [(HumanMessage if m.role == "user" else AIMessage)(m.content.text) for m in messages]


async def prompts_resources_tour():
  from typing import TypedDict

  from fastmcp.server.transforms import ResourcesAsTools
  from langchain.agents import create_agent
  from langchain.agents.middleware import ModelRequest, dynamic_prompt
  from langchain.mcp import MCPAdapter
  from langgraph.graph import START, StateGraph

  import nimbus.mcp.server as support

  model = get_chat_model()
  out = {}
  async with MCPAdapter(support.mcp) as adapter:
    client = adapter.client  # tools come from list_tools; prompts and resources from the FastMCP client
    tools = await adapter.list_tools()
    prompt = await client.get_prompt("support_reply", {"issue": "my order arrived with a torn bag", "tone": "warm"})
    out["MCP prompt → LangChain messages → model"] = (await model.ainvoke(to_langchain(prompt.messages))).text

    @dynamic_prompt
    async def with_policy(request: ModelRequest) -> str:
      policy = (await client.read_resource("nimbus://policies/refunds"))[0].text  # fresh on every model call
      return f"You are Nimbus support. Be brief. Follow this policy exactly: {policy}"

    agent = create_agent(model, tools, middleware=[with_policy])
    result = await agent.ainvoke({"messages": [HumanMessage("Can I still cancel an order that was already packed?")]})
    out["resource as always-on context"] = result["messages"][-1].text

    class State(TypedDict, total=False):
      question: str
      order_id: str
      context: dict
      answer: str

    async def load_context(state: State) -> dict:
      order = json.loads((await client.read_resource(f"nimbus://orders/{state['order_id']}"))[0].text)
      policy = (await client.read_resource("nimbus://policies/refunds"))[0].text
      return {"context": {"order": order, "policy": policy}}  # exactly what this request needs

    async def respond(state: State) -> dict:
      reply = await model.ainvoke(
        [
          SystemMessage(
            f"You are Nimbus support. Use only this context: {json.dumps(state['context'])}. One sentence."
          ),
          HumanMessage(state["question"]),
        ]
      )
      return {"answer": reply.text}

    graph = StateGraph(State).add_node("load_context", load_context).add_node("respond", respond)
    graph = graph.add_edge(START, "load_context").add_edge("load_context", "respond").compile()
    result = await graph.ainvoke({"question": "Can I still cancel this order?", "order_id": "NC-55022"})
    out["LangGraph: resources loaded in a node"] = f"{result['context']['order']} → {result['answer']}"
  tool_only = FastMCP("Nimbus for tool-only clients")
  tool_only.mount(support.mcp)
  tool_only.add_transform(ResourcesAsTools(tool_only))
  async with MCPAdapter(tool_only) as adapter:
    tools = await adapter.list_tools()
    out["ResourcesAsTools: tools"] = [t.name for t in tools]
    agent = create_agent(model, tools, system_prompt="Use read_resource for Nimbus data and policies. Be brief.")
    result = await agent.ainvoke(
      {"messages": [HumanMessage("Read the refund policy resource: are café drinks refundable?")]}
    )
    out["ResourcesAsTools: calls"] = [n for n, _ in _calls(result["messages"])]
    out["ResourcesAsTools: answer"] = result["messages"][-1].text
  await _close_models()
  return out


@lab(
  "mcp.apps", "MCP prompts and resources in LangChain and LangGraph apps", uses_model=True, calls="~7 calls to Mistral"
)
def apps():
  """The support_reply prompt run by a LangChain model, the refund policy as always-on context (dynamic_prompt), a
  LangGraph node that loads exactly the resources a request needs, and resources exposed as tools."""
  out = run(prompts_resources_tour())
  return [
    table(
      ["What", "Result"],
      [[k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v] for k, v in out.items()],
    ),
    callout(
      "MCPAdapter turns tools into LangChain tools; for prompts and resources use its FastMCP client (adapter.client). "
      "An MCP prompt becomes LangChain messages for any model. A resource can be always-on context: a dynamic_prompt "
      "middleware reads it on every model call, so a policy change applies at once. In LangGraph a node loads exactly "
      "the resources a request needs into state. For hosts that only understand tools, ResourcesAsTools exposes "
      "resources as list/read tools.",
      "decision",
      "Prompts and resources in apps",
    ),
  ] + [
    example(
      "mcp.l15.1",
      "adapter.client for prompts and resources",
      prompts_resources_tour,
      "",
      [table(["tools"], [["(from list_tools)"]])],
    ),
    example(
      "mcp.l15.2",
      "An MCP prompt → LangChain messages",
      prompts_resources_tour,
      "",
      [answer(out["MCP prompt → LangChain messages → model"], "reply")],
    ),
    example(
      "mcp.l15.3",
      "A resource as always-on context",
      prompts_resources_tour,
      "nimbus/mcp/client_agent.py does this in production.",
      [answer(out["resource as always-on context"], "answer")],
    ),
    example(
      "mcp.l15.4",
      "LangGraph: a node loads the resources",
      prompts_resources_tour,
      "",
      [code(out["LangGraph: resources loaded in a node"], "text")],
    ),
    example(
      "mcp.l15.5",
      "ResourcesAsTools",
      prompts_resources_tour,
      "",
      [table(["tools", "calls"], [[str(out["ResourcesAsTools: tools"]), str(out["ResourcesAsTools: calls"])]])],
    ),
  ]


# ── Lab 3: many servers, one agent ─────────────────────────────────────────────
CAFE_SERVER = '''
from fastmcp import FastMCP
mcp = FastMCP("Nimbus Café")
DRINKS = {"cappuccino": 180, "oat latte": 220, "cold brew": 200}

@mcp.tool
def price(item: str) -> str:
  """Price of a café DRINK, in rupees."""
  return f"{item}: ₹{DRINKS[item.lower()]} a cup" if item.lower() in DRINKS else f"{item} is not a café drink"

if __name__ == "__main__":
  mcp.run()
'''
ROASTERY_SERVER = '''
import sys
from fastmcp import FastMCP
mcp = FastMCP("Nimbus Roastery")
BAGS = {"nilgiri sunrise": 720, "chikmagalur classic": 560, "monsoon malabar reserve": 650}

@mcp.tool
def price(item: str) -> str:
  """Price of a 250 g BAG of beans, in rupees."""
  return f"{item}: ₹{BAGS[item.lower()]} a bag" if item.lower() in BAGS else f"{item} is not a bean blend"

@mcp.tool
def in_stock(item: str) -> bool:
  """Whether a bean blend is in stock."""
  return item.lower() != "monsoon malabar reserve"

if __name__ == "__main__":
  if "--http" in sys.argv:
    mcp.run(transport="http", host="127.0.0.1", port=int(sys.argv[-1]), show_banner=False)
  else:
    mcp.run()
'''


def write_servers() -> tuple[Path, Path]:
  WORK.mkdir(parents=True, exist_ok=True)
  cafe, roastery = WORK / "cafe_server.py", WORK / "roastery_server.py"
  cafe.write_text(CAFE_SERVER)
  roastery.write_text(ROASTERY_SERVER)
  return cafe, roastery


async def many_servers_tour(cafe: Path, roastery_url: str):
  from fastmcp.client.group import ClientGroup
  from langchain.agents import create_agent
  from langchain.mcp import MCPAdapter

  out = {}
  async with Client(_quiet(cafe)) as a, Client(roastery_url) as b:
    out["the clash"] = {
      a.server_info.name: [t.name for t in await a.list_tools()],
      b.server_info.name: [t.name for t in await b.list_tools()],
    }
  config = {
    "mcpServers": {
      "cafe": {"command": sys.executable, "args": [str(cafe)], "env": {"FASTMCP_SHOW_SERVER_BANNER": "false"}},
      "roastery": {"url": roastery_url},
    }
  }
  async with MCPAdapter(config) as adapter:  # one adapter, two servers, two transports, prefixed names
    tools = await adapter.list_tools()
    out["MCPConfig tools"] = [t.name for t in tools]
    agent = create_agent(
      get_chat_model(),
      tools,
      system_prompt=(
        "You help Nimbus customers. Drinks are on the cafe server; bags of beans on the roastery server. Be brief."
      ),
    )
    result = await agent.ainvoke(
      {
        "messages": [
          HumanMessage("How much is a cappuccino in the café, and a bag of Nilgiri Sunrise? Is it in stock?")
        ]
      }
    )
    out["the agent's calls"] = _calls(result["messages"])
    out["answer"] = result["messages"][-1].text
  group = ClientGroup({"cafe": Client(_quiet(cafe)), "roastery": Client(roastery_url, mode="legacy")})
  async with MCPAdapter(group) as adapter:
    out["ClientGroup tools"] = [t.name for t in await adapter.list_tools()]
  async with group:
    for name in ("cafe", "roastery"):
      route = await group.resolve_tool(f"{name}_price")
      out[f"{name} connection"] = f"protocol {route.client.protocol_version}, upstream tool {route.upstream_name!r}"
  await _close_models()
  return out


@lab(
  "mcp.many-servers",
  "Many servers, one agent: MCPConfig, ClientGroup, name clashes",
  uses_model=True,
  calls="~3 calls to Mistral",
)
def many_servers():
  """A café server and a roastery server that both offer `price`: one over stdio, one over HTTP, in one MCPConfig, an
  agent choosing between them, and a ClientGroup with one connection per server."""
  import subprocess

  import httpx

  cafe, roastery = write_servers()
  port = 8766
  process = subprocess.Popen(
    [sys.executable, str(roastery), "--http", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
  )
  try:
    for _ in range(100):
      try:
        httpx.post(f"http://127.0.0.1:{port}/mcp", timeout=0.5)
        break
      except httpx.TransportError:
        time.sleep(0.1)
    out = run(many_servers_tour(cafe, f"http://127.0.0.1:{port}/mcp"))
  finally:
    process.terminate()
    process.wait(timeout=10)
  return [
    table(
      ["What", "Result"],
      [[k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v] for k, v in out.items()],
    ),
    callout(
      "Two servers offering `price` would collide in a plain list. An MCPConfig ({'mcpServers': {...}}) puts many "
      "servers behind one adapter, prefixing each tool with its server's key, and mixes transports (here stdio and "
      "HTTP). The agent tells cafe_price from roastery_price by name and description. A ClientGroup gives each "
      "server its own connection, with its own auth, protocol era and handlers (the roastery here pinned to the "
      "legacy era).",
      "decision",
      "Many servers",
    ),
  ] + [
    example("mcp.l16.1", "The clash", write_servers, "", [code(json.dumps(out["the clash"]), "json")]),
    example(
      "mcp.l16.2",
      "MCPConfig: prefixed, mixed transports",
      many_servers_tour,
      "",
      [table(["tools"], [[t] for t in out["MCPConfig tools"]])],
    ),
    example(
      "mcp.l16.3",
      "The agent picks the right server",
      many_servers_tour,
      "",
      [table(["calls"], [[str(out["the agent's calls"])]]), answer(out["answer"], "answer")],
    ),
    example(
      "mcp.l16.4",
      "ClientGroup",
      many_servers_tour,
      "",
      [table(["connection", "detail"], [[k, out[k]] for k in ("cafe connection", "roastery connection")])],
    ),
  ]


# ── Lab 4: humans in the loop with MCP ─────────────────────────────────────────
async def hitl_tour():
  from langchain.agents import create_agent
  from langchain.mcp import MCPAdapter
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.types import Command

  import nimbus.mcp.server as support
  from academy.labs.mcp.m02_servers import booking_server
  from nimbus.mcp.client_agent import build_mcp_agent, is_destructive

  out = {}
  model = get_chat_model()
  async with MCPAdapter(support.mcp) as adapter:
    tools = await adapter.list_tools()
    out["destructive, per the server"] = sorted(t.name for t in tools if is_destructive(t))
    for thread, decision in (
      ("approve", {"type": "approve"}),
      ("reject", {"type": "reject", "message": "The customer changed their mind."}),
    ):
      support.store.orders["NC-55022"]["status"] = "processing"
      agent = await build_mcp_agent(model, adapter, InMemorySaver(), policy_uri="nimbus://policies/refunds")
      config = {"configurable": {"thread_id": thread}}
      result = await agent.ainvoke(
        {"messages": [HumanMessage("Please cancel order NC-55022, I ordered it twice.")]}, config
      )
      steps = []
      if "__interrupt__" not in result:  # seen: the agent asks "shall I go ahead?" in text first; the customer says yes
        steps.append(f"the agent asked first: {result['messages'][-1].text[:80]!r} → 'Yes, go ahead.'")
        result = await agent.ainvoke({"messages": [HumanMessage("Yes, go ahead and cancel it.")]}, config)
      while "__interrupt__" in result:
        value = result["__interrupt__"][0].value
        if "action_requests" in value:  # the approval gate (from destructiveHint)
          steps.append(f"approval for {value['action_requests'][0]['name']} → {decision['type']}")
          result = await agent.ainvoke(Command(resume={"decisions": [decision]}), config)
        else:  # the server asks the user (elicitation) → an interrupt
          question = value["requests"][0]
          steps.append(f"server asks {question['message']!r} → yes")
          result = await agent.ainvoke(
            Command(resume={"responses": {question["key"]: {"action": "accept", "content": {"confirm": True}}}}), config
          )
      out[f"{thread}: pauses"] = steps
      out[f"{thread}: reply"] = result["messages"][-1].text
      out[f"{thread}: order now"] = support.store.orders["NC-55022"]["status"]
  async with MCPAdapter(booking_server()) as adapter:
    agent = create_agent(
      model,
      await adapter.list_tools(),
      checkpointer=InMemorySaver(),
      system_prompt="You book tables at Nimbus cafés. Be brief.",
    )
    config = {"configurable": {"thread_id": "booking"}}
    result = await agent.ainvoke({"messages": [HumanMessage("Book me a table at the Bandra café.")]}, config)
    answers, asked = iter([{"people": 2}, {"time": "19:30"}]), []
    while "__interrupt__" in result:
      question = result["__interrupt__"][0].value["requests"][0]
      reply = next(answers)
      asked.append(f"{question['message']} → {reply}")
      result = await agent.ainvoke(
        Command(resume={"responses": {question["key"]: {"action": "accept", "content": reply}}}), config
      )
    out["booking: questions"] = asked
    out["booking: reply"] = result["messages"][-1].text
  await _close_models()
  return out


@lab(
  "mcp.hitl",
  "Humans in the loop with MCP: approvals from annotations, server questions as interrupts",
  uses_model=True,
  calls="~8 calls to Mistral",
)
def hitl():
  """The support agent on MCP tools: cancel_order pauses for approval because the server marks it destructive, then
  the server asks the customer to confirm (an interrupt); approved and rejected. Then a booking whose two questions
  become interrupts."""
  out = run(hitl_tour())
  return [
    table(
      ["What", "Result"],
      [[k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v] for k, v in out.items()],
    ),
    callout(
      "Let the SERVER say which tools are dangerous: destructiveHint arrives in each tool's metadata, and a "
      "HumanInTheLoopMiddleware `when` predicate gates exactly those, with no tool names hard-coded in the app. When "
      "an MCP tool asks the user a question (elicitation), the adapter pauses the agent with a LangGraph interrupt; "
      "the app shows the form and resumes with the answer. A checkpointer and a thread id make both pauses resumable, "
      "even hours later. Each resume re-runs the tool from the top, so its work before asking must be safe to repeat.",
      "decision",
      "Humans in the loop",
    ),
  ] + [
    example(
      "mcp.l17.1",
      "The server marks what's destructive",
      hitl_tour,
      "nimbus/mcp/client_agent.py: is_destructive.",
      [table(["destructive"], [[t] for t in out["destructive, per the server"]])],
    ),
    example(
      "mcp.l17.2",
      "HumanInTheLoopMiddleware + when",
      hitl_tour,
      "",
      [table(["thread", "pauses"], [[t, str(out[f"{t}: pauses"])] for t in ("approve", "reject")])],
    ),
    example(
      "mcp.l17.3",
      "Elicitation → interrupt",
      hitl_tour,
      "",
      [table(["questions"], [[q] for q in out["booking: questions"]])],
    ),
    example(
      "mcp.l17.4",
      "Resumable with a checkpointer",
      hitl_tour,
      "",
      [table(["thread", "order now"], [[t, out[f"{t}: order now"]] for t in ("approve", "reject")])],
    ),
  ]


# ── Lab 5: connections, caching, performance ──────────────────────────────────
async def connections_tour(url: str):
  from fastmcp.server.middleware import Middleware
  from langchain.mcp import MCPAdapter
  from mcp.client.caching import CacheConfig

  async def per_call_ms(tool, n=5):
    started = time.perf_counter()
    for _ in range(n):
      await tool.ainvoke({"name": "Cold Brew"})
    return round((time.perf_counter() - started) / n * 1000, 1)

  out = {}
  async with MCPAdapter(url) as adapter:
    tools = {t.name: t for t in await adapter.list_tools()}
  out["adapter closed: each call reconnects (ms per call)"] = await per_call_ms(tools["cafe_menu_item"])
  async with MCPAdapter(url) as adapter:
    tools = {t.name: t for t in await adapter.list_tools()}
    out["adapter open: one connection (ms per call)"] = await per_call_ms(tools["cafe_menu_item"])
    started = time.perf_counter()
    results = await asyncio.gather(
      *[tools["cafe_menu_item"].ainvoke({"name": n}) for n in ("Cappuccino", "Oat Latte", "Cold Brew", "Masala Chai")]
    )
    out["4 parallel calls on the open connection (ms)"] = (
      f"{(time.perf_counter() - started) * 1000:.1f} → {len(results)} results"
    )

  class CountLists(Middleware):
    def __init__(self):
      self.count = 0

    async def on_list_tools(self, context, call_next):
      self.count += 1
      return await call_next(context)

  counter = CountLists()
  server = FastMCP("Nimbus (cache hints)", cache_ttl=300)  # results may be reused for 300 s
  server.add_middleware(counter)

  @server.tool
  def price(item: str) -> int:
    """Price of a café drink in rupees."""
    return {"cold brew": 200}.get(item.lower(), 0)

  for label, cache in (("no cache", None), ("with cache", CacheConfig(target_id="nimbus-menu"))):
    counter.count = 0
    async with MCPAdapter(Client(server, cache=cache)) as adapter:
      for _ in range(3):
        await adapter.list_tools()
      await adapter.list_tools(cache_mode="refresh")  # force a refetch, e.g. after a deploy
    out[f"3 list_tools + 1 refresh, {label}: requests that reached the server"] = counter.count
  return out


@lab("mcp.connections", "Connections, caching and performance", calls="No model calls")
def connections():
  """The Nimbus server over HTTP: tool calls with the adapter closed (a connection per call) and open (one, reused),
  parallel calls on one connection, and tool-list caching with the server's freshness hints."""
  from academy.labs.mcp.m01_protocol import nimbus_http

  with nimbus_http(8767) as url:
    out = run(connections_tour(url))
  rows = list(out.items())
  return [
    table(["What", "Result"], rows),
    callout(
      f"Tools keep working after the adapter's block ends, but each call then opens its own connection "
      f"({rows[0][1]} ms per call here) against {rows[1][1]} ms with the adapter kept open around the run; parallel "
      "calls share that connection. The server's cache_ttl lets a caching client skip repeat tools/list round trips "
      "until a refresh. In a deployment: discover tools per run (servers change), but reuse connections and the cache "
      "underneath. Locally the difference is milliseconds; across a real network the reconnect cost grows.",
      "decision",
      "Connections",
    ),
  ] + [
    example(
      "mcp.l18.1", "Default: each call reconnects", connections_tour, "", [table(["ms per call"], [[rows[0][1]]])]
    ),
    example("mcp.l18.2", "Keep the adapter open", connections_tour, "", [table(["ms per call"], [[rows[1][1]]])]),
    example("mcp.l18.3", "Parallel calls share the connection", connections_tour, "", [code(str(rows[2][1]), "text")]),
    example(
      "mcp.l18.4",
      "Caching the tool list",
      connections_tour,
      "",
      [table(["case", "requests"], [list(r) for r in rows[3:]])],
    ),
    example(
      "mcp.l18.5",
      "In a deployment",
      connections_tour,
      "Discover per run; pool connections; cache lists.",
      [table(["advice"], [["discover per run, reuse connections"]])],
    ),
  ]


# ── Lab 6: agents served over MCP ──────────────────────────────────────────────
async def agent_server_tour():
  from langchain.agents import create_agent
  from langchain.mcp import MCPAdapter

  from nimbus.mcp.assistant_server import mcp as assistant_server

  out = {}
  async with Client(assistant_server) as client:
    info = (await client.list_tools())[0]
    out["the tool"] = f"{info.name}({list(info.input_schema['properties'])}): {info.description}"
    out["a direct call"] = (
      await client.call_tool("ask_nimbus", {"question": "Do loyalty points expire?", "customer_id": "c-1042"})
    ).data
  async with MCPAdapter(assistant_server) as adapter:
    concierge = create_agent(
      get_chat_model(),
      await adapter.list_tools(),
      system_prompt="You are a travel concierge. For anything about Nimbus Coffee, ask Nimbus. Be brief.",
    )
    result = await concierge.ainvoke(
      {"messages": [HumanMessage("I'm staying near Indiranagar. When does the Nimbus café there open?")]}
    )
    out["agent-to-agent: calls"] = _calls(result["messages"])
    out["agent-to-agent: answer"] = result["messages"][-1].text
  await _close_models()
  return out


async def langgraph_server_mcp(url: str):
  async with Client(f"{url}/mcp") as client:
    tools = {t.name: t for t in await client.list_tools()}
    assistant = next((t for n, t in tools.items() if "assistant" in n), None)
    result = (
      await client.call_tool(
        assistant.name, {"question": "What discount do subscriptions give?", "customer_id": "c-3301"}
      )
      if assistant
      else None
    )
  return list(tools), (assistant.input_schema if assistant else {}), (result.content[0].text if result else "")


@lab(
  "mcp.agent-server",
  "Serving your agent over MCP: any MCP host can use what you built",
  uses_model=True,
  calls="~10 calls to Mistral; starts and stops `langgraph dev`",
)
def agent_server():
  """The Nimbus assistant (Agents Module 7) as one MCP tool, used by another agent; then LangGraph Server, which
  exposes every hosted graph at /mcp."""
  from academy.labs.agents.m08_production import URL, start_server

  out = run(agent_server_tour())
  process = start_server()
  try:
    names, schema, reply = run(langgraph_server_mcp(URL))
  finally:
    process.terminate()
    process.wait()
  return [
    table(
      ["What", "Result"],
      [[k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v] for k, v in out.items()],
    ),
    table(
      ["LangGraph Server /mcp", "Result"],
      [
        ["tools (one per hosted graph)", ", ".join(names)],
        ["the assistant tool's input schema", json.dumps(schema)],
        ["its answer", reply[:300]],
      ],
    ),
    callout(
      "A whole LangChain agent can be ONE MCP tool: ask_nimbus wraps the router assistant, and another agent uses it "
      "like any other tool without knowing its insides (agent-to-agent through MCP). LangGraph Server goes further: "
      "every hosted graph is an MCP tool at /mcp automatically, and the graph's input and output schemas are the "
      "tool's interface (question and customer_id in, answer out), so callers don't receive its internal state.",
      "decision",
      "Agents over MCP",
    ),
  ] + [
    example(
      "mcp.l19.1",
      "An agent as one MCP tool",
      agent_server_tour,
      "nimbus/mcp/assistant_server.py.",
      [code(out["the tool"], "text"), answer(out["a direct call"], "answer")],
    ),
    example(
      "mcp.l19.2",
      "Agent-to-agent through MCP",
      agent_server_tour,
      "",
      [table(["calls"], [[str(out["agent-to-agent: calls"])]]), answer(out["agent-to-agent: answer"], "concierge")],
    ),
    example(
      "mcp.l19.3",
      "LangGraph Server: graphs at /mcp",
      langgraph_server_mcp,
      "server/assistant_graph.py in server/langgraph.json.",
      [table(["tools"], [[n] for n in names])],
    ),
    example(
      "mcp.l19.4",
      "Input and output schemas are the interface",
      langgraph_server_mcp,
      "",
      [code(json.dumps(schema, indent=1), "json")],
    ),
  ]


# ── Production path: the support agent as an MCP client ────────────────────────
def evaluate_mcp_agent() -> list:
  from langchain.mcp import MCPAdapter
  from langgraph.checkpoint.memory import InMemorySaver

  import nimbus.mcp.server as support
  from nimbus.config import get_settings
  from nimbus.mcp.client_agent import build_mcp_agent
  from nimbus.support.assistant import ToolRun
  from nimbus.support.tool_eval import ToolReport, load_requests, score

  requests = load_requests(get_settings().eval_tool_requests) + load_requests(
    PROJECT_DIR / "data" / "eval" / "policy_questions.json"
  )

  async def run_all():
    model = get_chat_model()
    reports = []
    async with MCPAdapter(support.mcp) as adapter:
      for name, policy in (("MCP tools only (Module 1)", False), ("MCP agent: tools + live policy + approvals", True)):
        agent = await build_mcp_agent(
          model, adapter, InMemorySaver(), policy=policy, policy_uri="nimbus://policies/refunds"
        )
        rows = []
        for n, request in enumerate(requests):
          started = time.perf_counter()
          result = await agent.ainvoke(
            {"messages": [HumanMessage(request["request"])]}, {"configurable": {"thread_id": f"{name}-{n}"}}
          )
          messages = result["messages"]
          calls = [
            {"name": c["name"], "args": c["args"]} for m in messages for c in getattr(m, "tool_calls", None) or []
          ]
          run_ = ToolRun(
            messages[-1].text,
            messages,
            sum(isinstance(m, AIMessage) for m in messages),
            calls,
            ms=(time.perf_counter() - started) * 1000,
          )
          rows.append((request, run_, score(request, run_)))
        reports.append(ToolReport(name, len(requests), rows))
    await _close_models()
    return reports

  return asyncio.run(run_all())


@lab(
  "mcp.client-agent-eval",
  "Evaluate: the support agent as an MCP client",
  uses_model=True,
  calls="~80 calls to Mistral: 20 requests × 2, about 4 minutes",
)
def client_agent_eval():
  """LangChain Module 3's 12 tool requests and Module 2's 8 policy questions, through an agent on the Nimbus server's
  tools alone, and through the production MCP agent (live policy resource, approvals from annotations)."""
  started = time.perf_counter()
  reports = evaluate_mcp_agent()
  seconds = time.perf_counter() - started
  rows = [r.summary() for r in reports]
  misses = [
    [r.variant, req["id"], req["request"], run_.answer[:200]]
    for r in reports
    for req, run_, s in r.rows
    if not s["success"]
  ]
  plain, full = rows
  return [
    scorecard(
      rows,
      [
        Metric("success", "success"),
        Metric("tools_ok", "right tools"),
        Metric("answer_ok", "right facts"),
        Metric("rounds", "model calls", "lower", "text"),
        Metric("tokens", "tokens", "lower", "int"),
        Metric("latency_p50_ms", "latency p50", "lower", "ms"),
      ],
      title=f"20 requests · {seconds:.0f} s",
    ),
    table(["Agent", "id", "request", "answer"], misses or [["—", "—", "every request right", ""]], title="Every miss"),
    callout(
      f"The production MCP agent answered {full['success']:.0%} against {plain['success']:.0%} with the tools alone. "
      "Everything it knows comes from the server: tools from list_tools, the policy from a resource read on every "
      "model call (a policy change needs no redeploy of the agent), and which tools need approval from the server's "
      "annotations. That makes the Nimbus servers the single source of truth for any agent or host that connects.",
      "decision",
      "Ship the MCP client agent",
    ),
  ]
