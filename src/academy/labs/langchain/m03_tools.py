"""LangChain · Module 3 labs: tools. Defining tools, what they return, prebuilt toolkits, and the tool-calling loop
by hand. Each learning point is a small snippet function that the lab runs; the Covers tab shows its code next to
what it produced."""

import json
import shutil
import sqlite3
import time
from contextlib import closing
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from academy.labs.runtime import Input, Metric, answer, callout, code, columns, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR, get_settings
from nimbus.llm import get_chat_model
from nimbus.support.assistant import answer_with_tools
from nimbus.support.tool_eval import evaluate_tools, load_requests, score, terse
from nimbus.support.tools import SupportStore, build_tools

SANDBOX = PROJECT_DIR / ".state" / "sandbox"
SUPPORT_DB = PROJECT_DIR / ".state" / "support_orders.db"


def _ms(started: float) -> int:
  return round((time.perf_counter() - started) * 1000)


@lru_cache
def _tools():
  return build_tools(SupportStore())


def _schema(t) -> dict:
  return t.tool_call_schema.model_json_schema()


# ── Lab 1: defining tools ───────────────────────────────────────────────────────
def a_tool_from_a_function():
  from langchain_core.tools import tool

  @tool
  def cafe_menu_item(name: str) -> dict:
    """Price in rupees and whether a café drink or snack is vegan."""
    return {"item": name, "price_rupees": 220, "vegan": True}

  name, description = cafe_menu_item.name, cafe_menu_item.description  # from the function name and docstring
  arguments = cafe_menu_item.tool_call_schema.model_json_schema()["properties"]  # from the type hints
  return name, description, arguments


def describe_every_argument():
  from langchain_core.tools import tool
  from pydantic import BaseModel, Field

  class RefundCheck(BaseModel):
    order_id: str = Field(pattern=r"^NC-\d{4,6}$", description="The order number, like NC-12093")
    opened: bool = Field(description="Whether the bag was opened; ask the customer if you don't know")

  @tool(args_schema=RefundCheck)
  def check_refund(order_id: str, opened: bool) -> str:
    """Check whether an order can be refunded under Nimbus policy."""
    return "eligible" if not opened else "opened bags can't be returned"

  arguments = check_refund.tool_call_schema.model_json_schema()["properties"]
  return arguments


def name_and_describe_it_yourself():
  from langchain_core.tools import tool

  @tool("loyalty_points", description="Nimbus Beans points earned for an amount in rupees (1 point per ₹10).")
  def points_for(amount_rupees: int) -> int:  # the function name stays internal
    return amount_rupees // 10

  return points_for.name, points_for.description


def call_a_tool_yourself(get_order):
  plain = get_order.invoke({"order_id": "NC-4410"})  # an args dict → just the result
  message = get_order.invoke(  # a ToolCall (what a model sends) → a ToolMessage
    {"name": "get_order", "args": {"order_id": "NC-4410"}, "id": "call_1", "type": "tool_call"}
  )
  return plain, message


def fail_with_a_message():
  from langchain_core.tools import ToolException, tool

  @tool
  def get_order(order_id: str) -> str:
    """Look up a Nimbus order."""
    raise ToolException(f"No order {order_id}. Ask the customer to check the number (it looks like NC-12093).")

  get_order.handle_tool_error = True  # the exception becomes a ToolMessage the model can read and act on
  message = get_order.invoke({"name": "get_order", "args": {"order_id": "NC-99999"}, "id": "c2", "type": "tool_call"})
  return message


def a_hidden_runtime_argument():
  from langchain.tools import ToolRuntime
  from langchain_core.tools import tool

  @tool
  def my_orders(status: str, runtime: ToolRuntime) -> str:
    """List the current customer's orders with a given status."""
    customer = runtime.context.customer_id  # who is asking: filled in by the agent, never by the model
    return f"orders of {customer} with status {status}"

  arguments = my_orders.tool_call_schema.model_json_schema()["properties"]  # `runtime` isn't in it
  return arguments


@lab("lc.tool-definitions", "Tools in depth: how a Python function becomes something a model can call")
def tool_definitions():
  tools = _tools()
  name, description, arguments = a_tool_from_a_function()
  described = describe_every_argument()
  custom_name, custom_description = name_and_describe_it_yourself()
  plain, message = call_a_tool_yourself(tools[0])
  failure = fail_with_a_message()
  hidden = a_hidden_runtime_argument()
  return [
    table(
      ["Tool", "Description the model reads", "Arguments"],
      [
        [
          t.name,
          t.description,
          ", ".join(f"{k}: {v.get('type', v.get('anyOf', ''))}" for k, v in _schema(t)["properties"].items()),
        ]
        for t in tools
      ],
      title="The Nimbus support tools, as the model sees them",
    ),
    table(
      ["Call", "Result"],
      [
        ["get_order.invoke({'order_id': 'NC-4410'})", plain],
        ["get_order.invoke(<ToolCall>)", f"{type(message).__name__}: {message.content}"],
        ["…its .artifact (kept by the app, not sent to the model)", json.dumps(message.artifact)[:160]],
        [
          "a tool raising ToolException, handled",
          f"{type(failure).__name__}(status={failure.status!r}): {failure.content}",
        ],
      ],
      title="Calling tools yourself, without a model",
    ),
    callout(
      "A tool is a function plus a **schema**: its name, its description and its arguments' types and descriptions. "
      "That schema is all the model knows about it, so write it for the model. Validate arguments with a Pydantic "
      "`args_schema`, and turn failures into messages (`ToolException` + `handle_tool_error`) the model can act on "
      "instead of exceptions that end the conversation. No model calls in this lab: tools are plain code.",
      "decision",
      "What to take away",
    ),
    example(
      "lc.l17.1",
      "@tool builds the schema from the function",
      a_tool_from_a_function,
      "Name from the function name, description from the docstring, arguments from the type hints.",
      [
        table(
          ["Variable", "Value"], [["name", name], ["description", description], ["arguments", json.dumps(arguments)]]
        )
      ],
    ),
    example(
      "lc.l17.2",
      "args_schema: describe each argument precisely",
      describe_every_argument,
      "A Pydantic model gives each argument a description and rules (here a pattern) the model sees and the "
      "tool enforces.",
      [code(json.dumps(described, indent=1))],
    ),
    example(
      "lc.l17.3",
      "A custom name and description",
      name_and_describe_it_yourself,
      "The model sees `loyalty_points`; your code keeps its own function name.",
      [table(["Attribute", "Value"], [["name", custom_name], ["description", custom_description]])],
    ),
    example(
      "lc.l17.4",
      "Call a tool yourself: an args dict, or a ToolCall",
      call_a_tool_yourself,
      "With an args dict you get the result; with a ToolCall you get the ToolMessage a model would receive "
      "(plus the artifact, for tools that keep one).",
      [table(["Variable", "Value"], [["plain", plain], ["message", f"{type(message).__name__}: {message.content}"]])],
    ),
    example(
      "lc.l17.5",
      "ToolException + handle_tool_error",
      fail_with_a_message,
      "The error becomes a ToolMessage with status 'error'; the model can ask the customer to check the number.",
      [table(["Attribute", "Value"], [["message.status", failure.status], ["message.content", failure.content]])],
    ),
    example(
      "lc.l17.6",
      "ToolRuntime: a hidden argument with the agent's context",
      a_hidden_runtime_argument,
      "Declare a `runtime: ToolRuntime` parameter to read who is asking (and the agent's state). It is not "
      "part of the schema, so the model can't fill it in or fake it. Module 4 uses it.",
      [table(["Variable", "Value"], [["arguments", json.dumps(hidden)]])],
    ),
  ]


# ── Lab 2: what tools return ────────────────────────────────────────────────────
def text_or_an_object(cafe_menu_item):
  message = cafe_menu_item.invoke(
    {"name": "cafe_menu_item", "args": {"name": "Oat Latte"}, "id": "c1", "type": "tool_call"}
  )
  content = message.content  # the dict the function returned, as JSON text: the model only ever reads text
  return content


def summary_for_the_model_record_for_the_app(get_order):
  message = get_order.invoke({"name": "get_order", "args": {"order_id": "NC-12500"}, "id": "c2", "type": "tool_call"})
  for_the_model = message.content  # short: costs few tokens
  for_the_app = message.artifact  # the full order record, never sent to the model
  return for_the_model, for_the_app


def return_an_image(model):
  import base64

  from langchain_core.tools import tool

  from nimbus.support.assistant import answer_with_tools

  @tool
  def receipt_image(bill_number: str) -> list:
    """The scanned café receipt for a bill number, like INB-20417."""
    data = base64.b64encode((PROJECT_DIR / "data/nimbus/receipt.png").read_bytes()).decode()
    return [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}]  # format Mistral accepts here

  run = answer_with_tools(model, [receipt_image], "What is the total on the receipt for bill INB-20417?")
  return run


def end_with_the_tool_output(model, escalate_to_human):
  from nimbus.support.assistant import answer_with_tools

  run = answer_with_tools(model, [escalate_to_human], "There was a piece of glass in my cold brew!")
  return run  # escalate_to_human has return_direct=True: its output is the answer, no model call after it


def never_name_an_argument_config():
  from langchain_core.tools import tool

  @tool
  def note(order_id: str, config: str) -> str:
    """Add a note to an order."""
    return f"{order_id}: {config}"

  shown_to_model = list(note.tool_call_schema.model_json_schema()["properties"])  # config IS in the schema
  try:
    note.invoke({"order_id": "NC-4410", "config": "gift wrap"})  # ...but LangChain keeps `config` for itself
  except TypeError as error:
    failure = error
  return shown_to_model, failure


@lab(
  "lc.tool-outputs",
  "Tool outputs: what a tool returns, and what the model actually sees",
  uses_model=True,
  calls="2 + 1 calls to Mistral",
)
def tool_outputs():
  model, tools = get_chat_model(), _tools()
  content = text_or_an_object(tools[2])
  short, full = summary_for_the_model_record_for_the_app(tools[0])
  image_run = return_an_image(model)
  direct_run = end_with_the_tool_output(model, tools[4])
  shown, failure = never_name_an_argument_config()
  return [
    table(
      ["Tool returns", "The model receives", "Model calls"],
      [
        ["a dict (cafe_menu_item)", content, "—"],
        ["(summary, record): content_and_artifact (get_order)", short, "—"],
        [
          "an image block (receipt_image)",
          f"the picture → answered: {image_run.answer.strip()[:120]}",
          image_run.rounds,
        ],
        [
          "return_direct=True (escalate_to_human)",
          f"nothing: the tool's output is the answer → {direct_run.answer}",
          direct_run.rounds,
        ],
      ],
      title="Four kinds of tool output",
    ),
    code(json.dumps(full, indent=1, ensure_ascii=False), "json", "The artifact the app kept (not sent to the model)"),
    callout(
      f"The model only reads text (or an image), so return a short summary and keep full records as artifacts: "
      f"get_order's summary is {len(short)} characters, the record {len(json.dumps(full))}. A `return_direct` tool "
      f"saves a model call ({direct_run.rounds} call instead of 2) and returns exactly the approved wording. "
      "Two traps: "
      f"`runtime` is injected and hidden, and an argument named `config` is shown to the model but never reaches your "
      f"function ({type(failure).__name__}). For images in tool results, Mistral needs the `image_url` block format; "
      "LangChain's standard `image` block is rejected there.",
      "decision",
      "What to take away",
    ),
    example(
      "lc.l18.1",
      "Text or an object: the model gets a ToolMessage either way",
      text_or_an_object,
      "Objects are serialised to JSON text in the ToolMessage.",
      [table(["Variable", "Value"], [["content", content]])],
    ),
    example(
      "lc.l18.2",
      "content_and_artifact: a summary for the model, the record for the app",
      summary_for_the_model_record_for_the_app,
      '`@tool(response_format="content_and_artifact")` and return `(content, artifact)`.',
      [table(["Variable", "Value"], [["for_the_model", short], ["for_the_app", json.dumps(full)[:200]]])],
    ),
    example(
      "lc.l18.3",
      "Returning an image",
      return_an_image,
      "Return content blocks instead of text. The model looked at the receipt and answered.",
      [
        table(
          ["Expression", "Value"],
          [["run.tool_calls", json.dumps(image_run.tool_calls)], ["run.answer", image_run.answer.strip()]],
        )
      ],
    ),
    example(
      "lc.l18.4",
      "return_direct=True: the tool's output is the final answer",
      end_with_the_tool_output,
      "The loop stops after the tool: no extra model call, and the customer gets the exact approved text.",
      [
        table(
          ["Expression", "Value"],
          [["run.rounds", direct_run.rounds], ["run.direct", direct_run.direct], ["run.answer", direct_run.answer]],
        )
      ],
    ),
    example(
      "lc.l18.5",
      "Reserved argument names: never call one `config`",
      never_name_an_argument_config,
      "`runtime` is injected and hidden from the model (lab 1). `config` is worse: the model is told to fill "
      "it in, but LangChain keeps it for the RunnableConfig, so your function never receives it.",
      [
        table(
          ["Variable", "Value"],
          [["shown_to_model", json.dumps(shown)], ["failure", f"{type(failure).__name__}: {failure}"]],
        )
      ],
    ),
  ]


# ── Lab 3: prebuilt tools ───────────────────────────────────────────────────────
def pick_from_a_toolkit():
  from langchain_community.agent_toolkits import FileManagementToolkit

  everything = FileManagementToolkit(root_dir=str(SANDBOX)).get_tools()
  chosen = FileManagementToolkit(
    root_dir=str(SANDBOX), selected_tools=["write_file", "read_file", "list_directory"]
  ).get_tools()
  return [t.name for t in everything], [t.name for t in chosen]


def files_in_one_folder(model):
  from langchain_community.agent_toolkits import FileManagementToolkit

  from nimbus.support.assistant import answer_with_tools

  file_tools = FileManagementToolkit(
    root_dir=str(SANDBOX), selected_tools=["write_file", "read_file", "list_directory"]
  ).get_tools()
  run = answer_with_tools(
    model,
    file_tools,
    "Write a file called shift_note.txt saying 'Grinder 2 needs cleaning', then list the files.",
    system="You manage files for the Nimbus café team. Use the tools.",
  )
  return run


def search_the_web():
  from langchain_community.tools import DuckDuckGoSearchResults

  search = DuckDuckGoSearchResults(num_results=3, output_format="list")  # no API key needed
  results = search.invoke("Monsoon Malabar coffee")
  return results


def query_a_database(model):
  from langchain_community.agent_toolkits import SQLDatabaseToolkit
  from langchain_community.utilities import SQLDatabase
  from sqlalchemy import create_engine

  from nimbus.support.assistant import answer_with_tools

  engine = create_engine(f"sqlite:///file:{SUPPORT_DB}?mode=ro&uri=true")  # read-only: the agent can't change data
  db = SQLDatabase(engine)
  sql_tools = SQLDatabaseToolkit(db=db, llm=model).get_tools()
  run = answer_with_tools(
    model,
    sql_tools,
    "How many delivered orders are there, and what is their total value in rupees?",
    system="You answer questions about the orders database. List the tables, check the schema, then query.",
    max_rounds=8,
  )
  engine.dispose()
  return [t.name for t in sql_tools], run


def keep_tools_in_bounds():
  from langchain_community.agent_toolkits import FileManagementToolkit
  from sqlalchemy import create_engine, text

  read_file = FileManagementToolkit(root_dir=str(SANDBOX), selected_tools=["read_file"]).get_tools()[0]
  escape = read_file.invoke({"file_path": "../../.env"})  # outside the folder: refused, the secrets stay put
  engine = create_engine(f"sqlite:///file:{SUPPORT_DB}?mode=ro&uri=true")
  try:
    with engine.connect() as db:
      db.execute(text("DELETE FROM orders"))  # read-only connection: refused
  except Exception as error:  # noqa: BLE001
    write = error
  engine.dispose()
  return escape, write


def _build_support_db() -> None:
  """A SQLite copy of the support orders for the SQL tools (rebuilt each run, opened read-only by the tools)."""
  SUPPORT_DB.parent.mkdir(parents=True, exist_ok=True)
  SUPPORT_DB.unlink(missing_ok=True)
  data = json.loads((PROJECT_DIR / "data/support/orders.json").read_text())["orders"]
  with closing(sqlite3.connect(SUPPORT_DB)) as db, db:
    db.execute(
      "CREATE TABLE orders (order_id TEXT PRIMARY KEY, customer TEXT, total INTEGER, status TEXT, ordered_on TEXT)"
    )
    db.executemany(
      "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
      [(o["order_id"], o["customer"], o["total"], o["status"], o["ordered_on"]) for o in data],
    )


@lab(
  "lc.prebuilt-tools",
  "Prebuilt tools and toolkits: files, web search and SQL, kept in bounds",
  uses_model=True,
  calls="about 8 calls to Mistral, plus a DuckDuckGo search",
)
def prebuilt_tools():
  shutil.rmtree(SANDBOX, ignore_errors=True)
  SANDBOX.mkdir(parents=True)
  _build_support_db()
  model = get_chat_model()
  everything, chosen = pick_from_a_toolkit()
  file_run = files_in_one_folder(model)
  try:
    results = search_the_web()
    search_rows = [[r.get("title", ""), r.get("link", "")] for r in results]
  except Exception as error:  # noqa: BLE001 — the web may be unreachable; the lab still teaches the rest
    results, search_rows = [], [[f"search failed: {type(error).__name__}", str(error)[:120]]]
  sql_names, sql_run = query_a_database(model)
  escape, write = keep_tools_in_bounds()
  with closing(sqlite3.connect(SUPPORT_DB)) as db:
    truth = db.execute("SELECT COUNT(*), SUM(total) FROM orders WHERE status = 'delivered'").fetchone()
  return [
    table(
      ["Toolkit", "Tools"],
      [
        ["FileManagementToolkit (all)", ", ".join(everything)],
        ["…selected_tools", ", ".join(chosen)],
        ["SQLDatabaseToolkit", ", ".join(sql_names)],
      ],
      title="Toolkits: bundles of related tools",
    ),
    columns(
      [
        answer(
          file_run.answer,
          "Files",
          {"tool calls": len(file_run.tool_calls), "files now": ", ".join(p.name for p in SANDBOX.iterdir())},
        ),
        answer(
          sql_run.answer,
          "SQL",
          {"tool calls": len(sql_run.tool_calls), "true answer": f"{truth[0]} orders, ₹{truth[1]}"},
        ),
      ]
    ),
    table(["Title", "Link"], search_rows, title="DuckDuckGo search: Monsoon Malabar coffee"),
    table(
      ["Limit", "What happened"],
      [
        ["read ../../.env from the sandbox", str(escape)[:160]],
        ["DELETE on the read-only database", f"{type(write).__name__}: {str(write).splitlines()[0][:140]}"],
      ],
      title="Keeping prebuilt tools in bounds",
    ),
    callout(
      "Prebuilt tools save writing code, but they act with your permissions: give file tools one folder "
      "(`root_dir`), choose only the tools you need (`selected_tools`), and connect SQL tools read-only. Check "
      "their answers against the data: the SQL answer above should match the true count and total.",
      "decision",
      "What to take away",
    ),
    example(
      "lc.l19.1",
      "A toolkit is a bundle: pick only what you need",
      pick_from_a_toolkit,
      "Every extra tool is something the model might misuse, and more schema tokens on every call.",
      [table(["Variable", "Tools"], [["everything", ", ".join(everything)], ["chosen", ", ".join(chosen)]])],
    ),
    example(
      "lc.l19.2",
      "File tools that work inside one folder",
      files_in_one_folder,
      "The model wrote and listed files, all inside the sandbox folder.",
      [
        table(
          ["Expression", "Value"],
          [["run.tool_calls", json.dumps(file_run.tool_calls)[:300]], ["run.answer", file_run.answer[:200]]],
        )
      ],
    ),
    example(
      "lc.l19.3",
      "Web search: DuckDuckGo, no API key",
      search_the_web,
      "Results as a list of {title, link, snippet} dicts.",
      [table(["Title", "Link"], search_rows)],
    ),
    example(
      "lc.l19.4",
      "SQL tools: the model explores the database and writes its own query",
      query_a_database,
      "It listed tables, read the schema, then queried.",
      [
        table(
          ["Expression", "Value"],
          [
            ["calls", " → ".join(c["name"] for c in sql_run.tool_calls)],
            ["answer", sql_run.answer[:200]],
            ["the truth", f"{truth[0]} delivered orders, ₹{truth[1]}"],
          ],
        )
      ],
    ),
    example(
      "lc.l19.5",
      "Limit what prebuilt tools can do",
      keep_tools_in_bounds,
      "A sandbox folder and a read-only database connection: mistakes and attacks stay harmless.",
      [table(["Variable", "Value"], [["escape", str(escape)[:160]], ["write", f"{type(write).__name__}"]])],
    ),
  ]


# ── Lab 4: the tool-calling loop by hand ────────────────────────────────────────
def what_the_model_sees(get_order):
  from langchain_core.utils.function_calling import convert_to_openai_tool

  schema = convert_to_openai_tool(get_order)  # the JSON the provider receives for this tool
  return schema


def the_model_asks(model, tools):
  bound = model.bind_tools(tools)  # the model can now ask for these tools
  reply = bound.invoke("Where is my order NC-78123?")
  calls = reply.tool_calls  # a request to call get_order: nothing has run yet
  return reply, calls


def _orders_for_customer():
  """A lab-only tool whose output another call needs: the order ids come first, the lookup second."""
  from langchain_core.tools import tool

  store = SupportStore()

  @tool
  def orders_for_customer(customer: str) -> str:
    """The order numbers and dates of a customer, by name (like 'Meena K.')."""
    found = [o for o in store.orders.values() if o["customer"].casefold() == customer.strip().casefold()]
    return ", ".join(f"{o['order_id']} ({o['ordered_on']})" for o in found) or f"No orders for {customer}."

  return orders_for_customer


def the_loop(model, tools, question):

  by_name = {t.name: t for t in tools}
  bound = model.bind_tools(tools)
  messages = [HumanMessage(question)]
  while True:
    reply = bound.invoke(messages)
    messages.append(reply)
    if not reply.tool_calls:  # no more requests: this is the answer
      break
    for call in reply.tool_calls:
      messages.append(by_name[call["name"]].invoke(call))  # run it, append the ToolMessage, ask again
  return messages


@lab(
  "lc.tool-loop",
  "The tool-calling loop by hand: what every agent does",
  uses_model=True,
  calls="1 + 2 + about 3 calls to Mistral",
)
def tool_loop():
  model, tools = get_chat_model(), _tools()
  schema = what_the_model_sees(tools[0])
  reply, calls = the_model_asks(model, tools)
  parallel = the_loop(model, tools, "Did I order twice by mistake? Check NC-55021 and NC-55022.")
  multi = the_loop(model, [*tools, _orders_for_customer()], "Is Meena K.'s most recent order on its way?")

  def transcript(messages):
    rows = []
    for m in messages:
      if isinstance(m, AIMessage) and m.tool_calls:
        rows.append(
          ["AIMessage", "asks for " + ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in m.tool_calls)]
        )
      elif isinstance(m, ToolMessage):
        rows.append(["ToolMessage", str(m.content)[:140]])
      else:
        rows.append([type(m).__name__, m.text[:140]])
    return rows

  turns = lambda ms: [m for m in ms if isinstance(m, AIMessage) and m.tool_calls]  # noqa: E731
  return [
    table(["Message", "Content"], transcript(multi), title="One conversation, message by message"),
    scorecard(
      [
        {
          "variant": "two lookups (NC-55021, NC-55022)",
          "model_calls": sum(isinstance(m, AIMessage) for m in parallel),
          "tool_calls": sum(len(m.tool_calls) for m in turns(parallel)),
          "tool_turns": len(turns(parallel)),
        },
        {
          "variant": "find the customer's orders, then look one up",
          "model_calls": sum(isinstance(m, AIMessage) for m in multi),
          "tool_calls": sum(len(m.tool_calls) for m in turns(multi)),
          "tool_turns": len(turns(multi)),
        },
      ],
      [
        Metric("model_calls", "model calls", "lower", "int"),
        Metric("tool_calls", "tool calls", "none", "int"),
        Metric("tool_turns", "turns with tool calls", "none", "int"),
      ],
      title="Parallel calls (one turn) vs multi-round calls",
    ),
    callout(
      "The model never runs a tool: it asks (`tool_calls`), your code runs it and sends back a ToolMessage, and the "
      "model decides what's next, until it answers without asking. Independent lookups come back as several calls "
      "in one turn; dependent ones need another round. Module 4's `create_agent` is exactly this loop, built for you.",
      "decision",
      "What to take away",
    ),
    example(
      "lc.l20.1",
      "What a tool looks like to the model",
      what_the_model_sees,
      "Name, description and a JSON schema of the arguments: that's all the model knows.",
      [code(json.dumps(schema, indent=1))],
    ),
    example(
      "lc.l20.2",
      "bind_tools(): the model asks, it never runs the tool",
      the_model_asks,
      "The reply has `tool_calls` and usually no text: it's a request for your code.",
      [table(["Expression", "Value"], [["reply.text", repr(reply.text)], ["calls", json.dumps(calls)]])],
    ),
    example(
      "lc.l20.3",
      "The loop: model → tool calls → ToolMessages → model …",
      the_loop,
      "Repeat until the model answers without asking for a tool.",
      [table(["Message", "Content"], transcript(multi))],
    ),
    example(
      "lc.l20.4",
      "Parallel calls in one turn, multi-round when one depends on another",
      the_loop,
      "The same loop, two questions: two independent lookups come back in one turn; finding a customer's latest "
      "order needs their order numbers first, so the lookup waits for a second round.",
      [table(["Message", "Content"], transcript(parallel) + [["—", "—"]] + transcript(multi))],
    ),
  ]


# ── Production path labs: the support tools ─────────────────────────────────────
HISTORY = PROJECT_DIR / ".state" / "tool_scorecards.jsonl"
TOOL_METRICS = [
  Metric("success", "success", "higher", "ratio", "Right tools, right arguments, and a correct answer"),
  Metric("tools_ok", "right tools", "higher", "ratio"),
  Metric("args_ok", "right args", "higher", "ratio"),
  Metric("answer_ok", "answer ok", "higher", "ratio"),
  Metric("rounds", "model calls", "lower", "ratio", "Per request"),
  Metric("tokens", "tokens", "lower", "int", "Per request, all calls"),
  Metric("cost_per_1k_usd", "$ / 1k", "lower", "usd"),
]


def _request_options() -> list[str]:
  return [f"{r['id']} · {r['request']}" for r in load_requests(get_settings().eval_tool_requests)]


@lab(
  "lc.tools-one",
  "Handle one support request with the tools",
  uses_model=True,
  calls="1–3 calls to Mistral",
  inputs=[
    Input("request", "Labelled request", "select", default=_request_options()[9], options=_request_options()),
    Input("custom", "…or type your own (overrides the selection)", "textarea", default=""),
  ],
)
def tools_one(request: str, custom: str):
  rid, text = request.split(" · ", 1)
  if custom.strip():
    rid, text = "custom", custom.strip()
  run = answer_with_tools(get_chat_model(), _tools(), text)
  views = [
    answer(
      run.answer,
      "The answer",
      {
        "model calls": run.rounds,
        "tool calls": len(run.tool_calls),
        "ms": round(run.ms),
        "ended by a tool": run.direct,
      },
    ),
    table(
      ["#", "Message", "Content"],
      [
        [
          i,
          type(m).__name__,
          (
            ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in m.tool_calls)
            if isinstance(m, AIMessage) and m.tool_calls
            else str(m.content)
          )[:160],
        ]
        for i, m in enumerate(run.messages, 1)
        if not isinstance(m, SystemMessage)
      ],
      title="The conversation",
    ),
  ]
  if run.artifacts:
    views.append(code(json.dumps(run.artifacts, indent=1, ensure_ascii=False)[:2000], "json", "Artifacts the app kept"))
  labelled = {r["id"]: r for r in load_requests(get_settings().eval_tool_requests)}
  if rid in labelled:
    s = score(labelled[rid], run)
    views.append(
      table(
        ["Check", "Result"],
        [[k, "✓" if v is True else ("✗" if v is False else v)] for k, v in s.items()],
        title="Against the label",
      )
    )
  return views


@lab(
  "lc.tools-eval",
  "Score the support tools on 12 labelled requests",
  uses_model=True,
  calls="about 22 calls to Mistral per toolset (about 2 minutes for both)",
  inputs=[Input("compare_terse", "Also run the same tools without descriptions", "bool", default=True)],
)
def tools_eval(compare_terse: bool):
  from datetime import UTC, datetime

  model, requests = get_chat_model(), load_requests(get_settings().eval_tool_requests)
  variants = [("described", _tools())] + ([("terse (no descriptions)", terse(_tools()))] if compare_terse else [])
  reports = [evaluate_tools(name, model, tools, requests) for name, tools in variants]
  at = datetime.now(UTC).isoformat(timespec="seconds")
  HISTORY.parent.mkdir(exist_ok=True)
  with HISTORY.open("a") as f:
    for r in reports:
      f.write(json.dumps({"at": at, **r.summary()}) + "\n")
  rows = [r.summary() for r in reports]
  misses = [
    [
      r.variant,
      req["id"],
      req["request"][:60],
      ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in run.tool_calls),
      run.answer[:100],
    ]
    for r in reports
    for req, run, s in r.rows
    if not s["success"]
  ]
  history = [json.loads(line) for line in HISTORY.read_text().splitlines() if line.strip()]
  views = [scorecard(rows, TOOL_METRICS, title=f"{len(requests)} labelled requests")]
  if misses:
    views.append(
      table(["Toolset", "Request", "Message", "Calls", "Answer"], misses, title=f"Requests not handled ({len(misses)})")
    )
  if len(rows) == 2:
    d, t = rows
    views.append(
      callout(
        f"Described: success {d['success']:.2f} at {d['tokens']:,} tokens per request; without descriptions: "
        f"{t['success']:.2f} at {t['tokens']:,} tokens. "
        f"Descriptions cost {d['tokens'] - t['tokens']:+,} tokens on every "
        "request, so keep the ones that carry rules (validation patterns, 'check before promising a refund', when to "
        "escalate) and measure the rest.",
        "decision",
        "What the scorecard shows",
      )
    )
  views.append(
    scorecard(
      [{**h, "variant": f"#{i + 1} · {h['variant']}"} for i, h in enumerate(history)], TOOL_METRICS, title="History"
    )
  )
  return views
