"""MCP · Module 2 labs: building servers. Tools in depth, resources, prompts and completions, context (logs,
progress, the server's own data), elicitation, sampling, lifespan and dependency injection, middleware, composition.
Then: should the refund policy reach the model as a resource the app reads, or only through tools?"""

import asyncio
import base64
import json
import time
import warnings
from typing import Annotated, Literal

from fastmcp import Client, Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from academy.labs.runtime import Metric, answer, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR
from nimbus.llm import get_chat_model

warnings.filterwarnings("ignore", message="The logging capability is deprecated")
RECEIPT = PROJECT_DIR / "data" / "nimbus" / "receipt.png"
PRICES = {"cappuccino": 180, "oat latte": 220, "cold brew": 200}


def run(coro):
  return asyncio.run(coro)


async def _close_models():
  from nimbus.llm.models import aclose_clients

  await aclose_clients()  # async model clients belong to this event loop: close them before it ends


# ── Lab 1: tools in depth ──────────────────────────────────────────────────────
class Address(BaseModel):
  street: str
  city: str
  pincode: Annotated[str, Field(pattern=r"^\d{6}$")]


class Quote(BaseModel):
  item: str
  size: str
  cups: int
  total_rupees: int


def tools_server():
  from fastmcp.tools import ToolResult
  from fastmcp.utilities.types import Image

  mcp = FastMCP("Nimbus Delivery")

  @mcp.tool
  def quote(
    item: Annotated[str, Field(description="Drink, e.g. 'oat latte'")],
    size: Literal["small", "medium", "large"],
    cups: Annotated[int, Field(ge=1, le=20)],
    address: Address,
  ) -> Quote:
    """Price a delivery order (validated before this code runs)."""
    if item.lower() not in PRICES:
      raise ToolError(f"{item!r} is not on the menu")
    total = (PRICES[item.lower()] + {"small": 0, "medium": 30, "large": 60}[size]) * cups
    return Quote(item=item, size=size, cups=cups, total_rupees=total)  # an output schema, structured data

  @mcp.tool
  def receipt_photo() -> Image:
    """The photo of a café receipt."""
    return Image(path=RECEIPT)  # an image block a vision model can look at

  @mcp.tool
  def sales_summary(day: str) -> ToolResult:
    """A day's sales: a short sentence for the model, every row for the app."""
    rows = [{"hour": h, "rupees": 2900 + h * 15} for h in range(7, 21)]
    return ToolResult(
      content=f"{day}: ₹{sum(r['rupees'] for r in rows)} in total.", structured_content={"day": day, "hourly": rows}
    )

  @mcp.tool(annotations={"readOnlyHint": True})
  def stock_level(item: str) -> int:
    """Bags of an item in stock."""
    return 12

  @mcp.tool(annotations={"destructiveHint": True, "idempotentHint": False})
  def write_off_stock(item: str, bags: int) -> str:
    """Remove spoiled bags from stock. Cannot be undone."""
    return f"Wrote off {bags} bags of {item}"

  @mcp.tool(timeout=0.5)
  async def supplier_eta(item: str) -> str:
    """Ask the supplier when an item arrives."""
    await asyncio.sleep(3)  # the supplier's API hangs
    return "tomorrow"

  return mcp


def broken_inventory() -> str:
  """Read the inventory database."""
  raise ConnectionError("postgres://admin:s3cret@10.0.0.7/inventory refused connection")  # internals!


async def tools_in_depth():
  mcp = tools_server()
  mcp.tool(broken_inventory)
  masked = FastMCP("Masked", mask_error_details=True)  # other exceptions' details stay on the server
  masked.tool(broken_inventory)
  out = {}
  async with Client(mcp) as client:
    tools = {t.name: t for t in await client.list_tools()}
    bad = {
      "item": "oat latte",
      "size": "huge",
      "cups": 50,
      "address": {"street": "12 Lake Rd", "city": "Chennai", "pincode": "60"},
    }
    result = await client.call_tool("quote", bad, raise_on_error=False)
    out["invalid arguments"] = " ".join(result.content[0].text.split())[:300]
    good = {**bad, "size": "large", "cups": 2, "address": {**bad["address"], "pincode": "600020"}}
    out["output schema"] = list(tools["quote"].output_schema["properties"])
    out["structured result"] = (await client.call_tool("quote", good)).structured_content
    block = (await client.call_tool("receipt_photo", {})).content[0]
    out["image"] = f"{block.type}, {block.mime_type}, {len(block.data):,} base64 characters"
    summary = await client.call_tool("sales_summary", {"day": "Monday"})
    out["ToolResult"] = (
      f"model reads: {summary.content[0].text} · app gets {len(summary.structured_content['hourly'])} rows"
    )
    out["annotations"] = {
      n: f"read_only={tools[n].annotations.read_only_hint}, destructive={tools[n].annotations.destructive_hint}"
      for n in ("stock_level", "write_off_stock")
    }
    out["ToolError"] = (
      (await client.call_tool("quote", {**good, "item": "pizza"}, raise_on_error=False)).content[0].text
    )
    out["other exception, default"] = (
      (await client.call_tool("broken_inventory", {}, raise_on_error=False)).content[0].text
    )
    started = time.perf_counter()
    timed_out = await client.call_tool("supplier_eta", {"item": "beans"}, raise_on_error=False)
    out["timeout"] = f"{timed_out.content[0].text} ({time.perf_counter() - started:.1f} s)"
  async with Client(masked) as client:
    out["other exception, masked"] = (
      (await client.call_tool("broken_inventory", {}, raise_on_error=False)).content[0].text
    )
  return out


@lab(
  "mcp.tools-depth",
  "Tools in depth: arguments, typed results, images, annotations, errors, timeouts",
  calls="No model calls",
)
def tools_depth():
  """A delivery quote with validated arguments and a typed result, a receipt photo, a sales summary with separate
  views, read-only and destructive annotations, errors with and without masking, and a timeout."""
  out = run(tools_in_depth())
  return [
    table(
      ["What", "Result"],
      [[k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v] for k, v in out.items()],
    ),
    callout(
      "Arguments are validated against the schema before your code runs (Field limits, Literal choices, nested "
      "models: every problem reported at once). Return a model for an output schema and structured data, an Image "
      "for an image block, or a ToolResult to give the model a sentence and the app the rows. Annotations tell "
      "hosts what a tool does to the world. ToolError messages reach the client; other exceptions leak their "
      "details unless the server masks them (look at the database password above). timeout= stops a hung call.",
      "decision",
      "Tools in depth",
    ),
  ] + [
    example("mcp.l05.1", "Arguments validated first", tools_server, "", [code(out["invalid arguments"], "text")]),
    example("mcp.l05.2", "Typed results", tools_server, "", [code(json.dumps(out["structured result"]), "json")]),
    example("mcp.l05.3", "Images", tools_server, "", [code(out["image"], "text")]),
    example(
      "mcp.l05.4", "ToolResult: content vs structured_content", tools_server, "", [code(out["ToolResult"], "text")]
    ),
    example("mcp.l05.5", "Annotations", tools_server, "", [code(json.dumps(out["annotations"], indent=1), "json")]),
    example(
      "mcp.l05.6",
      "Errors and mask_error_details",
      tools_in_depth,
      "",
      [
        table(
          ["case", "client sees"],
          [
            ["ToolError", out["ToolError"]],
            ["default", out["other exception, default"]],
            ["masked", out["other exception, masked"]],
          ],
        )
      ],
    ),
    example("mcp.l05.7", "timeout=", tools_server, "", [code(out["timeout"], "text")]),
  ]


# ── Lab 2: resources ───────────────────────────────────────────────────────────
async def resources_tour():
  from fastmcp.exceptions import ResourceError

  import nimbus.mcp.server as support

  extra = FastMCP("Media")

  @extra.resource("nimbus://media/receipt", mime_type="image/png")
  def receipt() -> bytes:
    """A café receipt photo."""
    return RECEIPT.read_bytes()  # bytes arrive as a base64 blob

  @extra.resource("staff://{path*}")
  def staff_file(path: str) -> str:
    """A file from the staff folder (a wildcard path)."""
    return (PROJECT_DIR / "data" / "nimbus" / path).read_text()

  out = {}
  async with Client(support.mcp) as client:
    out["static"] = [f"{r.uri} ({r.mime_type})" for r in await client.list_resources()]
    out["templates"] = [t.uri_template for t in await client.list_resource_templates()]
    out["policy"] = (await client.read_resource("nimbus://policies/refunds"))[0].text[:160]
    out["order with ?fields"] = (await client.read_resource("nimbus://orders/NC-78123?fields=status,total"))[0].text
    out["public document"] = (await client.read_resource("docs://products.txt"))[0].text[:120]
    try:
      await client.read_resource("docs://handbook.md")
    except Exception as error:  # noqa: BLE001 - the error is the point
      out["staff document"] = f"{type(error).__name__}: {error}"[:200]
    try:
      await client.read_resource("nimbus://orders/NC-99999")
    except Exception as error:  # noqa: BLE001
      out["unknown order"] = f"{type(error).__name__}: {error}"[:200]
  async with Client(extra) as client:
    blob = (await client.read_resource("nimbus://media/receipt"))[0]
    image = base64.b64decode(blob.blob)
    out["binary"] = f"{blob.mime_type}, {len(image):,} bytes, starts {image[:4]!r}"
    try:
      await client.read_resource("staff://../../pyproject.toml")
    except Exception as error:  # noqa: BLE001
      out["path traversal"] = f"{type(error).__name__}: {error}"[:200]
  del ResourceError
  return out


async def app_decides_context(question: str):
  import nimbus.mcp.server as support

  async with Client(support.mcp) as client:
    policy = (await client.read_resource("nimbus://policies/refunds"))[0].text  # the APP picks what to read...
    order = (await client.read_resource("nimbus://orders/NC-12093"))[0].text
  reply = await get_chat_model().ainvoke(
    [
      ("system", f"You are Nimbus support. Policy: {policy}\nOrder: {order}\nOne or two sentences."),
      ("human", question),
    ]
  )
  await _close_models()
  return reply.text  # ...the model just reads them


@lab(
  "mcp.resources",
  "Resources: data a server offers for the application to read",
  uses_model=True,
  calls="1 call to Mistral",
)
def resources():
  """The Nimbus server's resources: the refund policy, the menu, an order by id with ?fields, public documents
  (staff ones refused), a binary receipt, a wildcard path that refuses '..', and an answer from context the app
  chose."""
  out = run(resources_tour())
  reply = run(app_decides_context("My order NC-12093 arrived with a torn bag. What should I do?"))
  return [
    table(
      ["What", "Result"],
      [[k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v] for k, v in out.items()],
    ),
    answer(reply, "Answer from the policy and the order, read by the app"),
    callout(
      "A resource is a URI with content (text, JSON or base64 bytes), a name, a description and a MIME type; "
      "templates ({order_id}, {?fields}, {path*}) let one function serve many URIs, with path traversal refused "
      "before your code runs. ResourceError messages reach the client. The difference from a tool: the MODEL decides "
      "to call a tool, the APP decides which resources go into the context, as with the policy and the order here.",
      "decision",
      "Resources",
    ),
  ] + [
    example(
      "mcp.l06.1",
      "Static resources",
      resources_tour,
      "nimbus/mcp/server.py: refund_policy and menu.",
      [table(["resources"], [[r] for r in out["static"]])],
    ),
    example(
      "mcp.l06.2",
      "Resource templates",
      resources_tour,
      "",
      [table(["templates"], [[t] for t in out["templates"]]), code(out["order with ?fields"], "json")],
    ),
    example(
      "mcp.l06.3",
      "Query parameters, wildcards, traversal protection",
      resources_tour,
      "",
      [code(out["path traversal"], "text")],
    ),
    example("mcp.l06.4", "Binary resources", resources_tour, "", [code(out["binary"], "text")]),
    example(
      "mcp.l06.5",
      "ResourceError",
      resources_tour,
      "",
      [table(["case", "error"], [["staff document", out["staff document"]], ["unknown order", out["unknown order"]]])],
    ),
    example("mcp.l06.6", "Tools vs resources: who decides", app_decides_context, "", [answer(reply, "answer")]),
  ]


# ── Lab 3: prompts and completions ─────────────────────────────────────────────
def house_style_prompt():
  from fastmcp.prompts import Message

  mcp = FastMCP("Menu writer")

  @mcp.prompt
  def menu_description(item: str) -> list[Message]:
    """A menu description in the house style, from two examples."""
    return [
      Message("Describe 'Cold Brew' for the menu board."),
      Message("Slow-steeped for 18 hours. Smooth, low-acid, quietly strong.", role="assistant"),
      Message("Describe 'Masala Chai' for the menu board."),
      Message("Black tea, ginger and cardamom. Warm, spiced, familiar.", role="assistant"),
      Message(f"Describe '{item}' for the menu board. Same style: one short line, no title, no emoji."),
    ]

  return mcp


def to_langchain(messages) -> list:
  from langchain_core.messages import AIMessage, HumanMessage

  return [(HumanMessage if m.role == "user" else AIMessage)(m.content.text) for m in messages]


async def prompts_tour():
  from mcp.types import PromptReference, ResourceTemplateReference

  import nimbus.mcp.server as support

  model = get_chat_model()
  out = {}
  async with Client(support.mcp) as client:
    prompts = await client.list_prompts()
    out["prompts"] = [
      f"{p.name}({', '.join(a.name + ('' if a.required else '?') for a in p.arguments)})" for p in prompts
    ]
    rendered = await client.get_prompt(
      "support_reply", {"issue": "my cold brew bag arrived torn", "tone": "formal", "max_sentences": 2}
    )
    out["rendered"] = rendered.messages[0].content.text[:300]
    out["reply"] = (await model.ainvoke(to_langchain(rendered.messages))).text  # ANY model can run it
    out["complete order_id 'nc-55'"] = (
      await client.complete(
        ResourceTemplateReference(type="ref/resource", uri="nimbus://orders/{order_id}{?fields}"),
        {"name": "order_id", "value": "nc-55"},
      )
    ).values
    out["complete issue 'd'"] = (
      await client.complete(PromptReference(type="ref/prompt", name="support_reply"), {"name": "issue", "value": "d"})
    ).values
  async with Client(house_style_prompt()) as client:
    few_shot = await client.get_prompt("menu_description", {"item": "Oat Latte"})
    out["few-shot messages"] = len(few_shot.messages)
    out["menu line"] = (await model.ainvoke(to_langchain(few_shot.messages))).text
  await _close_models()
  return out


@lab("mcp.prompts", "Prompts and completions: templates the user picks", uses_model=True, calls="2 calls to Mistral")
def prompts():
  """The Nimbus support_reply prompt rendered and run, a multi-message house-style prompt, and completions for
  order ids and prompt arguments."""
  out = run(prompts_tour())
  return [
    table(
      ["What", "Result"],
      [[k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v] for k, v in out.items()],
    ),
    callout(
      "@mcp.prompt turns a function into a template with typed arguments; it can return a whole conversation (here "
      "two examples of the house style). A client fetches it with get_prompt and sends the messages to any model: "
      "the server owns the wording, the client owns the model. Completions suggest argument values as the user "
      "types, for prompts and for resource-template parameters.",
      "decision",
      "Prompts",
    ),
  ] + [
    example(
      "mcp.l07.1", "@mcp.prompt", prompts_tour, "nimbus/mcp/server.py: support_reply.", [code(out["rendered"], "text")]
    ),
    example(
      "mcp.l07.2",
      "Multi-message prompts",
      house_style_prompt,
      "",
      [table(["messages", "line"], [[out["few-shot messages"], out["menu line"]]])],
    ),
    example("mcp.l07.3", "get_prompt, then any model", prompts_tour, "", [answer(out["reply"], "reply")]),
    example(
      "mcp.l07.4",
      "Completions",
      prompts_tour,
      "",
      [
        table(
          ["field", "suggestions"],
          [["order_id 'nc-55'", str(out["complete order_id 'nc-55'"])], ["issue 'd'", str(out["complete issue 'd'"])]],
        )
      ],
    ),
  ]


# ── Lab 4: context: logging, progress, the server's own data ───────────────────
async def context_tour():
  from nimbus.mcp import ops

  logs, progress = [], []

  async def on_log(message):
    data = message.data if isinstance(message.data, dict) else {"msg": message.data}
    logs.append(f"{message.level}: {data.get('msg')}")

  async def on_progress(done, total, message):
    progress.append(f"{done:.0f}/{total:.0f} {message or ''}")

  server = FastMCP("Context demo")

  @server.resource("nimbus://notes/shift")
  def shift_notes() -> str:
    """Notes for the shift manager."""
    return "Oat milk is short at Indiranagar until Friday."

  @server.tool
  async def brief(ctx: Context) -> dict:
    """A shift briefing: reads the server's own resource, logs, and reports request info."""
    await ctx.info("Preparing the shift brief")  # a log message that travels to the CLIENT
    notes = (await ctx.read_resource("nimbus://notes/shift")).contents[0].content
    return {"notes": notes, "request_id": ctx.request_id, "server": ctx.fastmcp.name}

  async with Client(ops.mcp, progress_handler=on_progress) as client:
    schema = next(t for t in await client.list_tools() if t.name == "weekly_report").input_schema
    rows = (await client.call_tool("weekly_report", {})).structured_content
  async with Client(server, log_handler=on_log) as client:
    brief_result = (await client.call_tool("brief", {})).data
  return schema, rows, progress, logs, brief_result


@lab("mcp.context", "Context: logging, progress, and the server's own data inside a tool", calls="No model calls")
def context():
  """The ops server's weekly report reporting progress café by café, and a tool that logs to the client, reads its
  own server's resource, and reports the request id."""
  schema, rows, progress, logs, brief_result = run(context_tour())
  return [
    table(["Progress the client received"], [[p] for p in progress]),
    table(
      ["Café", "revenue"],
      [[r["cafe"], r["revenue_rupees"]] for r in (rows.get("result") if isinstance(rows, dict) else rows) or []],
    ),
    table(
      ["What", "Result"],
      [
        ["weekly_report's input schema", json.dumps(schema)],
        ["client logs", "; ".join(logs)],
        ["brief", json.dumps(brief_result, ensure_ascii=False)],
      ],
    ),
    callout(
      "Ask for ctx: Context and it never appears in the schema. ctx.report_progress sends progress the client can "
      "draw as a bar; ctx.info/warning send logs to the client (deprecated for clients in the 2026-07-28 spec: prefer "
      "progress for users and server-side logging for operations). A tool can read the server's own resources and "
      "prompts, and see the request id for tracing.",
      "decision",
      "Context",
    ),
  ] + [
    example("mcp.l08.1", "ctx: Context, not in the schema", context_tour, "", [code(json.dumps(schema), "json")]),
    example("mcp.l08.2", "Logs to the client", context_tour, "", [table(["log"], [[x] for x in logs])]),
    example(
      "mcp.l08.3",
      "report_progress",
      context_tour,
      "nimbus/mcp/ops.py: weekly_report.",
      [table(["progress"], [[p] for p in progress])],
    ),
    example("mcp.l08.4", "The server's own resources", context_tour, "", [code(brief_result["notes"], "text")]),
    example(
      "mcp.l08.5",
      "Request info",
      context_tour,
      "",
      [table(["request_id", "server"], [[brief_result["request_id"], brief_result["server"]]])],
    ),
  ]


# ── Lab 5: elicitation ─────────────────────────────────────────────────────────
def scripted_user(replies: list):
  """An elicitation handler that answers from a script (a real app shows a form)."""
  from fastmcp.client.elicitation import ElicitResult

  asked = []

  async def handler(message, response_type, params, context):
    reply = replies.pop(0)
    asked.append((message, reply))
    if reply in ("decline", "cancel"):
      return ElicitResult(action=reply)
    field = next(iter(params.requested_schema["properties"]))
    return response_type(**{field: reply})

  return handler, asked


def booking_server():
  from mcp.types import ElicitRequest, ElicitRequestFormParams, InputRequiredResult

  mcp = FastMCP("Tables")

  def ask(key, message, properties, request_state=None):
    form = ElicitRequestFormParams(
      message=message, requested_schema={"type": "object", "properties": properties, "required": list(properties)}
    )
    return InputRequiredResult(
      result_type="input_required",
      request_state=request_state,
      input_requests={key: ElicitRequest(method="elicitation/create", params=form)},
    )

  @mcp.tool
  async def book_table(cafe: str, ctx: Context) -> str | InputRequiredResult:
    """Book a table: asks for the party size, then a time."""
    answers = ctx.input_responses  # None on the first round
    if answers is None:
      return ask("party", f"How many people at {cafe}?", {"people": {"type": "integer", "minimum": 1, "maximum": 12}})
    if "party" in answers:
      if answers["party"].action != "accept":
        return "Booking cancelled."
      people = answers["party"].content["people"]
      return ask(
        "slot",
        f"Which time for {people}?",
        {"time": {"type": "string", "enum": ["18:00", "19:30", "21:00"]}},
        request_state=str(people),
      )
    if answers["slot"].action != "accept":
      return "Booking cancelled."
    return (
      f"Booked a table for {ctx.request_state} at {cafe}, {answers['slot'].content['time']}."  # request_state carried
    )

  return mcp


async def elicitation_tour():
  import nimbus.mcp.server as support

  out = {}
  handler, asked = scripted_user([4, "19:30"])
  async with Client(booking_server(), elicitation_handler=handler) as client:
    out["two questions, one call"] = ((await client.call_tool("book_table", {"cafe": "Indiranagar"})).data, list(asked))
  handler, asked = scripted_user(["decline"])
  async with Client(booking_server(), elicitation_handler=handler) as client:
    out["declined"] = ((await client.call_tool("book_table", {"cafe": "Bandra"})).data, list(asked))
  for mode, reply in (("auto", True), ("legacy", False)):
    support.store.orders["NC-55022"]["status"] = "processing"  # a fresh order each time
    handler, asked = scripted_user([reply])
    async with Client(support.mcp, mode=mode, elicitation_handler=handler) as client:
      out[f"cancel_order, {mode} era ({client.protocol_version})"] = (
        (await client.call_tool("cancel_order", {"order_id": "NC-55022"})).data,
        list(asked),
      )
  return out


@lab("mcp.elicitation", "Elicitation: a tool asks the user, mid-call", calls="No model calls")
def elicitation():
  """A table booking that asks two questions (with what it learned carried between rounds), a declined booking,
  and the Nimbus server's cancel_order asking for confirmation in both protocol eras."""
  out = run(elicitation_tour())
  return [
    table(
      ["Call", "asked the user", "result"],
      [[k, "; ".join(f"{q} → {a}" for q, a in v[1]), v[0]] for k, v in out.items()],
    ),
    callout(
      "Some information only the user can give. In the modern protocol the tool is a guard: it returns "
      "InputRequiredResult ('I need X'), the client asks the user and calls the tool again with the answer, and "
      "request_state (sealed by FastMCP) carries what was already learned, so the server holds nothing between "
      "rounds. The client's elicitation_handler shows the question and returns accept, decline or cancel. In the "
      "legacy era ctx.elicit() pauses mid-tool; cancel_order checks the era and supports both.",
      "decision",
      "Elicitation",
    ),
  ] + [
    example(
      "mcp.l09.1",
      "Why: only the user knows",
      booking_server,
      "",
      [table(["asked"], [[q] for q, _ in out["two questions, one call"][1]])],
    ),
    example("mcp.l09.2", "The guard pattern", booking_server, "", [code(out["two questions, one call"][0], "text")]),
    example("mcp.l09.3", "request_state", booking_server, "", [code(out["two questions, one call"][0], "text")]),
    example(
      "mcp.l09.4", "The client's elicitation_handler", scripted_user, "", [table(["declined"], [[out["declined"][0]]])]
    ),
    example(
      "mcp.l09.5",
      "Legacy: ctx.elicit()",
      elicitation_tour,
      "nimbus/mcp/server.py: cancel_order.",
      [table(["era", "result"], [[k, v[0]] for k, v in out.items() if "cancel_order" in k])],
    ),
  ]


# ── Lab 6: sampling ────────────────────────────────────────────────────────────
def client_model_handler(allow: bool = True):
  """The CLIENT's side: run the server's request on our own model, with our limits (or refuse)."""

  async def handler(messages, params, context) -> str:
    if not allow:
      raise ValueError("This app does not allow servers to use its model.")
    model = get_chat_model(max_tokens=min(params.max_tokens, 100), temperature=params.temperature or 0.2)  # WE choose
    conversation = [("system", params.system_prompt or "")] + [(m.role, m.content.text) for m in messages]
    return (await model.ainvoke(conversation)).text

  return handler


async def sampling_tour():
  from nimbus.mcp import ops

  out = {}
  async with Client(ops.mcp, sampling_handler=client_model_handler()) as client:
    out["Indiranagar digest (the client's model)"] = (
      await client.call_tool("review_digest", {"cafe": "Indiranagar"})
    ).data
    out["Hauz Khas (no reviews: no model needed)"] = (
      await client.call_tool("review_digest", {"cafe": "Hauz Khas"})
    ).data
  async with Client(ops.mcp, sampling_handler=client_model_handler(allow=False)) as client:
    try:
      await client.call_tool("review_digest", {"cafe": "Indiranagar"})
      out["a client that refuses"] = "it ran"
    except Exception as error:  # noqa: BLE001 - the refusal stops the call
      out["a client that refuses"] = f"{type(error).__name__}: {error}"[:200]
  await _close_models()
  return out


@lab(
  "mcp.sampling",
  "Sampling: a server borrows the client's language model",
  uses_model=True,
  calls="1 call to Mistral (the client's model)",
)
def sampling():
  """The ops server's review_digest: the server has no model; it asks the caller's model to summarise a café's low
  reviews, and a client can refuse."""
  out = run(sampling_tour())
  return [
    table(["Call", "Result"], list(out.items())),
    callout(
      "A server can call a model itself (its key, its bill) or use SAMPLING: ask the caller's model, on their "
      "provider, key, bill and privacy rules. In the modern protocol it's the same guard pattern as elicitation: "
      "the tool returns a sampling request, the client runs it and calls again with the completion. The client's "
      "sampling_handler decides which model and how many tokens, or refuses. The ops server offers a 'smart' digest "
      "without owning any model.",
      "decision",
      "Sampling",
    ),
  ] + [
    example(
      "mcp.l10.1",
      "Its own model, or the caller's",
      sampling_tour,
      "nimbus/mcp/ops.py: review_digest.",
      [table(["call", "result"], list(out.items())[:1])],
    ),
    example("mcp.l10.2", "The guard pattern again", sampling_tour, "", [answer(next(iter(out.values())), "digest")]),
    example(
      "mcp.l10.3",
      "The client's sampling_handler decides",
      client_model_handler,
      "",
      [table(["refusing client"], [[out["a client that refuses"]]])],
    ),
    example("mcp.l10.4", "Smart tools without a model", sampling_tour, "", [table(["server's model calls"], [[0]])]),
  ]


# ── Lab 7: lifespan and dependency injection ───────────────────────────────────
def lifespan_server(events: list):
  import sqlite3
  from contextlib import asynccontextmanager

  from fastmcp.dependencies import CallArgument, CurrentContext, Depends
  from fastmcp.server.lifespan import lifespan

  from nimbus.support.tools import SupportStore

  @lifespan
  async def database(server):
    events.append("lifespan: open database")
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.execute("create table orders (id text primary key, customer text, total integer, status text)")
    db.executemany(
      "insert into orders values (?, ?, ?, ?)",
      [(o["order_id"], o["customer_id"], o["total"], o["status"]) for o in SupportStore().orders.values()],
    )
    try:
      yield {"db": db}  # becomes ctx.lifespan_context
    finally:
      db.close()
      events.append("lifespan: close database")

  mcp = FastMCP("Nimbus DB", lifespan=database)

  def current_staff_member() -> str:
    return "ravi@nimbus"  # from the authenticated request in MCP Module 4: never from the model

  @asynccontextmanager
  async def audit_log():
    lines: list[str] = []
    events.append("dependency: open audit log")
    try:
      yield lines
    finally:
      events.append(f"dependency: flush {len(lines)} line(s)")

  def load_order(order_id: str = CallArgument(), ctx: Context = CurrentContext()) -> dict:  # noqa: B008
    row = ctx.lifespan_context["db"].execute("select * from orders where id = ?", (order_id.upper(),)).fetchone()
    if row is None:
      raise ToolError(f"No order {order_id}")
    return dict(zip(("id", "customer", "total", "status"), row, strict=True))

  @mcp.tool
  def count_orders(ctx: Context) -> int:
    """How many orders are in the database."""
    return ctx.lifespan_context["db"].execute("select count(*) from orders").fetchone()[0]

  @mcp.tool
  def flag_order(
    order_id: str,
    order: dict = Depends(load_order),  # noqa: B008 - FastMCP's injection works through defaults
    staff: str = Depends(current_staff_member),  # noqa: B008
    audit: list = Depends(audit_log),  # noqa: B008
    same_audit: list = Depends(audit_log),  # noqa: B008 - the same dependency twice: one instance per call
  ) -> str:
    """Flag an order for a teammate to review."""
    audit.append(f"{staff} flagged {order['id']}")
    return (
      f"{order['id']} (₹{order['total']}, {order['status']}) flagged by {staff}; "
      f"one audit log per call: {audit is same_audit}"
    )

  return mcp


async def lifespan_tour():
  events: list[str] = []
  mcp = lifespan_server(events)
  out = {}
  async with Client(mcp) as client:
    out["count_orders twice"] = [(await client.call_tool("count_orders", {})).data for _ in range(2)]
    out["events after two calls"] = list(events)
    tools = {t.name: t for t in await client.list_tools()}
    out["flag_order's arguments (what the model sees)"] = list(tools["flag_order"].input_schema["properties"])
    events.clear()
    out["flag_order"] = (await client.call_tool("flag_order", {"order_id": "nc-78123"})).data
    out["events during the call"] = list(events)
    result = await client.call_tool(
      "flag_order", {"order_id": "NC-78123", "staff": "attacker@evil"}, raise_on_error=False
    )
    out["a client sending staff=..."] = " ".join(result.content[0].text.split())[:200]
  out["after disconnecting"] = events[-1]
  return out


@lab(
  "mcp.lifespan",
  "Lifespan and dependency injection: shared resources, values the model can't set",
  calls="No model calls",
)
def lifespan_lab():
  """A server that opens its database once, tools that share it, injected values (the order the call is about, the
  staff member, a per-call audit log), and a client that tries to set an injected value."""
  out = run(lifespan_tour())
  return [
    table(
      ["What", "Result"],
      [[k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v] for k, v in out.items()],
    ),
    callout(
      "lifespan opens shared resources once when the server starts and closes them when it stops; tools reach them "
      "through ctx.lifespan_context. Depends() injects values that never appear in the schema: a per-call audit log "
      "(resolved once per call even when asked for twice, cleaned up after), the order a call is about "
      "(CallArgument), the staff member (from authentication in Module 4). A client can't override an injected value, "
      "even by sending an argument with its name.",
      "decision",
      "Lifespan and DI",
    ),
  ] + [
    example(
      "mcp.l11.1", "lifespan", lifespan_server, "", [table(["events"], [[e] for e in out["events after two calls"]])]
    ),
    example(
      "mcp.l11.2",
      "ctx.lifespan_context",
      lifespan_server,
      "",
      [table(["count_orders"], [[str(out["count_orders twice"])]])],
    ),
    example(
      "mcp.l11.3",
      "Depends(): not in the schema",
      lifespan_server,
      "",
      [table(["arguments"], [[str(out["flag_order's arguments (what the model sees)"])]])],
    ),
    example(
      "mcp.l11.4",
      "Per-request dependencies with cleanup",
      lifespan_server,
      "",
      [table(["events"], [[e] for e in out["events during the call"]])],
    ),
    example("mcp.l11.5", "CallArgument()", lifespan_server, "", [code(out["flag_order"], "text")]),
    example(
      "mcp.l11.6",
      "Injected values can't be overridden",
      lifespan_tour,
      "",
      [code(out["a client sending staff=..."], "text")],
    ),
  ]


# ── Lab 8: middleware ──────────────────────────────────────────────────────────
def middleware_server(*middleware):
  calls = {"price": 0}
  mcp = FastMCP("Nimbus Middleware")

  @mcp.tool
  def menu_price(item: str) -> int:
    """Price of a café drink in rupees."""
    calls["price"] += 1
    time.sleep(0.2)  # a slow query
    return PRICES[item]

  @mcp.tool(tags={"internal"})
  def staff_discount_code() -> str:
    """Today's staff discount code (internal)."""
    return "STAFF-77"

  @mcp.tool
  async def who_am_i(ctx: Context) -> str:
    """Who the server thinks is calling."""
    return await ctx.get_state("caller") or "unknown"

  for m in middleware:
    mcp.add_middleware(m)
  return mcp, calls


def middleware_classes():
  from fastmcp.server.middleware import Middleware, MiddlewareContext

  trace: list[str] = []

  class Announce(Middleware):
    def __init__(self, label):
      self.label = label

    async def on_message(self, context: MiddlewareContext, call_next):
      trace.append(f"{self.label} → {context.method}")
      result = await call_next(context)
      trace.append(f"{self.label} ← {context.method}")
      return result

  class HideInternal(Middleware):
    async def on_list_tools(self, context, call_next):
      return [t for t in await call_next(context) if "internal" not in t.tags]  # hide it from the list...

    async def on_call_tool(self, context, call_next):
      tool = await context.fastmcp_context.fastmcp.get_tool(context.message.name)
      if "internal" in tool.tags:
        raise ToolError("Unknown tool")  # ...AND refuse to run it: hiding alone isn't security
      return await call_next(context)

  class NormaliseAndStamp(Middleware):
    async def on_call_tool(self, context, call_next):
      if "item" in (context.message.arguments or {}):
        context.message.arguments["item"] = context.message.arguments["item"].strip().lower()  # request in
      result = await call_next(context)
      result.meta = {**(result.meta or {}), "served_by": "nimbus-mcp-1"}  # result out
      return result

  class IdentifyCaller(Middleware):
    async def on_call_tool(self, context, call_next):
      await context.fastmcp_context.set_state("caller", "front-desk-tablet-3")  # worked out once, read by tools
      return await call_next(context)

  return trace, Announce, HideInternal, NormaliseAndStamp, IdentifyCaller


async def middleware_tour():
  from fastmcp.server.middleware.caching import ResponseCachingMiddleware
  from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware

  trace, Announce, HideInternal, NormaliseAndStamp, IdentifyCaller = middleware_classes()
  out = {}
  mcp, _ = middleware_server(Announce("A"), Announce("B"))
  async with Client(mcp) as client:
    trace.clear()
    await client.call_tool("menu_price", {"item": "cold brew"})
    out["order"] = list(trace)
  mcp, _ = middleware_server(HideInternal())
  async with Client(mcp) as client:
    out["tools listed"] = [t.name for t in await client.list_tools()]
    out["calling the hidden tool"] = (
      (await client.call_tool("staff_discount_code", {}, raise_on_error=False)).content[0].text
    )
  mcp, _ = middleware_server(NormaliseAndStamp(), IdentifyCaller())
  async with Client(mcp) as client:
    result = await client.call_tool("menu_price", {"item": "  Oat Latte "})
    out["'  Oat Latte ' →"] = f"{result.data}, meta served_by={result.meta.get('served_by')}"
    out["who_am_i"] = (await client.call_tool("who_am_i", {})).data
  mcp, calls = middleware_server(ResponseCachingMiddleware())
  async with Client(mcp) as client:
    times = []
    for _ in range(2):
      started = time.perf_counter()
      await client.call_tool("menu_price", {"item": "cold brew"})
      times.append(round(time.perf_counter() - started, 2))
  out["caching: seconds per call"] = f"{times}; the tool ran {calls['price']}×"
  mcp, _ = middleware_server(RateLimitingMiddleware(max_requests_per_second=0.5, burst_capacity=5))
  async with Client(mcp) as client:
    results = []
    for _ in range(4):
      try:
        await client.call_tool("who_am_i", {})
        results.append("ok")
      except Exception as error:  # noqa: BLE001
        results.append(type(error).__name__)
  out["rate limit: 4 calls"] = results
  return out


@lab("mcp.middleware", "Server middleware: code around every request", calls="No model calls")
def middleware():
  """Two announcing middleware (first in, last out), an internal tool hidden and refused, a request normalised and a
  result stamped, request state passed to a tool, response caching and rate limiting."""
  out = run(middleware_tour())
  return [
    table(
      ["What", "Result"],
      [[k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v] for k, v in out.items()],
    ),
    callout(
      "Middleware wraps every request: A → B → handler → B → A, so order matters. Operation hooks (on_call_tool, "
      "on_list_tools, ...) time, audit and block calls; hiding a tool from the list isn't security unless the call "
      "is refused too. Middleware can change requests on the way in and results on the way out, and leave request "
      "state for tools. Built-ins cache responses and rate-limit (it counts every MCP request, connecting included). "
      "The Nimbus gateway runs an audit log, rate limiting and caching.",
      "decision",
      "Middleware",
    ),
  ] + [
    example(
      "mcp.l12.1",
      "The pipeline: first in, last out",
      middleware_classes,
      "",
      [table(["trace"], [[t] for t in out["order"]])],
    ),
    example(
      "mcp.l12.2",
      "Operation hooks",
      middleware_classes,
      "nimbus/mcp/gateway.py: Audit.",
      [code(json.dumps(out["tools listed"]), "json")],
    ),
    example(
      "mcp.l12.3",
      "Hide AND block",
      middleware_classes,
      "",
      [table(["listed", "calling it"], [[str(out["tools listed"]), out["calling the hidden tool"]]])],
    ),
    example(
      "mcp.l12.4", "Change requests and results", middleware_classes, "", [code(out["'  Oat Latte ' →"], "text")]
    ),
    example("mcp.l12.5", "Request state", middleware_classes, "", [code(out["who_am_i"], "text")]),
    example(
      "mcp.l12.6",
      "Caching and rate limiting",
      middleware_tour,
      "",
      [table(["caching", "rate limit"], [[out["caching: seconds per call"], str(out["rate limit: 4 calls"])]])],
    ),
  ]


# ── Lab 9: composition and proxies ─────────────────────────────────────────────
async def composition_tour():
  import os

  from fastmcp.client.transports import PythonStdioTransport
  from fastmcp.server import create_proxy
  from fastmcp.server.transforms import ToolTransform
  from fastmcp.tools.tool_transform import ArgTransformConfig, ToolTransformConfig

  import nimbus.mcp.server as support
  from nimbus.mcp.gateway import build_gateway

  out = {}
  staff, _ = build_gateway()
  public, _ = build_gateway(public=True)
  async with Client(staff) as client:
    out["staff gateway tools"] = [t.name for t in await client.list_tools()]
    out["staff gateway resources"] = [str(r.uri) for r in await client.list_resources()]
  async with Client(public) as client:
    out["public gateway tools"] = [t.name for t in await client.list_tools()]
    out["public calling ops_weekly_numbers"] = (
      (await client.call_tool("ops_weekly_numbers", {"cafe": "Bandra"}, raise_on_error=False)).content[0].text
    )
  child = FastMCP("Child")
  gateway = FastMCP("Live link")
  gateway.mount(child, namespace="kid")

  @child.tool
  def special_of_the_day() -> str:
    """Today's special (added after mounting)."""
    return "Coorg Dawn pour-over"

  async with Client(gateway) as client:
    out["a tool added to the child later"] = [t.name for t in await client.list_tools()]
  proxied = FastMCP("Proxy gateway")
  target = PythonStdioTransport(
    support.__file__, env={**os.environ, "FASTMCP_SHOW_SERVER_BANNER": "false", "FASTMCP_LOG_LEVEL": "WARNING"}
  )
  proxied.mount(create_proxy(target), namespace="shop")  # another server (a subprocess) served as ours
  async with Client(proxied) as client:
    out["proxied (stdio subprocess) get_order"] = (
      await client.call_tool("shop_get_order", {"order_id": "NC-12093"})
    ).structured_content.get("status")
  friendly = FastMCP("Friendly")
  friendly.mount(support.mcp)
  friendly.add_transform(
    ToolTransform(
      {
        "cafe_menu_item": ToolTransformConfig(
          name="how_much_is",
          description="How much a café drink or snack costs, and whether it's vegan.",
          arguments={"name": ArgTransformConfig(name="menu_item", description="e.g. 'Oat Latte'")},
        )
      }
    )
  )
  async with Client(friendly) as client:
    tool = next(t for t in await client.list_tools() if t.name == "how_much_is")
    renamed = await client.call_tool("how_much_is", {"menu_item": "Oat Latte"})
    out["ToolTransform"] = f"{tool.name}({list(tool.input_schema['properties'])}) → {renamed.structured_content}"
  return out


@lab("mcp.composition", "Composition and proxies: one server out of many", calls="No model calls")
def composition():
  """The Nimbus gateway (support + ops, prefixed), its public variant without internal tools, a live link to a child,
  a proxy for a stdio server, and a tool renamed without touching its code."""
  out = run(composition_tour())
  return [
    table(
      ["What", "Result"],
      [[k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v] for k, v in out.items()],
    ),
    callout(
      "mount(child, namespace=...) serves a child's tools, resources and prompts through a gateway, prefixed, and the "
      "link is live. create_proxy() serves another server (a subprocess or a remote URL) as if it were yours: expose "
      "a stdio-only server over HTTP, or put your middleware in front of a third-party one. Tags give one codebase a "
      "public gateway without the internal tools, and ToolTransform renames and reshapes a tool without touching its "
      "code. Nimbus runs one gateway for staff and one for customers.",
      "decision",
      "Composition",
    ),
  ] + [
    example(
      "mcp.l13.1",
      "mount(child, namespace=...)",
      composition_tour,
      "nimbus/mcp/gateway.py: build_gateway.",
      [table(["tools"], [[t] for t in out["staff gateway tools"]])],
    ),
    example(
      "mcp.l13.2", "A live link", composition_tour, "", [code(str(out["a tool added to the child later"]), "text")]
    ),
    example(
      "mcp.l13.3",
      "create_proxy()",
      composition_tour,
      "",
      [code(str(out["proxied (stdio subprocess) get_order"]), "text")],
    ),
    example(
      "mcp.l13.4",
      "Visibility by tag",
      composition_tour,
      "",
      [
        table(
          ["public tools", "calling an internal one"],
          [[", ".join(out["public gateway tools"]), out["public calling ops_weekly_numbers"]]],
        )
      ],
    ),
    example("mcp.l13.5", "ToolTransform", composition_tour, "", [code(out["ToolTransform"], "text")]),
  ]


# ── Production path: the policy as a resource ──────────────────────────────────
def evaluate_policy_context() -> list:
  from langchain.mcp import MCPAdapter

  import nimbus.mcp.server as support
  from nimbus.support.assistant import SYSTEM
  from nimbus.support.tool_eval import ToolReport, load_requests, score

  requests = load_requests(PROJECT_DIR / "data" / "eval" / "policy_questions.json")

  async def run_all():
    from nimbus.support.assistant import answer_with_tools_async

    model = get_chat_model()
    reports = []
    async with MCPAdapter(support.mcp) as adapter, Client(support.mcp) as client:
      tools = [t for t in await adapter.list_tools() if t.name != "cancel_order"]  # answering, not acting
      policy = (await client.read_resource("nimbus://policies/refunds"))[0].text  # the app reads it once
      for name, system in (
        ("tools only", SYSTEM),
        ("tools + policy resource in context", f"{SYSTEM}\n\nRefund policy (from nimbus://policies/refunds): {policy}"),
      ):
        rows = []
        for request in requests:
          run_ = await answer_with_tools_async(model, tools, request["request"], system=system)
          rows.append((request, run_, score(request, run_)))
        reports.append(ToolReport(name, len(requests), rows))
    await _close_models()
    return reports

  return asyncio.run(run_all())


@lab(
  "mcp.policy-eval",
  "Evaluate: the refund policy as a resource in context, or tools only",
  uses_model=True,
  calls="~30 calls to Mistral",
)
def policy_eval():
  """Eight policy questions (damaged bags, packed orders, refund timing, café drinks) answered with the Nimbus
  server's tools, with and without the refund-policy resource read into the system prompt."""
  started = time.perf_counter()
  reports = evaluate_policy_context()
  seconds = time.perf_counter() - started
  rows = [r.summary() for r in reports]
  misses = [
    [r.variant, req["id"], req["request"], run_.answer[:220]]
    for r in reports
    for req, run_, s in r.rows
    if not s["success"]
  ]
  tools_only, with_policy = rows
  return [
    scorecard(
      rows,
      [
        Metric("success", "right answers"),
        Metric("rounds", "model calls", "lower", "text"),
        Metric("tokens", "tokens", "lower", "int"),
        Metric("latency_p50_ms", "latency p50", "lower", "ms"),
      ],
      title=f"8 policy questions · {seconds:.0f} s",
    ),
    table(
      ["Variant", "id", "question", "answer"], misses or [["—", "—", "every answer right", ""]], title="Every miss"
    ),
    callout(
      f"With the policy read into context the agent answered {with_policy['success']:.0%}, with tools only "
      f"{tools_only['success']:.0%}, at {with_policy['tokens']:,} vs {tools_only['tokens']:,} tokens. No tool returns "
      "the policy's text, so with tools only the model answered from what it assumed (damaged-bag steps, refund "
      "timing, café drinks). The app decides what context the model needs: the policy is a resource the support "
      "agents read into their prompt (MCP Module 3 reads it fresh on every call).",
      "decision",
      "The policy goes into context",
    ),
  ]
