"""LangChain · Module 1 labs: chat models, messages, multimodal input, prompts, structured output, invocation,
errors, profiles and reasoning, and local models. Each lab calls real models and measures what it shows."""

import base64
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated, TypedDict

import httpx
import pymupdf
from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.tracers.context import collect_runs

from academy.labs.runtime import (
  Input,
  Metric,
  answer,
  callout,
  code,
  columns,
  example,
  images,
  lab,
  md,
  scorecard,
  table,
)
from nimbus.config import PROJECT_DIR, get_settings
from nimbus.llm import ConversationStore, Usage, classify, get_chat_model, model_name, supports
from nimbus.llm.models import KNOWN_PROFILES
from nimbus.support.evaluation import field_matches, load_tickets
from nimbus.support.prompts import CATEGORIES, EXAMPLES, RULES, SYSTEM_V1, TRIAGE_PROMPTS
from nimbus.support.schema import Category, Ticket, TicketTriage, Urgency
from nimbus.support.triage import Triager

DATA = PROJECT_DIR / "data" / "nimbus"
PROFILE_KEYS = (
  "max_input_tokens",
  "image_inputs",
  "pdf_inputs",
  "reasoning_output",
  "tool_calling",
  "structured_output",
)


def _ms(started: float) -> int:
  return round((time.perf_counter() - started) * 1000)


@lru_cache
def _tickets():
  return {lt.ticket.id: lt for lt in load_tickets(get_settings().eval_tickets)}


def _ticket_options() -> list[str]:
  return [f"{t['id']} · {t['text']}" for t in json.loads(get_settings().eval_tickets.read_text())]


# ── Lab 1: the chat model ───────────────────────────────────────────────────────
# Each learning point is a small function: the page shows its body as the code to learn from, and the lab runs it,
# so what you read is exactly what produced the output next to it.


def create_model():
  from langchain.chat_models import init_chat_model

  # The API key comes from the MISTRAL_API_KEY environment variable (ai-course/.env), never from code
  model = init_chat_model("ministral-14b-latest", model_provider="mistralai", temperature=0)
  return model


def create_models_for_other_providers():
  from langchain.chat_models import init_chat_model

  # Same function, same methods afterwards: only the model name and provider change
  groq = init_chat_model("openai/gpt-oss-20b", model_provider="groq")  # needs GROQ_API_KEY
  local = init_chat_model("qwen3:1.7b", model_provider="ollama")  # a model on your own computer
  return groq, local


def call_the_model(model, prompt):
  reply = model.invoke(prompt)
  return reply


def read_the_answer(reply):
  content = reply.content  # what the provider sent: a str here, a LIST of blocks for images or reasoning
  text = reply.text  # always just the text, as a str: use this for the answer
  return content, text


def a_reply_with_blocks():
  from langchain_core.messages import AIMessage

  # What a reasoning model sends back: content is a list of blocks, .text keeps only the text ones
  reply = AIMessage(
    content=[
      {"type": "reasoning", "reasoning": "91 = 7 × 13, so it has divisors other than 1 and itself."},
      {"type": "text", "text": "No, 91 is not prime."},
    ]
  )
  return reply.content, reply.text


def read_tokens_and_metadata(reply):
  usage = reply.usage_metadata  # what you're billed for: input, output and total tokens
  model_used = reply.response_metadata["model_name"]  # which model actually answered
  stop_reason = reply.response_metadata["finish_reason"]  # "stop" = finished; "length" = cut off by max_tokens
  return usage, model_used, stop_reason


def compare_temperatures(prompt, runs):
  from langchain.chat_models import init_chat_model

  answers = {}
  for temperature in (0.0, 1.0):
    model = init_chat_model("ministral-14b-latest", model_provider="mistralai", temperature=temperature)
    answers[temperature] = [model.invoke(prompt) for _ in range(runs)]
  return answers


@lab(
  "lc.chat-model",
  "Chat models: invoke, the AIMessage, tokens and temperature",
  uses_model=True,
  calls="1 + 2 × runs calls to Mistral (7 by default)",
  inputs=[
    Input(
      "prompt",
      "Prompt",
      "text",
      default="Suggest a name for a new cold brew at Nimbus Coffee. Reply with the name only.",
    ),
    Input("runs", "Runs per temperature", "number", default=3, min=2, max=4),
  ],
)
def chat_model(prompt: str, runs: int):
  """Call the model a few times at temperature 0 and 1: what comes back, what it cost, how much it varies."""
  model = create_model()
  groq, local = create_models_for_other_providers()
  started = time.perf_counter()
  reply = call_the_model(model, prompt)
  latency = _ms(started)
  content, text = read_the_answer(reply)
  block_content, block_text = a_reply_with_blocks()
  usage, model_used, stop_reason = read_tokens_and_metadata(reply)
  with collect_runs() as collected:  # records each model call with its start and end time, for the latency column
    replies = compare_temperatures(prompt, runs)
  timings = iter((run.end_time - run.start_time).total_seconds() * 1000 for run in collected.traced_runs)
  results = {t: [(m, next(timings)) for m in ms] for t, ms in replies.items()}
  answers = {t: [m.text.strip() for m in ms] for t, ms in replies.items()}
  anatomy = [
    ["type(reply)", type(reply).__name__],
    ["reply.content", repr(reply.content)[:160]],
    ["reply.text", repr(reply.text)[:160]],
    ["reply.usage_metadata", json.dumps(reply.usage_metadata)],
    [
      "reply.response_metadata",
      json.dumps({k: reply.response_metadata.get(k) for k in ("model_name", "finish_reason")}),
    ],
    ["reply.id", str(reply.id)],
  ]
  rows = []
  for temperature, measured in results.items():
    texts = [m.text.strip() for m, _ in measured]
    rows.append(
      {
        "variant": f"temperature {temperature:g}",
        "distinct": len(set(texts)),
        "output_tokens": sum(m.usage_metadata["output_tokens"] for m, _ in measured) / len(measured),
        "latency_ms": sum(ms for _, ms in measured) / len(measured),
      }
    )
  cold, hot = rows
  summary = [
    table(["Attribute", "Value"], anatomy, title="What invoke() returns: an AIMessage, not a string"),
    columns(
      [
        answer("\n".join(f"{i}. {m.text.strip()}" for i, (m, _) in enumerate(measured, 1)), f"temperature {t:g}")
        for t, measured in results.items()
      ],
      title=f"The same prompt, {runs} times each",
    ),
    scorecard(
      rows,
      [
        Metric("distinct", "distinct answers", "none", "int", "How many different answers came back"),
        Metric("output_tokens", "output tokens", "lower", "int"),
        Metric("latency_ms", "latency", "lower", "ms"),
      ],
    ),
    callout(
      f"Temperature 0 gave {cold['distinct']} distinct answer(s) in {runs} runs, temperature 1 gave {hot['distinct']}. "
      "Use 0 for extraction, classification and anything you test against labels; raise it only where variety is "
      "the point (names, marketing copy). Even at 0, providers don't promise identical outputs: test with labels, "
      "not exact strings.",
      "decision",
      "Which temperature",
    ),
  ]
  return summary + [
    example(
      "lc.l01.1",
      "Create a model with init_chat_model",
      create_model,
      "`init_chat_model(model, model_provider=...)` returns a chat model object. Settings like `temperature`, "
      "`timeout` and `max_retries` go in the same call. Nothing is sent to the provider yet.",
      [
        table(
          ["Expression", "Value"],
          [
            ["type(model).__name__", type(model).__name__],
            ["model.model", model.model],
            ["model.temperature", str(model.temperature)],
          ],
        )
      ],
    ),
    example(
      "lc.l01.1",
      "The same function for any provider",
      create_models_for_other_providers,
      "Each provider has its own package (`langchain-mistralai`, `langchain-groq`, `langchain-ollama`), and "
      "`init_chat_model` picks the right class. Your code after this line doesn't change.",
      [table(["Variable", "Class"], [["groq", type(groq).__name__], ["local", type(local).__name__]])],
    ),
    example(
      "lc.l01.2",
      "invoke() returns an AIMessage, not a string",
      call_the_model,
      "`invoke` sends one request and waits for the whole answer. The result is an **AIMessage** object: the "
      "answer plus everything the provider reported about it.",
      [
        table(
          ["Expression", "Value"],
          [
            ["prompt", prompt],
            ["type(reply).__name__", type(reply).__name__],
            ["reply.id", str(reply.id)],
            ["time taken", f"{latency:,} ms"],
          ],
        )
      ],
    ),
    example(
      "lc.l01.3",
      "Reading the answer: .content vs .text",
      read_the_answer,
      "For a plain text answer they look the same. They differ when the reply has several parts:",
      [table(["Expression", "Value"], [["content", repr(content)], ["text", repr(text)]])],
    ),
    example(
      "lc.l01.3",
      "When .content is a list of blocks",
      a_reply_with_blocks,
      "Reasoning models (and replies with images) put a **list of blocks** in `.content`. Code that expects a "
      "string breaks; `.text` still gives you just the answer. So read answers with `.text`.",
      [
        table(
          ["Expression", "Value"],
          [["reply.content", json.dumps(block_content, ensure_ascii=False)], ["reply.text", repr(block_text)]],
        )
      ],
    ),
    example(
      "lc.l01.4",
      "Token usage and response metadata",
      read_tokens_and_metadata,
      "Every reply reports its token usage: that's what you pay for and what counts against rate limits. "
      "`response_metadata` has the provider's details, including the exact model that answered.",
      [
        table(
          ["Variable", "Value"],
          [["usage", json.dumps(usage)], ["model_used", model_used], ["stop_reason", stop_reason]],
        )
      ],
    ),
    example(
      "lc.l01.5",
      "How temperature changes the output",
      compare_temperatures,
      f"The same prompt, {runs} times at temperature 0 and {runs} times at temperature 1. Temperature controls how "
      "much randomness goes into picking each word.",
      [
        table(
          ["Run", "temperature 0", "temperature 1"],
          [[i + 1, answers[0.0][i], answers[1.0][i]] for i in range(runs)],
        ),
      ],
    ),
  ]


# ── Lab 2: messages ─────────────────────────────────────────────────────────────
def start_a_conversation(model, name):
  from langchain_core.messages import HumanMessage, SystemMessage

  messages = [
    SystemMessage("You are Nimbus Coffee's assistant. Answer in one short sentence."),  # instructions
    HumanMessage(f"Hi, I'm {name}. I love the Nilgiri Sunrise blend."),  # the user
  ]
  reply = model.invoke(messages)  # an AIMessage: the model's turn
  return messages, reply


def ask_with_and_without_history(model, messages, reply):
  from langchain_core.messages import HumanMessage

  question = HumanMessage("What's my name, and which blend do I like?")
  forgetful = model.invoke([messages[0], question])  # a new request without the history: the model can't know
  remembers = model.invoke([*messages, reply, question])  # the app resends the whole conversation
  return question, forgetful, remembers


def use_shorthand():
  from langchain_core.messages import convert_to_messages

  shorthand = [
    ("system", "Be brief."),
    {"role": "user", "content": "Hello"},
    ("ai", "Hi! How can I help?"),
    "And a plain string?",
  ]
  messages = convert_to_messages(shorthand)  # model.invoke(shorthand) does this conversion for you
  return shorthand, messages


def save_and_load(conversation):
  import json

  from langchain_core.messages import messages_from_dict, messages_to_dict

  saved = json.dumps(messages_to_dict(conversation))  # a string: store it in a file or a database column
  loaded = messages_from_dict(json.loads(saved))  # the same message objects, ready to send again
  return saved, loaded


@lab(
  "lc.messages",
  "Messages: roles, statelessness, shorthand formats and saving a conversation",
  uses_model=True,
  calls="3 calls to Mistral",
  inputs=[Input("name", "Your name", "text", default="Priya")],
)
def messages_lab(name: str):
  model = get_chat_model()
  first, reply = start_a_conversation(model, name)
  question, forgetful, remembers = ask_with_and_without_history(model, first, reply)
  history = [*first, reply, question]

  store = ConversationStore(get_settings().conversations_dir)
  store.save("lab-messages", [*history, remembers])
  loaded = store.load("lab-messages")

  shorthand, converted = use_shorthand()
  saved, round_trip = save_and_load([*history, remembers])
  return [
    table(
      ["Role", "Class", "Content"],
      [[m.type, type(m).__name__, m.text] for m in [*history, remembers]],
      title="One conversation: the app keeps the list and resends it every time",
    ),
    columns(
      [
        answer(forgetful.text, "Without the history", {"input_tokens": forgetful.usage_metadata["input_tokens"]}),
        answer(remembers.text, "With the history", {"input_tokens": remembers.usage_metadata["input_tokens"]}),
      ],
      title="The model is stateless: it only knows what this request contains",
    ),
    table(
      ["You write", "LangChain makes"],
      [[repr(s), f"{type(m).__name__}({m.text!r})"] for s, m in zip(shorthand, converted, strict=True)],
      title="Shorthand formats, converted to messages",
    ),
    code(
      json.dumps(messages_to_dict(loaded[:2]), indent=1, ensure_ascii=False), title="Saved as JSON (first two messages)"
    ),
    callout(
      f"Saved {len(loaded)} messages to `.state/conversations/lab-messages.json` and loaded them back: "
      f"{'identical' if loaded == [*history, remembers] else 'DIFFERENT'} to the originals. Memory costs tokens: the "
      f"follow-up sent {remembers.usage_metadata['input_tokens']} input tokens with the history vs "
      f"{forgetful.usage_metadata['input_tokens']} without, and it grows with every turn.",
      "info",
      "Resuming a conversation",
    ),
    example(
      "lc.l02.1",
      "Build a conversation from message objects",
      start_a_conversation,
      "A **SystemMessage** sets the rules, a **HumanMessage** is the user, and the model's reply comes back as an "
      "**AIMessage**. A conversation is just a list of them.",
      [table(["#", "Class", "Content"], [[i, type(m).__name__, m.text] for i, m in enumerate([*first, reply], 1)])],
    ),
    example(
      "lc.l02.2",
      "Models are stateless: memory means resending the list",
      ask_with_and_without_history,
      "Each `invoke` is a new request. Without the earlier messages the model has nothing to go on; with them it "
      "answers correctly, and you pay for every resent token.",
      [
        table(
          ["Variable", "Answer", "Input tokens"],
          [
            ["forgetful", forgetful.text, forgetful.usage_metadata["input_tokens"]],
            ["remembers", remembers.text, remembers.usage_metadata["input_tokens"]],
          ],
        )
      ],
    ),
    example(
      "lc.l02.3",
      "Shorthand formats",
      use_shorthand,
      "Tuples, role dicts and plain strings are converted to message objects. `invoke` accepts them directly.",
      [
        table(
          ["shorthand", "messages"],
          [[repr(x), f"{type(m).__name__}({m.text!r})"] for x, m in zip(shorthand, converted, strict=True)],
        )
      ],
    ),
    example(
      "lc.l02.4",
      "Save a conversation as JSON and load it back",
      save_and_load,
      "`messages_to_dict` turns messages into plain JSON (metadata and token usage included); `messages_from_dict` "
      "rebuilds them. Save after each turn, load to resume.",
      [
        table(
          ["Expression", "Value"],
          [
            ["len(saved)", f"{len(saved):,} characters"],
            ["saved[:120]", saved[:120] + "…"],
            ["loaded == conversation", str(round_trip == [*history, remembers])],
          ],
        )
      ],
    ),
  ]


# ── Lab 3: multimodal input ─────────────────────────────────────────────────────
def _data_uri(path: Path) -> str:
  return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"


def _pdf_page_png(path: Path, page: int = 0, zoom: float = 1.5) -> bytes:
  with pymupdf.open(path) as pdf:
    return pdf[page].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")


MULTIMODAL_CASES = {
  "One photo: read a café receipt": (
    ["receipt_photo.png"],
    "What was bought, and what is the total? List items, then the total.",
  ),
  "Two images: compare them": (
    ["receipt.png", "receipt_photo.png"],
    "Are these the same bill? Compare bill number and total.",
  ),
  "A chart": (["sales_chart.png"], "Which quarter had the highest revenue, and what was the full-year total?"),
  "A scanned PDF page, as an image": (
    ["scanned_notice.pdf"],
    "When is the roastery closed, and when do delayed orders ship?",
  ),
  "Provider-native block (same receipt)": (["receipt_photo.png"], "What is the total on this receipt?"),
}


def send_an_image(model, path):
  import base64

  from langchain_core.messages import HumanMessage

  image = base64.b64encode(path.read_bytes()).decode()  # path = data/nimbus/receipt_photo.png
  message = HumanMessage(
    content=[  # a list of blocks instead of a string
      {"type": "text", "text": "What is the total on this receipt?"},
      {"type": "image", "base64": image, "mime_type": "image/png"},
    ]
  )
  reply = model.invoke([message])
  return reply


def image_by_url_or_bytes(image):
  url_block = {"type": "image", "url": "https://example.com/receipt.png"}  # the provider downloads it
  bytes_block = {"type": "image", "base64": image, "mime_type": "image/png"}  # the image travels in the request
  return url_block, bytes_block


def send_two_images(model, first, second):
  import base64

  from langchain_core.messages import HumanMessage

  message = HumanMessage(
    content=[
      {"type": "text", "text": "Are these the same bill? Compare bill number and total."},
      {"type": "image", "base64": base64.b64encode(first.read_bytes()).decode(), "mime_type": "image/png"},
      {"type": "image", "base64": base64.b64encode(second.read_bytes()).decode(), "mime_type": "image/png"},
    ]
  )
  reply = model.invoke([message])
  return reply


def send_a_provider_native_block(model, path):
  import base64

  from langchain_core.messages import HumanMessage

  data_uri = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
  message = HumanMessage(
    content=[
      {"type": "text", "text": "What is the total on this receipt?"},
      {"type": "image_url", "image_url": data_uri},  # Mistral's own format: passed through as it is
    ]
  )
  reply = model.invoke([message])
  return reply


def send_a_pdf_page(model, pdf_path):
  import base64

  import pymupdf
  from langchain_core.messages import HumanMessage

  with pymupdf.open(pdf_path) as pdf:  # close the file when done
    png = pdf[0].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).tobytes("png")  # page 1 → an image
  message = HumanMessage(
    content=[
      {"type": "text", "text": "When is the roastery closed?"},
      {"type": "image", "base64": base64.b64encode(png).decode(), "mime_type": "image/png"},
    ]
  )
  reply = model.invoke([message])
  return reply


@lru_cache
def _multimodal_examples() -> tuple:
  """The code examples make their own model calls: run them once per server start, then reuse."""
  model = get_chat_model()
  receipt, photo = DATA / "receipt.png", DATA / "receipt_photo.png"
  one = send_an_image(model, photo)
  url_block, bytes_block = image_by_url_or_bytes(base64.b64encode(photo.read_bytes()).decode())
  two = send_two_images(model, receipt, photo)
  native = send_a_provider_native_block(model, photo)
  pdf = send_a_pdf_page(model, DATA / "scanned_notice.pdf")

  def reply_table(reply):
    return table(
      ["Expression", "Value"], [["reply.text", reply.text], ["input tokens", reply.usage_metadata["input_tokens"]]]
    )

  return (
    example(
      "lc.l03.1",
      "Text and an image in one message",
      send_an_image,
      "The message content is a **list of blocks**: a text block and an image block.",
      [reply_table(one)],
    ),
    example(
      "lc.l03.2",
      "An image as base64 or as a URL",
      image_by_url_or_bytes,
      "Base64 puts the bytes in the request. A URL block makes the provider download the image, so it must be "
      "publicly reachable (not sent here: example.com has no receipt).",
      [
        code(
          json.dumps(
            {"url_block": url_block, "bytes_block": {**bytes_block, "base64": bytes_block["base64"][:40] + "…"}},
            indent=1,
          )
        )
      ],
    ),
    example(
      "lc.l03.3",
      "Several images in one message",
      send_two_images,
      "Add one image block per picture; the model sees them together and can compare them.",
      [reply_table(two)],
    ),
    example(
      "lc.l03.4",
      "A provider-native block",
      send_a_provider_native_block,
      "LangChain's standard blocks work with every provider. A provider's own format is passed through "
      "unchanged, so this code only works with that provider.",
      [reply_table(native)],
    ),
    example(
      "lc.l03.5",
      "A PDF page, sent as an image",
      send_a_pdf_page,
      "Render the page to PNG and send it like any image: it works with every vision model.",
      [reply_table(pdf)],
    ),
  )


@lab(
  "lc.multimodal",
  "Multimodal input: images, several images and PDF pages in one message",
  uses_model=True,
  calls="1 call to Mistral (images count as input tokens), plus 4 the first time for the code examples",
  inputs=[
    Input("case", "What to send", "select", default=next(iter(MULTIMODAL_CASES)), options=list(MULTIMODAL_CASES))
  ],
)
def multimodal(case: str):
  files, question = MULTIMODAL_CASES[case]
  model = get_chat_model()
  pictures, blocks = [], []
  for name in files:
    path = DATA / name
    png = _pdf_page_png(path) if path.suffix == ".pdf" else path.read_bytes()
    uri = f"data:image/png;base64,{base64.b64encode(png).decode()}"
    pictures.append({"src": uri, "caption": name + (" (page 1, rendered to PNG)" if path.suffix == ".pdf" else "")})
    if case.startswith("Provider-native"):
      blocks.append({"type": "image_url", "image_url": uri})  # Mistral's own format, passed through unchanged
    else:
      blocks.append({"type": "image", "base64": base64.b64encode(png).decode(), "mime_type": "image/png"})
  message = HumanMessage(content=[{"type": "text", "text": question}, *blocks])
  started = time.perf_counter()
  reply = model.invoke([message])
  shown = [
    {**b, "base64": b["base64"][:40] + "…"} if "base64" in b else {**b, "image_url": b["image_url"][:50] + "…"}
    for b in blocks
  ]
  return [
    images(pictures, title="Sent to the model"),
    answer(reply.text, question, {"input_tokens": reply.usage_metadata["input_tokens"], "latency_ms": _ms(started)}),
    code(
      json.dumps([{"type": "text", "text": question}, *shown], indent=1), title="The message content: a list of blocks"
    ),
    callout(
      f"Images cost input tokens too: this message used {reply.usage_metadata['input_tokens']} input tokens for "
      f"{len(files)} image(s) and one short question. Check the model's profile before sending images "
      f"(`image_inputs` is {supports(model, 'image_inputs')} for {model_name(model)}), and for PDFs render pages to "
      "images: that works with every vision model, while native PDF input is provider-specific. A URL block "
      '(`{"type": "image", "url": "https://..."}`) makes the provider download the image instead.',
      "info",
      "What this shows",
    ),
    *_multimodal_examples(),
  ]


# ── Lab 4: prompt templates ─────────────────────────────────────────────────────
HARD_TICKETS = ("t08", "t10", "t11", "t15", "t17", "t20")


def fill_a_template(ticket):
  from langchain_core.prompts import ChatPromptTemplate

  template = ChatPromptTemplate.from_messages([("system", "You help {company} customers."), ("human", "{question}")])
  value = template.invoke({"company": "Nimbus Coffee", "question": ticket})  # a PromptValue
  messages = value.to_messages()  # what model.invoke(value) would send
  return template, value, messages


def insert_the_history(ticket):
  from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

  template = ChatPromptTemplate.from_messages(
    [("system", "You help Nimbus Coffee customers."), MessagesPlaceholder("history"), ("human", "{question}")]
  )
  history = [("human", "Hi, order NC-90112 again."), ("ai", "Sorry to hear that! What happened?")]
  messages = template.invoke({"history": history, "question": ticket}).to_messages()
  return messages


def fix_a_variable_up_front():
  from langchain_core.prompts import ChatPromptTemplate

  template = ChatPromptTemplate.from_messages([("system", "You help {company} customers."), ("human", "{question}")])
  nimbus = template.partial(company="Nimbus Coffee")  # filled in once, for every call
  return template.input_variables, nimbus.input_variables


def teach_by_example(ticket):
  from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

  examples = [
    {"message": "My bag arrived torn, order NC-20417.", "answer": "order_problem"},
    {"message": "Please pause my subscription for a month.", "answer": "subscription"},
  ]
  few_shot = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=ChatPromptTemplate.from_messages([("human", "{message}"), ("ai", "{answer}")]),
  )
  template = ChatPromptTemplate.from_messages(
    [("system", "Reply with the category only."), few_shot, ("human", "{message}")]
  )
  messages = template.invoke({"message": ticket}).to_messages()
  return messages


def _template_examples(ticket: str) -> list:
  template, value, messages = fill_a_template(ticket)
  inserted = insert_the_history(ticket)
  before, after = fix_a_variable_up_front()
  shown = teach_by_example(ticket)

  def message_table(ms):
    return table(["#", "Class", "Content"], [[i, type(m).__name__, m.text[:120]] for i, m in enumerate(ms, 1)])

  return [
    example(
      "lc.l04.1",
      "Fill a template: ChatPromptTemplate → PromptValue → messages",
      fill_a_template,
      "`{company}` and `{question}` are variables. Invoking the template returns a **PromptValue**, which any "
      "chat model accepts, or which you can turn into messages or a string.",
      [
        table(
          ["Expression", "Value"],
          [
            ["type(value).__name__", type(value).__name__],
            ["template.input_variables", ", ".join(template.input_variables)],
            ["value.to_string()", value.to_string()[:160]],
          ],
        ),
        message_table(messages),
      ],
    ),
    example(
      "lc.l04.2",
      "MessagesPlaceholder: insert a whole list of messages",
      insert_the_history,
      "The placeholder becomes as many messages as you pass: here, the two earlier turns of the chat.",
      [message_table(inserted)],
    ),
    example(
      "lc.l04.3",
      "partial(): fix some variables up front",
      fix_a_variable_up_front,
      "A partial template only needs the variables that change per call.",
      [table(["Template", "Variables still needed"], [["template", ", ".join(before)], ["nimbus", ", ".join(after)]])],
    ),
    example(
      "lc.l04.4",
      "Few-shot: teach the model by example",
      teach_by_example,
      "Each example becomes a human message and an AI message, placed before the real question.",
      [message_table(shown)],
    ),
  ]


@lab(
  "lc.prompt-templates",
  "Prompt templates: variables, history placeholders, partials and few-shot examples",
  uses_model=True,
  calls="12 calls to Mistral when comparing, otherwise none",
  inputs=[
    Input(
      "ticket",
      "Ticket",
      "textarea",
      default="I'm taking you to consumer court. Three damaged deliveries in a row (latest NC-90112) "
      "and no replacement.",
    ),
    Input(
      "compare", "Compare zero-shot (v1) with few-shot (v2) on 6 hard tickets (12 model calls)", "bool", default=True
    ),
  ],
)
def prompt_templates(ticket: str, compare: bool):
  simple = ChatPromptTemplate.from_messages([("system", "You help {company} customers."), ("human", "{question}")])
  value = simple.invoke({"company": "Nimbus Coffee", "question": ticket})
  chat = ChatPromptTemplate.from_messages(
    [("system", "You help {company} customers."), MessagesPlaceholder("history"), ("human", "{question}")]
  ).partial(company="Nimbus Coffee")
  with_history = chat.invoke(
    {
      "history": [("human", "Hi, order NC-90112 again."), ("ai", "Sorry to hear that! What happened?")],
      "question": ticket,
    }
  )
  few_shot = TRIAGE_PROMPTS["v2"].invoke({"ticket": ticket}).to_messages()
  views = [
    table(
      ["Step", "Result"],
      [
        ["template.invoke({...})", type(value).__name__],
        ["value.to_messages()", ", ".join(type(m).__name__ for m in value.to_messages())],
        ["value.to_string()", value.to_string()[:200]],
        ["template.input_variables", ", ".join(simple.input_variables)],
        ["after .partial(company=...)", ", ".join(chat.input_variables)],
      ],
      title="A template produces a PromptValue: messages, ready for any chat model",
    ),
    table(
      ["#", "Role", "Content"],
      [[i, m.type, m.text[:140]] for i, m in enumerate(with_history.to_messages(), 1)],
      title="MessagesPlaceholder inserted the whole history",
    ),
    table(
      ["#", "Role", "Content"],
      [[i, m.type, m.text[:140]] for i, m in enumerate(few_shot, 1)],
      title=f"Production prompt v2: rules + {len(EXAMPLES)} worked examples + the ticket ({len(few_shot)} messages)",
    ),
  ]
  if not compare:
    return views + _template_examples(ticket)
  model = get_chat_model()
  hard = [_tickets()[i] for i in HARD_TICKETS]
  rows = []
  for version in ("v1", "v2"):
    results = Triager(model, version, get_settings().triage_method).triage_many(
      [lt.ticket for lt in hard], concurrency=3
    )
    matches = [field_matches(r, lt.expected) for lt, r in zip(hard, results, strict=True)]
    usage = Usage()
    for r in results:
      usage.add(r.raw)
    rows.append(
      {
        "variant": f"{version}: {'zero-shot' if version == 'v1' else 'rules + few-shot'}",
        "category": sum(m["category"] for m in matches) / len(hard),
        "urgency": sum(m["urgency"] for m in matches) / len(hard),
        "needs_human": sum(m["needs_human"] for m in matches) / len(hard),
        "all": sum(all(m.values()) for m in matches) / len(hard),
        "input_tokens": usage.per_call()["input_tokens"],
      }
    )
  zero, few = rows
  return views + [
    scorecard(
      rows,
      [
        Metric("category", "category", "higher", "ratio"),
        Metric("urgency", "urgency", "higher", "ratio"),
        Metric("needs_human", "needs_human", "higher", "ratio"),
        Metric("all", "all fields right", "higher", "ratio"),
        Metric("input_tokens", "input tokens", "lower", "int", "Per ticket: examples make every request bigger"),
      ],
      title="6 hard tickets (legal threats, chargebacks, Hinglish, Tamil, contamination)",
    ),
    callout(
      f"Few-shot changed all-fields-right from {zero['all']:.2f} to {few['all']:.2f} on these tickets, and costs "
      f"{few['input_tokens'] - zero['input_tokens']:+.0f} input tokens per ticket. Examples teach the edge cases words "
      "struggle to describe; they are paid on every call. The full 20-ticket scorecard is in the production path.",
      "decision",
      "Zero-shot or few-shot",
    ),
    *_template_examples(ticket),
  ]


# ── Lab 5: output parsers vs structured output ─────────────────────────────────
class TriageDict(TypedDict):
  """How to route one customer support message."""

  # The same fields, allowed values and descriptions as TicketTriage: the only difference is that nothing validates.
  category: Annotated[Category, ..., "The ticket's category"]
  urgency: Annotated[Urgency, ..., TicketTriage.model_fields["urgency"].description]
  order_id: Annotated[str | None, ..., TicketTriage.model_fields["order_id"].description]
  language: Annotated[str, ..., TicketTriage.model_fields["language"].description]
  needs_human: Annotated[bool, ..., TicketTriage.model_fields["needs_human"].description]


def text_with_str_parser(model):
  from langchain_core.output_parsers import StrOutputParser

  reply = model.invoke("Name one way to brew coffee. One word.")
  text = StrOutputParser().invoke(reply)  # AIMessage → plain str
  return text


def dict_with_json_parser(model, message):
  from langchain_core.output_parsers import JsonOutputParser

  parser = JsonOutputParser()
  prompt = (
    "Give the category (order_problem, refund_return, subscription or other) and whether it is urgent, "
    f"as JSON with the keys category and urgent.\n{parser.get_format_instructions()}\nMessage: {message}"
  )
  data = parser.invoke(model.invoke(prompt))  # the reply's text → a dict
  return data


def object_with_pydantic_parser(model, message):
  from typing import Literal

  from langchain_core.exceptions import OutputParserException
  from langchain_core.output_parsers import PydanticOutputParser
  from pydantic import BaseModel

  class Routing(BaseModel):
    category: Literal["order_problem", "refund_return", "subscription", "other"]
    urgent: bool

  parser = PydanticOutputParser(pydantic_object=Routing)
  prompt = f"Route this customer message.\n{parser.get_format_instructions()}\nMessage: {message}"
  routing = parser.invoke(model.invoke(prompt))  # a validated Routing object
  try:
    parser.parse('{"category": "late delivery"}')  # not an allowed category, and urgent is missing
  except OutputParserException as error:
    failure = error
  return routing, failure


def object_with_structured_output(model, message):
  from typing import Literal

  from pydantic import BaseModel, Field

  class Routing(BaseModel):
    category: Literal["order_problem", "refund_return", "subscription", "other"]
    urgent: bool = Field(description="true if money or safety is at risk")

  routing = model.with_structured_output(Routing).invoke(message)  # no format instructions needed
  return routing


def nested_objects(model):
  from pydantic import BaseModel

  class Item(BaseModel):
    name: str
    quantity: int

  class Order(BaseModel):
    items: list[Item]  # a list of objects inside an object
    deliver_to: str

  order = model.with_structured_output(Order).invoke(
    "Send 2 bags of Nilgiri Sunrise and one Monsoon Malabar to Priya in Indiranagar."
  )
  return order


def plain_dict_with_typeddict(model, message):
  from typing import Annotated, TypedDict

  class Routing(TypedDict):
    category: Annotated[str, ..., "order_problem, refund_return, subscription or other"]
    urgent: Annotated[bool, ..., "true if money or safety is at risk"]

  routing = model.with_structured_output(Routing).invoke(message)  # a plain dict: nothing validates it
  return routing


def keep_the_raw_reply(model, message):
  from pydantic import BaseModel

  class Routing(BaseModel):
    category: str
    urgent: bool

  out = model.with_structured_output(Routing, include_raw=True).invoke(message)
  raw, parsed, error = out["raw"], out["parsed"], out["parsing_error"]  # never raises on a bad reply
  return raw, parsed, error


@lru_cache
def _structured_examples() -> tuple:
  """The code examples make their own model calls: run them once per server start, then reuse."""
  model = get_chat_model()
  message = "My order NC-12093 came with the wrong beans and I want my money back."
  text = text_with_str_parser(model)
  data = dict_with_json_parser(model, message)
  routing, failure = object_with_pydantic_parser(model, message)
  structured = object_with_structured_output(model, message)
  order = nested_objects(model)
  plain = plain_dict_with_typeddict(model, message)
  raw, parsed, error = keep_the_raw_reply(model, message)
  shown = lambda value: code(repr(value), "python")  # noqa: E731
  return (
    example(
      "lc.l05.1",
      "StrOutputParser: AIMessage → str",
      text_with_str_parser,
      "The simplest parser: it returns the reply's text.",
      [shown(text)],
    ),
    example(
      "lc.l05.2",
      "JsonOutputParser: format instructions in, a dict out",
      dict_with_json_parser,
      "The parser's format instructions go into the prompt; it parses the JSON in the reply. Nothing checks "
      "the fields or values.",
      [shown(data)],
    ),
    example(
      "lc.l05.3",
      "PydanticOutputParser: a validated object, or an exception",
      object_with_pydantic_parser,
      "Same idea, but the result is validated against the schema. A reply that doesn't fit raises "
      "`OutputParserException`.",
      [shown(routing), code(str(failure).splitlines()[0][:300], "text")],
    ),
    example(
      "lc.l06.1",
      "with_structured_output: ask the provider for the schema",
      object_with_structured_output,
      "No format instructions in your prompt: the provider receives the schema (as a tool or a JSON schema) "
      "and LangChain validates the reply into your class.",
      [shown(structured)],
    ),
    example(
      "lc.l06.2",
      "Nested schemas: a list of objects from free text",
      nested_objects,
      "Fields can be other models and lists of them.",
      [shown(order)],
    ),
    example(
      "lc.l06.3",
      "TypedDict schemas → plain dicts",
      plain_dict_with_typeddict,
      '`Annotated[type, ..., "description"]` describes each field. You get a dict back, not validated.',
      [shown(plain)],
    ),
    example(
      "lc.l06.4",
      "include_raw=True: the raw reply next to the result",
      keep_the_raw_reply,
      "Returns a dict instead of raising: the AIMessage (tokens, metadata), the parsed object, and the error "
      "if parsing failed.",
      [
        table(
          ["Key", "Value"],
          [
            ["raw", f"AIMessage, {raw.usage_metadata['input_tokens']} input tokens"],
            ["parsed", repr(parsed)],
            ["parsing_error", repr(error)],
          ],
        )
      ],
    ),
  )


# Every method gets the same instructions (role, categories, routing rules), so only the extraction method differs.
METHOD_PROMPT = ChatPromptTemplate.from_messages(
  [("system", SYSTEM_V1 + "\n\n" + RULES + "{format}"), ("human", "{ticket}")]
).partial(categories=CATEGORIES)


def _parser_method(parser):
  def run(model, ticket: Ticket):
    raw = model.invoke(
      METHOD_PROMPT.invoke({"ticket": ticket.text, "format": "\n\n" + parser.get_format_instructions()})
    )
    parsed = parser.invoke(raw)
    return (parsed.model_dump() if hasattr(parsed, "model_dump") else parsed), raw

  return run


def _structured_method(schema, method: str | None = None):
  def run(model, ticket: Ticket):
    kwargs = {"method": method} if method else {}
    structured = model.with_structured_output(schema, include_raw=True, **kwargs)
    out = structured.invoke(METHOD_PROMPT.invoke({"ticket": ticket.text, "format": ""}))
    if out["parsing_error"] or out["parsed"] is None:
      raise ValueError(f"parse failed: {out['parsing_error'] or 'no structured reply'}")
    parsed = out["parsed"]
    return (parsed.model_dump() if hasattr(parsed, "model_dump") else parsed), out["raw"]

  return run


PARSE_METHODS = {
  "JsonOutputParser (schema in the prompt)": _parser_method(JsonOutputParser(pydantic_object=TicketTriage)),
  "PydanticOutputParser (schema in the prompt)": _parser_method(PydanticOutputParser(pydantic_object=TicketTriage)),
  "with_structured_output(Pydantic), tool calling": _structured_method(TicketTriage),
  "with_structured_output(Pydantic), JSON schema": _structured_method(TicketTriage, "json_schema"),
  "with_structured_output(TypedDict), tool calling": _structured_method(TriageDict),
}


@lab(
  "lc.structured-output",
  "Structured output: output parsers vs with_structured_output, measured",
  uses_model=True,
  calls="5 × tickets calls to Mistral (25 by default, about a minute), plus 7 the first time for the code examples",
  inputs=[Input("tickets", "Tickets per method (5 methods → 5 × this many calls)", "number", default=5, min=2, max=8)],
)
def structured_output(tickets: int):
  model = get_chat_model()
  chosen = list(_tickets().values())[:: max(1, 20 // tickets)][:tickets]
  rows, sample = [], {}
  for name, run in PARSE_METHODS.items():
    ok = correct = 0
    usage, latency = Usage(), 0
    for lt in chosen:
      started = time.perf_counter()
      try:
        parsed, raw = run(model, lt.ticket)
        usage.add(raw)
        ok += 1
        correct += all(parsed.get(f) == lt.expected[f] for f in ("category", "urgency", "order_id", "needs_human"))
        sample.setdefault(name, parsed)
      except Exception as error:  # noqa: BLE001 — a failed parse is a result here
        sample.setdefault(name, {"error": f"{type(error).__name__}: {str(error)[:160]}"})
      latency += _ms(started)
    per = usage.per_call()
    rows.append(
      {
        "variant": name,
        "valid": ok / len(chosen),
        "correct": correct / len(chosen),
        "input_tokens": per["input_tokens"],
        "output_tokens": per["output_tokens"],
        "latency_ms": latency / len(chosen),
      }
    )
  text_only = StrOutputParser().invoke(AIMessage("order_problem"))
  try:
    PydanticOutputParser(pydantic_object=TicketTriage).parse('{"category": "late delivery", "urgency": "very high"}')
    failure = "(parsed?)"
  except Exception as error:  # noqa: BLE001
    failure = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
  return [
    scorecard(
      rows,
      [
        Metric("valid", "valid", "higher", "ratio", "Share of replies that parsed into the schema"),
        Metric("correct", "fields right", "higher", "ratio", "category, urgency, order_id and needs_human all match"),
        Metric("input_tokens", "input tokens", "lower", "int"),
        Metric("output_tokens", "output tokens", "lower", "int"),
        Metric("latency_ms", "latency", "lower", "ms"),
      ],
      title=f"{len(chosen)} tickets × {len(PARSE_METHODS)} ways to get structured data",
    ),
    code(
      json.dumps(sample, indent=1, ensure_ascii=False, default=str)[:3000],
      title="What each method returned (first ticket)",
    ),
    table(
      ["Case", "Result"],
      [
        ["StrOutputParser on AIMessage('order_problem')", repr(text_only)],
        ["PydanticOutputParser on a reply that breaks the schema", failure],
      ],
      title="Parsers: plain text, and what a parse failure looks like",
    ),
    callout(
      _which_structured(rows),
      "decision",
      "Which to use",
    ),
    *_structured_examples(),
  ]


def _which_structured(rows: list[dict]) -> str:
  best = max(rows, key=lambda r: (r["valid"], r["correct"]))
  cheapest = min(rows, key=lambda r: r["input_tokens"])
  tied = len({(r["valid"], r["correct"]) for r in rows}) == 1
  accuracy = (
    f"With the same instructions, every method got the same fields right ({best['correct']:.2f}): the method "
    "doesn't make the model smarter. "
    if tied
    else f"Most fields right: **{best['variant']}** ({best['correct']:.2f}); with 5 tickets, one ticket is 0.2. "
  )
  return (
    accuracy
    + f"The cost differs: **{cheapest['variant']}** sent {cheapest['input_tokens']:.0f} input tokens per ticket "
    f"against up to {max(r['input_tokens'] for r in rows):.0f}, because JSON-schema mode passes the schema outside "
    "the prompt while parsers paste format instructions into it. Parsers also parse whatever text comes back; "
    "`with_structured_output` asks the provider for structured output, and with a Pydantic schema validates types, "
    "enums, the order-id pattern and nested lists (`include_raw=True` hands you the raw reply and the parsing error "
    "instead of an exception). A TypedDict returns a plain dict that nothing checks."
  )


# ── Lab 6: invocation methods ───────────────────────────────────────────────────
def invoke_one_by_one(model, prompts):
  replies = [model.invoke(p) for p in prompts]  # each call waits for the previous one
  return replies


def batch_them(model, prompts):
  replies = model.batch(prompts, config={"max_concurrency": len(prompts)})  # the calls run at the same time
  return replies


def batch_them_async(model, prompts):
  import asyncio

  async def main():
    return await model.abatch(prompts)  # in an async web server, just `await` this

  replies = asyncio.run(main())
  return replies


def stream_the_answer(model, prompt):
  chunks, full = [], None
  for chunk in model.stream(prompt):
    chunks.append(chunk.text)  # show each piece to the user as it arrives
    full = chunk if full is None else full + chunk  # chunks add up to one message
  return chunks, full


def the_async_versions(model, prompt):
  import asyncio

  async def main():
    reply = await model.ainvoke(prompt)
    pieces = [chunk.text async for chunk in model.astream(prompt)]
    return reply, pieces

  reply, pieces = asyncio.run(main())
  return reply, pieces


@lru_cache
def _invocation_examples() -> tuple:
  """Examples that make their own calls (stream, ainvoke, astream): run once per server start, then reuse."""
  model = get_chat_model()
  prompt = "Name three Nimbus Coffee blends in one short sentence."
  chunks, full = stream_the_answer(model, prompt)
  reply, pieces = the_async_versions(model, prompt)
  return (
    example(
      "lc.l07.3",
      "stream(): the answer in pieces",
      stream_the_answer,
      "Each chunk is a small `AIMessageChunk`. Show `chunk.text` as it arrives; adding the chunks together "
      "gives the same message `invoke` would return.",
      [
        table(
          ["Expression", "Value"],
          [
            ["len(chunks)", len(chunks)],
            ["chunks[:6]", " | ".join(repr(c) for c in chunks[:6])],
            ["full.text", full.text if full else ""],
          ],
        )
      ],
    ),
    example(
      "lc.l07.4",
      "The async versions: ainvoke, abatch, astream",
      the_async_versions,
      "The same methods with an `a` in front, awaited. Use them inside async code (an async web server), so "
      "one request waiting for the model doesn't block the others.",
      [table(["Expression", "Value"], [["reply.text", reply.text], ["len(pieces)", len(pieces)]])],
    ),
  )


@lab(
  "lc.invocation",
  "Invocation: invoke, batch, stream and their async versions, timed",
  uses_model=True,
  calls="3 × n + 1 calls to Mistral (10 by default), plus 3 the first time for the code examples",
  inputs=[Input("n", "Tickets per method", "number", default=3, min=2, max=5)],
)
def invocation(n: int):
  # No client-side limiter here, so the timings show the methods themselves; production keeps the limiter.
  model = get_chat_model(requests_per_second=None)
  prompts = [
    [HumanMessage(f"Classify in one word (order, refund, subscription, question, other): {lt.ticket.text}")]
    for lt in list(_tickets().values())[:n]
  ]
  started = time.perf_counter()
  loop = invoke_one_by_one(model, prompts)
  loop_ms = _ms(started)
  started = time.perf_counter()
  batched = batch_them(model, prompts)
  batch_ms = _ms(started)
  started = time.perf_counter()
  async_out = batch_them_async(model, prompts)
  async_ms = _ms(started)
  stream_prompt = "In three short sentences, explain how Nimbus roasts its beans every Monday."
  started, first_ms, chunks, full = time.perf_counter(), None, [], None
  for chunk in model.stream(stream_prompt):
    first_ms = first_ms if first_ms is not None else _ms(started)
    chunks.append(chunk)
    full = chunk if full is None else full + chunk  # chunks add up to one message
  stream_ms = _ms(started)
  limiter_s = 100 / get_settings().chat_requests_per_second
  return [
    scorecard(
      [
        {"variant": f"for loop of invoke() × {n}", "wall_ms": loop_ms, "per_item_ms": loop_ms / n},
        {"variant": f"batch() × {n}", "wall_ms": batch_ms, "per_item_ms": batch_ms / n},
        {"variant": f"abatch() × {n} (async)", "wall_ms": async_ms, "per_item_ms": async_ms / n},
      ],
      [Metric("wall_ms", "total time", "lower", "ms"), Metric("per_item_ms", "per ticket", "lower", "ms")],
      title=f"{n} tickets, three ways",
    ),
    table(
      ["Ticket", "invoke", "batch", "abatch"],
      [
        [i + 1, a.text.strip()[:30], b.text.strip()[:30], c.text.strip()[:30]]
        for i, (a, b, c) in enumerate(zip(loop, batched, async_out, strict=True))
      ],
      title="Same answers, whichever method",
    ),
    table(
      ["Measure", "Value"],
      [
        ["time to first chunk", f"{first_ms} ms"],
        ["time to the full answer", f"{stream_ms} ms"],
        ["chunks received", len(chunks)],
        ["first chunks", " | ".join(repr(c.text) for c in chunks[:6])],
        ["sum of chunks == full text", str(full is not None and full.text == "".join(c.text for c in chunks))],
      ],
      title="stream(): the answer arrives in pieces",
    ),
    md(f"**Streamed answer:** {full.text if full else ''}"),
    callout(
      f"batch() took {batch_ms:,} ms for {n} tickets vs {loop_ms:,} ms in a loop: calls overlap instead of waiting in "
      f"line. Streaming showed the first words after {first_ms} ms instead of {stream_ms:,} ms. In production the "
      f"provider's rate limit still caps throughput: at {get_settings().chat_requests_per_second} requests/s, 100 "
      f"tickets take at least {limiter_s:.0f} s whatever the concurrency. Use the async methods inside an async web "
      "server, so a waiting request doesn't block the others.",
      "decision",
      "When to use which",
    ),
    example(
      "lc.l07.1",
      "invoke(): one input, one output",
      invoke_one_by_one,
      "In a loop, each call starts only when the previous one has finished.",
      [
        table(["Ticket", "reply.text"], [[i + 1, m.text.strip()[:60]] for i, m in enumerate(loop)]),
        table(["Measure", "Value"], [["total time", f"{loop_ms:,} ms"]]),
      ],
    ),
    example(
      "lc.l07.2",
      "batch(): many inputs, run concurrently",
      batch_them,
      "`max_concurrency` caps how many calls run at once (keep it within your rate limit).",
      [table(["Measure", "Value"], [["total time", f"{batch_ms:,} ms"], ["vs the loop", f"{loop_ms:,} ms"]])],
    ),
    example(
      "lc.l07.4",
      "abatch(): the async version of batch",
      batch_them_async,
      "`asyncio.run` starts an event loop for this example; inside async code you'd `await model.abatch(...)`.",
      [table(["Measure", "Value"], [["total time", f"{async_ms:,} ms"]])],
    ),
    *_invocation_examples(),
  ]


# ── Lab 7: errors and timeouts ──────────────────────────────────────────────────
def call_with_a_timeout():
  from langchain.chat_models import init_chat_model

  model = init_chat_model("ministral-14b-latest", model_provider="mistralai", timeout=1, max_retries=0)
  caught = None
  try:
    model.invoke("Write 300 words about the history of coffee in India.")  # takes longer than 1 s
  except Exception as error:  # noqa: BLE001 — httpx.ReadTimeout
    caught = error
  return caught


def let_the_client_retry():
  from langchain.chat_models import init_chat_model

  # max_retries: temporary failures (timeouts, network errors, 429, 5xx) are retried with a growing wait
  model = init_chat_model("ministral-14b-latest", model_provider="mistralai", timeout=1, max_retries=2)
  caught = None
  try:
    model.invoke("Write 300 words about the history of coffee in India.")
  except Exception as error:  # noqa: BLE001
    caught = error
  return caught


def use_a_bad_api_key():
  from langchain.chat_models import init_chat_model

  model = init_chat_model("ministral-14b-latest", model_provider="mistralai", api_key="not-a-real-key", max_retries=0)
  caught = None
  try:
    model.invoke("OK?")
  except Exception as error:  # noqa: BLE001 — 401: retrying can never fix this
    caught = error
  return caught


def ask_for_an_unknown_model():
  from langchain.chat_models import init_chat_model

  model = init_chat_model("ministral-999b-nonexistent", model_provider="mistralai", max_retries=0)
  caught = None
  try:
    model.invoke("OK?")
  except Exception as error:  # noqa: BLE001 — 400 Invalid model
    caught = error
  return caught


def hit_a_rate_limit():
  from langchain.chat_models import init_chat_model

  model = init_chat_model("magistral-small-latest", model_provider="mistralai", max_retries=0)  # free plan: 429
  caught = None
  try:
    model.invoke("OK?")
  except Exception as error:  # noqa: BLE001
    caught = error
  return caught


def decide_what_to_do(errors):
  import httpx

  def what_to_do(error):
    status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
    if isinstance(error, httpx.TimeoutException) or status in (408, 429, 500, 502, 503, 504):
      return "try again later"  # temporary: queue it and retry
    return "fix the request"  # permanent: bad key, unknown model, input too long...

  decisions = [what_to_do(error) for error in errors]
  return decisions


@lab(
  "lc.errors",
  "Errors and timeouts: retryable vs permanent, and what to do with each",
  uses_model=True,
  calls="7 failing requests (timeouts are still billed for what was generated)",
)
def errors():
  cases = [
    ("Timeout 1 s, max_retries=0", call_with_a_timeout),
    ("Timeout 1 s, max_retries=2", let_the_client_retry),
    ("Invalid API key", use_a_bad_api_key),
    ("Unknown model name", ask_for_an_unknown_model),
    ("Rate limited (magistral on the free plan)", hit_a_rate_limit),
  ]
  rows, caught = [], {}
  for label, run in cases:
    started = time.perf_counter()
    error = run()
    caught[run] = (error, _ms(started))
    if error is None:
      rows.append([label, "—", "—", "succeeded", "", "", _ms(started)])
      continue
    failure = classify(error)
    rows.append(
      [label, type(error).__name__, failure.status or "—", failure.action, failure.reason, failure.detail, _ms(started)]
    )
  decisions = decide_what_to_do([e for e, _ in caught.values() if e is not None])

  def outcome(run):
    error, ms = caught[run]
    return table(
      ["Expression", "Value"],
      [
        ["type(caught).__name__", type(error).__name__ if error else "None (it worked)"],
        ["str(caught)", str(error)[:160] if error else ""],
        ["time until it failed", f"{ms:,} ms"],
      ],
    )

  return [
    table(
      ["Case", "Exception", "HTTP", "Action", "Reason", "Provider says", "Elapsed ms"],
      rows,
      title="Real failures, classified",
    ),
    callout(
      "**retry** failures are temporary (timeouts, network errors, 429 rate limits, 5xx): the client retries them "
      f"with backoff (`max_retries`): the second timeout took {rows[1][-1] / max(rows[0][-1], 1):.0f}× as long "
      "(three attempts plus waits) before giving up; after that, queue the work and try later. "
      "**fix** failures (bad key, unknown model, input too long) fail "
      "at once and retrying can never help: alert someone and fix the request or the configuration. A timeout still "
      "costs tokens: the provider keeps generating after you stop waiting.",
      "decision",
      "Try again later, or fix the request",
    ),
    example(
      "lc.l08.1",
      "timeout: stop waiting after N seconds",
      call_with_a_timeout,
      "`timeout` is in seconds (whole seconds for Mistral). With `max_retries=0` the first timeout is final.",
      [outcome(call_with_a_timeout)],
    ),
    example(
      "lc.l08.2",
      "max_retries: temporary failures are retried for you",
      let_the_client_retry,
      "The same timeout with two retries: three attempts with waits in between, so it fails much later.",
      [outcome(let_the_client_retry)],
    ),
    example(
      "lc.l08.2",
      "A rate limit (429) is a temporary failure too",
      hit_a_rate_limit,
      "With retries on, the client would wait and try again; here they're off, so you see the 429.",
      [outcome(hit_a_rate_limit)],
    ),
    example(
      "lc.l08.3",
      "A bad API key fails at once",
      use_a_bad_api_key,
      "401: no retry can fix a wrong key.",
      [outcome(use_a_bad_api_key)],
    ),
    example(
      "lc.l08.3",
      "So does an unknown model",
      ask_for_an_unknown_model,
      "400 with the provider's message: fix the configuration.",
      [outcome(ask_for_an_unknown_model)],
    ),
    example(
      "lc.l08.4",
      "Sort errors in your own code: try again later, or fix the request",
      decide_what_to_do,
      "One small function turns any error into a decision your code can act on.",
      [table(["Case", "Decision"], [[label, d] for (label, _), d in zip(cases, decisions, strict=True)])],
    ),
  ]


# ── Lab 8: model profiles and reasoning models ──────────────────────────────────
PUZZLES = {
  "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much is the ball?": "0.05",
  "A café sells 3 cappuccinos for every 2 lattes. It sold 45 drinks of those two. How many lattes?": "18",
}


def _reasoning_text(message) -> str:
  return "".join(b.get("reasoning", "") for b in message.content_blocks if b.get("type") == "reasoning")


def read_a_profile():
  from langchain.chat_models import init_chat_model

  reasoner = init_chat_model("openai/gpt-oss-20b", model_provider="groq")
  profile = reasoner.profile  # a dict of what the model can do; no request is sent
  return profile


def supply_a_profile():
  from langchain.chat_models import init_chat_model

  model = init_chat_model("ministral-14b-latest", model_provider="mistralai")
  missing = model.profile  # None: the installed package has no data for this model
  model = init_chat_model(
    "ministral-14b-latest",
    model_provider="mistralai",
    profile={"max_input_tokens": 262_144, "image_inputs": True, "tool_calling": True, "structured_output": True},
  )
  return missing, model.profile


def check_before_sending(model, image_block):
  if (model.profile or {}).get("image_inputs"):
    content = [{"type": "text", "text": "What is the total on this receipt?"}, image_block]
  else:
    content = "What is the total on this receipt? (the photo was left out: this model can't read images)"
  return content


def think(reasoner, question, effort):
  reply = reasoner.invoke(question, reasoning_effort=effort)  # "low", "medium" or "high"
  reasoning = "".join(b.get("reasoning", "") for b in reply.content_blocks if b["type"] == "reasoning")
  answer = reply.text  # the text blocks only: the reasoning is kept separate
  thinking_tokens = reply.usage_metadata["output_token_details"]["reasoning"]  # billed as output tokens
  return reply, reasoning, answer, thinking_tokens


def stream_a_reasoning_model(reasoner):
  thinking, answer, order = "", "", []
  for chunk in reasoner.stream("Is 91 a prime number? One short sentence.", reasoning_effort="low"):
    for block in chunk.content_blocks:
      if block["type"] == "reasoning":
        thinking += block.get("reasoning", "")
      elif block["type"] == "text":
        answer += block.get("text", "")
      if not order or order[-1] != block["type"]:
        order.append(block["type"])  # which kind of block arrived, in order
  return thinking, answer, order


@lru_cache
def _reasoning_stream_example() -> dict:
  thinking, answer_text, order = stream_a_reasoning_model(get_chat_model(role="reasoning"))
  return example(
    "lc.l09.6",
    "Streaming a reasoning model: the reasoning first, then the answer",
    stream_a_reasoning_model,
    "Chunks carry reasoning blocks first; text blocks with the answer come after. A UI can show the thinking as "
    "it happens, then the answer.",
    [
      table(
        ["Variable", "Value"], [["order", " → ".join(order)], ["thinking", thinking[:200]], ["answer", answer_text]]
      )
    ],
  )


@lab(
  "lc.reasoning-profiles",
  "Model profiles and reasoning models: what a model can do, and thinking before answering",
  uses_model=True,
  calls="4 calls to Groq, plus 1 the first time for the code examples",
  inputs=[Input("puzzle", "Question", "select", default=next(iter(PUZZLES)), options=list(PUZZLES))],
)
def reasoning_profiles(puzzle: str):
  s = get_settings()
  default, reasoner = get_chat_model(), get_chat_model(role="reasoning")
  local_profile = KNOWN_PROFILES.get(s.local_model, {})
  profiles = [
    [f"{s.chat_model} (default)", *[str((default.profile or {}).get(k)) for k in PROFILE_KEYS]],
    [f"{s.reasoning_model} (reasoning)", *[str((reasoner.profile or {}).get(k)) for k in PROFILE_KEYS]],
    [f"{s.local_model} (local)", *[str(local_profile.get(k)) for k in PROFILE_KEYS]],
  ]
  gate = [
    [name, "send the image" if (m.profile or {}).get("image_inputs") else "send text only (OCR first)"]
    for name, m in ((s.chat_model, default), (s.reasoning_model, reasoner))
  ]
  rows, answers, thoughts = [], [], []
  for effort in ("low", "medium", "high"):
    started = time.perf_counter()
    reply, reasoning, _, _ = think(reasoner, puzzle + " Answer with the number only.", effort)
    thoughts.append((effort, reply, reasoning))
    details = reply.usage_metadata.get("output_token_details") or {}
    rows.append(
      {
        "variant": f"reasoning_effort={effort}",
        "correct": 1.0 if PUZZLES[puzzle] in reply.text else 0.0,
        "reasoning_tokens": details.get("reasoning", 0),
        "output_tokens": reply.usage_metadata["output_tokens"],
        "latency_ms": _ms(started),
      }
    )
    answers.append(answer(f"**{reply.text.strip()}**\n\n*Reasoning:* {_reasoning_text(reply)[:400]}", effort))
  order, started = [], time.perf_counter()
  for chunk in reasoner.stream("Is 91 a prime number? One short sentence.", reasoning_effort="low"):
    for block in chunk.content_blocks:
      if block.get("type") in ("reasoning", "text") and (block.get("reasoning") or block.get("text")):
        if not order or order[-1][0] != block["type"]:
          order.append([block["type"], _ms(started), 0])
        order[-1][2] += 1
  return [
    table(
      ["Model", *PROFILE_KEYS],
      profiles,
      title="model.profile (the default and local rows are filled in by Nimbus: the package has no data for them)",
    ),
    table(["Model", "Given a receipt photo, Nimbus would…"], gate, title="Use the profile before sending"),
    scorecard(
      rows,
      [
        Metric("correct", "correct", "higher", "ratio"),
        Metric("reasoning_tokens", "reasoning tokens", "lower", "int"),
        Metric("output_tokens", "output tokens (billed)", "lower", "int"),
        Metric("latency_ms", "latency", "lower", "ms"),
      ],
      title=f"{s.reasoning_model}: more thinking, more tokens",
    ),
    columns(answers, title="The answers, with the start of the reasoning"),
    table(
      ["Block type", "First arrived at (ms)", "Chunks"],
      order,
      title="Streaming a reasoning model: reasoning first, then the answer",
    ),
    callout(
      "Reasoning tokens are billed as output tokens and add latency. Use a reasoning model (and higher effort) for "
      "multi-step problems; for classification and extraction a fast non-reasoning model is usually as accurate and "
      "far cheaper. Read capabilities from `model.profile`, and supply a profile when the package has none.",
      "decision",
      "When to pay for thinking",
    ),
    example(
      "lc.l09.1",
      "model.profile: what a model can do",
      read_a_profile,
      "Context size, image input, reasoning output, tool calling, structured output: read it in code instead of "
      "guessing from the model's name.",
      [table(["Key", "Value"], [[k, str(v)] for k, v in read_a_profile().items() if k in PROFILE_KEYS])],
    ),
    example(
      "lc.l09.2",
      "When a profile is missing, supply one",
      supply_a_profile,
      "Some models have no profile data in the installed package. Pass your own (from the provider's model card).",
      [
        table(
          ["Variable", "Value"],
          [[n, json.dumps(v)] for n, v in zip(("missing", "model.profile"), supply_a_profile(), strict=True)],
        )
      ],
    ),
    example(
      "lc.l09.3",
      "Use the profile: only send what the model can handle",
      check_before_sending,
      "Check the capability, then build the message. Here: the same code for a model that reads images and one "
      "that doesn't.",
      [
        table(
          ["Model", "content"],
          [
            [
              name,
              json.dumps(check_before_sending(m, {"type": "image", "base64": "…", "mime_type": "image/png"}))[:140],
            ]
            for name, m in ((s.chat_model, default), (s.reasoning_model, reasoner))
          ],
        )
      ],
    ),
    example(
      "lc.l09.4",
      "Reasoning comes back as reasoning blocks",
      think,
      "A reasoning model's reply has **reasoning** content blocks before the **text** block. `.text` gives "
      "only the answer; the reasoning tokens are counted (and billed) as output tokens.",
      [
        table(
          ["Variable", "Value"],
          [
            ["answer", thoughts[0][1].text.strip()],
            ["reasoning", thoughts[0][2][:300]],
            ["thinking_tokens", rows[0]["reasoning_tokens"]],
          ],
        )
      ],
    ),
    example(
      "lc.l09.5",
      "reasoning_effort: more or less thinking",
      think,
      "The same function with `effort` set to low, medium and high: more thinking costs more tokens and time.",
      [
        table(
          ["effort", "thinking_tokens", "answer"],
          [[e, r["reasoning_tokens"], m.text.strip()] for (e, m, _), r in zip(thoughts, rows, strict=True)],
        )
      ],
    ),
    _reasoning_stream_example(),
  ]


# ── Lab 9: local models ─────────────────────────────────────────────────────────
@tool
def order_status(order_id: str) -> str:
  """Look up where a Nimbus order is. order_id looks like NC-12345."""
  return f"{order_id}: packed, ships tomorrow"


def _ollama_ready(url: str, model: str) -> str | None:
  try:
    tags = httpx.get(f"{url}/api/tags", timeout=2).json()
  except httpx.HTTPError:
    return (
      f"Ollama isn't running at {url}. Start it with `ollama serve` (and keep it running), then run this lab again."
    )
  if not any(m["name"] == model for m in tags.get("models", [])):
    return f"Ollama is running but `{model}` isn't downloaded. Run `ollama pull {model}` (about 1.4 GB)."
  return None


def ask_a_local_model(question):
  from langchain.chat_models import init_chat_model

  local = init_chat_model("qwen3:1.7b", model_provider="ollama")  # Ollama at http://localhost:11434, no API key
  reply = local.invoke(question)
  return reply


def same_methods_locally(question):
  from langchain.chat_models import init_chat_model

  local = init_chat_model("qwen3:1.7b", model_provider="ollama", reasoning=False)
  pieces = [chunk.text for chunk in local.stream(question)]  # stream, batch, async: all the same as in the cloud
  return pieces


def turn_thinking_off(question):
  from langchain.chat_models import init_chat_model

  thinking = init_chat_model("qwen3:1.7b", model_provider="ollama")  # qwen3 thinks before answering by default
  fast = init_chat_model("qwen3:1.7b", model_provider="ollama", reasoning=False)
  slow_reply, fast_reply = thinking.invoke(question), fast.invoke(question)
  return slow_reply, fast_reply


def call_a_tool_locally():
  from langchain.chat_models import init_chat_model
  from langchain_core.tools import tool

  @tool
  def order_status(order_id: str) -> str:
    """Look up where a Nimbus order is. order_id looks like NC-12345."""
    return f"{order_id}: packed, ships tomorrow"

  local = init_chat_model("qwen3:1.7b", model_provider="ollama")
  reply = local.bind_tools([order_status]).invoke("Where is my order NC-20417?")
  return reply.tool_calls


def swap_cloud_for_local(question):
  from langchain.chat_models import init_chat_model

  def answer(model, question):  # written once: works with any chat model
    return model.invoke(question).text

  cloud = init_chat_model("ministral-14b-latest", model_provider="mistralai")
  local = init_chat_model("qwen3:1.7b", model_provider="ollama", reasoning=False)
  answers = {"cloud": answer(cloud, question), "local": answer(local, question)}
  return answers


def _local_examples(question: str) -> list:
  first = ask_a_local_model(question)
  pieces = same_methods_locally(question)
  started = time.perf_counter()
  slow_reply, fast_reply = turn_thinking_off(question)
  both_ms = _ms(started)
  tool_calls = call_a_tool_locally()
  answers = swap_cloud_for_local(question)
  return [
    example(
      "lc.l10.1",
      "Ollama: a local model, called like any provider",
      ask_a_local_model,
      'Ollama serves open models on your computer; `init_chat_model` talks to it with `model_provider="ollama"`.',
      [
        table(
          ["Expression", "Value"], [["type(local).__name__", "ChatOllama"], ["reply.text", first.text.strip()[:200]]]
        )
      ],
    ),
    example(
      "lc.l10.2",
      "The same interface: stream, batch, async",
      same_methods_locally,
      "Everything from the invocation lab works unchanged.",
      [
        table(
          ["Expression", "Value"], [["len(pieces)", len(pieces)], ["''.join(pieces)", "".join(pieces).strip()[:200]]]
        )
      ],
    ),
    example(
      "lc.l10.3",
      "Thinking on or off",
      turn_thinking_off,
      "Small reasoning models spend most of their time thinking. `reasoning=False` skips it.",
      [
        table(
          ["Variable", "Output tokens", "Text"],
          [
            ["slow_reply", (slow_reply.usage_metadata or {}).get("output_tokens", 0), slow_reply.text.strip()[:120]],
            ["fast_reply", (fast_reply.usage_metadata or {}).get("output_tokens", 0), fast_reply.text.strip()[:120]],
          ],
        ),
        table(["Measure", "Value"], [["both calls", f"{both_ms:,} ms"]]),
      ],
    ),
    example(
      "lc.l10.4",
      "Tool calling with a local model",
      call_a_tool_locally,
      "`bind_tools` works the same: the model asks for the tool with arguments (your code runs it).",
      [code(json.dumps(tool_calls, indent=1))],
    ),
    example(
      "lc.l10.5",
      "Swap a cloud model for a local one",
      swap_cloud_for_local,
      "Code written against the chat model interface doesn't care which model it gets.",
      [table(["Model", "Answer"], [[k, v.strip()[:200]] for k, v in answers.items()])],
    ),
  ]


@lab(
  "lc.local-models",
  "Local models with Ollama: same interface, no key, no rate limit",
  uses_model=True,
  calls="2 calls to Mistral, the rest local (free)",
  inputs=[Input("question", "Question", "text", default="Name two ways to brew coffee at home. One short line.")],
)
def local_models(question: str):
  s = get_settings()
  problem = _ollama_ready(s.ollama_url, s.local_model)
  if problem:
    return [callout(problem, "warning", "Local model not available")]
  cloud, local = get_chat_model(), get_chat_model(role="local")
  local_fast = get_chat_model(role="local", reasoning=False)
  rows, outs = [], []
  for label, model in (
    (f"{s.chat_model} (cloud)", cloud),
    (f"{s.local_model}, thinking on", local),
    (f"{s.local_model}, thinking off", local_fast),
  ):
    started = time.perf_counter()
    reply = model.invoke(question)
    rows.append(
      {
        "variant": label,
        "latency_ms": _ms(started),
        "output_tokens": (reply.usage_metadata or {}).get("output_tokens", 0),
      }
    )
    outs.append(answer(reply.text.strip()[:400], label))
  tool_rows = []
  for label, model in (("thinking on", local), ("thinking off", local_fast)):
    for request in ("Where is my order NC-20417?", "Use the order_status tool to check order NC-20417."):
      calls = model.bind_tools([order_status]).invoke(request).tool_calls
      tool_rows.append([label, request, json.dumps(calls[0]["args"]) if calls else "— (answered in text instead)"])
  sample = list(_tickets().values())[:4]
  swap = Triager(local_fast, "v2", get_settings().triage_method).triage_many(
    [lt.ticket for lt in sample], concurrency=1
  )
  swap_rows = []
  for lt, res in zip(sample, swap, strict=True):
    right = field_matches(res, lt.expected)
    swap_rows.append(
      [res.ticket_id, "✓" if res.triage else "✗", f"{sum(right.values())}/{len(right)}", res.error or ""]
    )
  return [
    scorecard(
      rows,
      [Metric("latency_ms", "latency", "lower", "ms"), Metric("output_tokens", "output tokens", "lower", "int")],
      title="The same question, cloud vs local",
    ),
    columns(outs),
    table(["Mode", "Request", "Tool call"], tool_rows, title="Tool calling with the local model"),
    table(
      ["Ticket", "Valid", "Fields right", "Error"],
      swap_rows,
      title="Swap: the production Triager, unchanged, on the local model (thinking off)",
    ),
    callout(
      "Only the model name changed: `invoke`, `stream`, tools and structured output work the same. A 1.7B local model "
      "is free, private and offline, but slower on a laptop and weaker on hard instructions; turning thinking off "
      "makes small reasoning models much faster, but here it also stopped using the tool unless told to. Run the "
      "triage scorecard with the local model before trusting it.",
      "decision",
      "Cloud or local",
    ),
    *_local_examples(question),
  ]


# ── Production path labs: the Nimbus triage service ─────────────────────────────
HISTORY = PROJECT_DIR / ".state" / "triage_scorecards.jsonl"
TRIAGE_METRICS = [
  Metric("valid", "valid", "higher", "ratio", "Share of replies that matched the schema"),
  Metric("category_acc", "category", "higher", "ratio"),
  Metric("urgency_acc", "urgency", "higher", "ratio"),
  Metric("order_id_acc", "order id", "higher", "ratio"),
  Metric("language_acc", "language", "higher", "ratio"),
  Metric("needs_human_acc", "needs human", "higher", "ratio"),
  Metric("all_correct", "all right", "higher", "ratio", "Tickets with every scored field right"),
  Metric("input_tokens", "input tok", "lower", "int", "Per ticket"),
  Metric("cost_per_1k_usd", "$ / 1k", "lower", "usd", "Cost of 1,000 tickets at a reference paid price"),
  Metric("latency_p95_ms", "p95", "lower", "ms", "Per ticket, including time queued behind the rate limiter"),
]


@lab(
  "lc.triage-one",
  "Triage one ticket, then stream a reply draft",
  uses_model=True,
  calls="2 calls to Mistral (triage + reply)",
  inputs=[
    Input(
      "ticket", "Ticket from the evaluation set", "select", default=_ticket_options()[16], options=_ticket_options()
    ),
    Input("custom", "…or type your own (overrides the selection)", "textarea", default=""),
    Input(
      "prompt_version",
      "Prompt version",
      "select",
      default=get_settings().triage_prompt_version,
      options=list(TRIAGE_PROMPTS),
    ),
  ],
)
def triage_one(ticket: str, custom: str, prompt_version: str):
  from nimbus.support.triage import stream_reply

  s = get_settings()
  ticket_id, text = ticket.split(" · ", 1)
  if custom.strip():
    ticket_id, text = "custom", custom.strip()
  model = get_chat_model()
  result = Triager(model, prompt_version, s.triage_method).triage(Ticket(id=ticket_id, text=text))
  views = []
  if result.triage is None:
    return [callout(f"{result.error} (action: {result.action})", "warning", "Triage failed")]
  expected = _tickets()[ticket_id].expected if ticket_id in _tickets() else None
  checks = field_matches(result, expected) if expected else {}
  views.append(
    table(
      ["Field", "Model", "Label", ""],
      [
        [
          f,
          str(getattr(result.triage, f)),
          str(expected[f]) if expected else "—",
          ("✓" if checks[f] else "✗") if expected else "",
        ]
        for f in ("category", "urgency", "order_id", "language", "needs_human")
      ]
      + [
        ["sentiment", result.triage.sentiment, "—", ""],
        ["summary", result.triage.summary, "—", ""],
        ["products", ", ".join(p.name for p in result.triage.products) or "—", "—", ""],
      ],
      title=f"TicketTriage (prompt {prompt_version}): a validated object",
    )
  )
  started, first_ms, parts = time.perf_counter(), None, []
  for piece in stream_reply(model, Ticket(id=ticket_id, text=text), result.triage):
    first_ms = first_ms if first_ms is not None else _ms(started)
    parts.append(piece)
  usage = result.raw.usage_metadata
  views += [
    answer(
      "".join(parts),
      "Reply draft (streamed)",
      {"first chunk": f"{first_ms} ms", "total": f"{_ms(started)} ms", "chunks": len(parts)},
    ),
    table(
      ["Measure", "Value"],
      [
        ["triage latency", f"{result.latency_ms:,.0f} ms"],
        ["input tokens", usage["input_tokens"]],
        ["output tokens", usage["output_tokens"]],
        ["model", model_name(model)],
        ["structured output method", s.triage_method],
        ["finish reason", result.raw.response_metadata.get("finish_reason", "—")],
      ],
      title="The raw AIMessage behind it (include_raw=True)",
    ),
  ]
  if s.triage_prompt_version != prompt_version:
    views.append(callout(f"Production uses prompt {s.triage_prompt_version}; this run used {prompt_version}.", "info"))
  return views


@lab(
  "lc.triage-eval",
  "Score triage on 20 labelled tickets",
  uses_model=True,
  calls="20 calls to Mistral (about a minute)",
  inputs=[
    Input(
      "prompt_version",
      "Prompt version",
      "select",
      default=get_settings().triage_prompt_version,
      options=list(TRIAGE_PROMPTS),
    ),
    Input(
      "method",
      "Structured output method",
      "select",
      default=get_settings().triage_method,
      options=["function_calling", "json_schema"],
    ),
  ],
)
def triage_eval(prompt_version: str, method: str):
  from datetime import UTC, datetime

  from nimbus.support.evaluation import FIELDS, evaluate

  s = get_settings()
  model = get_chat_model()
  triager = Triager(model, prompt_version, method)
  report = evaluate(triager, list(_tickets().values()), model_name(model), s.triage_concurrency)
  record = {"at": datetime.now(UTC).isoformat(timespec="seconds"), **report.summary()}
  HISTORY.parent.mkdir(exist_ok=True)
  with HISTORY.open("a") as f:
    f.write(json.dumps(record) + "\n")
  history = [json.loads(line) for line in HISTORY.read_text().splitlines() if line.strip()]
  misses = [
    [
      lt.ticket.id,
      lt.ticket.text[:80],
      ", ".join(f"{f}: {getattr(r.triage, f) if r.triage else '—'} (want {lt.expected[f]})" for f in FIELDS if not m[f])
      or (r.error or ""),
    ]
    for lt, r, m in report.results
    if not all(m.values())
  ]
  previous = history[-2] if len(history) > 1 else None
  delta = ""
  if previous:
    delta = (
      f"vs the previous run ({previous['prompt_version']}, {previous.get('method', 'function_calling')}): "
      f"all fields right {previous['all_correct']:.2f} → "
      f"{record['all_correct']:.2f}, cost per 1k tickets ${previous['cost_per_1k_usd']:.2f} → "
      f"${record['cost_per_1k_usd']:.2f}."
    )
  return [
    scorecard(
      [{**record, "variant": f"this run · {prompt_version} · {method}"}],
      TRIAGE_METRICS,
      title=f"{report.tickets} tickets, prompt {prompt_version}, {model_name(model)}",
    ),
    callout(
      f"{report.tickets} tickets took {report.wall_s:.0f} s at {s.triage_concurrency} in parallel under a "
      f"{s.chat_requests_per_second} requests/s limit. {delta}",
      "info",
      "This run",
    ),
    table(["Ticket", "Message", "What was wrong"], misses, title=f"Tickets with a wrong field ({len(misses)})"),
    scorecard(
      [
        {**h, "variant": f"#{i + 1} · {h['prompt_version']} · {h.get('method', 'function_calling')}"}
        for i, h in enumerate(history)
      ],
      TRIAGE_METRICS,
      title="Scorecard history",
    ),
  ]
