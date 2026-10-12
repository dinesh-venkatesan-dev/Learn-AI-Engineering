"""LangChain · Module 2 labs: chains with LCEL. Runnables and `|`, parallel steps and passthrough, functions in
chains, routing, retries / fallbacks / config, callbacks and events. Each learning point is a small snippet function
that the lab runs; the Covers tab shows its code next to what it produced."""

import asyncio
import json
import time
from functools import lru_cache

from langchain.chat_models import init_chat_model

from academy.labs.runtime import Input, Metric, answer, callout, code, columns, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR, get_settings
from nimbus.llm import get_chat_model
from nimbus.support.evaluation import load_tickets
from nimbus.support.pipeline import build_pipeline, run, stream_events, support_pipeline
from nimbus.support.resilience import drill

TICKET = "I received Nilgiri Sunrise instead of the Monsoon Malabar Reserve I ordered (NC-12093). Please fix it."


def _ms(started: float) -> int:
  return round((time.perf_counter() - started) * 1000)


@lru_cache
def _tickets():
  return {lt.ticket.id: lt for lt in load_tickets(get_settings().eval_tickets)}


# ── Lab 1: runnables and | ──────────────────────────────────────────────────────
def three_steps_by_hand(model, ticket):
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate

  prompt = ChatPromptTemplate.from_template("Summarise this support message in one short sentence:\n{ticket}")
  parser = StrOutputParser()
  prompt_value = prompt.invoke({"ticket": ticket})  # dict → PromptValue
  message = model.invoke(prompt_value)  # PromptValue → AIMessage
  summary = parser.invoke(message)  # AIMessage → str
  return prompt_value, message, summary


def the_same_with_a_pipe(model, ticket):
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate

  prompt = ChatPromptTemplate.from_template("Summarise this support message in one short sentence:\n{ticket}")
  chain = prompt | model | StrOutputParser()  # a RunnableSequence: each output is the next step's input
  summary = chain.invoke({"ticket": ticket})
  return chain, summary


def batch_and_stream_for_free(chain, tickets):
  summaries = chain.batch([{"ticket": t} for t in tickets])  # every chain has batch()...
  pieces = [piece for piece in chain.stream({"ticket": tickets[0]})]  # ...and stream(), with no extra code
  return summaries, pieces


@lab(
  "lc.runnables",
  "Runnables and the | operator: one interface, chained",
  uses_model=True,
  calls="1 + 1 + 3 calls to Mistral (5)",
  inputs=[Input("ticket", "Support message", "textarea", default=TICKET)],
)
def runnables(ticket: str):
  model = get_chat_model()
  prompt_value, message, by_hand = three_steps_by_hand(model, ticket)
  chain, piped = the_same_with_a_pipe(model, ticket)
  others = [lt.ticket.text for lt in list(_tickets().values())[:2]]
  summaries, pieces = batch_and_stream_for_free(chain, [ticket, *others])
  steps = [
    ["prompt.invoke(dict)", type(prompt_value).__name__, prompt_value.to_string()[:90]],
    ["model.invoke(prompt_value)", type(message).__name__, message.text[:90]],
    ["parser.invoke(message)", "str" if isinstance(by_hand, str) else type(by_hand).__name__, by_hand[:90]],
  ]
  return [
    table(["Step", "Returns", "Value"], steps, title="What flows between the steps"),
    columns(
      [answer(by_hand, "Three calls by hand"), answer(piped, "prompt | model | parser")],
      title="The same result, written two ways",
    ),
    table(
      ["Chain method", "Result"],
      [
        ["batch() of 3", " · ".join(s[:60] for s in summaries)],
        ["stream() chunks", f"{len(pieces)} pieces: " + " | ".join(repr(p) for p in pieces[:5])],
      ],
      title="Methods every chain gets",
    ),
    callout(
      "Prompts, models and parsers are all **Runnables**: they share `invoke`, `batch`, `stream` and the async "
      "versions. `|` connects them into a **RunnableSequence**, which is a Runnable too, so a whole chain can be "
      "batched, streamed, retried or used as one step of a bigger chain.",
      "decision",
      "Why it matters",
    ),
    example(
      "lc.l11.1",
      "Prompts, models and parsers share one interface",
      three_steps_by_hand,
      "Each piece has `invoke`; the output of one is exactly what the next one takes.",
      [
        table(
          ["Variable", "Type"],
          [
            ["prompt_value", type(prompt_value).__name__],
            ["message", type(message).__name__],
            ["summary", "str" if isinstance(by_hand, str) else type(by_hand).__name__],
          ],
        )
      ],
    ),
    example(
      "lc.l11.2",
      "What flows between steps, by hand",
      three_steps_by_hand,
      "dict → PromptValue → AIMessage → str: three invoke calls, written out.",
      [table(["Step", "Returns", "Value"], steps)],
    ),
    example(
      "lc.l11.3",
      "The same with prompt | model | parser",
      the_same_with_a_pipe,
      "`|` builds a RunnableSequence that passes each output to the next step.",
      [table(["Expression", "Value"], [["type(chain).__name__", type(chain).__name__], ["summary", piped]])],
    ),
    example(
      "lc.l11.4",
      "A chain gets batch() and stream() for free",
      batch_and_stream_for_free,
      "No extra code: the chain inherits every Runnable method.",
      [
        table(
          ["Variable", "Value"],
          [
            ["summaries", json.dumps(summaries, ensure_ascii=False)[:300]],
            ["pieces[:6]", " | ".join(repr(p) for p in pieces[:6])],
          ],
        )
      ],
    ),
  ]


# ── Lab 2: parallel steps and passthrough ───────────────────────────────────────
def run_in_parallel(model, ticket):
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.runnables import RunnableParallel

  summary = ChatPromptTemplate.from_template("Summarise in one sentence: {ticket}") | model | StrOutputParser()
  mood = ChatPromptTemplate.from_template("One word for the customer's mood: {ticket}") | model | StrOutputParser()
  both = RunnableParallel(summary=summary, mood=mood)  # both get the same input, at the same time
  result = both.invoke({"ticket": ticket})  # {"summary": "...", "mood": "..."}
  return result


def a_dict_is_parallel_too(model, ticket):
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate

  reply = ChatPromptTemplate.from_template("Write a two-sentence reply to: {ticket}\nTheir mood: {mood}") | model
  mood = ChatPromptTemplate.from_template("One word for the customer's mood: {ticket}") | model | StrOutputParser()
  chain = {"ticket": lambda x: x["ticket"], "mood": mood} | reply  # the dict becomes a RunnableParallel
  result = chain.invoke({"ticket": ticket})
  return result


def keep_the_input(ticket):
  from langchain_core.runnables import RunnableParallel, RunnablePassthrough

  step = RunnableParallel(original=RunnablePassthrough(), length=lambda x: len(x["ticket"]))
  result = step.invoke({"ticket": ticket})  # the input, unchanged, next to a computed value
  return result


def add_keys_with_assign(ticket):
  from langchain_core.runnables import RunnablePassthrough

  step = RunnablePassthrough.assign(
    words=lambda x: len(x["ticket"].split()), has_order_id=lambda x: "NC-" in x["ticket"]
  )
  result = step.invoke({"ticket": ticket})  # the input dict, plus the new keys
  return result


def pick_one_key(model, ticket):
  from operator import itemgetter

  from langchain_core.prompts import ChatPromptTemplate

  prompt = ChatPromptTemplate.from_template("Translate to Hindi: {text}")
  chain = {"text": itemgetter("ticket")} | prompt | model  # itemgetter takes one key out of the input dict
  result = chain.invoke({"ticket": ticket, "customer_id": "c-1042"})
  return result


def _one_after_the_other(model, ticket: str) -> dict:
  """The same two chains as run_in_parallel, invoked in sequence: the timing baseline."""
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate

  summary = ChatPromptTemplate.from_template("Summarise in one sentence: {ticket}") | model | StrOutputParser()
  mood = ChatPromptTemplate.from_template("One word for the customer's mood: {ticket}") | model | StrOutputParser()
  return {"summary": summary.invoke({"ticket": ticket}), "mood": mood.invoke({"ticket": ticket})}


@lab(
  "lc.parallel",
  "RunnableParallel, RunnablePassthrough and itemgetter: shaping data between steps",
  uses_model=True,
  calls="2 parallel + 2 sequential + 2 + 1 calls to Mistral (7)",
  inputs=[Input("ticket", "Support message", "textarea", default=TICKET)],
)
def parallel(ticket: str):
  model = get_chat_model()
  started = time.perf_counter()
  together = run_in_parallel(model, ticket)
  parallel_ms = _ms(started)
  started = time.perf_counter()
  one_by_one = _one_after_the_other(model, ticket)
  sequential_ms = _ms(started)
  reply = a_dict_is_parallel_too(model, ticket)
  kept = keep_the_input(ticket)
  assigned = add_keys_with_assign(ticket)
  translated = pick_one_key(model, ticket)
  return [
    scorecard(
      [
        {"variant": "RunnableParallel (summary + mood together)", "ms": parallel_ms},
        {"variant": "the same two chains, one after the other", "ms": sequential_ms},
      ],
      [Metric("ms", "time", "lower", "ms")],
      title="Two independent steps: in parallel vs in sequence",
    ),
    columns(
      [
        answer(together["summary"], "summary"),
        answer(together["mood"], "mood"),
        answer(one_by_one["mood"], "mood (sequential run)"),
      ]
    ),
    table(
      ["Step", "Output"],
      [
        ["{'ticket': ..., 'mood': mood} | reply", reply.text[:200]],
        ["RunnablePassthrough + length", json.dumps({**kept, "original": "…"})],
        [
          "RunnablePassthrough.assign(words, has_order_id)",
          json.dumps({k: v for k, v in assigned.items() if k != "ticket"}),
        ],
        ["{'text': itemgetter('ticket')} | prompt | model", translated.text[:200]],
      ],
      title="Shaping the data between steps",
    ),
    callout(
      f"Run side by side, the two calls took {parallel_ms:,} ms instead of {sequential_ms:,} ms one after the other. "
      "Use RunnableParallel (or a plain dict) whenever steps don't depend on each other; use `assign` to keep the "
      "input and add to it, and `itemgetter` to hand a step just the key it needs.",
      "decision",
      "When to use which",
    ),
    example(
      "lc.l12.1",
      "RunnableParallel: several runnables on the same input",
      run_in_parallel,
      "Each branch gets the same input; the result is a dict with one key per branch.",
      [code(json.dumps(together, ensure_ascii=False, indent=1))],
    ),
    example(
      "lc.l12.2",
      "A plain dict is shorthand for RunnableParallel",
      a_dict_is_parallel_too,
      "Inside a chain, `{key: runnable_or_function}` is converted for you.",
      [answer(reply.text, "result.text")],
    ),
    example(
      "lc.l12.3",
      "RunnablePassthrough: pass the input through unchanged",
      keep_the_input,
      "Useful when a later step needs the original input next to something computed.",
      [code(json.dumps(kept, ensure_ascii=False, indent=1))],
    ),
    example(
      "lc.l12.4",
      "RunnablePassthrough.assign: add keys to the input dict",
      add_keys_with_assign,
      "The whole input stays; the new keys are added.",
      [code(json.dumps(assigned, ensure_ascii=False, indent=1))],
    ),
    example(
      "lc.l12.5",
      "itemgetter: pick one key for a step",
      pick_one_key,
      "The prompt only needs `text`; `customer_id` stays out of it.",
      [answer(translated.text, "result.text")],
    ),
  ]


# ── Lab 3: functions in chains ──────────────────────────────────────────────────
def wrap_a_function():
  from langchain_core.runnables import RunnableLambda

  def mask_order_ids(text: str) -> str:
    import re

    return re.sub(r"NC-\d+", "NC-•••••", text)

  masker = RunnableLambda(mask_order_ids)  # now it has invoke, batch, stream...
  masked = masker.invoke("Order NC-12093 is late")
  many = masker.batch(["NC-1 and NC-2", "no id here"])
  return masked, many


def functions_convert_automatically(model, ticket):
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate

  def clean(text: str) -> dict:
    return {"ticket": " ".join(text.split())[:500]}  # tidy the text, then shape it for the prompt

  chain = clean | ChatPromptTemplate.from_template("Reply in one sentence: {ticket}") | model | StrOutputParser()
  result = chain.invoke("   " + ticket + "\n\n\n")  # `clean` became a RunnableLambda when piped
  return result


def one_input_only():
  from langchain_core.runnables import RunnableLambda

  def price(x: dict) -> int:  # several values → one dict argument
    return x["bags"] * x["price_per_bag"]

  total = RunnableLambda(price).invoke({"bags": 3, "price_per_bag": 650})
  return total


def write_steps_as_python(model, ticket):
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.runnables import chain

  @chain
  def short_reply(text: str) -> str:  # normal Python, and still a Runnable
    if "NC-" not in text:
      return "Could you share your order number (it starts with NC-)?"  # no model call needed
    prompt = ChatPromptTemplate.from_template("Write a one-sentence apology for: {ticket}")
    return model.invoke(prompt.invoke({"ticket": text})).text

  with_id = short_reply.invoke(ticket)
  without_id = short_reply.invoke("My beans never came")
  return with_id, without_id


@lab(
  "lc.lambda",
  "RunnableLambda and @chain: plain Python functions inside chains",
  uses_model=True,
  calls="2 calls to Mistral",
  inputs=[Input("ticket", "Support message", "textarea", default=TICKET)],
)
def lambda_lab(ticket: str):
  model = get_chat_model()
  masked, many = wrap_a_function()
  converted = functions_convert_automatically(model, ticket)
  total = one_input_only()
  with_id, without_id = write_steps_as_python(model, ticket)
  return [
    table(
      ["What ran", "Result"],
      [
        ["RunnableLambda(mask_order_ids).invoke(...)", masked],
        ["…batch([...])", json.dumps(many)],
        ["clean | prompt | model | parser", converted],
        ["RunnableLambda(price).invoke({...})", total],
        ["@chain short_reply (has an order id)", with_id],
        ["@chain short_reply (no order id)", without_id],
      ],
      title="Python functions as chain steps",
    ),
    callout(
      "Any function becomes a step: wrap it in `RunnableLambda`, or just pipe it next to a Runnable. A step takes "
      "exactly one input, so pass a dict when it needs several values. `@chain` lets you write branching, "
      "multi-step logic as normal Python (here: ask for an order number without calling the model at all).",
      "decision",
      "When to use which",
    ),
    example(
      "lc.l13.1",
      "RunnableLambda: any function becomes a Runnable",
      wrap_a_function,
      "The function gains invoke, batch, stream and the async versions.",
      [table(["Variable", "Value"], [["masked", masked], ["many", json.dumps(many)]])],
    ),
    example(
      "lc.l13.2",
      "Functions convert automatically next to |",
      functions_convert_automatically,
      "A plain function piped with a Runnable is wrapped for you.",
      [answer(converted, "result")],
    ),
    example(
      "lc.l13.3",
      "One input only: pass a dict for several values",
      one_input_only,
      "Steps receive exactly one argument.",
      [table(["Variable", "Value"], [["total", total]])],
    ),
    example(
      "lc.l13.4",
      "@chain: multi-step logic as normal Python",
      write_steps_as_python,
      "Use if/else, loops and early returns; the result is still a Runnable you can batch or stream.",
      [
        table(
          ["Call", "Result"],
          [["short_reply.invoke(ticket)", with_id], ["short_reply.invoke('My beans never came')", without_id]],
        )
      ],
    ),
  ]


# ── Lab 4: routing ──────────────────────────────────────────────────────────────
def classify_the_route(model):
  from typing import Literal

  from langchain_core.prompts import ChatPromptTemplate
  from pydantic import BaseModel

  class Route(BaseModel):
    team: Literal["orders", "subscriptions", "safety", "general"]

  classifier = ChatPromptTemplate.from_template(
    "Which team should handle this message: orders, subscriptions, safety or general?\n{ticket}"
  ) | model.with_structured_output(Route, method="json_schema")
  return classifier


def route_with_a_function(model, classifier):
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.runnables import RunnableLambda, RunnablePassthrough

  def team_chain(team: str):
    return (
      ChatPromptTemplate.from_template(f"You are the Nimbus {team} team. Reply in one sentence: {{ticket}}")
      | model
      | StrOutputParser()
    )

  chains = {team: team_chain(team) for team in ("orders", "subscriptions", "general")}

  def pick(x: dict):  # returning a Runnable makes LCEL run it with the same input
    if x["route"].team == "safety":
      return RunnableLambda(lambda _: "A teammate will call you right away.")  # a Runnable, so it runs
    return chains[x["route"].team]

  routed = RunnablePassthrough.assign(route=classifier) | RunnableLambda(pick)
  return routed


def route_with_a_branch(model, classifier):
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.runnables import RunnableBranch, RunnablePassthrough

  general = ChatPromptTemplate.from_template("Reply in one sentence: {ticket}") | model | StrOutputParser()
  branch = RunnableBranch(
    (lambda x: x["route"].team == "safety", lambda _: "A teammate will call you right away."),
    (
      lambda x: x["route"].team == "orders",
      ChatPromptTemplate.from_template("You are the orders team. One sentence: {ticket}") | model | StrOutputParser(),
    ),
    general,  # the default when no condition matches
  )
  routed = RunnablePassthrough.assign(route=classifier) | branch
  return routed


ROUTE_TICKETS = ("t01", "t02", "t07", "t18")
EXPECTED_TEAM = {"t01": "orders", "t02": "subscriptions", "t07": "safety", "t18": "general"}


@lab(
  "lc.routing",
  "Routing: send each input down a different chain",
  uses_model=True,
  calls="about 18 calls to Mistral (each ticket is classified once per method)",
)
def routing():
  model = get_chat_model()
  classifier = classify_the_route(model)
  by_function = route_with_a_function(model, classifier)
  by_branch = route_with_a_branch(model, classifier)
  rows, function_out, branch_out = [], {}, {}
  for tid in ROUTE_TICKETS:
    text = _tickets()[tid].ticket.text
    team = classifier.invoke({"ticket": text}).team
    function_out[tid] = by_function.invoke({"ticket": text})
    branch_out[tid] = by_branch.invoke({"ticket": text})
    rows.append(
      [tid, text[:60], team, EXPECTED_TEAM[tid], "✓" if team == EXPECTED_TEAM[tid] else "✗", function_out[tid][:80]]
    )
  right = sum(r[4] == "✓" for r in rows)
  return [
    table(
      ["Ticket", "Message", "Routed to", "Expected", "", "Reply (function routing)"],
      rows,
      title=f"{len(rows)} tickets, routed by a classifier",
    ),
    callout(
      f"The classifier sent {right} of {len(rows)} tickets to the expected team. The safety ticket got a fixed "
      "message with no model call: routing is also how you keep risky cases away from generated text. Prefer a "
      "function that returns a chain (any logic, easy to test); `RunnableBranch` is the declarative alternative "
      "with (condition, runnable) pairs and a default.",
      "decision",
      "Function or RunnableBranch",
    ),
    example(
      "lc.l14.1",
      "A classifier chain decides the route",
      classify_the_route,
      "Structured output with a Literal: the route can only be one of the allowed teams.",
      [table(["Ticket", "team"], [[r[0], r[2]] for r in rows])],
    ),
    example(
      "lc.l14.2",
      "Route with a function that returns a chain (recommended)",
      route_with_a_function,
      "Plain Python decides; the Runnable it returns runs on the same input. Return a Runnable: a plain "
      "function would be handed back as the result instead of being run.",
      [table(["Ticket", "Reply"], [[t, function_out[t][:160]] for t in ROUTE_TICKETS])],
    ),
    example(
      "lc.l14.3",
      "The same with RunnableBranch",
      route_with_a_branch,
      "(condition, runnable) pairs, checked in order, and a default last.",
      [table(["Ticket", "Reply"], [[t, branch_out[t][:160]] for t in ROUTE_TICKETS])],
    ),
  ]


# ── Lab 5: retries, fallbacks and config ────────────────────────────────────────
def retry_a_flaky_step():
  from langchain_core.runnables import RunnableLambda

  attempts = []

  def flaky(text: str) -> str:
    attempts.append(len(attempts) + 1)
    if len(attempts) < 3:
      raise ConnectionError("temporary network problem")  # fails twice, then works
    return f"processed: {text}"

  reliable = RunnableLambda(flaky).with_retry(
    retry_if_exception_type=(ConnectionError,), stop_after_attempt=3, wait_exponential_jitter=False
  )
  result = reliable.invoke("order NC-12093")
  return result, attempts


def fall_back_to_another_provider():
  from langchain.chat_models import init_chat_model

  # The primary is "down" (nothing listens at this address); the backup is a different provider
  primary = init_chat_model(
    "ministral-14b-latest", model_provider="mistralai", endpoint="http://127.0.0.1:9", max_retries=0
  )
  backup = init_chat_model("openai/gpt-oss-20b", model_provider="groq")
  safe = primary.with_fallbacks([backup])  # tried in order until one succeeds
  reply = safe.invoke("Say 'the fallback answered' and nothing else.")
  return reply


def pass_and_read_config():
  from langchain_core.runnables import RunnableConfig, RunnableLambda

  def tag_aware(text: str, config: RunnableConfig) -> str:  # ask for `config` to receive it
    return f"{text} | tags={config.get('tags')} | ticket_id={config.get('metadata', {}).get('ticket_id')}"

  result = RunnableLambda(tag_aware).invoke(
    "hello", config={"run_name": "greeting", "tags": ["support", "lab"], "metadata": {"ticket_id": "t13"}}
  )
  return result


def change_a_setting_per_call(model):
  from langchain_core.runnables import ConfigurableField

  configurable = model.configurable_fields(temperature=ConfigurableField(id="temperature", name="Temperature"))
  prompt = "Invent a name for a Nimbus coffee blend. Reply with the name only."
  calm = [configurable.invoke(prompt, config={"configurable": {"temperature": 0}}).text for _ in range(2)]
  wild = [configurable.invoke(prompt, config={"configurable": {"temperature": 1}}).text for _ in range(2)]
  return calm, wild


@lab(
  "lc.reliability",
  "Fallbacks, retries and RunnableConfig: robust, configurable chains",
  uses_model=True,
  calls="1 call to Groq + 4 to Mistral",
)
def reliability():
  started = time.perf_counter()
  result, attempts = retry_a_flaky_step()
  retry_ms = _ms(started)
  started = time.perf_counter()
  reply = fall_back_to_another_provider()
  fallback_ms = _ms(started)
  configured = pass_and_read_config()
  calm, wild = change_a_setting_per_call(
    init_chat_model("ministral-14b-latest", model_provider="mistralai", temperature=0)
  )
  return [
    table(
      ["Technique", "What happened", "Time"],
      [
        [
          "with_retry(stop_after_attempt=3)",
          f"failed on attempts 1–2, succeeded on {attempts[-1]}: {result}",
          f"{retry_ms:,} ms",
        ],
        [
          "with_fallbacks([groq model])",
          f"Mistral unreachable → answered by {reply.response_metadata.get('model_name')}: {reply.text.strip()}",
          f"{fallback_ms:,} ms",
        ],
        ["config (run_name, tags, metadata)", configured, "—"],
        ["configurable_fields(temperature)", f"temperature 0: {calm} · temperature 1: {wild}", "—"],
      ],
      title="Making a chain robust and configurable",
    ),
    callout(
      "Retry what can succeed on a second try (a flaky network call, a reply that didn't parse); fall back to a "
      "different provider for outages, because retrying a provider that is down only adds waiting. Config carries "
      "tags and metadata to every step (and into traces), and configurable fields change a setting per call "
      "without rebuilding the chain.",
      "decision",
      "Retry, or fall back",
    ),
    example(
      "lc.l15.1",
      "with_retry(): re-run a step that fails transiently",
      retry_a_flaky_step,
      "Only retry exception types that can go away; cap the attempts.",
      [table(["Variable", "Value"], [["attempts", str(attempts)], ["result", result]])],
    ),
    example(
      "lc.l15.2",
      "with_fallbacks(): switch to a backup when the primary fails",
      fall_back_to_another_provider,
      "The backup runs only if the primary raises. Use a different provider, so one outage can't take both down.",
      [
        table(
          ["Expression", "Value"],
          [
            ["reply.response_metadata['model_name']", reply.response_metadata.get("model_name")],
            ["reply.text", reply.text.strip()],
            ["time", f"{fallback_ms:,} ms"],
          ],
        )
      ],
    ),
    example(
      "lc.l15.3",
      "RunnableConfig: run_name, tags, metadata, read inside a step",
      pass_and_read_config,
      "Add a `config` parameter to a function step to read the config of the current call.",
      [table(["Variable", "Value"], [["result", configured]])],
    ),
    example(
      "lc.l15.4",
      "configurable_fields(): change a setting per call",
      change_a_setting_per_call,
      'One chain object; the setting comes from `config["configurable"]` on each call.',
      [table(["Variable", "Value"], [["calm", json.dumps(calm)], ["wild", json.dumps(wild)]])],
    ),
  ]


# ── Lab 6: callbacks and events ─────────────────────────────────────────────────
def a_callback_handler(model, ticket):
  from langchain_core.callbacks import BaseCallbackHandler
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate

  class EventLog(BaseCallbackHandler):
    def __init__(self):
      self.events = []

    def on_chain_start(self, serialized, inputs, **kwargs):
      self.events.append(("chain start", kwargs.get("name")))

    def on_chat_model_start(self, serialized, messages, **kwargs):
      self.events.append(("model start", f"{len(messages[0])} message(s)"))

    def on_llm_end(self, response, **kwargs):
      usage = response.generations[0][0].message.usage_metadata
      self.events.append(("model end", f"{usage['total_tokens']} tokens"))

    def on_chain_end(self, outputs, **kwargs):
      self.events.append(("chain end", str(outputs)[:40]))

  log = EventLog()
  chain = ChatPromptTemplate.from_template("Reply in one sentence: {ticket}") | model | StrOutputParser()
  reply = chain.invoke({"ticket": ticket}, config={"callbacks": [log]})  # every step reports to it
  return log.events, reply


def stream_every_event(model, ticket):
  import asyncio

  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate

  chain = ChatPromptTemplate.from_template("Reply in one sentence: {ticket}") | model | StrOutputParser()

  async def collect():
    seen = []
    async for event in chain.astream_events({"ticket": ticket}, version="v2"):
      seen.append((event["event"], event["name"]))  # on_chain_start, on_chat_model_stream, on_parser_end...
    return seen

  events = asyncio.run(collect())
  return events


@lab(
  "lc.callbacks",
  "Callbacks and astream_events: seeing inside a running chain",
  uses_model=True,
  calls="2 calls to Mistral",
  inputs=[Input("ticket", "Support message", "textarea", default=TICKET)],
)
def callbacks(ticket: str):
  model = get_chat_model()
  events, reply = a_callback_handler(model, ticket)
  streamed = stream_every_event(model, ticket)
  kinds: dict[str, int] = {}
  for kind, _ in streamed:
    kinds[kind] = kinds.get(kind, 0) + 1
  return [
    table(["Event", "Detail"], [list(e) for e in events], title="A callback handler, passed in config"),
    table(
      ["astream_events kind", "Count"], [[k, v] for k, v in kinds.items()], title="The same run as an event stream"
    ),
    answer(reply, "The reply"),
    callout(
      "Callbacks see every start and end inside a chain: Nimbus uses one (`UsageRecorder`) to record each model call's "
      "model, tokens and time. `astream_events` gives the same events as an async stream: a UI can show the "
      "classification as soon as it's ready, then the reply token by token.",
      "decision",
      "What they're for",
    ),
    example(
      "lc.l16.1",
      "A callback handler: methods called at each start and end",
      a_callback_handler,
      "Subclass BaseCallbackHandler and implement the events you care about.",
      [table(["Event", "Detail"], [list(e) for e in events])],
    ),
    example(
      "lc.l16.2",
      "Pass callbacks through config: every step reports",
      a_callback_handler,
      "Callbacks in `config` are inherited by every step of the chain: the prompt, the model and the parser "
      "all reported to the same handler.",
      [table(["Variable", "Value"], [["len(events)", len(events)], ["reply", reply]])],
    ),
    example(
      "lc.l16.3",
      "astream_events: the events as an async stream",
      stream_every_event,
      "One event per start, end and streamed token, from every step.",
      [table(["Kind", "Count"], [[k, v] for k, v in kinds.items()])],
    ),
  ]


# ── Production path labs: the support pipeline ──────────────────────────────────
HISTORY = PROJECT_DIR / ".state" / "pipeline_drills.jsonl"
DRILL_METRICS = [
  Metric("success", "success", "higher", "ratio", "Tickets that got a triage and a reply"),
  Metric("category_acc", "category", "higher", "ratio"),
  Metric("latency_p50_ms", "p50", "lower", "ms"),
  Metric("latency_p95_ms", "p95", "lower", "ms"),
  Metric("tokens_per_ticket", "tokens", "lower", "int", "Per ticket, both steps"),
  Metric("cost_per_1k_usd", "$ / 1k", "lower", "usd", "At a reference paid price"),
]


def _ticket_options() -> list[str]:
  return [f"{lt.ticket.id} · {lt.ticket.text}" for lt in _tickets().values()]


@lab(
  "lc.pipeline-one",
  "Run one ticket through the pipeline, streamed",
  uses_model=True,
  calls="2 calls to Mistral (1 for safety tickets)",
  inputs=[Input("ticket", "Ticket", "select", default=_ticket_options()[12], options=_ticket_options())],
)
def pipeline_one(ticket: str):
  ticket_id, text = ticket.split(" · ", 1)
  pipeline = support_pipeline()
  timeline, tokens, started = [], [], time.perf_counter()

  async def watch():
    async for kind, value in stream_events(pipeline, text):
      if kind == "triage":
        timeline.append(
          ["triage ready", _ms(started), f"{value.category} · {value.urgency} · needs_human={value.needs_human}"]
        )
      else:
        if not tokens:
          timeline.append(["first reply token", _ms(started), repr(value)])
        tokens.append(value)

  asyncio.run(watch())
  timeline.append(["reply finished", _ms(started), f"{len(tokens)} pieces"])
  result = run(pipeline, ticket_id, text)
  return [
    table(
      ["Moment", "ms", "Detail"], timeline, title="Streamed with astream_events: what the support agent sees, when"
    ),
    answer("".join(tokens), "The reply"),
    table(
      ["Step", "Model", "Input tokens", "Output tokens", "ms"],
      [
        [
          next((t for t in c.tags if t in ("triage", "reply")), "?"),
          c.model,
          c.input_tokens,
          c.output_tokens,
          round(c.ms),
        ]
        for c in result.calls
      ],
      title="The same ticket with invoke(): every model call, recorded by a callback",
    ),
  ]


@lab(
  "lc.pipeline-drill",
  "Outage drill: the pipeline when the main provider is down",
  uses_model=True,
  calls="4 × tickets × 2 calls (Mistral and Groq, about 2 minutes for 6 tickets)",
  inputs=[Input("tickets", "Tickets", "number", default=6, min=3, max=10)],
)
def pipeline_drill(tickets: int):
  from datetime import UTC, datetime

  s = get_settings()
  primary, backup = get_chat_model(), get_chat_model(role="fallback")
  down = init_chat_model(s.chat_model, model_provider=s.chat_provider, endpoint="http://127.0.0.1:9", max_retries=0)
  sample = list(_tickets().values())[:: max(1, 20 // tickets)][:tickets]
  conditions = {
    "healthy, no fallback": build_pipeline(primary, None, s.triage_prompt_version, s.triage_method),
    "Mistral down, no fallback": build_pipeline(down, None, s.triage_prompt_version, s.triage_method),
    "Mistral down, Groq fallback": build_pipeline(down, backup, s.triage_prompt_version, s.triage_method),
    "healthy, Groq fallback ready": build_pipeline(primary, backup, s.triage_prompt_version, s.triage_method),
  }
  reports = [drill(name, p, sample) for name, p in conditions.items()]
  at = datetime.now(UTC).isoformat(timespec="seconds")
  HISTORY.parent.mkdir(exist_ok=True)
  with HISTORY.open("a") as f:
    for r in reports:
      f.write(json.dumps({"at": at, **r.summary()}) + "\n")
  rows = [{**r.summary(), "variant": r.condition} for r in reports]
  healthy, down_only, with_fb, ready = rows
  return [
    scorecard(rows, DRILL_METRICS, title=f"{len(sample)} tickets under four conditions"),
    table(
      ["Condition", "Calls answered by"],
      [
        [r.condition, ", ".join(f"{m}: {n}" for m, n in r.answered_by.items()) or "— (no call succeeded)"]
        for r in reports
      ],
      title="Which model actually answered",
    ),
    callout(
      f"With Mistral down, the pipeline without a fallback served {down_only['success']:.0%} of tickets; with the Groq "
      f"fallback it served {with_fb['success']:.0%} (category accuracy {with_fb['category_acc']:.2f} vs "
      f"{healthy['category_acc']:.2f} on the healthy primary). The backup isn't free: "
      f"p95 {with_fb['latency_p95_ms']:,} ms "
      f"vs {healthy['latency_p95_ms']:,} ms, and ${with_fb['cost_per_1k_usd']:.2f} "
      f"vs ${healthy['cost_per_1k_usd']:.2f} "
      f"per 1,000 tickets ({with_fb['tokens_per_ticket']:,} vs {healthy['tokens_per_ticket']:,} tokens per ticket, "
      "because it's a reasoning model). That's fine for an outage, so the fallback stays on: when the primary is "
      f"healthy it never runs (answered by: {', '.join(ready['answered_by'])}).",
      "decision",
      "What the drill shows",
    ),
  ]
