"""Shipping to production · Module 1 labs: what every call costs, where the tokens go, and the levers that cut them."""

import json
import time
from functools import lru_cache

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage

from academy.labs.runtime import Input, Metric, answer, bars, callout, code, columns, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR
from nimbus.llm import get_chat_model, model_name

DATA = PROJECT_DIR / "data" / "nimbus"
HELP_FILES = ["faq.html", "refunds_and_returns.html", "subscriptions.txt", "products.txt", "privacy_notice.md"]
SHORT_PROMPT = "You are Nimbus Coffee's support assistant. Answer in at most two short sentences."
FOLLOW_UPS = ["And do the points expire?", "How many points would a ₹1,300 order earn?"]
RELEVANT = (
  "Loyalty: 1 point for every 10 rupees spent. 500 points can be redeemed for a free 250 g bag. "
  "Points expire 12 months after they are earned."
)


@lru_cache
def _help_centre() -> str:
  """Every customer-facing help document, as text: what a 'put everything in the prompt' assistant sends."""
  parts = []
  for name in HELP_FILES:
    raw = (DATA / name).read_text()
    text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True) if name.endswith(".html") else raw
    parts.append(f"[{name}]\n{text}")
  return "\n\n".join(parts)


# ── Lab 1: tokens and cost ──────────────────────────────────────────────────────
# Each learning point is a small function: the page shows its body, and the lab runs it.


def read_usage(model, messages):
  reply = model.invoke(messages)
  usage = reply.usage_metadata  # input, output and total tokens (+ details like reasoning) for THIS call
  return reply, usage


def estimate_before_sending(messages):
  from langchain_core.messages.utils import count_tokens_approximately

  estimate = count_tokens_approximately(messages)  # about 4 characters per token: no API call, instant
  return estimate


def total_for_a_run(calls):
  from langchain_core.callbacks import get_usage_metadata_callback

  with get_usage_metadata_callback() as callback:  # adds up every model call made inside the block
    replies = {label: read_usage(model, messages) for label, (model, messages) in calls.items()}
  totals = callback.usage_metadata  # {model name: summed usage}, across providers
  return replies, totals


def price_it(usage):
  # Keep prices in configuration: they change. USD per million tokens; a reference paid model (Claude Sonnet 5.5,
  # Oct 2026), because the course runs on free plans.
  price_per_million = {"input": 2.00, "output": 10.00}
  cost = (
    usage["input_tokens"] * price_per_million["input"] + usage["output_tokens"] * price_per_million["output"]
  ) / 1_000_000
  cost_per_1k_requests = cost * 1000
  return cost_per_1k_requests


def a_conversation_resends_everything(model, system, questions):
  from langchain_core.messages import HumanMessage, SystemMessage

  messages = [SystemMessage(system)]
  input_per_turn = []
  for question in questions:
    messages.append(HumanMessage(question))
    reply = model.invoke(messages)  # the WHOLE history goes again, every turn: an agent loop does the same
    messages.append(reply)
    input_per_turn.append(reply.usage_metadata["input_tokens"])
  return input_per_turn


def cheaper_settings():
  from nimbus.llm import get_chat_model

  capped = get_chat_model(max_tokens=80)  # a hard cap on output tokens: the most expensive kind
  small_prompt = [SystemMessage(f"{SHORT_PROMPT}\n\n{RELEVANT}")]  # only what this question needs
  return capped, small_prompt


@lab(
  "prod.tokens-cost",
  "Tokens and cost: what every call costs, and the levers that cut it",
  uses_model=True,
  calls="6 calls: 5 to Mistral, 1 to Groq",
  inputs=[
    Input("question", "Customer question", "text", default="How do Nimbus Beans points work?"),
  ],
)
def tokens_cost(question: str):
  """The same question three ways (everything in the prompt, a short capped prompt, a reasoning model), then a
  three-turn conversation: estimated vs actual tokens, totals across providers, cost per 1,000 requests."""
  model = get_chat_model()
  groq = get_chat_model(role="reasoning")
  capped, small_prompt = cheaper_settings()
  everything = [SystemMessage(f"{SHORT_PROMPT}\n\nHelp centre:\n{_help_centre()}"), HumanMessage(question)]
  calls = {
    "everything in the prompt": (model, everything),
    "relevant text + max_tokens=80": (capped, [*small_prompt, HumanMessage(question)]),
    "reasoning model (Groq)": (groq, [*small_prompt, HumanMessage(question)]),
  }
  estimates = {label: estimate_before_sending(messages) for label, (_, messages) in calls.items()}
  started = time.perf_counter()
  replies, totals = total_for_a_run(calls)
  seconds = time.perf_counter() - started
  rows = []
  for label, (_, usage) in replies.items():
    details = usage.get("output_token_details") or {}
    rows.append(
      {
        "variant": label,
        "estimated_in": estimates[label],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "reasoning_tokens": details.get("reasoning", 0),
        "cost_per_1k_usd": round(price_it(usage), 3),
      }
    )
  turns = a_conversation_resends_everything(model, f"{SHORT_PROMPT}\n\n{_help_centre()}", [question, *FOLLOW_UPS])
  first, *_ = replies.values()
  return [
    scorecard(
      rows,
      [
        Metric("estimated_in", "estimated input", "none", "int", "count_tokens_approximately, before sending"),
        Metric("input_tokens", "input tokens", "lower", "int", "What the provider counted"),
        Metric("output_tokens", "output tokens", "lower", "int", "Reasoning tokens included"),
        Metric("reasoning_tokens", "of which reasoning", "lower", "int", "Thinking you pay for but never see"),
        Metric("cost_per_1k_usd", "$ per 1,000", "lower", "usd", "At $2 / $10 per million in / out tokens"),
      ],
      title=f"One question, three ways ({seconds:.1f} s for all three)",
    ),
    columns([answer(reply.text, label) for label, (reply, _) in replies.items()], title="The answers"),
    bars(
      [{"label": f"turn {i + 1}", "values": {"input tokens": n}} for i, n in enumerate(turns)],
      title=f"A {len(turns)}-turn conversation with the help centre in the prompt: input tokens per turn",
      unit="tokens",
    ),
    callout(
      f"Putting every help document in the prompt cost {rows[0]['input_tokens']:,} input tokens for one question, "
      f"against {rows[1]['input_tokens']:,} with only the relevant text: "
      f"{rows[0]['cost_per_1k_usd'] / max(rows[1]['cost_per_1k_usd'], 1e-9):.1f}× the price. In a conversation every "
      f"turn resends the history ({turns[0]:,} → {turns[-1]:,} input tokens by turn {len(turns)}), and agents loop "
      "the same way. Levers, biggest first: send less context (retrieval), cap output with max_tokens, use a "
      "smaller model for easy jobs, and measure per request so you can see which one moved the number.",
      "decision",
      "Where the money goes",
    ),
  ] + [
    example(
      "lc.l59.1",
      "usage_metadata on every reply",
      read_usage,
      "Every AIMessage carries the token counts the provider billed for that call.",
      [code(_json(first[1]), "json", "usage")],
    ),
    example(
      "lc.l59.2",
      "Estimating tokens before sending",
      estimate_before_sending,
      "`count_tokens_approximately` counts about 4 characters per token. It's free and instant: good enough for "
      "budgets and trimming, not for billing (compare with what the provider counted).",
      [
        table(
          ["Messages", "Estimated", "Provider counted"],
          [[r["variant"], r["estimated_in"], r["input_tokens"]] for r in rows],
        )
      ],
    ),
    example(
      "lc.l59.3",
      "Totals for a whole run, across models",
      total_for_a_run,
      "`get_usage_metadata_callback()` records every model call inside the `with` block, whichever provider, and "
      "sums them per model name. Here: two Mistral calls and one Groq call.",
      [code(_json(totals), "json", "callback.usage_metadata")],
    ),
    example(
      "lc.l59.4",
      "Turning tokens into money",
      price_it,
      "Cost is input tokens × input price + output tokens × output price. Output tokens usually cost several times "
      "more, and a reasoning model's hidden thinking is output.",
      [table(["Call", "$ per 1,000 requests"], [[r["variant"], f"${r['cost_per_1k_usd']:.3f}"] for r in rows])],
    ),
    example(
      "lc.l59.5",
      "Why conversations and agents cost more than you expect",
      a_conversation_resends_everything,
      "A model has no memory: each turn resends the system prompt and every earlier message. Input tokens grow every "
      "turn, and an agent's tool loop does exactly the same within one request.",
      [
        table(
          ["Turn", "Question", "Input tokens"],
          [[i + 1, q, n] for i, (q, n) in enumerate(zip([question, *FOLLOW_UPS], turns, strict=True))],
        )
      ],
    ),
    example(
      "lc.l59.6",
      "Levers: less context, capped output, a smaller model",
      cheaper_settings,
      "`max_tokens` caps the output (a cut-off answer ends with finish reason `length`, so check it). Sending only "
      "the relevant text is the biggest saving here.",
      [
        table(
          ["Call", "Input", "Output", "Finish reason", "Model"],
          [
            [
              label,
              usage["input_tokens"],
              usage["output_tokens"],
              reply.response_metadata.get("finish_reason"),
              model_name(calls[label][0]),
            ]
            for label, (reply, usage) in replies.items()
          ],
        )
      ],
    ),
  ]


def _json(value) -> str:
  import json

  return json.dumps(value, indent=2, default=str)


# ── Lab 2: staying under the provider's rate limit ─────────────────────────────
def spaced_calls(requests_per_second: float | None, n: int = 5) -> list[float]:
  from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
  from langchain_core.messages import AIMessage
  from langchain_core.rate_limiters import InMemoryRateLimiter

  limiter = (
    InMemoryRateLimiter(requests_per_second=requests_per_second, check_every_n_seconds=0.05, max_bucket_size=1)
    if requests_per_second
    else None
  )
  model = GenericFakeChatModel(messages=iter([AIMessage("ok")] * n), rate_limiter=limiter)  # any chat model takes one
  started, times = time.perf_counter(), []
  for _ in range(n):
    model.invoke("hi")
    times.append(round(time.perf_counter() - started, 2))
  return times


def one_limiter_per_provider():
  from nimbus.llm.models import rate_limiter

  # Provider limits count every request from your key, whichever model it names: the default and the small model
  # share ONE limiter (cached per provider), so together they stay under the limit.
  return rate_limiter("mistralai", 0.4) is rate_limiter("mistralai", 0.4)


@lab("prod.rate-limiter", "InMemoryRateLimiter: stay under the provider's limit", calls="No model calls")
def rate_limiter_lab():
  """Five calls to a fake model with no limiter and with 2 requests per second, and why Nimbus keeps one limiter per
  provider."""
  free = spaced_calls(None)
  limited = spaced_calls(2.0)
  shared = one_limiter_per_provider()
  view = table(
    ["", "call finished at (s)"],
    [["no limiter", ", ".join(map(str, free))], ["2 requests/second", ", ".join(map(str, limited))]],
  )
  return [
    view,
    callout(
      f"With the limiter, five calls were spaced about 0.5 s apart ({limited[-1]:.1f} s in all) instead of firing at "
      f"once ({free[-1]:.2f} s): bursts never reach the provider as 429s. One limiter per provider is shared by every "
      f"model on it: {shared}.",
      "info",
      "Space requests out",
    ),
    example("lc.l58.3", "InMemoryRateLimiter", spaced_calls, "", [view]),
  ]


# ── Lab 3: what each call cost, recorded as runs ───────────────────────────────
def collect_runs(model, question: str):
  from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

  collector = RunCollectorCallbackHandler()  # the runs a tracer would send, kept in memory: nothing leaves this machine
  model.invoke(question, {"callbacks": [collector]})
  run = collector.traced_runs[0]
  usage = run.outputs["generations"][0][0]["message"]["kwargs"]["usage_metadata"]  # tokens, on the run itself
  return run, usage


def log_an_unseen_call(question: str):
  import os

  import httpx
  from langsmith.run_trees import RunTree

  # A model call LangChain doesn't see (Mistral's REST API through httpx): describe it as an "llm" run yourself
  run = RunTree(name="mistral-rest", run_type="llm", inputs={"messages": [{"role": "user", "content": question}]})
  run.extra = {"metadata": {"ls_provider": "mistral", "ls_model_name": "ministral-14b-latest"}}
  body = {"model": "ministral-14b-latest", "messages": [{"role": "user", "content": question}], "max_tokens": 40}
  with httpx.Client(timeout=30) as http:
    reply = http.post(
      "https://api.mistral.ai/v1/chat/completions",
      json=body,
      headers={"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}"},
    ).json()
  usage = reply["usage"]
  run.end(
    outputs={
      "choices": reply["choices"],
      "usage_metadata": {
        "input_tokens": usage["prompt_tokens"],
        "output_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
      },
    }
  )
  return run  # with a LangSmith key: run.post() sends it; here it stays local


def priced(runs_usage: list[tuple[str, dict]]):
  from nimbus.config import get_settings
  from nimbus.llm.routing import cost

  # LangSmith multiplies tokens by its price table (Settings → Model pricing); the same, in our config
  return [(model, usage, cost({model: usage}, get_settings())) for model, usage in runs_usage]


def run_stats(runs: list) -> dict:
  # What client.get_run_stats(project_name=...) returns for a LangSmith project, computed from runs we hold
  latencies = sorted((r.end_time - r.start_time).total_seconds() for r in runs)
  tokens = [r.outputs["generations"][0][0]["message"]["kwargs"]["usage_metadata"]["total_tokens"] for r in runs]
  return {
    "run_count": len(runs),
    "total_tokens": sum(tokens),
    "latency_p50": latencies[len(latencies) // 2],
    "latency_p99": latencies[-1],
    "error_rate": sum(r.error is not None for r in runs) / len(runs),
  }


@lab(
  "smith.cost", "Tokens and cost on traces", uses_model=True, calls="4 calls to Mistral (3 through LangChain, 1 REST)"
)
def smith_cost():
  """Three LangChain calls recorded as runs (in memory: no key, nothing sent), a REST call LangChain can't see
  described as an llm run with provider and model metadata, each priced from config, and project-style totals."""
  model = get_chat_model()
  questions = ["Name a coffee in two words.", "What is cold brew, in ten words?", "Say thanks in Tamil."]
  collected = [collect_runs(model, q) for q in questions]
  rest = log_an_unseen_call("Name one Indian coffee region in two words.")
  rows = [(r.extra.get("metadata", {}).get("ls_model_name", "?"), u) for r, u in collected]
  rows.append((rest.extra["metadata"]["ls_model_name"], rest.outputs["usage_metadata"]))
  prices = priced(rows)
  stats = run_stats([r for r, _ in collected])
  runs = table(
    ["run", "model", "input", "output", "US$"],
    [
      [
        f"run {i + 1}" if i < 3 else "REST (logged by hand)",
        m,
        str(u["input_tokens"]),
        str(u["output_tokens"]),
        f"{c:.7f}",
      ]
      for i, (m, u, c) in enumerate(prices)
    ],
  )
  return [
    runs,
    table(
      ["get_run_stats-style totals", "value"],
      [[k, f"{v:.2f}" if isinstance(v, float) else str(v)] for k, v in stats.items()],
    ),
    callout(
      "Every LangChain run carries its token usage; a call LangChain can't see needs ls_provider, ls_model_name and "
      "usage_metadata added by hand, or it shows no tokens and no cost. Prices live in config, and LangSmith's own "
      "table needs an entry for providers it doesn't price. Nothing here was sent anywhere: add a LangSmith key to "
      "send the same runs.",
      "info",
      "Tokens and cost per run",
    ),
    example("smith.l07.1", "Tokens on every LangChain run", collect_runs, "", [runs]),
    example(
      "smith.l07.2",
      "Logging a call LangChain can't see",
      log_an_unseen_call,
      "",
      [code(json.dumps(rest.outputs["usage_metadata"]), "json")],
    ),
    example("smith.l07.3", "Cost from a price table", priced, "", [runs]),
    example("smith.l07.4", "Project totals", run_stats, "", [code(json.dumps(stats, default=str), "json")]),
  ]


# ── Lab 4: when the provider fails ─────────────────────────────────────────────
def a_timeout():
  from nimbus.llm import get_chat_model

  impatient = get_chat_model(timeout_s=1, max_retries=0)  # never wait forever for one call
  started = time.perf_counter()
  try:
    impatient.invoke("Write a 400-word story about a coffee bean.")
    return f"answered in {time.perf_counter() - started:.1f} s"
  except Exception as error:  # noqa: BLE001
    return f"{type(error).__name__} after {time.perf_counter() - started:.1f} s"


def retries_with_backoff():
  from langchain_core.runnables import RunnableLambda

  attempts = []

  def flaky(prompt: str) -> str:
    attempts.append(time.perf_counter())
    if len(attempts) < 3:
      raise ConnectionError("503 overloaded")  # transient: worth retrying (a 401 never is)
    return "ok"

  result = (
    RunnableLambda(flaky)
    .with_retry(stop_after_attempt=4, wait_exponential_jitter=True, retry_if_exception_type=(ConnectionError,))
    .invoke("hi")
  )
  gaps = [round(b - a, 2) for a, b in zip(attempts, attempts[1:], strict=False)]
  return result, len(attempts), gaps


def broken_primary():
  from nimbus.llm import get_chat_model

  return get_chat_model(api_key="not-a-real-key", max_retries=0)  # what a provider outage looks like to us


def fallback_answers():
  from nimbus.llm import get_chat_model
  from nimbus.llm.resilience import with_fallback

  resilient = with_fallback(broken_primary(), get_chat_model(role="fallback"))  # Mistral → Groq
  reply = resilient.invoke("In five words: why do people like cold brew?")
  return reply.response_metadata.get("model_name"), reply.text


def circuit_breaker():
  from nimbus.llm import get_chat_model
  from nimbus.llm.resilience import CircuitBreaker

  breaker = CircuitBreaker(broken_primary(), get_chat_model(role="fallback"), threshold=2, cooldown_s=30)
  rows = []
  for n in range(1, 5):
    started = time.perf_counter()
    breaker.invoke("In three words: what is a latte?")
    rows.append([f"request {n}", breaker.last_route, breaker.state, f"{time.perf_counter() - started:.2f} s"])
  return rows


@lab(
  "prod.reliability",
  "When the provider fails: timeouts, retries, fallbacks, a circuit breaker",
  uses_model=True,
  calls="~6 calls: 1 to Mistral, ~5 to Groq",
)
def reliability():
  """A 1-second timeout on a long answer, a flaky step retried with backoff, Mistral down and Groq answering, and a
  circuit breaker that stops calling the broken provider."""
  timeout = a_timeout()
  result, attempts, gaps = retries_with_backoff()
  answered_by, text = fallback_answers()
  breaker = circuit_breaker()
  view = table(["request", "route", "circuit", "time"], breaker)
  return [
    table(
      ["", "result"],
      [
        ["timeout (1 s)", timeout],
        ["retries", f"{result!r} after {attempts} attempts, waiting {gaps} s between them"],
        ["fallback", f"answered by {answered_by}: {text}"],
      ],
    ),
    view,
    callout(
      f"The fallback answered while the primary was down. With the breaker, requests 1 and 2 tried the broken "
      f"primary first; from request 3 the circuit was {breaker[2][2]} and the primary was skipped ({breaker[2][3]} "
      f"against {breaker[0][3]}). Retry only transient errors: a bad key fails the same way every time.",
      "decision",
      "Timeouts, transient retries, a fallback and a breaker",
    ),
    example("prod.l06.1", "Timeouts on the client", a_timeout, "", [code(timeout, "text")]),
    example(
      "prod.l06.2",
      "Retries with backoff, for transient errors",
      retries_with_backoff,
      "",
      [code(f"{attempts} attempts, gaps {gaps}", "text")],
    ),
    example(
      "prod.l06.3",
      "A fallback on another provider",
      fallback_answers,
      "nimbus/llm/resilience.py with_fallback",
      [answer(text, f"answered by {answered_by}")],
    ),
    example("prod.l06.4", "A circuit breaker", circuit_breaker, "nimbus/llm/resilience.py CircuitBreaker", [view]),
  ]


# ── Lab 5: the right model, priced, within budget ──────────────────────────────
ROUTED = [
  "How much is a cold brew?",
  "Is the Oat Latte vegan?",
  "My order arrived late and the bag was torn, can I get a refund, and how long will it take?",
  "Compare your three blends for a French press, and say which suits someone who dislikes bitterness.",
]


def usage_per_request(model, question):
  from langchain_core.callbacks import get_usage_metadata_callback

  with get_usage_metadata_callback() as usage:  # every model call inside the block, summed per model
    reply = model.invoke(question)
  return reply.text, usage.usage_metadata


def routed_and_priced(question: str):
  from nimbus.config import get_settings
  from nimbus.llm import get_chat_model
  from nimbus.llm.routing import cost, route

  role = route(question)  # a cheap rule, before any model call
  messages = [SystemMessage(f"{POLICY} Answer in one or two sentences."), HumanMessage(question)]
  text, usage = usage_per_request(get_chat_model(role=role), messages)
  return role, text, usage, cost(usage, get_settings())


def budget_app(budget: int):
  from fastapi.testclient import TestClient
  from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
  from langchain_core.messages import AIMessage

  from nimbus.api.app import create_app
  from nimbus.api.limits import key_hash
  from nimbus.config import Settings

  settings = Settings(
    api_key_hashes={key_hash("k-free-1"): "free", key_hash("k-pro-1"): "pro"}, daily_token_budget=budget
  )
  usage = {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}  # a fake that reports usage, like a real one
  replies = iter([AIMessage("A short reply.", usage_metadata=usage)] * 50)
  return TestClient(create_app(settings=settings, chat_model=GenericFakeChatModel(messages=replies)))


def spend_until_refused(client, key: str, n: int = 8) -> list[str]:
  out = []
  for _ in range(n):
    r = client.post("/support/reply", json={"message": "Where is my order?"}, headers={"X-API-Key": key})
    out.append(str(r.status_code))
  return out


@lab(
  "prod.routing-budgets",
  "Spend on the right model: routing, cost per request, budgets",
  uses_model=True,
  calls="4 calls to Mistral",
)
def routing_budgets():
  """Four questions routed to the small or the default model, each priced from config; then per-caller daily token
  budgets in Redis, refused with 429 before any model call. Needs Redis (docker compose up -d)."""
  import redis as redis_lib

  rows = []
  for q in ROUTED:
    role, text, usage, usd = routed_and_priced(q)
    tokens = sum(u["total_tokens"] for u in usage.values())
    rows.append([q, role, ", ".join(usage), str(tokens), f"{usd:.7f}", text[:140]])
  view = table(["question", "routed to", "model", "tokens", "US$", "answer"], rows)
  try:
    with budget_app(40) as client:  # a tiny budget (25 tokens per reply), so this lab reaches it
      client.app.state.redis.delete(*client.app.state.redis.keys("tokens:*") or ["none"])
      statuses = spend_until_refused(client, "k-pro-1", 3)
      other = client.post("/support/reply", json={"message": "hi"}, headers={"X-API-Key": "k-free-1"})
    budget = table(
      ["", "statuses"], [["caller A, budget 40 tokens", " → ".join(statuses)], ["caller B", str(other.status_code)]]
    )
    refused = "429" in statuses
  except (redis_lib.RedisError, AttributeError) as error:
    budget, refused = table(["", "statuses"], [["Redis unavailable", str(error)[:100]]]), False
  return [
    view,
    budget,
    callout(
      f"{sum(r[1] == 'small' for r in rows)} of {len(rows)} questions went to the small model, at "
      f"{min(float(r[4]) for r in rows) / max(float(r[4]) for r in rows):.0%} of the most expensive request's cost. "
      + (
        "The budget refused caller A with 429 once today's tokens were used, before calling the model, while caller "
        "B still had theirs."
        if refused
        else "The budget didn't refuse anything: check Redis."
      ),
      "decision",
      "Route, price and budget every request",
    ),
    example("prod.l08.1", "Usage per request", usage_per_request, "", [view]),
    example("prod.l08.2", "Routing with a rule", routed_and_priced, "nimbus/llm/routing.py route", [view]),
    example("prod.l08.3", "Cost from a price table in config", routed_and_priced, "settings.model_prices", [view]),
    example(
      "prod.l08.4",
      "Per-caller daily budgets in Redis",
      spend_until_refused,
      "nimbus/api/limits.py TokenBudget",
      [budget],
    ),
  ]


# ── Lab 6: keeping abuse out ───────────────────────────────────────────────────
def keys_as_hashes():
  from nimbus.api.limits import key_hash

  # The database holds sha256(key) → plan. The plain key was shown to the customer once; a leaked table leaks no keys.
  return {key_hash("k-free-1"): "free", key_hash("k-pro-1"): "pro"}


@lab("prod.abuse", "Keep abuse out: hashed keys, plans, rate limits, request limits", calls="No model calls")
def abuse():
  """Requests with no key, a wrong key and a valid one; five more on the free and the pro plan against a Redis
  fixed-window limiter; a 10,000-character message. Needs Redis (docker compose up -d)."""
  import redis as redis_lib

  stored = keys_as_hashes()
  try:
    with budget_app(1_000_000) as client:
      client.app.state.redis.delete(*client.app.state.redis.keys("ratelimit:*") or ["none"])
      who = [
        [label, str(client.post("/support/reply", json={"message": "hi"}, headers=h).status_code)]
        for label, h in (
          ("no key", {}),
          ("a wrong key", {"X-API-Key": "guess"}),
          ("a valid key", {"X-API-Key": "k-free-1"}),
        )
      ]
      plans, retry = [], "—"
      for label, key in (("free (3/min)", "k-free-1"), ("pro (20/min)", "k-pro-1")):
        codes = []
        for _ in range(5):
          r = client.post("/support/reply", json={"message": "hi"}, headers={"X-API-Key": key})
          codes.append(r.status_code)
          if r.status_code == 429:
            retry = r.headers.get("retry-after")
        plans.append([label, ", ".join(map(str, codes))])
      big = client.post("/support/reply", json={"message": "x" * 10_000}, headers={"X-API-Key": "k-pro-1"})
  except (redis_lib.RedisError, AttributeError) as error:
    return [callout(f"Rate limits need Redis: {error}", "warning", "No Redis")]
  return [
    table(["what the database stores", "plan"], [[h[:20] + "…", p] for h, p in stored.items()]),
    table(["request", "status"], who),
    table(["plan", "5 more requests"], plans),
    table(
      ["", "value"],
      [["Retry-After on a 429", f"{retry} s"], ["a 10,000-character message", f"{big.status_code}: {big.text[:90]}"]],
    ),
    callout(
      "Unknown keys get 401; each plan has its own per-minute limit in Redis, shared by every server process, and a "
      "429 says when to come back. Request size is capped by the model of the body, so a huge prompt never reaches "
      "the model.",
      "decision",
      "Keys, plans and limits",
    ),
    example(
      "prod.l09.1",
      "API keys stored as hashes",
      keys_as_hashes,
      "nimbus/api/limits.py caller_plan",
      [table(["hash", "plan"], [[h[:20] + "…", p] for h, p in stored.items()])],
    ),
    example("prod.l09.2", "Plans", keys_as_hashes, "settings.plan_limits", [table(["plan", "codes"], plans)]),
    example(
      "prod.l09.3",
      "A Redis rate limiter",
      keys_as_hashes,
      "nimbus/api/limits.py rate_limited",
      [table(["plan", "codes"], plans)],
    ),
    example(
      "prod.l09.4",
      "Request limits",
      keys_as_hashes,
      "ReplyRequest: message ≤ 2,000 characters",
      [code(big.text[:200], "json")],
    ),
  ]


# ── Production path: an outage drill, and what a request costs ─────────────────
POLICY = (
  "Nimbus policy: a cold brew costs ₹200; the Oat Latte is vegan and costs ₹220; refunds within 14 days of delivery "
  "for unopened bags; loyalty: 1 point per ₹10, points expire after 12 months; deliveries take 2 days in Mumbai, 3 "
  "to 5 days elsewhere."
)
DRILL = [  # (question, a fact the right answer has)
  ("How much is a cold brew?", r"200"),
  ("Is the Oat Latte vegan?", r"(?i)yes|vegan"),
  ("When do points expire?", r"12"),
  ("How long does delivery to Mumbai take?", r"2|two"),
  ("How many points for ₹500?", r"50"),
  ("My bag arrived torn and I opened it, can I get a refund, and how long do I have?", r"(?i)unopened|not|no"),
  ("I live in Pune, how long will delivery take, and will my points expire before it arrives?", r"3|three"),
  ("Compare the cold brew and the Oat Latte on price, and say which one is vegan.", r"220"),
  ("I bought ₹1,300 of beans, how many points do I get, and when do they expire?", r"130"),
  ("Can I return an unopened bag after 10 days, and what if it's 20 days?", r"14"),
]


async def drill(models, outage: bool) -> dict:
  import re

  from langchain_core.callbacks import get_usage_metadata_callback

  from nimbus.config import get_settings
  from nimbus.llm.routing import cost

  right, failed, seconds, usd = 0, 0, [], 0.0
  for question, fact in DRILL:
    messages = [SystemMessage(f"{POLICY} Answer in one sentence."), HumanMessage(question)]
    started = time.perf_counter()
    try:
      with get_usage_metadata_callback() as usage:
        reply = await models(messages, question)
      usd += cost(usage.usage_metadata, get_settings())
      right += re.search(fact, reply.text) is not None
    except Exception:  # noqa: BLE001 - the provider was down and nothing answered
      failed += 1
    seconds.append(time.perf_counter() - started)
  ordered = sorted(seconds)
  return {
    "answered": round((len(DRILL) - failed) / len(DRILL), 3),
    "right": round(right / len(DRILL), 3),
    "p50_ms": round(ordered[len(ordered) // 2] * 1000),
    "usd_per_1k": round(usd / len(DRILL) * 1000, 4),
    "outage": outage,
  }


def evaluate_resilience() -> list[dict]:
  import asyncio

  from nimbus.config import get_settings
  from nimbus.llm.routing import RoutedModel

  async def run_all():
    settings = get_settings()
    rows = []
    for outage in (False, True):
      broken = {"api_key": "not-a-real-key", "max_retries": 0} if outage else {}
      single = get_chat_model(**broken)
      routed = RoutedModel.from_settings(settings, **broken)
      for name, call in (
        ("default model only", lambda m, q, single=single: single.ainvoke(m)),
        ("routed, with fallback and breaker", routed.ainvoke),
      ):
        rows.append({"variant": f"{name}{' · Mistral down' if outage else ''}", **await drill(call, outage)})
    from nimbus.llm.models import aclose_clients

    await aclose_clients()
    return rows

  return asyncio.run(run_all())


@lab(
  "prod.resilience-eval",
  "Evaluate: an outage drill, and what a request costs",
  uses_model=True,
  calls="~40 calls: ~30 to Mistral, ~10 to Groq",
)
def resilience_eval():
  """Ten support questions (five simple, five with several parts) answered by the default model alone and by the
  routed setup (small or default model, each behind a breaker with Groq as backup), with Mistral up and with
  Mistral down: answered, right, latency and cost per 1,000 requests."""
  rows = evaluate_resilience()
  up = [r for r in rows if not r["outage"]]
  down = [r for r in rows if r["outage"]]
  return [
    scorecard(
      [{k: v for k, v in r.items() if k != "outage"} for r in rows],
      [
        Metric("answered", "answered"),
        Metric("right", "right facts"),
        Metric("p50_ms", "latency p50", "lower", "ms"),
        Metric("usd_per_1k", "US$ per 1,000 requests", "lower", "text"),
      ],
      title="10 questions, Mistral up and down",
    ),
    callout(
      f"With Mistral down the default model alone answered {down[0]['answered']:.0%}; the routed setup "
      f"{down[1]['answered']:.0%}, from Groq, with the breaker skipping the dead provider after two failures. With "
      f"Mistral up, routing cost US${up[1]['usd_per_1k']} per 1,000 requests against US${up[0]['usd_per_1k']} for "
      f"the default model alone, at {up[1]['right']:.0%} against {up[0]['right']:.0%} right facts.",
      "decision",
      "Ship routing with a fallback and a breaker",
    ),
  ]
