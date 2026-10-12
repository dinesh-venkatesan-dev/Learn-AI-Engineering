"""Production · Module 4 labs: evaluation, datasets and prompts. The support reply evaluated on a dataset with code
checks and LLM judges (langsmith.evaluate, nothing uploaded), agent trajectories checked with agentevals, versions
compared example by example, tests in pytest, prompts versioned and released behind a gate, and quality watched on
live traffic. Steps that need LangSmith's servers (hosted datasets, the Prompt Hub, managed online evaluators) run
only when Nimbus has its own key."""

import json
import random
import time
import uuid
from functools import cache

from langchain_core.messages import AIMessage

from academy.labs.runtime import Metric, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR, get_settings
from nimbus.observability.tracing import langsmith_ready
from nimbus.support import reply_eval as rv

NEEDS_KEY = "needs LangSmith: set NIMBUS_LANGSMITH_API_KEY and NIMBUS_LANGSMITH_TRACING=true in ai-course/.env"
V3_EXTRA = " When the customer reports a problem, start by apologising."
V2 = (
  "You are Nimbus Coffee's support assistant. Reply in a {tone} tone, in at most {max_words} words. Nimbus policy: "
  "a cold brew costs ₹200; the Oat Latte is vegan (oat milk); loyalty points: 1 point per ₹10, they expire 12 months "
  "after they are earned; deliveries take 2 days in Mumbai, 3 to 5 days elsewhere; unopened bags can be returned "
  "within 14 days of delivery. Answer from the policy when it covers the question. Never promise refunds or dates "
  "for a specific order: say a teammate will check."
)


def server(fn):
  if not langsmith_ready(get_settings()):
    return NEEDS_KEY
  from nimbus.observability.tracing import tracing_client

  return fn(tracing_client(get_settings()))


@cache
def judge_model():
  from nimbus.llm.judges import judge_chat_model

  return judge_chat_model()  # Groq gpt-oss in JSON-schema mode: judges need structured output


_ANSWERS: dict[tuple, dict] = {}


def answers(ref: str, split: str | None = None, model=None, label: str = "") -> dict:
  """The support reply's outputs for a prompt version, computed once per process (labs share them)."""
  key = (ref, split, label)
  if key not in _ANSWERS:
    _ANSWERS[key] = rv.answer_all(ref, rv.load_examples(split), model)
  return _ANSWERS[key]


def ensure_candidate(version: str = "v2") -> str:
  from nimbus.prompts.registry import PromptRegistry

  text, note = {
    "v2": (V2, "v2: the policy in the prompt"),
    "v3": (V2 + V3_EXTRA, "v3: v2 + apologise first when something went wrong"),
  }[version]
  return PromptRegistry().push("support-reply", [("system", text)], {"temperature": 0}, note)


# ── Lab 1: evaluation basics ───────────────────────────────────────────────────
def correctness_judge():
  from openevals.llm import create_llm_as_judge
  from openevals.prompts import CORRECTNESS_PROMPT

  # A judge from a prompt with {inputs}, {outputs} and {reference_outputs}: it returns a score and its reason
  return create_llm_as_judge(prompt=CORRECTNESS_PROMPT, judge=judge_model(), feedback_key="correctness")


def first_experiment():
  examples = rv.load_examples()  # 1. a dataset: inputs + reference outputs
  outputs = answers("support-reply:production")  # 2. the target: the support reply, prompt version "production"
  hard = [e for e in examples if e["split"] == "hard"]
  judge = correctness_judge()

  def judged(inputs, outputs, reference_outputs):  # 3. an LLM judge next to the code checks
    return judge(inputs=inputs, outputs=outputs, reference_outputs={"answer": reference_outputs["answer"]})

  code_checks = rv.run_experiment(outputs, examples, rv.CODE_EVALUATORS)  # 4. run, read the scores
  with_judge = rv.run_experiment({e["id"]: outputs[e["id"]] for e in hard}, hard, [rv.facts, judged])
  return rv.scores(code_checks), rv.scores(with_judge)


@lab(
  "lc.evaluation",
  "Evaluation: does the app give good answers?",
  uses_model=True,
  calls="~12 calls to Mistral, 6 to Groq",
)
def evaluation():
  """The 12-example support reply dataset, the reply as the target, code checks on every example and an LLM judge on
  the six fact questions."""
  code_scores, judged = first_experiment()
  rows = [
    [
      e["id"],
      e["split"],
      e["inputs"]["message"],
      str(code_scores[e["id"]].get("facts")),
      str(code_scores[e["id"]].get("no_promises")),
      str(judged.get(e["id"], {}).get("correctness", "—")),
    ]
    for e in rv.load_examples()
  ]
  view = table(["id", "split", "message", "facts", "no promises", "judge: correct"], rows)
  avg = rv.averages(code_scores)
  agree = sum(str(judged[i].get("correctness")) == str(bool(code_scores[i]["facts"])) for i in judged)

  def split_rate(split: str) -> float:
    ids = [e["id"] for e in rv.load_examples(split)]
    return sum(code_scores[i]["facts"] for i in ids) / len(ids)

  return [
    view,
    callout(
      f"Facts present: {split_rate('smoke'):.0%} of the smoke examples and {split_rate('hard'):.0%} of the fact "
      f"questions; no promises: {avg['no_promises']:.0%}. The judge agreed with the code check on {agree} of "
      f"{len(judged)} fact questions.",
      "info",
      "A dataset, a target, evaluators",
    ),
    example("lc.l54.1", "A dataset", rv.load_examples, "data/eval/support_reply_dataset.json", [view]),
    example("lc.l54.2", "A target", rv.answer_all, "", [view]),
    example("lc.l54.3", "Code checks and an LLM judge", correctness_judge, "", [view]),
    example("lc.l54.4", "Running it, reading scores", first_experiment, "", [code(json.dumps(avg), "json")]),
  ]


# ── Lab 2: agent evals ─────────────────────────────────────────────────────────
AGENT_CASES = [
  {
    "question": "Can I get a refund on NC-90112? I haven't opened it.",
    "reference": [("get_order", {"order_id": "NC-90112"}), ("check_refund", {"order_id": "NC-90112", "opened": False})],
  },
  {"question": "How many loyalty points did NC-4410 earn?", "reference": [("loyalty_points", {"order_id": "NC-4410"})]},
  {"question": "Is the Oat Latte vegan, and how much is it?", "reference": [("cafe_menu_item", {"name": "Oat Latte"})]},
]


def trajectory(messages) -> list[dict]:
  """The agent's messages in the OpenAI format agentevals reads."""
  from langchain_core.messages import convert_to_openai_messages

  return convert_to_openai_messages(messages)


def reference_trajectory(question: str, calls: list[tuple[str, dict]]) -> list[dict]:
  tool_calls = [{"function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls]
  return [{"role": "user", "content": question}, {"role": "assistant", "content": "", "tool_calls": tool_calls}]


def match_modes():
  from agentevals.trajectory.match import create_trajectory_match_evaluator

  # strict: same calls, same order · unordered · superset: at least these · subset: nothing beyond these
  return {
    mode: create_trajectory_match_evaluator(trajectory_match_mode=mode)
    for mode in ("strict", "unordered", "superset", "subset")
  }


def ignore_args():
  from agentevals.trajectory.match import create_trajectory_match_evaluator

  return create_trajectory_match_evaluator(trajectory_match_mode="superset", tool_args_match_mode="ignore")


def trajectory_judge():
  from agentevals.trajectory.llm import TRAJECTORY_ACCURACY_PROMPT, create_trajectory_llm_as_judge

  return create_trajectory_llm_as_judge(prompt=TRAJECTORY_ACCURACY_PROMPT, judge=judge_model())  # no reference


def single_step(model, question: str):
  from nimbus.support.tools import SupportStore, build_tools

  # One decision, one model call, no tool executed: does the model pick the right tool and arguments?
  reply = model.bind_tools(build_tools(SupportStore())).invoke(question)
  return [(c["name"], c["args"]) for c in reply.tool_calls]


def run_agent_cases():
  from langchain.agents import create_agent

  from nimbus.llm import get_chat_model
  from nimbus.support.tools import SupportStore, build_tools

  agent = create_agent(
    get_chat_model(), build_tools(SupportStore()), system_prompt="You are Nimbus support. Use the tools. Be brief."
  )
  out = []
  for case in AGENT_CASES:
    result = agent.invoke({"messages": [("user", case["question"])]})
    out.append((case, trajectory(result["messages"])))
  return out


@lab(
  "prod.agent-evals",
  "Agent evals: how it got there, not just the answer",
  uses_model=True,
  calls="~10 calls to Mistral, 3 to Groq",
)
def agent_evals():
  """Three support questions through an agent with the order tools: its trajectories checked against reference tool
  calls in four match modes, with arguments compared and ignored, judged by an LLM without a reference, and one
  decision tested on its own."""
  from nimbus.llm import get_chat_model

  runs = run_agent_cases()
  modes, loose, judge = match_modes(), ignore_args(), trajectory_judge()
  rows = []
  for case, traj in runs:
    reference = reference_trajectory(case["question"], case["reference"])
    used = [(c["function"]["name"]) for m in traj for c in m.get("tool_calls") or []]
    scored = {mode: ev(outputs=traj, reference_outputs=reference)["score"] for mode, ev in modes.items()}
    rows.append(
      [
        case["question"],
        ", ".join(used),
        *[str(scored[m]) for m in modes],
        str(loose(outputs=traj, reference_outputs=reference)["score"]),
        str(judge(outputs=traj)["score"]),
      ]
    )
  passes = sum(r[3] == "True" for r in rows)
  step = single_step(get_chat_model(), "How many loyalty points did order NC-4410 earn?")
  view = table(
    ["question", "tools used", "strict", "unordered", "superset", "subset", "args ignored", "LLM judge"], rows
  )
  return [
    view,
    table(
      ["single step: 'How many loyalty points did NC-4410 earn?'", "tool calls"],
      [["the model chose", json.dumps(step)]],
    ),
    callout(
      f"Suite pass rate (unordered match): {passes}/{len(rows)}. Strict match fails an agent that looked something "
      "up in another order; superset accepts extra calls; subset forbids them. Comparing arguments catches a right "
      "tool called for the wrong order; the judge grades the path when no single path is right.",
      "info",
      "Trajectories",
    ),
    example("lc.l55.1", "A trajectory", trajectory, "", [view]),
    example("lc.l55.2", "Strict match", match_modes, "", [view]),
    example("lc.l55.3", "Unordered, superset, subset", match_modes, "", [view]),
    example("lc.l55.4", "Tool arguments: compared or ignored", ignore_args, "", [view]),
    example("lc.l55.5", "An LLM judge on the trajectory", trajectory_judge, "", [view]),
    example("lc.l55.6", "A small suite, a pass rate", run_agent_cases, "", [code(f"{passes}/{len(rows)}", "text")]),
    example("smith.l18.1", "Single step", single_step, "", [code(json.dumps(step), "json")]),
    example("smith.l18.2", "Trajectory match (agentevals)", match_modes, "", [view]),
    example("smith.l18.3", "A trajectory judge", trajectory_judge, "", [view]),
  ]


# ── Lab 3: datasets ────────────────────────────────────────────────────────────
def splits() -> dict[str, int]:
  examples = rv.load_examples()
  return {s: sum(e["split"] == s for e in examples) for s in sorted({e["split"] for e in examples})}


def dataset_version(examples: list[dict]) -> str:
  import hashlib

  # Every change makes a new version; tag the one your scores refer to ("v1"), so they stay comparable
  return hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest()[:12]


def from_production(feedback_rows: list[dict]) -> list[dict]:
  # Turn rated traces into examples: the question, and what a good answer must contain (a person adds that)
  return [
    {
      "id": f"p{i}",
      "split": "from-production",
      "inputs": {"message": f["question"], "tone": "warm"},
      "reference_outputs": {"must_contain": [], "must_not": "", "answer": f.get("correction", "")},
      "metadata": {"source": "feedback", "run_id": f["run_id"]},
    }
    for i, f in enumerate(feedback_rows)
  ]


def schema_check():
  from pydantic import BaseModel, ValidationError

  class Inputs(BaseModel):  # what an example's inputs must look like
    message: str
    tone: str = "warm"

  try:
    Inputs.model_validate({"text": "wrong field"})
  except ValidationError as error:
    return str(error).splitlines()[0]


def hosted_dataset():
  def create(client):
    dataset = client.create_dataset("nimbus-support-reply", description="Support reply examples")
    client.create_examples(
      dataset_id=dataset.id,
      examples=[
        {"inputs": e["inputs"], "outputs": e["reference_outputs"], "metadata": e["metadata"], "split": e["split"]}
        for e in rv.load_examples()
      ],
    )
    return dataset.id

  return server(create)


@lab("smith.datasets", "Datasets: the fixed set every version is tested against", calls="No model calls")
def datasets():
  """The support reply dataset: its splits, a version hash and a tag, examples made from rated production traces, a
  schema that rejects a malformed example, and the same dataset in LangSmith when Nimbus has a key."""
  examples = rv.load_examples()
  version = dataset_version(examples)
  new = from_production(
    [{"question": "Do you ship to Goa?", "run_id": str(uuid.uuid4()), "correction": "Yes, in 3 to 5 days."}]
  )
  rejected = schema_check()
  hosted = hosted_dataset()
  return [
    table(["split", "examples"], [[k, str(v)] for k, v in splits().items()]),
    table(
      ["", "value"],
      [
        ["version (content hash)", version],
        ["tag", "v1 → " + version],
        ["an example from a rated trace", json.dumps(new[0])[:200]],
        ["a malformed example", rejected],
        ["in LangSmith", str(hosted)],
      ],
    ),
    example(
      "smith.l13.1",
      "Examples: inputs, references, metadata",
      rv.load_examples,
      "",
      [code(json.dumps(examples[0]), "json")],
    ),
    example("smith.l13.2", "Splits", splits, "", [code(json.dumps(splits()), "json")]),
    example("smith.l13.3", "Versions and tags", dataset_version, "", [code(version, "text")]),
    example("smith.l13.4", "Examples from production", from_production, "", [code(json.dumps(new[0])[:300], "json")]),
    example("smith.l13.5", "Schemas", schema_check, "", [code(str(rejected), "text")]),
  ]


# ── Lab 4: experiments and evaluators ──────────────────────────────────────────
def a_code_evaluator(outputs: dict, reference_outputs: dict) -> dict:
  import re

  found = all(re.search(p, outputs["answer"]) for p in reference_outputs["must_contain"])
  return {"key": "facts", "score": int(found), "comment": "" if found else "a required fact is missing"}


def counts_model_calls(run, example) -> dict:
  # An evaluator can look INSIDE the run: here, how many model calls the target made
  calls = sum(1 for child in run.child_runs or [] if child.run_type == "llm")
  return {"key": "model_calls", "score": calls}


def all_right(outputs: list[dict], reference_outputs: list[dict]) -> dict:
  import re

  # A summary evaluator sees every example at once: "everything right", not an average
  ok = [
    all(re.search(p, o["answer"]) for p in r["must_contain"]) for o, r in zip(outputs, reference_outputs, strict=True)
  ]
  return {"key": "all_facts_right", "score": int(all(ok))}


def inside_the_run():
  from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
  from langsmith import evaluate

  fake = GenericFakeChatModel(messages=iter([AIMessage("A cold brew costs ₹200.")] * 10))
  examples = rv.load_examples("hard")[:2]

  def target(inputs: dict) -> dict:  # a target that calls a model: the evaluator counts the calls
    return {"answer": fake.invoke(inputs["message"]).text}

  return evaluate(
    target, data=[rv.as_example(e) for e in examples], evaluators=[counts_model_calls], upload_results=False
  )


@lab(
  "smith.experiments", "Experiments and evaluators", uses_model=True, calls="~12 calls to Mistral (shared with Lab 1)"
)
def experiments():
  """The support reply dataset through evaluate(): code evaluators returning a score, several scores and a
  category, an evaluator looking inside the run, a summary evaluator, experiment names, and adding evaluators to a
  finished experiment (LangSmith)."""
  examples = rv.load_examples()
  outputs = answers("support-reply:production")
  results = rv.run_experiment(
    outputs, examples, [a_code_evaluator, rv.no_promises, rv.length, rv.tone_label], prefix="support-reply-v1"
  )
  per = rv.scores(results)
  summary = all_right([outputs[e["id"]] for e in examples], [e["reference_outputs"] for e in examples])
  inside = [r["evaluation_results"]["results"][0].score for r in inside_the_run()]
  existing = server(lambda client: "evaluate_existing(experiment_name, evaluators=[...]) re-scores without re-running")
  view = table(
    ["example", "facts", "no_promises", "words", "short_enough", "opens_with"],
    [
      [k, *[str(v.get(c)) for c in ("facts", "no_promises", "words", "short_enough", "opens_with")]]
      for k, v in per.items()
    ],
  )
  return [
    view,
    table(
      ["", "value"],
      [
        ["averages", json.dumps(rv.averages(per))],
        ["summary: every fact right", str(summary["score"])],
        ["model calls seen inside the run", str(inside)],
        ["evaluate_existing", str(existing)],
      ],
    ),
    example("smith.l14.1", "Target, data, evaluators", rv.run_experiment, "", [view]),
    example("smith.l14.2", "A code evaluator", a_code_evaluator, "", [view]),
    example(
      "smith.l14.3",
      "evaluate()",
      rv.run_experiment,
      "upload_results=False here; with a key, an experiment in LangSmith",
      [view],
    ),
    example("smith.l14.4", "Reading results", rv.scores, "", [code(json.dumps(rv.averages(per)), "json")]),
    example("smith.l14.5", "experiment_prefix and metadata", rv.run_experiment, "prefix='support-reply-v1'", [view]),
    example("smith.l15.1", "evaluate_existing()", experiments, "", [code(str(existing), "text")]),
    example("smith.l15.2", "Several scores", rv.length, "", [view]),
    example("smith.l15.3", "Categories", rv.tone_label, "", [view]),
    example("smith.l15.4", "Looking inside the run", counts_model_calls, "", [code(str(inside), "text")]),
    example("smith.l15.5", "Summary evaluators", all_right, "", [code(json.dumps(summary), "json")]),
  ]


# ── Lab 5: LLM-as-judge ────────────────────────────────────────────────────────
GRADED = [  # answers whose right grade we already know
  ("How much is a cold brew?", "A cold brew costs ₹200.", True),
  ("How much is a cold brew?", "A cold brew is ₹150.", False),
  ("When do points expire?", "They expire 12 months after you earn them.", True),
  ("When do points expire?", "Points never expire.", False),
  ("How long does Mumbai delivery take?", "About two days.", True),
  ("How long does Mumbai delivery take?", "Usually a week.", False),
]
REFERENCES = {
  "How much is a cold brew?": "A cold brew costs ₹200.",
  "When do points expire?": "Points expire 12 months after they are earned.",
  "How long does Mumbai delivery take?": "Deliveries in Mumbai take 2 days.",
}


def own_rubric():
  from openevals.llm import create_llm_as_judge

  rubric = (
    "You grade a coffee shop's support reply. The customer wrote: {inputs}. The reply: {outputs}. Score true only "
    "if the reply is polite AND promises no refund, delivery date or other action for a specific order."
  )
  return create_llm_as_judge(prompt=rubric, judge=judge_model(), feedback_key="safe_and_polite")


def conciseness_judge():
  from openevals.llm import create_llm_as_judge
  from openevals.prompts import CONCISENESS_PROMPT

  return create_llm_as_judge(prompt=CONCISENESS_PROMPT, judge=judge_model(), feedback_key="conciseness")


def agreement() -> tuple[int, list]:
  judge = correctness_judge()
  rows = []
  for question, reply, right in GRADED:
    verdict = judge(
      inputs={"message": question}, outputs={"answer": reply}, reference_outputs={"answer": REFERENCES[question]}
    )
    rows.append((question, reply, right, bool(verdict["score"])))
  return sum(r[2] == r[3] for r in rows), rows


@lab("smith.judges", "LLM-as-judge: a model grades what code can't", calls="~10 calls to Groq")
def judges():
  """A correctness judge checked against six answers whose grade we know, a prebuilt conciseness rubric, and Nimbus's
  own rubric for safe, polite replies."""
  agreed, rows = agreement()
  concise = conciseness_judge()(inputs="How much is a cold brew?", outputs="A cold brew costs ₹200.")
  safe = own_rubric()(
    inputs="Can I get my refund today?", outputs="Absolutely, your refund will be in your account today!"
  )
  view = table(["question", "answer", "right grade", "judge"], [[q, a, str(r), str(j)] for q, a, r, j in rows])
  return [
    view,
    table(
      ["rubric", "score", "reason"],
      [
        ["conciseness (prebuilt)", str(concise["score"]), concise["comment"][:200]],
        ["safe and polite (ours)", str(safe["score"]), safe["comment"][:200]],
      ],
    ),
    callout(
      f"The judge agreed with the known grades on {agreed} of {len(rows)} answers. Check a judge like this before "
      "trusting it, keep code checks for what code can check, and judge samples: every judge call costs tokens.",
      "decision" if agreed == len(rows) else "warning",
      "Trust a judge after checking it",
    ),
    example("smith.l16.1", "create_llm_as_judge", correctness_judge, "", [view]),
    example("smith.l16.2", "Prebuilt and your own rubrics", own_rubric, "", [code(safe["comment"][:300], "text")]),
    example("smith.l16.3", "Check the judge", agreement, "", [view]),
    example("smith.l16.4", "Judges in an experiment", first_experiment, "Lab 1 judged the six fact questions", [view]),
  ]


# ── Lab 6: comparing versions ──────────────────────────────────────────────────
def compare_models():
  from nimbus.llm import get_chat_model

  examples = rv.load_examples()
  base = rv.scores(rv.run_experiment(answers("support-reply:production"), examples, [rv.facts, rv.no_promises]))
  small = get_chat_model(role="small").with_retry(stop_after_attempt=4, wait_exponential_jitter=True)
  small_out = answers("support-reply:production", model=small, label="small")
  cand = rv.scores(rv.run_experiment(small_out, examples, [rv.facts, rv.no_promises]))
  return base, cand, small_out


def repetitions():
  from langsmith import evaluate

  examples = rv.load_examples("smoke")[:2]
  outputs = answers("support-reply:production")
  results = evaluate(
    lambda inputs: next(outputs[e["id"]] for e in examples if e["inputs"] == inputs),
    data=[rv.as_example(e) for e in examples],
    evaluators=[rv.no_promises],
    upload_results=False,
    num_repetitions=3,  # each example several times: one lucky or unlucky answer doesn't decide a release
  )
  return len(list(results))


def pairwise(question: str, a: str, b: str) -> str:
  order = [("A", a), ("B", b)]
  random.shuffle(order)  # randomize the order: judges prefer whichever answer comes first
  prompt = (
    f"A customer asked a coffee shop: {question}\nWhich reply is better (accurate, helpful, no promises)?\n"
    f"Reply 1: {order[0][1]}\nReply 2: {order[1][1]}\nAnswer with 1 or 2 only."
  )
  pick = judge_model().invoke(prompt).text.strip()
  return order[0][0] if pick.startswith("1") else order[1][0]


@lab(
  "smith.compare",
  "Comparing versions: better, and where worse?",
  uses_model=True,
  calls="~12 calls to Mistral, ~3 to Groq",
)
def compare():
  """The production prompt on the default model and on the small model: per-example regressions, cost and speed,
  repetitions, and a pairwise judge with the order randomized."""
  base, cand, small_out = compare_models()
  big_out = answers("support-reply:production")
  broken = rv.regressions(base, cand)
  reps = repetitions()
  picks = [
    pairwise(e["inputs"]["message"], big_out[e["id"]]["answer"], small_out[e["id"]]["answer"])
    for e in rv.load_examples("smoke")[:3]
  ]
  cost = table(
    ["version", "facts", "no promises", "tokens / answer", "p50 ms"],
    [
      [
        name,
        f"{rv.averages(s)['facts']:.0%}",
        f"{rv.averages(s)['no_promises']:.0%}",
        f"{sum(o['tokens'] for o in out.values()) / len(out):.0f}",
        f"{sorted(o['ms'] for o in out.values())[len(out) // 2]:.0f}",
      ]
      for name, s, out in (("ministral-14b (production)", base, big_out), ("ministral-8b", cand, small_out))
    ],
  )
  return [
    cost,
    table(["regression (right before, wrong now)", "check"], [list(b) for b in broken] or [["none", ""]]),
    table(
      ["", "value"],
      [
        ["runs with num_repetitions=3 on 2 examples", str(reps)],
        ["pairwise picks (A = 14b, B = 8b)", ", ".join(picks)],
      ],
    ),
    callout(
      f"The small model broke {len(broken)} check(s) the production model passed. Averages can hide that: compare per "
      "example, and repeat runs before deciding.",
      "decision" if not broken else "warning",
      "Compare per example",
    ),
    example("smith.l17.1", "Same dataset, new version", compare_models, "", [cost]),
    example(
      "smith.l17.2",
      "Regressions",
      rv.regressions,
      "",
      [table(["example", "check"], [list(b) for b in broken] or [["none", ""]])],
    ),
    example("smith.l17.3", "Cost and speed", compare_models, "read_project(include_stats=True) in LangSmith", [cost]),
    example("smith.l17.4", "Repetitions", repetitions, "", [code(f"{reps} runs", "text")]),
    example("smith.l17.5", "Pairwise, order randomized", pairwise, "", [code(", ".join(picks), "text")]),
  ]


# ── Lab 7: tests in pytest ─────────────────────────────────────────────────────
@lab("smith.pytest", "Tests in pytest, results in LangSmith", uses_model=True, calls="2 calls to Mistral, 1 to Groq")
def pytest_lab():
  """tests/test_reply_quality.py: @pytest.mark.langsmith tests that log inputs, outputs, reference outputs and
  feedback; run here with tracking off (nothing uploaded)."""
  import os
  import subprocess
  import sys

  env = {**os.environ, "LANGSMITH_TEST_TRACKING": "false", "LANGSMITH_TEST_SUITE": "nimbus-support-reply"}
  result = subprocess.run(
    [
      sys.executable,
      "-m",
      "pytest",
      "-q",
      "-m",
      "integration",
      "tests/test_reply_quality.py",
      "-p",
      "no:cacheprovider",
    ],
    cwd=PROJECT_DIR,
    env=env,
    capture_output=True,
    text=True,
    timeout=300,
    check=False,
  )
  from academy.labs.agents.m08_production import _test_module

  tests = _test_module("test_reply_quality")
  return [
    code(result.stdout[-1500:], "text", "pytest -m integration tests/test_reply_quality.py"),
    example(
      "smith.l20.1",
      "@pytest.mark.langsmith",
      tests.test_no_promises_to_an_angry_customer,
      "",
      [code(result.stdout[-400:], "text")],
    ),
    example(
      "smith.l20.2",
      "Logging inputs, outputs, references",
      tests.test_no_promises_to_an_angry_customer,
      "",
      [code(result.stdout[-400:], "text")],
    ),
    example(
      "smith.l20.3",
      "trace_feedback for a judge",
      tests.test_a_judge_finds_the_reply_polite,
      "",
      [code(result.stdout[-400:], "text")],
    ),
    example(
      "smith.l20.4",
      "Suites and experiments",
      pytest_lab,
      "LANGSMITH_TEST_SUITE, LANGSMITH_EXPERIMENT",
      [code(result.stdout[-400:], "text")],
    ),
  ]


# ── Lab 8: local evaluations and backtesting ───────────────────────────────────
def local_evaluation():
  import uuid as uuid_lib
  from datetime import UTC, datetime

  from langsmith import evaluate
  from langsmith.schemas import Example

  # Examples built in code: no dataset in LangSmith, nothing stored
  data = [
    Example(
      id=uuid_lib.uuid4(),
      inputs={"message": "hi"},
      outputs={"must_contain": [r"(?i)hello|hi"], "must_not": ""},
      dataset_id=uuid_lib.uuid4(),
      created_at=datetime.now(UTC),
    )
  ]
  return list(evaluate(lambda inputs: {"answer": "Hi there!"}, data=data, evaluators=[rv.facts], upload_results=False))


def quality_gate(per_example: dict, threshold: float = 0.95) -> tuple[bool, float]:
  score = rv.averages(per_example)["no_promises"]
  return score >= threshold, score  # in CI: exit non-zero when it's False


def backtest(candidate_ref: str):
  # Replay real questions (here: the production answers) through the candidate and compare, question by question
  production = answers("support-reply:production")
  candidate = answers(candidate_ref)
  changed = [
    (i, production[i]["answer"], candidate[i]["answer"])
    for i in production
    if production[i]["answer"] != candidate[i]["answer"]
  ]
  return changed


@lab(
  "smith.local-backtest",
  "Local evaluations, a quality gate, and backtesting",
  uses_model=True,
  calls="~12 calls to Mistral",
)
def local_backtest():
  """An evaluation on plain Python objects with nothing stored, a CI quality gate on the production prompt's scores,
  and the candidate prompt replayed on production's questions."""
  candidate = ensure_candidate()
  plain = local_evaluation()
  per = rv.scores(rv.run_experiment(answers("support-reply:production"), rv.load_examples(), [rv.no_promises]))
  passed, score = quality_gate(per)
  changed = backtest(f"support-reply:{candidate}")
  view = table(["question", "production said", "candidate says"], [[i, a[:120], b[:120]] for i, a, b in changed[:6]])
  return [
    table(
      ["", "value"],
      [
        ["evaluation on plain objects", str(plain[0]["evaluation_results"]["results"][0].score)],
        ["quality gate: no_promises ≥ 95%", f"{'pass' if passed else 'FAIL'} ({score:.0%})"],
        ["backtest: answers that changed", f"{len(changed)} of {len(answers('support-reply:production'))}"],
      ],
    ),
    view,
    example(
      "smith.l21.1",
      "upload_results=False, plain objects",
      local_evaluation,
      "",
      [code(str(plain[0]["evaluation_results"]["results"][0].score), "text")],
    ),
    example("smith.l21.2", "A quality gate", quality_gate, "", [code(f"{passed} ({score:.0%})", "text")]),
    example("smith.l21.3", "Backtesting on real questions", backtest, "", [view]),
  ]


# ── Lab 9: the prompt registry and the Prompt Hub ──────────────────────────────
def registry_tour():
  from nimbus.prompts.registry import PromptRegistry

  registry = PromptRegistry()
  candidate = ensure_candidate()  # every push is a commit, named by its hash
  production = registry.pull("support-reply:production")  # what the app runs: a moving tag
  pinned = registry.pull(f"support-reply:{candidate}")  # one exact version
  return registry.history("support-reply"), production.commit, pinned.commit


def hub_calls():
  def calls(client):
    from langchain_core.prompts import ChatPromptTemplate

    url = client.push_prompt(
      "nimbus-support-reply", object=ChatPromptTemplate.from_messages([("system", V2), ("human", "{message}")])
    )
    return url

  return server(calls)


def cached_pull():
  from nimbus.prompts.registry import PromptRegistry

  registry = PromptRegistry(cache_s=60)
  started = time.perf_counter()
  registry.pull("support-reply:production")
  first = time.perf_counter() - started
  started = time.perf_counter()
  registry.pull("support-reply:production")  # cached: skip_cache=True right after pushing a new version
  return first * 1000, (time.perf_counter() - started) * 1000


@lab("smith.prompt-hub", "Prompts as versions, not code", calls="No model calls")
def prompt_hub():
  """The support reply prompt in Nimbus's registry: commits, the production tag, a pinned version, model settings
  stored with it, and cached pulls; the same operations in LangSmith's Prompt Hub when Nimbus has a key."""
  history, production, pinned = registry_tour()
  first, second = cached_pull()
  hub = hub_calls()
  view = table(["commit", "note", "tag"], [[h["commit"], h["note"], h["tag"]] for h in history])
  return [
    view,
    table(
      ["", "value"],
      [
        ["production tag → commit", production],
        ["pinned pull", pinned],
        ["pull, then cached pull", f"{first:.2f} ms, {second:.3f} ms"],
        ["push to the Prompt Hub", str(hub)],
      ],
    ),
    example("smith.l22.1", "push: every push is a commit", registry_tour, "nimbus/prompts/registry.py", [view]),
    example(
      "smith.l22.2",
      "pull at runtime",
      registry_tour,
      "routers/support.py system_prompt pulls support-reply:production",
      [view],
    ),
    example("smith.l22.3", "Pinning: commit or tag", registry_tour, "", [view]),
    example(
      "smith.l22.4",
      "Prompt + model settings",
      registry_tour,
      "push(..., model={'temperature': 0}): stored with the version",
      [view],
    ),
    example("smith.l22.5", "Pulls are cached", cached_pull, "", [code(f"{first:.2f} ms → {second:.3f} ms", "text")]),
  ]


# ── Lab 10: releasing prompts safely ───────────────────────────────────────────
def release(candidate: str) -> dict:
  from nimbus.prompts.registry import PromptRegistry

  registry = PromptRegistry()
  examples = rv.load_examples()
  production = registry.pull("support-reply:production", skip_cache=True).commit
  base = rv.scores(rv.run_experiment(answers("support-reply:production"), examples, [rv.facts, rv.no_promises]))
  cand = rv.scores(rv.run_experiment(answers(f"support-reply:{candidate}"), examples, [rv.facts, rv.no_promises]))
  ok, broken = rv.gate(base, cand)  # promote only if nothing got worse
  return {
    "production": production,
    "candidate": candidate,
    "ok": ok,
    "broken": broken,
    "base": rv.averages(base),
    "cand": rv.averages(cand),
  }


def promote_and_roll_back(candidate: str):
  from nimbus.prompts.registry import PromptRegistry

  registry = PromptRegistry()
  previous = registry.tag("support-reply", "production", candidate)  # the app picks it up on its next pull
  registry.tag("support-reply", "production", previous)  # rollback: the same move, back
  return previous


@lab("smith.release", "Releasing prompts safely", uses_model=True, calls="~12 calls to Mistral (the candidate)")
def release_lab():
  """The production prompt and a candidate (the policy in the prompt) on the same dataset; a gate that promotes only
  if no check got worse; promotion and rollback as tag moves (shown, then undone: the capstone releases for real)."""
  candidate = ensure_candidate()
  result = release(candidate)
  previous = promote_and_roll_back(candidate)
  view = table(
    ["version", "facts", "no promises"],
    [
      ["production " + result["production"], f"{result['base']['facts']:.0%}", f"{result['base']['no_promises']:.0%}"],
      ["candidate " + candidate, f"{result['cand']['facts']:.0%}", f"{result['cand']['no_promises']:.0%}"],
    ],
  )
  return [
    view,
    table(
      ["gate", "value"],
      [
        ["promote?", str(result["ok"])],
        ["what got worse", json.dumps(result["broken"]) or "nothing"],
        ["rollback goes back to", str(previous)],
      ],
    ),
    example("smith.l23.1", "The app runs the production tag", release, "", [view]),
    example("smith.l23.2", "A candidate is a commit, evaluated", release, "", [view]),
    example("smith.l23.3", "A gate, then a tag move", promote_and_roll_back, "", [code(str(result["ok"]), "text")]),
    example("smith.l23.4", "Rollback", promote_and_roll_back, "", [code(str(previous), "text")]),
  ]


# ── Lab 11: monitoring and online evaluation ───────────────────────────────────
def monitoring_numbers(outputs: dict) -> dict:
  # In LangSmith: one list_runs(select=[...]) call, then plain Python. Here, from the answers' own measurements
  ms = sorted(o["ms"] for o in outputs.values())
  return {
    "requests": len(ms),
    "error_rate": 0.0,
    "latency_p50_ms": round(ms[len(ms) // 2]),
    "latency_p95_ms": round(ms[int(len(ms) * 0.95) - 1]),
    "tokens": sum(o["tokens"] for o in outputs.values()),
  }


def online_by_hand(outputs: dict, sample: int = 3) -> list:
  examples = {e["id"]: e for e in rv.load_examples()}
  feedback = []
  for i, out in outputs.items():  # cheap code checks on everything
    feedback.append((i, "no_promises", rv.no_promises(out, examples[i]["reference_outputs"])["score"]))
  judge = correctness_judge()
  for i in random.sample(sorted(i for i in outputs if examples[i]["split"] == "hard"), sample):  # a judge on a sample
    verdict = judge(
      inputs=examples[i]["inputs"],
      outputs=outputs[i],
      reference_outputs={"answer": examples[i]["reference_outputs"]["answer"]},
    )
    feedback.append((i, "correctness", int(verdict["score"])))
  return feedback  # attached to the runs as feedback (create_feedback), in Nimbus's store or LangSmith


@lab("smith.monitoring", "Monitoring and online evaluation", uses_model=True, calls="~3 calls to Groq")
def monitoring():
  """Traffic, latency and tokens computed from the production answers, code checks on every answer and a judge on a
  sample attached as feedback, and what LangSmith's UI adds (managed evaluators, dashboards, alerts, rules)."""
  outputs = answers("support-reply:production")
  numbers = monitoring_numbers(outputs)
  feedback = online_by_hand(outputs)
  managed = server(
    lambda client: "client.evaluators: stored in LangSmith, attached to the project with a filter and a sampling rate"
  )
  return [
    table(["metric", "value"], [[k, str(v)] for k, v in numbers.items()]),
    table(["run", "feedback key", "score"], [[a, b, str(c)] for a, b, c in feedback]),
    table(
      ["in LangSmith", "what"],
      [
        ["managed online evaluators", str(managed)],
        ["dashboards and alerts", "error rate, latency, feedback score → Slack or a webhook"],
        ["automation rules", "matching traces → a dataset or an annotation queue"],
      ],
    ),
    example("smith.l24.1", "Monitoring numbers", monitoring_numbers, "", [code(json.dumps(numbers), "json")]),
    example(
      "smith.l24.2",
      "Online evaluation by hand",
      online_by_hand,
      "",
      [table(["run", "key", "score"], [[a, b, str(c)] for a, b, c in feedback])],
    ),
    example("smith.l24.3", "Managed online evaluators", monitoring, "", [code(str(managed), "text")]),
    example("smith.l24.4", "Dashboards, alerts, rules", monitoring, "", [code("LangSmith UI", "text")]),
  ]


# ── Production path: the improvement loop ──────────────────────────────────────
def user_rates(example: dict, out: dict) -> int:
  """A stand-in for real users: 👎 when a fact they asked for is missing or the reply promised something."""
  facts_ok = rv.facts(out, example["reference_outputs"])["score"]
  safe = rv.no_promises(out, example["reference_outputs"])["score"]
  return int(facts_ok and safe)


def improvement_loop() -> dict:
  from nimbus.prompts.registry import PromptRegistry

  registry = PromptRegistry()
  examples = rv.load_examples()
  # A repeatable lab: roll production back to v1 (the shipped prompt) first, so each run replays the release
  registry.tag("support-reply", "production", registry.history("support-reply")[0]["commit"])
  production = registry.pull("support-reply:production", skip_cache=True).commit
  live = answers("support-reply:production")  # 1. production answers, traced with the prompt version
  thumbs = {e["id"]: user_rates(e, live[e["id"]]) for e in examples}  # 2. users rate them
  complaints = [e for e in examples if not thumbs[e["id"]]]  # 3. triage: the 👎 become a regression set
  smoke = [e for e in examples if e["split"] == "smoke"]
  suite = complaints + [e for e in smoke if e not in complaints]
  base = rv.scores(rv.run_experiment({e["id"]: live[e["id"]] for e in suite}, suite, [rv.facts, rv.no_promises]))
  attempts, released = [], None
  for version in ("v2", "v3"):  # 4. a fix is a new commit; a blocked fix is revised and evaluated again
    candidate = ensure_candidate(version)
    out = answers(f"support-reply:{candidate}")
    cand = rv.scores(rv.run_experiment({e["id"]: out[e["id"]] for e in suite}, suite, [rv.facts, rv.no_promises]))
    ok, broken = rv.gate(base, cand)  # 5. release only if nothing got worse
    attempts.append({"version": version, "commit": candidate, "ok": ok, "broken": broken, "scores": rv.averages(cand)})
    if ok:
      registry.tag("support-reply", "production", candidate)
      released = candidate
      break
  verified = answers("support-reply:production", label="after-release") if released else live  # 6. the live app
  return {
    "production_before": production,
    "released": released,
    "attempts": attempts,
    "thumbs_down_before": len(complaints),
    "resolved": sum(user_rates(e, verified[e["id"]]) for e in complaints),
    "thumbs_up_after": sum(user_rates(e, verified[e["id"]]) for e in examples),
    "base": rv.averages(base),
    "complaints": [
      (e["id"], e["inputs"]["message"], live[e["id"]]["answer"][:100], verified[e["id"]]["answer"][:120])
      for e in complaints
    ],
  }


@lab(
  "prod.improvement-loop", "Evaluate: the improvement loop, end to end", uses_model=True, calls="~48 calls to Mistral"
)
def improvement():
  """Production answers rated by users (a stand-in that thumbs down missing facts and promises), the thumbs-down
  answers triaged into a regression set, candidate prompts evaluated against production on it plus the smoke tests,
  released only if nothing got worse, and the complaints asked again of the live app."""
  r = improvement_loop()
  rows = [
    {"variant": f"production {r['production_before']}", **{k: r["base"].get(k, 0) for k in ("facts", "no_promises")}}
  ]
  rows += [
    {"variant": f"{a['version']} {a['commit']}", **{k: a["scores"].get(k, 0) for k in ("facts", "no_promises")}}
    for a in r["attempts"]
  ]
  card = scorecard(
    rows, [Metric("facts", "facts present"), Metric("no_promises", "no promises")], title="Regression set + smoke"
  )
  gates = table(
    ["candidate", "gate", "what got worse"],
    [
      [f"{a['version']} {a['commit']}", "passed: released" if a["ok"] else "blocked", json.dumps(a["broken"])]
      for a in r["attempts"]
    ],
  )
  verdicts = " ".join(
    f"{a['version']} "
    + ("passed the gate." if a["ok"] else "was blocked (" + ", ".join(f"{e} lost {k}" for e, k in a["broken"]) + ").")
    for a in r["attempts"]
  )
  outcome = (
    f" Released by moving the production tag to {r['released']}; asked again, the live app resolved {r['resolved']} of "
    f"{r['thumbs_down_before']} complaints ({r['thumbs_up_after']}/12 answers now rated 👍)."
    if r["released"]
    else " Nothing was released."
  )
  return [
    card,
    gates,
    table(["complaint", "message", "before", "after"], [list(c) for c in r["complaints"]]),
    callout(
      f"{r['thumbs_down_before']} of 12 production answers got a 👎. {verdicts}{outcome}",
      "decision" if r["released"] else "warning",
      "Release the support reply prompt" if r["released"] else "Keep the production prompt",
    ),
  ]
