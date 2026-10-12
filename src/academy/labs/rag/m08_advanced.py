"""RAG · Module 8 labs: agentic RAG (one tool per source), corrective RAG and Self-RAG as LangGraph graphs,
GraphRAG over extracted triples, and multimodal RAG with images. Then: the knowledge agent against the answerer."""

import json
import time
from functools import lru_cache
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from academy.labs import cache
from academy.labs.rag import corpus
from academy.labs.runtime import Metric, answer, callout, code, columns, example, images, lab, scorecard, table
from nimbus.config import PROJECT_DIR, get_settings
from nimbus.llm import get_chat_model
from nimbus.llm.usage import UsageRecorder
from nimbus.rag.agent import build_knowledge_agent
from nimbus.rag.answer import Answerer, format_sources
from nimbus.rag.ingestion.embeddings import get_embeddings
from nimbus.rag.retriever import Retriever

DATA = get_settings().data_dir
AGENTIC = json.loads((PROJECT_DIR / "data" / "eval" / "agentic.json").read_text())


@lru_cache
def _retriever() -> Retriever:
  s = get_settings()
  return Retriever(s, get_embeddings(s), get_chat_model())


def _calls(messages) -> list[tuple[str, dict]]:
  return [(c["name"], c["args"]) for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]


def _correct(text: str, expect: list[str]) -> bool:
  norm = text.replace(",", "").replace("**", "").casefold()
  return all(e.casefold() in norm for e in expect)


# ── Lab 1: agentic RAG ──────────────────────────────────────────────────────────
# Each learning point is a small function: the "What it covers" tab shows its body, and the lab runs it.


def retrieval_as_a_tool(retriever):
  from langchain_core.tools import tool

  @tool
  def search_documents(query: str) -> str:
    """Search Nimbus Coffee's documents. Pass a complete, standalone question."""
    _, hits = retriever.search(query, k=4)
    return format_sources([d for d, _ in hits])  # numbered sources the model can cite

  return search_documents


def an_agent_that_decides(model, search_tool, questions):
  from langchain.agents import create_agent

  agent = create_agent(
    model, tools=[search_tool], system_prompt="Answer from the documents when needed; just reply to small talk."
  )
  runs = {q: agent.invoke({"messages": [("user", q)]})["messages"] for q in questions}
  return runs  # look at which questions triggered a search, and how many


def one_tool_per_source(retriever, data_dir):
  from nimbus.rag.agent import build_tools, structured_data

  db = structured_data(data_dir)
  try:
    tools = build_tools(retriever, db)  # documents (search) and structured data (SQL)
    return {t.name: t.description.splitlines()[0] for t in tools}
  finally:
    db.close()


@lab(
  "rag.agentic",
  "Agentic RAG: the agent decides whether, where and how often to search",
  uses_model=True,
  calls="~12 calls to Mistral",
)
def agentic():
  """A search tool alone, then production's knowledge agent with a tool per source, on small talk, single and
  multi-part questions."""
  model = get_chat_model()
  search = retrieval_as_a_tool(_retriever())
  questions = [
    "Hi! Who are you?",
    "How many days of sick leave do I get?",
    "What's the learning budget, and how often are laptops replaced?",
  ]
  runs = an_agent_that_decides(model, search, questions)
  tools = one_tool_per_source(_retriever(), DATA)
  multi = [q for q in AGENTIC if q["kind"] in ("aggregate", "multi_hop")][-2:]
  with build_knowledge_agent(get_settings(), _retriever(), model) as agent:
    multi_runs = {q["question"]: agent.invoke({"messages": [("user", q["question"])]})["messages"] for q in multi}
  return [
    table(
      ["Question", "searches", "answer"],
      [[q, " | ".join(a["query"] for _, a in _calls(m)) or "(no search)", m[-1].text[:160]] for q, m in runs.items()],
      title="One search tool: the agent decides",
    ),
    table(
      ["Question", "tool calls", "answer"],
      [
        [q, " → ".join(f"{n}({next(iter(a.values()))[:60]})" for n, a in _calls(m)), m[-1].text[:160]]
        for q, m in multi_runs.items()
      ],
      title="The knowledge agent: a tool per source",
    ),
    callout(
      "A pipeline searches every time; an agent searches when the question needs it (here: "
      f"{len(_calls(runs[questions[0]]))} searches for small talk) and as often as it decides: for the two-part "
      f"question it made {len(_calls(runs[questions[2]]))} search(es)"
      + (
        " with both parts in one query, which worked because one chunk held both. "
        if len(_calls(runs[questions[2]])) == 1
        else ". "
      )
      + "With a tool per source it also picks the right place: SQL for counting over the menu, documents for policies, "
      "both for a question that spans them. The cost is more model calls and less predictable latency; the "
      "evaluation lab measures both against the answerer.",
      "decision",
      "Agentic RAG",
    ),
  ] + [
    example(
      "lc.l46.1",
      "Retrieval wrapped in a @tool",
      retrieval_as_a_tool,
      "The docstring tells the model what the tool is for; the result is numbered sources.",
      [table(["tool", "description"], [[search.name, search.description]])],
    ),
    example(
      "lc.l46.2",
      "It searches only when it needs to",
      an_agent_that_decides,
      "",
      [table(["Question", "searches"], [[q, len(_calls(m))] for q, m in runs.items()])],
    ),
    example(
      "lc.l46.3",
      "Several searches for a multi-part question",
      an_agent_that_decides,
      "",
      [
        table(
          ["Question", "queries"],
          [[q, " | ".join(a["query"] for _, a in _calls(m))] for q, m in list(runs.items())[2:]],
        )
      ],
    ),
    example(
      "rag.l44.1",
      "One tool per knowledge source",
      one_tool_per_source,
      "Production's knowledge agent: document search and read-only SQL over the menu and orders.",
      [table(["tool", "for"], [[k, v] for k, v in tools.items()])],
    ),
    example(
      "rag.l44.2",
      "The agent chooses the tool, or none",
      an_agent_that_decides,
      "",
      [
        table(["Question", "tools"], [[q, ", ".join(n for n, _ in _calls(m)) or "none"] for q, m in multi_runs.items()])
      ],
    ),
    example(
      "rag.l44.3",
      "Multi-hop: several tools in sequence",
      one_tool_per_source,
      "",
      [
        table(["Question", "calls in order"], [[q, " → ".join(n for n, _ in _calls(m))] for q, m in multi_runs.items()])
      ],
    ),
  ]


# ── Lab 2: corrective RAG ───────────────────────────────────────────────────────
class Grade(BaseModel):
  relevant: bool = Field(description="Does this chunk help answer the question?")


class CragState(TypedDict, total=False):
  question: str
  query: str
  docs: list[Document]
  relevant: list[Document]
  tries: int
  answer: str
  path: list[str]


def grade_chunks(model, question, docs):
  graded = [
    (d, cache.structured(model, Grade, f"Question: {question}\n\nChunk:\n{d.page_content}\n\nIs it relevant?").relevant)
    for d in docs
  ]
  return [d for d, ok in graded if ok]  # an LLM as a yes/no grader, one call per chunk


def build_crag(model, retriever):
  from langgraph.graph import END, START, StateGraph

  def retrieve(state):
    _, hits = retriever.search(state["query"], k=3)
    return {"docs": [d for d, _ in hits], "path": state.get("path", []) + [f"retrieve({state['query']})"]}

  def grade(state):
    relevant = grade_chunks(model, state["question"], state["docs"])
    return {"relevant": relevant, "path": state["path"] + [f"grade: {len(relevant)}/{len(state['docs'])} relevant"]}

  def rewrite(state):
    query = cache.text(
      model,
      "Rewrite this as a better search query for a coffee company's documents. Reply with the query "
      f"only.\n{state['question']}",
    )
    return {"query": query, "tries": state.get("tries", 0) + 1, "path": state["path"] + ["rewrite"]}

  def generate(state):
    reply = cache.text(
      model,
      [
        ("system", "Answer only from the sources, in one sentence.\n\n" + format_sources(state["relevant"])),
        ("human", state["question"]),
      ],
    )
    return {"answer": reply, "path": state["path"] + ["generate"]}

  def give_up(state):
    return {"answer": "I couldn't find this in the documents.", "path": state["path"] + ["give up"]}

  def after_grading(state) -> Literal["generate", "rewrite", "give_up"]:
    if state["relevant"]:
      return "generate"
    return "rewrite" if state.get("tries", 0) < 1 else "give_up"  # one retry, then say so

  graph = StateGraph(CragState)
  for name, fn in (
    ("retrieve", retrieve),
    ("grade", grade),
    ("rewrite", rewrite),
    ("generate", generate),
    ("give_up", give_up),
  ):
    graph.add_node(name, fn)
  graph.add_edge(START, "retrieve")
  graph.add_edge("retrieve", "grade")
  graph.add_conditional_edges("grade", after_grading)
  graph.add_edge("rewrite", "retrieve")
  graph.add_edge("generate", END)
  graph.add_edge("give_up", END)
  return graph.compile()


@lab(
  "rag.crag",
  "Corrective RAG: grade the chunks before answering",
  uses_model=True,
  calls="~15 calls to Mistral (cached after the first run)",
)
def crag():
  """A LangGraph that grades retrieved chunks, rewrites and retries when none are relevant, and gives up honestly."""
  model = get_chat_model()
  graph = build_crag(model, _retriever())
  questions = ["How long is parental leave?", "perks??", "What is the CEO's annual salary?"]
  runs = {q: graph.invoke({"question": q, "query": q}) for q in questions}
  return [
    table(
      ["Question", "path", "answer"],
      [[q, " → ".join(r["path"]), r["answer"]] for q, r in runs.items()],
      title="The path each question took",
    ),
    code(graph.get_graph().draw_mermaid(), "text", title="The graph"),
    callout(
      "Corrective RAG checks the evidence before using it: an LLM grades each chunk, a question with nothing relevant "
      "gets a rewritten query and a second search, and if that fails too the graph says so instead of answering from "
      "noise. "
      + (
        "Graders can be too strict: it gave up on 'perks??' although the handbook lists benefits. "
        if any("give up" in r["path"] for q, r in runs.items() if q == "perks??")
        else ""
      )
      + "Each grade is a model call per chunk; production's answerer covers the same failure more cheaply (reranked "
      "chunks and an explicit 'answerable' field), so CRAG suits corpora where retrieval often returns nothing useful.",
      "decision",
      "Corrective RAG",
    ),
  ] + [
    example(
      "rag.l45.1",
      "Grade each chunk",
      grade_chunks,
      "",
      [table(["Question", "relevant chunks"], [[q, r["path"][1]] for q, r in runs.items()])],
    ),
    example(
      "rag.l45.2",
      "Nothing relevant: rewrite and search again",
      build_crag,
      "",
      [
        table(
          ["Question", "path"],
          [[q, " → ".join(r["path"])] for q, r in runs.items() if "rewrite" in r["path"]]
          or [["(no question needed a rewrite)", ""]],
        )
      ],
    ),
    example(
      "rag.l45.3",
      "Still nothing: say so",
      build_crag,
      "",
      [table(["Question", "answer"], [[q, r["answer"]] for q, r in runs.items()])],
    ),
    example(
      "rag.l45.4",
      "A LangGraph with conditional edges",
      build_crag,
      "`after_grading` returns the next node's name.",
      [code(graph.get_graph().draw_mermaid(), "text")],
    ),
  ]


# ── Lab 3: Self-RAG ─────────────────────────────────────────────────────────────
class Supported(BaseModel):
  supported: bool = Field(description="Is EVERY claim in the answer supported by the sources?")
  unsupported: list[str] = Field(description="Claims not supported by the sources")


class Addresses(BaseModel):
  addresses: bool = Field(description="Does the answer actually address the question?")


def hallucination_grader(model, sources, answer_text):
  return cache.structured(
    model, Supported, f"Sources:\n{sources}\n\nAnswer:\n{answer_text}\n\nIs every claim supported by the sources?"
  )


def answer_grader(model, question, answer_text):
  return cache.structured(
    model, Addresses, f"Question: {question}\nAnswer: {answer_text}\n\nDoes the answer address the question?"
  )


def self_rag_loop(model, question, sources, first_answer, max_tries):
  attempts, current = [], first_answer
  for attempt in range(max_tries):
    grounded = hallucination_grader(model, sources, current)
    useful = answer_grader(model, question, current)
    attempts.append((current, grounded.supported, useful.addresses, grounded.unsupported))
    if grounded.supported and useful.addresses:
      break
    current = cache.text(
      model,
      [
        ("system", f"Answer ONLY with facts from the sources, in one sentence. Attempt {attempt + 2}.\n\n{sources}"),
        ("human", question),
      ],
    )
  return attempts  # regenerate while unsupported, give up after max_tries


@lab(
  "rag.self-rag",
  "Self-RAG: check the answer before returning it",
  uses_model=True,
  calls="~8 calls to Mistral (cached after the first run)",
)
def self_rag():
  """Graders for support and relevance, run on a fabricated answer and on a regenerated one."""
  model = get_chat_model()
  question = "How long is parental leave, and is it paid?"
  _, hits = _retriever().search(question, k=3)
  sources = format_sources([d for d, _ in hits])
  fabricated = "Parents get 52 weeks of paid parental leave, plus a 10,000 rupee baby bonus."
  attempts = self_rag_loop(model, question, sources, fabricated, max_tries=3)
  off_topic = answer_grader(model, question, "Hotels are reimbursed up to 6,000 rupees per night.")
  return [
    table(
      ["Attempt", "answer", "supported?", "addresses the question?", "unsupported claims"],
      [[i + 1, a, "✓" if s else "✗", "✓" if u else "✗", "; ".join(c)] for i, (a, s, u, c) in enumerate(attempts)],
      title=question,
    ),
    callout(
      f"The graders caught the fabricated answer ({len(attempts[0][3])} unsupported claim(s)), and the loop "
      f"regenerated until both graders passed ({len(attempts)} attempt(s)). Each check is a model call, and graders "
      "can be wrong too: production prefers checks code can do exactly (citation quotes verified against their "
      "sources) and keeps graders for offline evaluation (Module 9).",
      "decision",
      "Self-RAG",
    ),
  ] + [
    example(
      "rag.l46.1",
      "Hallucination grader",
      hallucination_grader,
      "",
      [table(["answer", "supported", "unsupported"], [[attempts[0][0], attempts[0][1], "; ".join(attempts[0][3])]])],
    ),
    example(
      "rag.l46.2",
      "Answer grader",
      answer_grader,
      "",
      [
        table(
          ["answer", "addresses the question?"],
          [["Hotels are reimbursed up to 6,000 rupees per night.", off_topic.addresses]],
        )
      ],
    ),
    example(
      "rag.l46.3",
      "Regenerate, give up after a few tries",
      self_rag_loop,
      "",
      [table(["attempt", "supported", "addresses"], [[i + 1, s, u] for i, (_, s, u, _) in enumerate(attempts)])],
    ),
    example(
      "rag.l46.4",
      "The graders catch a fabricated answer",
      self_rag_loop,
      "",
      [answer(attempts[0][0], title="fabricated"), answer(attempts[-1][0], title="after the loop")],
    ),
  ]


# ── Lab 4: GraphRAG ─────────────────────────────────────────────────────────────
GRAPH_QUESTION = "Which of the blends Nimbus sells comes from the same region as its bean supplier?"


class Triple(BaseModel):
  subject: str
  relation: str
  object: str


class Triples(BaseModel):
  triples: list[Triple]


class Entities(BaseModel):
  names: list[str]


def extract_triples(model, docs):
  prompt = (
    "Extract factual (subject, relation, object) triples from the text. Use short canonical entity names "
    "(e.g. 'Nimbus Coffee Co.', 'Chikmagalur', 'Green Valley Estates') and short verb relations.\n\nText:\n{text}"
  )
  return [t for d in docs for t in cache.structured(model, Triples, prompt.format(text=d.page_content)).triples]


def build_graph(triples):
  import networkx as nx

  graph = nx.MultiDiGraph()  # entities are nodes, relations are edges
  for t in triples:
    if t.subject.strip() and t.object.strip():
      graph.add_edge(t.subject.strip(), t.object.strip(), relation=t.relation)
  return graph


def walk_from_the_question(model, graph, question):
  import networkx as nx

  picked = cache.structured(
    model,
    Entities,
    "Which of these entities does the question mention or clearly refer to (e.g. 'bean supplier' → the supplier "
    f"company)? Copy names exactly.\nQuestion: {question}\nEntities: {sorted(graph.nodes)}",
  )
  facts = set()
  for node in [n for n in picked.names if n in graph]:
    for a, b, data in nx.ego_graph(graph.to_undirected(as_view=True), node, radius=2).edges(data=True):  # 2 hops
      facts.add(f"{a} {data['relation']} {b}" if graph.has_edge(a, b) else f"{b} {data['relation']} {a}")
  return picked.names, sorted(facts)


@lab(
  "rag.graphrag",
  "GraphRAG: follow relationships between entities",
  uses_model=True,
  calls="~6 calls to Mistral (cached after the first run)",
)
def graphrag():
  """Triples from three documents, a networkx graph, a two-hop walk from the question's entities, and vector search for
  the same multi-hop question."""
  model = get_chat_model()
  docs = [
    d for d in corpus.documents("v3-tesseract") if d.metadata["source"] in ("products.txt", "supplier_contract.docx")
  ]
  started = time.perf_counter()
  triples = extract_triples(model, docs)
  graph = build_graph(triples)
  start, facts = walk_from_the_question(model, graph, GRAPH_QUESTION)
  graph_answer = cache.text(
    model, "Facts:\n" + "\n".join(facts) + f"\n\nUsing only these facts, answer in one sentence: {GRAPH_QUESTION}"
  )
  seconds = time.perf_counter() - started
  _, hits = _retriever().search(GRAPH_QUESTION, k=3)
  vector_answer = cache.text(
    model,
    [
      ("system", "Answer only from the sources, in one sentence.\n\n" + format_sources([d for d, _ in hits])),
      ("human", GRAPH_QUESTION),
    ],
  )
  hubs = sorted(graph.degree, key=lambda x: -x[1])[:6]
  return [
    table(
      ["subject", "relation", "object"],
      [[t.subject, t.relation, t.object] for t in triples[:14]],
      title=f"{len(triples)} triples from {len(docs)} documents ({seconds:.1f} s)",
    ),
    table(
      ["entity", "connections"],
      hubs,
      title=f"The graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges",
    ),
    columns(
      [
        answer(graph_answer, "graph walk", meta={"start": ", ".join(start), "facts": len(facts)}),
        answer(
          vector_answer,
          "vector search (production retriever)",
          meta={"sources": ", ".join(d.metadata["source"] for d, _ in hits)},
        ),
      ],
      title=GRAPH_QUESTION,
    ),
    callout(
      "The answer needs two facts from two documents: the supplier is in Chikmagalur (the contract) and a blend comes "
      "from Chikmagalur (the product guide). A graph connects them through the shared node; vector search finds them "
      "only if both chunks happen to rank high. GraphRAG costs an extraction call per document and its quality depends "
      "on consistent entity names; it pays off for corpora full of relationships (org charts, supply chains), not for "
      "a policy FAQ.",
      "decision",
      "GraphRAG",
    ),
  ] + [
    example(
      "rag.l47.1",
      "Extract (subject, relation, object) triples",
      extract_triples,
      "",
      [table(["subject", "relation", "object"], [[t.subject, t.relation, t.object] for t in triples[:8]])],
    ),
    example(
      "rag.l47.2",
      "Store them in a graph",
      build_graph,
      "",
      [table(["nodes", "edges"], [[graph.number_of_nodes(), graph.number_of_edges()]])],
    ),
    example(
      "rag.l47.3",
      "Walk from the question's entities",
      walk_from_the_question,
      "",
      [table(["start", "facts within 2 hops"], [[", ".join(start), "\n".join(facts[:12])]])],
    ),
    example(
      "rag.l47.4",
      "Against vector search",
      walk_from_the_question,
      "",
      [table(["way", "answer"], [["graph", graph_answer], ["vector", vector_answer]])],
    ),
  ]


# ── Lab 5: multimodal RAG ───────────────────────────────────────────────────────
IMAGES = ["sales_chart.png", "receipt.png"]


def describe_images(model):
  from nimbus.rag.ingestion.ocr import image_message, open_image

  docs = []
  for name in IMAGES:
    description = cache.text(
      model,
      [
        image_message(
          open_image(DATA / name),
          "Describe this image for a search index: what it is, its title, and every number and label in it. Plain "
          "text, 3-5 sentences.",
        )
      ],
    )
    docs.append(Document(description, metadata={"source": name, "doc_type": "image", "image_path": str(DATA / name)}))
  return docs  # the description is what gets embedded; the image path rides in metadata


def one_index(image_docs):
  from langchain_core.vectorstores import InMemoryVectorStore

  chunks = [Document(c.page_content, metadata={**c.metadata, "doc_type": "text"}) for c in corpus.chunks(300, 50, "v2")]
  store = InMemoryVectorStore.from_documents(chunks + image_docs, corpus.embeddings())  # text and images together
  return store


def mixed_context_answer(model, question, hits):
  import base64
  from pathlib import Path

  from langchain_core.messages import HumanMessage

  texts = [d for d in hits if d.metadata["doc_type"] != "image"]
  blocks = [
    {
      "type": "text",
      "text": "Answer from these sources and the attached image(s), in one sentence.\n\n"
      f"{format_sources(texts)}\n\nQuestion: {question}",
    }
  ]
  for d in hits:
    if d.metadata["doc_type"] == "image":  # the ORIGINAL image, not just its description
      blocks.append(
        {
          "type": "image",
          "base64": base64.b64encode(Path(d.metadata["image_path"]).read_bytes()).decode(),
          "mime_type": "image/png",
        }
      )
  return cache.text(model, [HumanMessage(content_blocks=blocks)])


@lab(
  "rag.multimodal",
  "Multimodal RAG: charts and images in the knowledge base",
  uses_model=True,
  calls="3 calls to Mistral (vision; cached after the first run)",
)
def multimodal():
  """Images described by a vision model, indexed with the text chunks, answered from text plus the original image."""
  model = get_chat_model()
  image_docs = describe_images(model)
  store = one_index(image_docs)
  question = "Which quarter had the highest revenue in 2025, and by how much did it beat the lowest quarter?"
  hits = store.similarity_search(question, k=4)
  reply = mixed_context_answer(model, question, hits)
  from nimbus.rag.ingestion.ocr import open_image

  thumbs = []
  for name in IMAGES:
    import base64
    import io

    buffer = io.BytesIO()
    image = open_image(DATA / name).convert("RGB")
    image.thumbnail((360, 360))
    image.save(buffer, format="PNG")
    thumbs.append({"src": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(), "caption": name})
  return [
    images(thumbs, title="The images"),
    table(["image", "description (what gets embedded)"], [[d.metadata["source"], d.page_content] for d in image_docs]),
    table(
      ["rank", "source", "type"],
      [[i + 1, d.metadata["source"], d.metadata["doc_type"]] for i, d in enumerate(hits)],
      title=question,
    ),
    answer(reply, title="Answer from text chunks + the original chart"),
    callout(
      "A text embedding model can't read an image, so a vision model describes each one and the description is "
      "indexed beside the text chunks, with the image path in metadata. At answer time the model gets the original "
      "image too, because a description can miss the detail the question asks about. Production's loaders v3 OCR "
      "images to text (Module 3); descriptions add what OCR can't (what a chart shows).",
      "decision",
      "Multimodal",
    ),
  ] + [
    example(
      "rag.l48.1",
      "Describe each image with a vision model",
      describe_images,
      "",
      [table(["image", "description"], [[d.metadata["source"], d.page_content] for d in image_docs])],
    ),
    example(
      "rag.l48.2",
      "Index the description, keep the image path",
      describe_images,
      "",
      [table(["metadata"], [[json.dumps(d.metadata)] for d in image_docs])],
    ),
    example(
      "rag.l48.3",
      "One index for text and images",
      one_index,
      "",
      [
        table(
          ["rank", "source", "type"],
          [[i + 1, d.metadata["source"], d.metadata["doc_type"]] for i, d in enumerate(hits)],
        )
      ],
    ),
    example(
      "rag.l48.4", "Answer from text + the original image", mixed_context_answer, "", [answer(reply, title=question)]
    ),
  ]


# ── Production path: the knowledge agent ────────────────────────────────────────
def evaluate_agentic() -> list[dict]:
  model = get_chat_model()
  answerer = Answerer(get_settings(), _retriever(), model)
  with build_knowledge_agent(get_settings(), _retriever(), model) as agent:
    return _evaluate_with(answerer, agent)


def _evaluate_with(answerer, agent) -> list[dict]:
  rows = []
  for case in AGENTIC:
    q = case["question"]
    t = time.perf_counter()
    r = answerer.answer(q)
    a_text = r.answer
    a_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    rec2 = UsageRecorder()
    msgs = agent.invoke({"messages": [HumanMessage(q)]}, {"callbacks": [rec2]})["messages"]
    g_ms = (time.perf_counter() - t) * 1000
    rows.append(
      {
        "case": case,
        "answerer": (a_text, r.answerable, r.recorder, a_ms, 1),
        "agent": (msgs[-1].text, msgs, rec2, g_ms, len(_calls(msgs))),
      }
    )
  return rows


def _ok(case: dict, text: str, searched: int | None = None) -> bool:
  from academy.labs.rag.m07_generation import refused

  if case["kind"] == "small_talk":
    return searched == 0 if searched is not None else True
  if case["kind"] == "unanswerable":
    return refused(text)
  return _correct(text, case["expect"])


@lab(
  "rag.agentic-eval",
  "Evaluate: the knowledge agent vs the answerer",
  uses_model=True,
  calls="~45 calls to Mistral (15 questions, two ways): about 2 minutes",
)
def agentic_eval():
  """15 questions of five kinds: documents, aggregates over the menu and orders, multi-hop, small talk, unanswerable."""
  started = time.perf_counter()
  results = evaluate_agentic()
  seconds = time.perf_counter() - started
  kinds = ["documents", "aggregate", "multi_hop", "small_talk", "unanswerable"]
  rows = []
  for who in ("answerer", "agent"):
    row = {"variant": "answerer (Module 7)" if who == "answerer" else "knowledge agent (production)"}
    for kind in kinds:
      cases = [r for r in results if r["case"]["kind"] == kind]
      if who == "agent":
        ok = sum(_ok(r["case"], r["agent"][0], r["agent"][4] if kind == "small_talk" else None) for r in cases)
      else:
        ok = sum(
          _ok(r["case"], r["answerer"][0], 1 if kind == "small_talk" else None) for r in cases
        )  # it always searches
      row[kind] = f"{ok}/{len(cases)}"
    usage = [r[who][2].usage() for r in results]
    row["tokens"] = round(sum(u.input_tokens + u.output_tokens for u in usage) / len(results))
    row["p50_ms"] = round(sorted(r[who][3] for r in results)[len(results) // 2])
    rows.append(row)
  detail = [
    [
      r["case"]["kind"],
      r["case"]["question"],
      r["answerer"][0][:90],
      " → ".join(n for n, _ in _calls(r["agent"][1])) or "none",
      r["agent"][0][:90],
    ]
    for r in results
  ]
  return [
    scorecard(
      rows,
      [
        *[Metric(k, k.replace("_", " "), "none", "text") for k in kinds],
        Metric("tokens", "tokens / question", "lower", "int"),
        Metric("p50_ms", "latency p50", "lower", "ms"),
      ],
      title=f"15 questions · {seconds:.0f} s",
    ),
    table(["kind", "question", "answerer", "agent tools", "agent"], detail),
    callout(
      f"Answerer: documents {rows[0]['documents']}, aggregates {rows[0]['aggregate']} (it searches text for 'how "
      f"many'), small talk {rows[0]['small_talk']} (it searches even for 'hi'). Knowledge agent: documents "
      f"{rows[1]['documents']}, aggregates {rows[1]['aggregate']} via SQL, small talk {rows[1]['small_talk']}, at "
      f"{rows[1]['tokens'] / max(rows[0]['tokens'], 1):.1f}× the tokens per question (every step resends the tools "
      "and the conversation). Production uses both: the agent as the entry point for open questions, the answerer's "
      "verified citations when a question is about the documents.",
      "decision",
      "Ship the knowledge agent",
    ),
  ]
