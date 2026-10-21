"""RAG · Module 4 labs: how documents are cut into chunks (by characters, tokens, structure and meaning), how chunks
get back the context they lost, and which configuration retrieves best on the 45 evaluation questions."""

import json
import re
import time

import numpy as np
from langchain_core.documents import Document

from academy.labs.rag import corpus
from academy.labs.runtime import (
  Input,
  Metric,
  bars,
  callout,
  code,
  columns,
  example,
  highlight,
  lab,
  scorecard,
  spans_of,
  table,
)
from nimbus.config import get_settings
from nimbus.llm import get_chat_model
from nimbus.rag.evaluation import evaluate

DATA = get_settings().data_dir
QUESTIONS = {  # question → a phrase the right chunk must contain
  "What is the hotel limit per night?": "6,000 rupees per night",
  "How often are passwords rotated?": "180 days",
  "How much does Nilgiri Sunrise cost?": "720 rupees",
  "When are salaries paid?": "last working day",
  "How many days of sick leave do employees get?": "12 days per year",
  "Who approves remote work outside India?": "Meera Iyer",
}


def _text() -> str:
  return (DATA / "handbook.md").read_text() + "\n\n" + (DATA / "products.txt").read_text()


def _top1(chunks: list[str]) -> int:
  """How many QUESTIONS get the chunk with their answer at rank 1."""
  vectors = np.array(corpus.embeddings().embed_documents(chunks))
  found = 0
  for q, phrase in QUESTIONS.items():
    best = chunks[int(np.argmax(vectors @ np.array(corpus.embeddings().embed_query(q))))]
    found += phrase in " ".join(best.split())
  return found


# ── Lab 1: splitting by characters ──────────────────────────────────────────────
# Each learning point is a small function: the "What it covers" tab shows its body, and the lab runs it.


def why_split(question):
  # One vector for the whole handbook has to stand for every topic in it at once
  whole = (DATA / "handbook.md").read_text()
  paragraph = next(p for p in whole.split("\n\n") if "Travel and expenses" in p)
  vectors = corpus.embeddings().embed_documents([whole, paragraph])
  q = corpus.embeddings().embed_query(question)
  similarity = {"whole handbook": float(np.dot(vectors[0], q)), "travel section": float(np.dot(vectors[1], q))}
  return similarity


def fixed_size(text, size):
  from langchain_text_splitters import CharacterTextSplitter

  chunks = CharacterTextSplitter(separator="", chunk_size=size, chunk_overlap=0).split_text(text)  # every N characters
  return chunks


def on_paragraphs(text, size):
  from langchain_text_splitters import CharacterTextSplitter

  chunks = CharacterTextSplitter(separator="\n\n", chunk_size=size, chunk_overlap=0).split_text(text)
  return chunks  # natural boundaries, but a long paragraph is never cut: sizes are uneven


def recursive(text, size, overlap):
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  splitter = RecursiveCharacterTextSplitter(
    chunk_size=size, chunk_overlap=overlap
  )  # tries "\n\n", then "\n", then " ", then characters, until every piece fits
  chunks = splitter.split_text(text)
  return chunks


def overlap_at_a_boundary(overlap):
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  sentence = "Unused leave can be carried over, up to a maximum of 8 days, into the next year."
  without = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0).split_text(sentence)
  with_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=overlap).split_text(sentence)
  return without, with_overlap  # with overlap, the words at each cut appear in both neighbours


def split_documents_keeps_metadata():
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  doc = Document((DATA / "products.txt").read_text(), metadata={"source": "products.txt", "doc_type": "catalogue"})
  chunks = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, add_start_index=True).split_documents([doc])
  return chunks  # each chunk: the parent's metadata + start_index (where it begins in the document)


def size_on_real_questions(sizes):
  results = {}
  for size in sizes:
    chunks = corpus.chunks(size, size // 6, "v3-tesseract")  # the full corpus, production's splitter
    vectors = corpus.chunk_vectors(size, size // 6, "v3-tesseract")

    def retrieve(query, k, chunks=chunks, vectors=vectors):
      q = np.array(corpus.embeddings().embed_query(query))
      return [chunks[i] for i in np.argsort(-(vectors @ q))[:k]]

    report = evaluate(retrieve, corpus.questions(), [c.page_content for c in chunks], k=5)
    results[size] = (len(chunks), report.at["hit"][5], report.mrr)
  return results


def _highlighted(text: str, ways: dict[str, list[str]], size: int, overlap: int) -> list:
  """The same opening passage under each splitter, every chunk in its own colour."""
  window = min(len(text), max(900, size * 4))
  views = []
  for name, chunks in ways.items():
    spans = [(a, min(b, window)) for a, b in spans_of(text, chunks) if a < window]
    cut_words = sum(1 for _, b in spans if 0 < b < window and text[b - 1].isalnum() and text[b].isalnum())
    shared = sum(max(0, spans[i][1] - spans[i + 1][0]) for i in range(len(spans) - 1))
    title = f"{name}: {len(spans)} chunks here, {cut_words} cut mid-word, {shared} characters in two chunks"
    views.append(highlight(text[:window], spans, title=title))
  return views + [_overlap_reality(size, overlap)]


def _overlap_reality(size: int, overlap: int) -> dict:
  """How often recursive splitting's overlap actually repeats text, measured on every v1 document."""
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
  pairs = shared = 0
  for d in corpus.documents("v1"):
    spans = spans_of(d.page_content, splitter.split_text(d.page_content))
    pairs += max(0, len(spans) - 1)
    shared += sum(1 for i in range(len(spans) - 1) if spans[i][1] > spans[i + 1][0])
  return callout(
    f"With chunk_size={size} and chunk_overlap={overlap}, {'only ' if shared < pairs / 4 else ''}"
    f"**{shared} of {pairs}** neighbouring chunk pairs in the whole corpus share any text. The recursive splitter "
    "carries over only whole pieces (a paragraph, a line or a word) that fit inside the overlap. When a chunk ends "
    "at a paragraph break, that paragraph is longer than the overlap, so nothing is repeated. Overlap only takes "
    "effect where a chunk has to be cut inside a paragraph: set a small chunk size to see it.",
    "info",
    "How often overlap really applies",
  )


@lab(
  "rag.splitters",
  "Splitting by characters: fixed, paragraphs, recursive, and overlap",
  inputs=[
    Input("size", "Chunk size (characters)", "number", default=200, min=50, max=1000),
    Input("overlap", "Overlap (characters)", "number", default=20, min=0, max=100),
  ],
)
def splitters(size: int, overlap: int):
  """The handbook and product guide split three ways; each version searched with six questions."""
  text = _text()
  overlap = min(overlap, size - 1)
  ways = {
    "fixed size": fixed_size(text, size),
    "on paragraphs": on_paragraphs(text, size),
    "recursive": recursive(text, size, overlap),
  }
  rows = []
  for name, chunks in ways.items():
    lengths = [len(c) for c in chunks]
    rows.append(
      {
        "variant": name,
        "chunks": len(chunks),
        "avg": round(sum(lengths) / len(lengths)),
        "largest": max(lengths),
        "over_size": sum(n > size for n in lengths),
        "mid_word": sum(1 for c in chunks if re.match(r"^[a-z0-9]", c) or re.search(r"[a-z]$", c)),
        "top1": f"{_top1(chunks)}/{len(QUESTIONS)}",
      }
    )
  demo_overlap = max(1, min(overlap, 25))  # the demo sentence is cut at 50 characters: overlap must be smaller
  without, with_overlap = overlap_at_a_boundary(demo_overlap)
  sentence = "Unused leave can be carried over, up to a maximum of 8 days, into the next year."
  kept = split_documents_keeps_metadata()
  similarity = why_split("What is the hotel limit per night?")
  sweep = size_on_real_questions((150, 300, 500, 1000))
  return [
    scorecard(
      rows,
      [
        Metric("chunks", "chunks", "none", "int"),
        Metric("avg", "average size", "none", "int"),
        Metric("over_size", f"chunks over {size}", "lower", "int"),
        Metric("mid_word", "start or end mid-word", "lower", "int"),
        Metric("top1", "answer in the top chunk", "none", "text", f"{len(QUESTIONS)} questions"),
      ],
      title=f"chunk_size={size}, overlap={overlap} (recursive only)",
    ),
    *_highlighted(text, ways, size, overlap),
    columns(
      [code("\n\n— chunk —\n\n".join(c[:300] for c in chunks[3:5]), "text", name) for name, chunks in ways.items()],
      title="Chunks 4 and 5 of each",
    ),
    table(
      ["chunk_size", "chunks", "hit@5 (45 questions)", "MRR"],
      [[s, n, round(h, 3), round(m, 3)] for s, (n, h, m) in sweep.items()],
      title="Chunk size on the whole corpus and all 45 questions (recursive, overlap ≈ size/6)",
    ),
    callout(
      "Fixed-size chunks cut words in half; splitting on paragraphs keeps sentences whole but can't make a long "
      "paragraph smaller; recursive splitting tries paragraphs, then lines, then words, so every chunk fits at "
      "the most natural boundary available. That's why it's the default. Size is a trade-off you measure: too "
      "small and a fact is separated from its subject, too big and every chunk is about everything.",
      "decision",
      "Recursive splitting, size measured",
    ),
  ] + [
    example(
      "lc.l41.1",
      "Why split: one vector can't stand for everything",
      why_split,
      "The whole handbook's vector averages every topic; the travel section's vector is about travel.",
      [table(["Text", "Similarity to the question"], [[k, round(v, 3)] for k, v in similarity.items()])],
    ),
    example(
      "rag.l19.1",
      "Whole documents are too big and too vague",
      why_split,
      "The same comparison: the focused piece is much closer to the question than the whole document.",
      [table(["Text", "Similarity"], [[k, round(v, 3)] for k, v in similarity.items()])],
    ),
    example(
      "rag.l19.2",
      "Fixed size: simple, but cuts words",
      fixed_size,
      "Every N characters, wherever they fall.",
      [code("\n— chunk —\n".join(repr(c) for c in ways["fixed size"][3:6]), "text")],
    ),
    example(
      "rag.l19.3",
      "On a separator: natural boundaries, uneven sizes",
      on_paragraphs,
      "Split at blank lines; a paragraph longer than the size is kept whole.",
      [table(["Chunk sizes"], [[str([len(c) for c in ways["on paragraphs"]])]])],
    ),
    example(
      "lc.l41.2",
      "RecursiveCharacterTextSplitter: chunk_size and chunk_overlap",
      recursive,
      "The usual default: natural boundaries and a size limit.",
      [table(["Chunk sizes"], [[str([len(c) for c in ways["recursive"]])]])],
    ),
    example(
      "rag.l19.4",
      "Recursive: paragraphs → lines → words",
      recursive,
      "Each chunk ends at the largest boundary that keeps it under the size.",
      [code(ways["recursive"][5], "text", "chunk 6")],
    ),
    example(
      "lc.l41.3",
      "Overlap: no sentence lost at a boundary",
      overlap_at_a_boundary,
      "With overlap, the words at each cut appear in both neighbouring chunks.",
      [
        table(["Overlap", "Chunks"], [["0", " | ".join(without)], [str(demo_overlap), " | ".join(with_overlap)]]),
        highlight(sentence, spans_of(sentence, with_overlap), title=f"chunk_size=50, overlap={demo_overlap}"),
      ],
    ),
    example(
      "rag.l19.5",
      "Overlap repeats a little text",
      overlap_at_a_boundary,
      "The cost: repeated text is stored and embedded twice.",
      [
        table(
          ["Overlap", "characters stored"],
          [["0", sum(map(len, without))], [str(demo_overlap), sum(map(len, with_overlap))]],
        )
      ],
    ),
    example(
      "lc.l41.4",
      "split_documents(): metadata is kept",
      split_documents_keeps_metadata,
      "Each chunk gets its parent's metadata plus `start_index`.",
      [
        table(
          ["Chunk", "metadata", "Text"],
          [[i, json.dumps(c.metadata), c.page_content[:60] + "…"] for i, c in enumerate(kept[:4])],
        )
      ],
    ),
    example(
      "rag.l19.6",
      "Chunk size is a trade-off, measured",
      size_on_real_questions,
      "The full corpus at four sizes, scored on all 45 questions.",
      [
        table(
          ["chunk_size", "chunks", "hit@5", "MRR"], [[s, n, round(h, 3), round(m, 3)] for s, (n, h, m) in sweep.items()]
        )
      ],
    ),
  ]


# ── Lab 2: tokens ───────────────────────────────────────────────────────────────
def count_tokens(texts):
  import tiktoken

  encoding = tiktoken.get_encoding("cl100k_base")  # a common tokenizer; every model family has its own
  counts = {t: [encoding.decode([i]) for i in encoding.encode(t)] for t in texts}
  return counts  # text → its tokens


def the_silent_limit(embeddings):
  import numpy as np

  fact = "Hotels are reimbursed up to 6,000 rupees per night. "
  filler = "The committee reviewed the quarterly logistics schedule in detail. " * 60  # ~700 tokens
  long_chunk = fact + filler
  a = np.array(embeddings.embed_documents([long_chunk])[0])
  b = np.array(embeddings.embed_documents([long_chunk + "Laptops are replaced every 3 years."])[0])
  identical = bool(np.allclose(a, b))  # True: the laptop sentence never reached the model (512-token limit)
  return long_chunk, identical


def token_splitter(text, tokens):
  from langchain_text_splitters import TokenTextSplitter

  chunks = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=tokens, chunk_overlap=0).split_text(text)
  return chunks  # exactly `tokens` tokens each, wherever that falls


def recursive_in_tokens(text, tokens):
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=tokens, chunk_overlap=tokens // 6
  )  # natural boundaries, sized in tokens
  chunks = splitter.split_text(text)
  return chunks


@lab(
  "rag.tokens",
  "Tokens: measure chunks the way models do",
  inputs=[Input("tokens", "Chunk size (tokens)", "number", default=60, min=20, max=200)],
)
def tokens_lab(tokens: int):
  """Characters vs tokens, the embedding model's silent limit, and two token-sized splitters."""
  import tiktoken

  encoding = tiktoken.get_encoding("cl100k_base")
  samples = [
    "Hotels are reimbursed up to 6,000 rupees per night.",
    "GSTIN 29ABCDE1234F1Z5",
    "Chikmagalur",
    "నమస్కారం (hello in Telugu)",
  ]
  counts = count_tokens(samples)
  long_chunk, identical = the_silent_limit(corpus.embeddings())
  text = (DATA / "handbook.md").read_text()
  exact = token_splitter(text, tokens)
  natural = recursive_in_tokens(text, tokens)
  sizes = lambda chunks: [len(encoding.encode(c)) for c in chunks]  # noqa: E731
  rows = [
    {
      "variant": "TokenTextSplitter",
      "chunks": len(exact),
      "max_tokens": max(sizes(exact)),
      "mid_word": sum(1 for c in exact if re.match(r"^[a-z]", c.lstrip()) or re.search(r"[a-z]$", c)),
    },
    {
      "variant": "Recursive, sized in tokens",
      "chunks": len(natural),
      "max_tokens": max(sizes(natural)),
      "mid_word": sum(1 for c in natural if re.match(r"^[a-z]", c.lstrip()) or re.search(r"[a-z]$", c)),
    },
  ]
  ratios = [len(t) / max(len(encoding.encode(t)), 1) for t in samples]
  return [
    table(
      ["Text", "characters", "tokens", "chars per token", "tokens"],
      [
        [t, len(t), len(tok), round(r, 1), " | ".join(tok[:10])]
        for (t, tok), r in zip(counts.items(), ratios, strict=True)
      ],
      title="Characters vs tokens",
    ),
    scorecard(
      rows,
      [
        Metric("chunks", "chunks", "none", "int"),
        Metric("max_tokens", "largest chunk (tokens)", "none", "int"),
        Metric("mid_word", "chunks cut mid-word", "lower", "int"),
      ],
      title=f"The handbook in {tokens}-token chunks",
    ),
    columns(
      [
        code("\n— chunk —\n".join(exact[1:3]), "text", "TokenTextSplitter"),
        code("\n— chunk —\n".join(natural[1:3]), "text", "recursive, in tokens"),
      ]
    ),
    callout(
      f"English averages ~4 characters per token, but codes and other scripts take far more tokens. The embedding "
      "model reads at most 512 tokens and ignores the rest without any error: a "
      f"{len(encoding.encode(long_chunk))}-token chunk with one more sentence at the end gave "
      f"{'the identical' if identical else 'a different'} vector. Size "
      "chunks in tokens when a model limit matters, stay well under it, and prefer natural boundaries.",
      "decision",
      "Tokens",
    ),
  ] + [
    example(
      "rag.l20.1",
      "Models count tokens, not characters",
      count_tokens,
      "Common words are one token; codes, rare names and other scripts split into many.",
      [table(["Text", "tokens"], [[t, len(tok)] for t, tok in counts.items()])],
    ),
    example(
      "rag.l20.2",
      "The embedding model's silent limit",
      the_silent_limit,
      "bge-small reads 512 tokens. Text after that changes nothing.",
      [
        table(
          ["chunk tokens", "vectors identical with an extra sentence?"], [[len(encoding.encode(long_chunk)), identical]]
        )
      ],
    ),
    example(
      "rag.l20.3",
      "TokenTextSplitter: exact token counts",
      token_splitter,
      "Every chunk has exactly the size, so boundaries ignore words and sentences.",
      [table(["Token counts"], [[str(sizes(exact))]])],
    ),
    example(
      "rag.l20.4",
      "Recursive splitting, sized in tokens",
      recursive_in_tokens,
      "Natural boundaries, measured with the tokenizer.",
      [table(["Token counts"], [[str(sizes(natural))]])],
    ),
  ]


# ── Lab 3: structure-aware splitting ────────────────────────────────────────────
def markdown_sections():
  from langchain_text_splitters import MarkdownHeaderTextSplitter

  splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "document"), ("##", "section")])
  sections = splitter.split_text((DATA / "handbook.md").read_text())
  return sections  # one per section; the headings become metadata


def then_by_size(sections):
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  small = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0).split_documents(sections)
  return small  # long sections split again, each piece keeps its section metadata


def html_sections():
  from langchain_text_splitters import HTMLHeaderTextSplitter

  splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "topic"), ("h3", "question")])
  pieces = splitter.split_text((DATA / "faq.html").read_text())
  return [p for p in pieces if "question" in p.metadata and p.page_content != p.metadata["question"]]


def code_at_definitions():
  from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

  source = (DATA / "loyalty.py").read_text()
  plain = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=0).split_text(source)
  python = RecursiveCharacterTextSplitter.from_language(Language.PYTHON, chunk_size=350, chunk_overlap=0).split_text(
    source
  )  # separators: "\nclass ", "\ndef ", "\n\tdef ", then lines...
  return plain, python


@lab("rag.structure", "Structure-aware splitting: sections, web pages and code")
def structure_lab():
  """Split where the document itself says a new part begins: Markdown headings, HTML headings, code definitions."""
  sections = markdown_sections()
  small = then_by_size(sections)
  faq = html_sections()
  plain, python = code_at_definitions()
  starts = lambda chunks: [c.strip().splitlines()[0][:50] for c in chunks]  # noqa: E731
  at_def = lambda chunks: sum(c.strip().startswith(("def ", "class ", '"""', "from ", "import ")) for c in chunks)  # noqa: E731
  rows = [
    {"variant": "plain recursive", "chunks": len(plain), "start_at_definition": f"{at_def(plain)}/{len(plain)}"},
    {
      "variant": "from_language(PYTHON)",
      "chunks": len(python),
      "start_at_definition": f"{at_def(python)}/{len(python)}",
    },
  ]
  return [
    table(
      ["section", "text"],
      [[d.metadata.get("section", "—"), d.page_content[:90] + "…"] for d in sections],
      title="handbook.md, one Document per section",
    ),
    scorecard(
      rows,
      [
        Metric("chunks", "chunks", "none", "int"),
        Metric("start_at_definition", "chunks starting at a definition", "none", "text"),
      ],
      title="loyalty.py at 350 characters",
    ),
    columns(
      [
        code("\n".join(starts(plain)), "text", "plain: chunk starts"),
        code("\n".join(starts(python)), "text", "Python-aware: chunk starts"),
      ]
    ),
    callout(
      "Documents tell you where their parts begin: headings in Markdown and HTML, class and def lines in code. "
      "Split there first and size second, and record the headings as metadata: chunks become citable ('Leave "
      "policy') and filterable. On this corpus the documents are short, so section splitting barely changed "
      "retrieval (the evaluation shows it); it matters most for long documents with many sections.",
      "decision",
      "Structure first",
    ),
  ] + [
    example(
      "lc.l41.5",
      "MarkdownHeaderTextSplitter",
      markdown_sections,
      "Splits on headings and records them as metadata.",
      [table(["metadata", "text"], [[json.dumps(d.metadata), d.page_content[:70]] for d in sections[:4]])],
    ),
    example(
      "rag.l21.1",
      "One chunk per section, headings as metadata",
      markdown_sections,
      "Each section knows which document and section it belongs to.",
      [table(["section", "characters"], [[d.metadata.get("section"), len(d.page_content)] for d in sections])],
    ),
    example(
      "rag.l21.2",
      "Two stages: sections, then size",
      then_by_size,
      f"{len(sections)} sections → {len(small)} chunks; every chunk keeps its section.",
      [table(["metadata", "text"], [[json.dumps(d.metadata), d.page_content] for d in small[2:6]])],
    ),
    example(
      "rag.l21.3",
      "HTMLHeaderTextSplitter",
      html_sections,
      "The same for web pages: topic and question headings become metadata.",
      [
        table(
          ["topic / question", "answer"],
          [[f"{p.metadata['topic']} / {p.metadata['question']}", p.page_content] for p in faq],
        )
      ],
    ),
    example(
      "rag.l21.4",
      "Code: split at class and function boundaries",
      code_at_definitions,
      "The Python splitter prefers `class` and `def` lines as boundaries.",
      [
        table(
          ["plain", "Python-aware"],
          [[a, b] for a, b in zip(starts(plain) + [""] * 5, starts(python) + [""] * 5, strict=False) if a or b],
        )
      ],
    ),
  ]


# ── Lab 4: semantic chunking ────────────────────────────────────────────────────
TOPICS_TEXT = (
  "Monsoon Malabar Reserve is a low-acidity blend with notes of dark chocolate. It is roasted dark and suits espresso. "
  "Nilgiri Sunrise is a light roast with citrus and honey notes. It is best for pour-over coffee. "
  "The Bengaluru roastery roasts beans every Monday. Its capacity doubled to 40 tonnes a month in 2025. "
  "The roastery will run on 70% solar power by the end of 2026. "
  "Head office staff may work remotely on Tuesdays and Thursdays. Café staff cannot work remotely. "
  "Working from outside India needs approval from the Head of People."
)
TOPIC_OF = ["beans"] * 4 + ["roastery"] * 3 + ["remote work"] * 3  # the true topic of each sentence


def split_sentences(text):
  import re

  sentences = re.split(r"(?<=[.!?])\s+", text.strip())  # simple: good enough for clean prose
  return sentences


def neighbour_similarity(sentences, embeddings):
  import numpy as np

  vectors = np.array(embeddings.embed_documents(sentences))
  similarities = [float(vectors[i] @ vectors[i + 1]) for i in range(len(sentences) - 1)]
  return similarities  # how alike each sentence is to the next


def break_at_drops(sentences, similarities, percentile):
  import numpy as np

  threshold = np.percentile(similarities, percentile)  # the lowest similarities count as topic changes
  chunks, current = [], [sentences[0]]
  for sentence, similarity in zip(sentences[1:], similarities, strict=True):
    if similarity < threshold:
      chunks.append(" ".join(current))
      current = []
    current.append(sentence)
  chunks.append(" ".join(current))
  return chunks, threshold


def _mixed(chunks: list[str], sentences: list[str]) -> int:
  """Chunks that contain sentences from more than one topic."""
  topic = dict(zip(sentences, TOPIC_OF, strict=True))
  return sum(len({topic[s] for s in sentences if s in c}) > 1 for c in chunks)


@lab(
  "rag.semantic-chunking",
  "Semantic chunking: split where the topic changes",
  inputs=[Input("percentile", "Break below percentile", "number", default=25, min=5, max=95)],
)
def semantic_chunking(percentile: int):
  """Three topics with no headings: break where neighbouring sentences stop being alike, against fixed-size chunks."""
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  sentences = split_sentences(TOPICS_TEXT)
  similarities = neighbour_similarity(sentences, corpus.embeddings())
  semantic, threshold = break_at_drops(sentences, similarities, percentile)
  fixed = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0).split_text(TOPICS_TEXT)
  cut = lambda chunks: sum(not any(c.startswith(x) for x in sentences) for c in chunks)  # noqa: E731
  rows = [
    {
      "variant": f"semantic (below p{percentile})",
      "chunks": len(semantic),
      "mixed_topics": _mixed(semantic, sentences),
      "cut_sentences": cut(semantic),
      "embeddings": len(sentences),
    },
    {
      "variant": "fixed 250 characters",
      "chunks": len(fixed),
      "mixed_topics": _mixed(fixed, sentences),
      "cut_sentences": cut(fixed),
      "embeddings": 0,
    },
  ]
  return [
    bars(
      [
        {
          "label": f"s{i}→s{i + 1}"
          + (f"  ({TOPIC_OF[i]} → {TOPIC_OF[i + 1]})" if TOPIC_OF[i] != TOPIC_OF[i + 1] else ""),
          "values": {"similarity": round(s, 3)},
        }
        for i, s in enumerate(similarities)
      ],
      title=f"Similarity of each sentence to the next (break below {threshold:.3f})",
      max_value=1.0,
    ),
    scorecard(
      rows,
      [
        Metric("chunks", "chunks", "none", "int"),
        Metric("mixed_topics", "chunks mixing topics", "lower", "int"),
        Metric("cut_sentences", "chunks starting mid-sentence", "lower", "int"),
        Metric("embeddings", "extra embeddings to split", "lower", "int"),
      ],
      title="Three topics, no headings",
    ),
    columns([code("\n\n".join(semantic), "text", "semantic chunks"), code("\n\n".join(fixed), "text", "fixed chunks")]),
    callout(
      f"Semantic chunks mixed topics {rows[0]['mixed_topics']} time(s) and never cut a sentence; fixed chunks mixed "
      f"{rows[1]['mixed_topics']} and started {rows[1]['cut_sentences']} chunk(s) mid-sentence. The biggest similarity "
      "drop isn't always the topic change: 'The Bengaluru roastery roasts beans every Monday' mentions beans, so a "
      "small embedding model keeps it with the bean sentences. Semantic chunking costs one embedding per sentence at "
      "indexing time and a threshold to tune (move the percentile and watch). For documents with headings, splitting "
      "on the headings is cheaper and more reliable.",
      "decision",
      "Semantic chunking",
    ),
  ] + [
    example(
      "rag.l22.1",
      "Split into sentences",
      split_sentences,
      "A regular expression on sentence-ending punctuation.",
      [table(["#", "sentence", "topic"], [[i, s, TOPIC_OF[i]] for i, s in enumerate(sentences)])],
    ),
    example(
      "rag.l22.2",
      "Similarity of each sentence to the next",
      neighbour_similarity,
      "Embed every sentence; compare neighbours.",
      [table(["pair", "similarity"], [[f"s{i}→s{i + 1}", round(s, 3)] for i, s in enumerate(similarities)])],
    ),
    example(
      "rag.l22.3",
      "Break where similarity drops",
      break_at_drops,
      f"Threshold: the {percentile}th percentile of the similarities ({threshold:.3f}).",
      [table(["chunk"], [[c] for c in semantic])],
    ),
    example(
      "rag.l22.4",
      "Against fixed-size chunks",
      break_at_drops,
      "Fixed chunks mix topics inside one chunk.",
      [table(["way", "chunks mixing topics"], [[r["variant"], r["mixed_topics"]] for r in rows])],
    ),
  ]


# ── Lab 5: chunk enrichment ─────────────────────────────────────────────────────
ENRICH_QUESTION = "How much does a bag of Monsoon Malabar Reserve cost?"
ENRICH_ANSWER = "650 rupees"


def small_chunks_lose_their_subject(document):
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  pieces = RecursiveCharacterTextSplitter(
    chunk_size=110, chunk_overlap=0, separators=[". ", "\n"], keep_separator="end"
  ).split_text(document)
  answer_chunk = next(p for p in pieces if ENRICH_ANSWER in p)  # "A 250 g bag costs 650 rupees." — of WHAT?
  return pieces, answer_chunk


def contextual_headers(pieces):

  from nimbus.rag.ingestion.chunking import add_context_headers

  chunks = [Document(p, metadata={"source": "products.txt", "section": "Signature blends"}) for p in pieces]
  headed = add_context_headers(chunks)  # "[products.txt › Signature blends]\n..." before embedding
  return headed


def contextual_retrieval(pieces, document, model):

  from nimbus.rag.ingestion.chunking import add_llm_context

  chunks = [Document(p, metadata={"source": "products.txt"}) for p in pieces]
  enriched = add_llm_context(chunks, {"products.txt": document}, model)  # one sentence per chunk, cached
  return enriched


def rank_of_the_answer(texts):
  import numpy as np

  vectors = np.array(corpus.embeddings().embed_documents(texts))
  order = np.argsort(-(vectors @ np.array(corpus.embeddings().embed_query(ENRICH_QUESTION))))
  rank = next(r for r, i in enumerate(order, start=1) if ENRICH_ANSWER in texts[i])
  return rank  # 1 = the answer chunk came first


@lab(
  "rag.enrichment",
  "Chunk enrichment: give chunks back the context they lost",
  uses_model=True,
  calls="one call to Mistral per chunk the first time (~10), then cached",
)
def enrichment():
  """The product guide in small chunks: plain, with a contextual header, and with a model-written context sentence."""
  document = (DATA / "products.txt").read_text()
  pieces, answer_chunk = small_chunks_lose_their_subject(document)
  headed = contextual_headers(pieces)
  started = time.perf_counter()
  enriched = contextual_retrieval(pieces, document, get_chat_model())
  seconds = time.perf_counter() - started
  ranks = {
    "plain chunks": rank_of_the_answer(pieces),
    "contextual header": rank_of_the_answer([d.page_content for d in headed]),
    "model-written context": rank_of_the_answer([d.page_content for d in enriched]),
  }
  context = next(d.metadata["context"] for d in enriched if ENRICH_ANSWER in d.page_content)
  rows = [
    {"variant": name, "rank": rank, "index_cost": cost}
    for (name, rank), cost in zip(ranks.items(), ("free", "free", f"{len(pieces)} model calls, once"), strict=True)
  ]
  return [
    scorecard(
      rows,
      [
        Metric("rank", "rank of the answer chunk", "lower", "int"),
        Metric("index_cost", "indexing cost", "none", "text"),
      ],
      title=f"Q: {ENRICH_QUESTION} ({len(pieces)} chunks; context step {seconds:.1f} s)",
    ),
    table(
      ["Version", "The answer chunk as embedded"],
      [
        ["plain", answer_chunk],
        ["header", next(d.page_content for d in headed if ENRICH_ANSWER in d.page_content)],
        ["model context", next(d.page_content for d in enriched if ENRICH_ANSWER in d.page_content)],
      ],
    ),
    callout(
      f"'{answer_chunk}' never names the blend, so it ranked {ranks['plain chunks']}. A header naming the source "
      f"and section ranked it {ranks['contextual header']}: it says where the chunk is, not what 'it' is. The "
      f"model-written sentence ('{context}') names the product and ranked it {ranks['model-written context']}. "
      "Contextual retrieval costs one model call per chunk at indexing time, never per question, and the result "
      "is cached. The evaluation below measures both on the whole corpus.",
      "decision",
      "Enrichment",
    ),
  ] + [
    example(
      "rag.l23.1",
      "Small chunks lose their subject",
      small_chunks_lose_their_subject,
      "The chunk with the price never says which blend it's about.",
      [code(answer_chunk, "text"), table(["rank for the question"], [[ranks["plain chunks"]]])],
    ),
    example(
      "rag.l23.2",
      "Contextual headers: document and section in front",
      contextual_headers,
      "Free and instant; helps when the source and section name the topic.",
      [
        code(next(d.page_content for d in headed if ENRICH_ANSWER in d.page_content), "text"),
        table(["rank"], [[ranks["contextual header"]]]),
      ],
    ),
    example(
      "rag.l23.3",
      "Contextual retrieval: a model situates each chunk",
      contextual_retrieval,
      "The model sees the whole document and writes one sentence per chunk; the sentence is embedded with the chunk.",
      [table(["context sentence", "rank"], [[context, ranks["model-written context"]]])],
    ),
    example(
      "rag.l23.4",
      "Measure it: rank of the answer chunk",
      rank_of_the_answer,
      "The same question against each version of the chunks.",
      [table(["Version", "rank"], [[k, v] for k, v in ranks.items()])],
    ),
  ]


# ── Production path: chunking v2 ────────────────────────────────────────────────
CONFIGS = [  # (label, chunk_size, overlap, strategy, context)
  ("v1 · recursive 500/80", 500, 80, "recursive", "none"),
  ("recursive 300/50", 300, 50, "recursive", "none"),
  ("sections 300/50", 300, 50, "sections", "none"),
  ("recursive 500/80 + headers", 500, 80, "recursive", "headers"),
  ("recursive 300/50 + headers", 300, 50, "recursive", "headers"),
  ("recursive 300/50 + model context", 300, 50, "recursive", "llm"),
]


def evaluate_chunking(k: int = 5, include_llm: bool = True) -> list[dict]:
  rows = []
  for label, size, overlap, strategy, context in CONFIGS:
    if context == "llm" and not include_llm:
      continue
    chunks = corpus.chunks(size, overlap, "v3-tesseract", strategy, context)
    vectors = corpus.chunk_vectors(size, overlap, "v3-tesseract", strategy, context)

    def retrieve(query, n, chunks=chunks, vectors=vectors):
      q = np.array(corpus.embeddings().embed_query(query))
      return [chunks[i] for i in np.argsort(-(vectors @ q))[:n]]

    report = evaluate(retrieve, corpus.questions(), [c.page_content for c in chunks], k=k)
    rows.append(
      {
        "variant": label,
        "chunks": len(chunks),
        "avg_chars": round(sum(len(c.page_content) for c in chunks) / len(chunks)),
        "hit1": report.at["hit"][1],
        "hit": report.at["hit"][k],
        "mrr": round(report.mrr, 3),
        "ndcg": report.at["ndcg"][k],
        "context_tokens": round(sum(len(c.page_content) for c in chunks) / len(chunks) * k / 4),
      }
    )
  return rows


@lab(
  "rag.chunking-eval",
  "Evaluate: six chunking configurations on all 45 questions",
  uses_model=True,
  calls="~100 Mistral calls the first time (model context per chunk), then cached",
  inputs=[
    Input("k", "Top k", "number", default=5, min=1, max=10),
    Input("include_llm", "Include model-written context", "bool", default=True),
  ],
)
def chunking_eval(k: int, include_llm: bool):
  """Loaders v3 (Tesseract), the same embeddings and questions; only the chunking changes."""
  started = time.perf_counter()
  rows = evaluate_chunking(k, include_llm)
  seconds = time.perf_counter() - started
  free = [r for r in rows if "model context" not in r["variant"]]
  best = max(free, key=lambda r: (round(r["ndcg"], 2), r["hit"], -r["context_tokens"]))  # rank quality, then cost
  base = rows[0]
  return [
    scorecard(
      rows,
      [
        Metric("chunks", "chunks", "none", "int"),
        Metric("avg_chars", "avg characters", "none", "int"),
        Metric("hit1", "hit@1", "higher", "ratio"),
        Metric("hit", f"hit@{k}", "higher", "ratio"),
        Metric("mrr", "MRR", "higher", "ratio"),
        Metric("ndcg", f"nDCG@{k}", "higher", "ratio"),
        Metric("context_tokens", f"≈ tokens sent per question (top {k})", "lower", "int"),
      ],
      title=f"45 questions · {seconds:.1f} s",
    ),
    callout(
      f"Best free configuration (by nDCG@{k}, then hit@{k}, then tokens): {best['variant']}: nDCG {best['ndcg']:.2f}, "
      f"hit@{k} {best['hit']:.2f}, MRR {best['mrr']:.3f}, against v1's {base['ndcg']:.2f}, {base['hit']:.2f} and "
      f"{base['mrr']:.3f}. Naming each chunk's source and section helped every size for free; splitting by section "
      "first changed little on these short documents. "
      + (
        f"Model-written context scored nDCG {llm['ndcg']:.2f}: below the free headers on the whole corpus, at a model "
        "call per chunk, so production keeps it behind a setting (NIMBUS_CHUNK_CONTEXT=llm) for corpora where chunks "
        "lose their subject. "
        if (llm := next((r for r in rows if "model context" in r["variant"]), None))
        else ""
      )
      + "Production ships the winner as its defaults (chunk_size, chunk_overlap, chunk_context).",
      "decision",
      "Ship chunking v2",
    ),
  ]
