"""RAG · Module 2 labs: Documents and metadata, then every format a company stores: text, CSV, JSON, HTML, PDF (text
and tables), Word, Excel, PowerPoint, email, and your own sources. Each lab compares a naive way with a careful one
and measures what the difference does to search."""

import json
import os
import re
import time
import tracemalloc
from functools import lru_cache

import numpy as np
from langchain_core.documents import Document

from academy.labs.rag import corpus
from academy.labs.runtime import (
  Input,
  Metric,
  answer,
  bars,
  callout,
  code,
  columns,
  example,
  lab,
  scorecard,
  table,
)
from nimbus.config import get_settings
from nimbus.llm import get_chat_model
from nimbus.rag.evaluation import evaluate, is_relevant
from nimbus.rag.ingestion.chunking import build_splitter, split_documents

DATA = get_settings().data_dir
SUPPORT_ORDERS = DATA.parent / "support" / "orders.json"
os.environ.setdefault("USER_AGENT", "nimbus-academy")  # WebBaseLoader identifies itself to the sites it fetches


def _ms(started: float) -> float:
  return (time.perf_counter() - started) * 1000


def _rank(query: str, docs: list[Document], target) -> int | None:
  """1-based rank of the first doc for which target(doc) is true, by cosine similarity (normalised vectors)."""
  vectors = np.array(corpus.embeddings().embed_documents([d.page_content for d in docs]))
  q = np.array(corpus.embeddings().embed_query(query))
  order = np.argsort(-(vectors @ q))
  return next((r for r, i in enumerate(order, start=1) if target(docs[i])), None)


def _short(text: str, n: int = 120) -> str:
  text = " ".join(text.split())
  return text if len(text) <= n else text[: n - 1] + "…"


# ── Lab 1: Documents and metadata ───────────────────────────────────────────────
# Each learning point is a small function: the "What it covers" tab shows its body, and the lab runs it.


def make_a_document():
  from langchain_core.documents import Document

  doc = Document(
    page_content="Hotels are reimbursed up to 6,000 rupees per night.",  # the text that gets embedded and searched
    metadata={"source": "handbook.md", "section": "Travel and expenses"},  # facts ABOUT the text: not embedded
    id="handbook-travel-1",  # identity: what an index uses to update or delete it
  )
  return doc


def design_metadata(text):
  from langchain_core.documents import Document

  handbook = Document(
    text,
    metadata={
      "source": "handbook.md",  # citations: "according to handbook.md"
      "doc_type": "policy",  # filtering: search only policies
      "department": "people",  # routing and ownership
      "updated_at": "2026-01-10",  # freshness: prefer or require recent documents
      "access": "all_staff",  # permissions: who may see it
    },
  )
  return handbook


def metadata_travels_with_chunks(handbook):
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  chunks = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0).split_documents([handbook])
  same = all(
    {k: c.metadata[k] for k in handbook.metadata} == handbook.metadata for c in chunks
  )  # every chunk carries its parent's metadata
  return chunks, same


def stable_ids(chunks):
  import hashlib

  def chunk_id(chunk):
    # Same source + same text → the same id, every time. Production hashes content AND metadata the same way.
    return hashlib.sha256(f"{chunk.metadata['source']}::{chunk.page_content}".encode()).hexdigest()[:16]

  ids = [chunk_id(c) for c in chunks]
  return ids, chunk_id


def _index_runs(chunks: list[Document], chunk_id) -> list[dict]:
  """Index the same file three times (first run, unchanged re-run, after one edit) with random and stable ids."""
  import uuid

  edited = [*chunks]
  edited[1] = Document(edited[1].page_content + " (Updated in 2026.)", metadata=dict(edited[1].metadata))
  runs = [("first run", chunks), ("re-run, nothing changed", chunks), ("re-run after editing one chunk", edited)]
  rows = []
  for label, ids_of in (
    ("random ids (uuid4)", lambda batch: [str(uuid.uuid4()) for _ in batch]),
    ("stable ids (hash of source + text)", lambda batch: [chunk_id(c) for c in batch]),
  ):
    store: dict[str, str] = {}
    counts = []
    for _, batch in runs:
      for key, c in zip(ids_of(batch), batch, strict=True):
        store[key] = c.page_content  # an upsert: an existing id is replaced, a new id is added
      counts.append(len(store))
    distinct = len({c.page_content for _, b in runs for c in b})
    rows.append(
      {
        "variant": label,
        "after_run_1": counts[0],
        "after_run_2": counts[1],
        "after_edit": counts[2],
        "duplicates": counts[2] - len(chunks),
        "note": f"{distinct - len(chunks)} chunk changed",
      }
    )
  return rows


@lab(
  "rag.documents",
  "Documents and metadata: text, facts about it, and a stable id",
  inputs=[Input("chunk_size", "Chunk size (characters)", "number", default=250, min=100, max=1000)],
)
def documents_lab(chunk_size: int):
  """The handbook as a Document with designed metadata, split into chunks, indexed three times with two id schemes."""
  doc = make_a_document()
  handbook = design_metadata((DATA / "handbook.md").read_text())
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  chunks = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0).split_documents([handbook])
  ids, chunk_id = stable_ids(chunks)
  rows = _index_runs(chunks, chunk_id)
  small_chunks, same = metadata_travels_with_chunks(handbook)
  small_ids, _ = stable_ids(small_chunks)
  return [
    scorecard(
      rows,
      [
        Metric("after_run_1", "rows after run 1", "none", "int"),
        Metric("after_run_2", "after an unchanged re-run", "lower", "int"),
        Metric("after_edit", "after editing one chunk", "lower", "int"),
        Metric("duplicates", "stale or duplicate rows", "lower", "int", "Rows beyond the file's current chunks"),
      ],
      title=f"handbook.md in {len(chunks)} chunks, indexed three times",
    ),
    table(
      ["#", "id (stable)", "page_content", "metadata"],
      [[i, ids[i], _short(c.page_content, 80), json.dumps(c.metadata)] for i, c in enumerate(chunks[:6])],
      title="The first chunks: every one carries the document's metadata",
    ),
    callout(
      f"With random ids every re-run adds the whole file again ({rows[0]['after_run_2']} rows after an unchanged "
      f"re-run). With stable ids the re-run changes nothing, and an edit adds one new row: the old version still "
      "needs deleting, which is what production's record manager does (it remembers every id per source and "
      "removes the ones that disappeared). Put in metadata everything you'll filter, cite or check permissions on: "
      "it can't be added to millions of chunks later without re-indexing.",
      "decision",
      "Ids and metadata",
    ),
  ] + [
    example(
      "lc.l40.1",
      "A Document: page_content + metadata",
      make_a_document,
      "The unit every loader, splitter, vector store and retriever passes around. Only `page_content` is "
      "embedded; `metadata` rides along for filtering, citations and management.",
      [
        table(
          ["Attribute", "Value"],
          [["page_content", doc.page_content], ["metadata", json.dumps(doc.metadata)], ["id", doc.id]],
        )
      ],
    ),
    example(
      "rag.l05.1",
      "Three parts: text, metadata, id",
      make_a_document,
      "`page_content` is what search compares with the question, `metadata` is what you filter and cite by, and "
      "`id` is how an index finds this exact chunk again to update or delete it.",
      [code(json.dumps({"page_content": doc.page_content, "metadata": doc.metadata, "id": doc.id}, indent=2), "json")],
    ),
    example(
      "rag.l05.2",
      "Designing metadata for a knowledge base",
      design_metadata,
      "Each key exists for a job: citing the source, filtering by type, routing to an owner, preferring fresh "
      "documents, and enforcing who may read it.",
      [table(["Key", "Value"], [[k, v] for k, v in handbook.metadata.items()])],
    ),
    example(
      "rag.l05.3",
      "Metadata is copied to every chunk",
      metadata_travels_with_chunks,
      f"Splitting made {len(small_chunks)} chunks; every one has the parent's metadata: `{same}`. That's why a "
      "chunk found deep inside a file can still be cited and filtered.",
      [
        table(
          ["Chunk", "Text", "metadata"],
          [[i, _short(c.page_content, 70), json.dumps(c.metadata)] for i, c in enumerate(small_chunks[3:6], start=3)],
        )
      ],
    ),
    example(
      "rag.l05.4",
      "Stable ids: re-indexing replaces instead of duplicating",
      stable_ids,
      "A hash of the source and the text: the same chunk gets the same id on every run, so an upsert replaces it. "
      "Edit the text and the id changes, so the old row can be found and deleted.",
      [
        table(
          ["Chunk", "id", "recomputed", "same?"],
          [[i, small_ids[i], chunk_id(c), small_ids[i] == chunk_id(c)] for i, c in enumerate(small_chunks[:3])],
        )
      ],
    ),
  ]


# ── Lab 2: text, CSV and JSON ───────────────────────────────────────────────────
ORDER_QUESTIONS = {
  "What is the status of order NC-78123?": "NC-78123",
  "What did Ravi order?": "NC-78123",
  "Which order was cancelled?": "NC-13001",
  "What did Lakshmi order and how much did it cost?": "NC-66310",
  "Which order is being packed right now?": "NC-12500",
  "When did Farah place her order?": "NC-90112",
}


def text_loader():
  from langchain_community.document_loaders import TextLoader

  docs = TextLoader("data/nimbus/handbook.md", encoding="utf-8").load()  # one file → one Document
  return docs


def csv_loader():
  from langchain_community.document_loaders import CSVLoader

  rows = CSVLoader(
    "data/nimbus/menu.csv",
    content_columns=["item", "price_rupees", "category", "vegan"],  # what gets embedded
    metadata_columns=["category", "vegan"],  # copied into metadata: usable as filters
  ).load()  # one row → one Document
  return rows


def directory_loader():
  from langchain_community.document_loaders import DirectoryLoader, TextLoader

  loader = DirectoryLoader(
    "data/nimbus", glob="*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
  )  # every matching file
  docs = loader.load()
  return docs


def load_lazily():
  from langchain_community.document_loaders import DirectoryLoader, TextLoader

  loader = DirectoryLoader("data/nimbus", glob="*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
  seen = []
  for doc in loader.lazy_load():  # a generator: one Document at a time, nothing held in a list
    seen.append(doc.metadata["source"])
    if len(seen) == 2:
      break  # stop early: the remaining files are never read
  return seen


def json_records_three_ways(orders):
  import json

  from langchain_core.documents import Document

  from nimbus.rag.ingestion.loaders import record_text

  def raw(o):  # the record as JSON text: braces, quotes and keys
    return json.dumps(o, ensure_ascii=False)

  def labelled(o):  # "key: value" lines, nested items flattened (production's JsonLoader)
    return record_text(o)

  def sentence(o):  # what a person would write
    items = ", ".join(f"{i['qty']} × {i['name']}" for i in o["items"])
    return (
      f"Order {o['order_id']} was placed by {o['customer']} on {o['ordered_on']} for {items}. "
      f"Total {o['total']} rupees. Status: {o['status']}."
    )

  ways = {
    name: [Document(fn(o), metadata={"source": "orders.json", "order_id": o["order_id"]}) for o in orders]
    for name, fn in (("raw JSON", raw), ("key: value lines", labelled), ("written sentence", sentence))
  }
  return ways


@lab(
  "rag.text-csv-json",
  "Text, CSV and JSON: one Document per file, per row, per record",
  inputs=[
    Input(
      "question", "Question about orders", "select", default=next(iter(ORDER_QUESTIONS)), options=list(ORDER_QUESTIONS)
    )
  ],
)
def text_csv_json(question: str):
  """JSON orders phrased three ways, searched among every other chunk of the corpus: which phrasing gets found?"""
  fields = ("order_id", "customer", "items", "total", "status", "ordered_on", "delivered_on")
  orders = [{k: o[k] for k in fields} for o in json.loads(SUPPORT_ORDERS.read_text())["orders"]]
  ways = json_records_three_ways(orders)
  others = list(corpus.chunks())
  rows, mine = [], {}
  for name, docs in ways.items():
    ranks = []
    for q, order_id in ORDER_QUESTIONS.items():
      rank = _rank(q, [*others, *docs], lambda d, o=order_id: d.metadata.get("order_id") == o)
      ranks.append(rank)
      if q == question:
        mine[name] = rank
    rows.append(
      {
        "variant": name,
        "mrr": round(sum(1 / r for r in ranks if r) / len(ranks), 3),
        "top1": round(sum(r == 1 for r in ranks) / len(ranks), 3),
        "top3": round(sum(bool(r) and r <= 3 for r in ranks) / len(ranks), 3),
        "chars": round(sum(len(d.page_content) for d in docs) / len(docs)),
      }
    )
  text_docs = text_loader()
  rows_csv = csv_loader()
  directory = directory_loader()
  lazy = load_lazily()
  target = ORDER_QUESTIONS[question]
  return [
    scorecard(
      rows,
      [
        Metric("mrr", "MRR", "higher", "ratio", "1 / rank of the right order, averaged over the questions"),
        Metric("top1", "right order first", "higher", "ratio"),
        Metric("top3", "in the top 3", "higher", "ratio"),
        Metric("chars", "characters per record", "lower", "int"),
      ],
      title=f"{len(orders)} orders among {len(others)} corpus chunks: how each phrasing ranks for "
      f"{len(ORDER_QUESTIONS)} questions",
    ),
    table(
      ["Phrasing", "Rank of the right order", "The record as embedded"],
      [
        [
          name,
          mine[name] or "—",
          _short(next(d for d in ways[name] if d.metadata["order_id"] == target).page_content, 160),
        ]
        for name in ways
      ],
      title=question,
    ),
    callout(
      _phrasing_verdict(rows),
      "decision",
      "Phrase structured data as text",
    ),
  ] + [
    example(
      "lc.l40.2",
      "TextLoader: one file → one Document",
      text_loader,
      "The simplest loader: the file's text becomes `page_content`, its path goes in `metadata['source']`.",
      [
        table(
          ["Documents", "Characters", "metadata"],
          [[len(text_docs), len(text_docs[0].page_content), json.dumps(text_docs[0].metadata)]],
        )
      ],
    ),
    example(
      "rag.l06.1",
      "Markdown is kept as Markdown",
      text_loader,
      "TextLoader doesn't render Markdown: the `#` headings stay, and the splitter uses them as boundaries later.",
      [code(text_docs[0].page_content[:300], "markdown", "page_content (start)")],
    ),
    example(
      "lc.l40.3",
      "CSVLoader: one row → one Document",
      csv_loader,
      "Each row becomes `column: value` lines. `metadata_columns` copies chosen columns into metadata too.",
      [
        table(
          ["Row", "page_content", "metadata"],
          [
            [
              d.metadata["row"],
              d.page_content.replace("\n", " · "),
              json.dumps({k: v for k, v in d.metadata.items() if k != "source"}),
            ]
            for d in rows_csv[:4]
          ],
        )
      ],
    ),
    example(
      "rag.l06.2",
      "Choosing which columns become metadata",
      csv_loader,
      "`category` and `vegan` are both in the text (so a question can match them) and in metadata (so a filter "
      "like `vegan == 'yes'` can select rows exactly).",
      [
        table(
          ["item", "metadata"],
          [
            [d.page_content.splitlines()[0], json.dumps({k: d.metadata[k] for k in ("category", "vegan")})]
            for d in rows_csv
          ],
        )
      ],
    ),
    example(
      "lc.l40.4",
      "DirectoryLoader: every matching file in a folder",
      directory_loader,
      "A glob picks the files and `loader_cls` loads each. Production uses its own folder loader (lab 6) because "
      "each file type needs a different loader.",
      [table(["source", "characters"], [[d.metadata["source"], len(d.page_content)] for d in directory])],
    ),
    example(
      "lc.l40.5",
      "lazy_load(): a generator",
      load_lazily,
      "`lazy_load()` yields Documents one at a time; stop early and the rest are never read. `load()` is just "
      "`list(lazy_load())`.",
      [table(["Read before stopping"], [[s] for s in lazy])],
    ),
    example(
      "rag.l06.3",
      "JSON records → readable text + metadata",
      json_records_three_ways,
      "There's no single right JSON loader: you decide how a record reads. The exact fields go in metadata for "
      "filtering and display.",
      [
        table(
          ["Phrasing", "page_content", "metadata"],
          [[n, _short(d[2].page_content, 150), json.dumps(d[2].metadata)] for n, d in ways.items()],
        )
      ],
    ),
    example(
      "rag.l06.4",
      "The phrasing decides what search finds",
      json_records_three_ways,
      "The scorecard above: the same four orders, three phrasings, ranked among every other chunk.",
      [table(["Phrasing", "MRR", "first", "top 3"], [[r["variant"], r["mrr"], r["top1"], r["top3"]] for r in rows])],
    ),
  ]


def _phrasing_verdict(rows: list[dict]) -> str:
  ranked = sorted(rows, key=lambda r: -r["mrr"])
  scores = ", ".join(f"{r['variant']} {r['mrr']:.2f}" for r in rows)
  if ranked[0]["mrr"] - ranked[-1]["mrr"] < 0.05:
    lead = f"MRR: {scores}. On these records the phrasing barely mattered: the order ids and names are distinctive."
  else:
    lead = f"MRR: {scores}. {ranked[0]['variant'].capitalize()} ranked best and {ranked[-1]['variant']} worst."
  return (
    lead + " Raw JSON spends characters on braces, quotes and key names that mean nothing to an embedding model. "
    "Production writes each record as labelled lines, which works for any JSON without a hand-written template; a "
    "sentence template is worth it for your few most-asked record types."
  )


# ── Lab 3: HTML ─────────────────────────────────────────────────────────────────
BOILERPLATE = ["Home | Shop", "newsletter", "Privacy policy", "Cookie settings", "window.analytics"]


def load_html_as_is():
  from langchain_community.document_loaders import BSHTMLLoader

  doc = BSHTMLLoader("data/nimbus/faq.html").load()[0]  # all visible text, the <title> in metadata
  return doc


def keep_the_main_content(html):
  from bs4 import BeautifulSoup

  soup = BeautifulSoup(html, "html.parser")
  for junk in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
    junk.decompose()  # menus, banners and footers repeat on every page: noise in every chunk
  main = soup.find("main") or soup.body
  text = main.get_text("\n", strip=True)
  return text


def headings_as_markdown(html):
  from bs4 import BeautifulSoup

  soup = BeautifulSoup(html, "html.parser")
  for junk in soup(["script", "style", "nav", "header", "footer"]):
    junk.decompose()
  lines = []
  for element in (soup.find("main") or soup.body).find_all(["h1", "h2", "h3", "p", "li"]):
    prefix = {"h1": "# ", "h2": "## ", "h3": "### "}.get(element.name, "")  # structure the splitter can use
    lines.append(prefix + element.get_text(" ", strip=True))
  markdown = "\n\n".join(lines)
  return markdown


def load_a_web_page(folder):
  import threading
  from functools import partial
  from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

  from langchain_community.document_loaders import WebBaseLoader

  class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
      pass

  server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=folder))  # a local site: no internet
  thread = threading.Thread(target=server.serve_forever)
  thread.start()
  try:
    url = f"http://127.0.0.1:{server.server_port}/faq.html"
    page = WebBaseLoader(url).load()[0]  # the same call works for any public URL
  finally:
    server.shutdown()
    server.server_close()
    thread.join()
  return url, page


@lab("rag.html", "HTML and web pages: keep the content, drop the clutter")
def html_lab():
  """The help-centre page three ways: everything visible, the main content, and the main content with headings."""
  html = (DATA / "faq.html").read_text()
  raw = load_html_as_is()
  main = keep_the_main_content(html)
  markdown = headings_as_markdown(html)
  splitter = build_splitter(220, 0)
  variants = {
    "BSHTMLLoader (everything)": raw.page_content,
    "main content only": main,
    "main + Markdown headings": markdown,
  }
  rows, chunk_views = [], {}
  for name, text in variants.items():
    chunks = split_documents([Document(text, metadata={"source": "faq.html"})], splitter, 20)
    noise = sum(phrase.casefold() in text.casefold() for phrase in BOILERPLATE)
    starts = sum(c.page_content.lstrip().startswith("#") for c in chunks)
    rows.append(
      {
        "variant": name,
        "chars": len(text),
        "boilerplate": noise,
        "chunks": len(chunks),
        "chunks_with_heading": round(starts / max(len(chunks), 1), 3),
      }
    )
    chunk_views[name] = chunks
  url, page = load_a_web_page(str(DATA))
  return [
    scorecard(
      rows,
      [
        Metric("chars", "characters", "lower", "int"),
        Metric("boilerplate", "boilerplate phrases", "lower", "int", "Menu, newsletter banner, footer links, scripts"),
        Metric("chunks", "chunks (220 chars)", "none", "int"),
        Metric("chunks_with_heading", "chunks that start at a heading", "higher", "ratio"),
      ],
      title="faq.html three ways",
    ),
    columns(
      [
        code("\n\n— chunk —\n\n".join(c.page_content for c in chunks[:3]), "markdown", name)
        for name, chunks in chunk_views.items()
      ],
      title="The first three chunks of each",
    ),
    callout(
      "Everything visible includes the menu, the newsletter banner and the footer: repeated on every page of a "
      "real site, they make thousands of chunks look alike. Production keeps the main content (cleaning step). "
      "Turning headings into Markdown also lets the splitter cut at section boundaries, so each chunk starts with "
      "the question it answers.",
      "decision",
      "HTML",
    ),
  ] + [
    example(
      "rag.l07.1",
      "BSHTMLLoader: all visible text",
      load_html_as_is,
      "The page's `<title>` goes in metadata, and every visible string becomes text, including the clutter.",
      [
        table(["metadata"], [[json.dumps(raw.metadata)]]),
        code(" ".join(raw.page_content.split())[:600], "text", "page_content"),
      ],
    ),
    example(
      "rag.l07.2",
      "Keep only the main content",
      keep_the_main_content,
      "Remove the tags that hold site furniture, then take `<main>` (or `<body>`).",
      [code(main, "text")],
    ),
    example(
      "rag.l07.3",
      "Keep the structure: headings as Markdown",
      headings_as_markdown,
      "`## Orders and delivery` and `### How long does delivery take?` survive as Markdown, so chunks can start "
      "at a heading and carry it.",
      [code(markdown, "markdown")],
    ),
    example(
      "rag.l07.4",
      "WebBaseLoader: fetch a page over HTTP",
      load_a_web_page,
      "Here a local web server serves the data folder, so no internet is needed; the call is the same for any URL. "
      "For whole sites, RecursiveUrlLoader follows links and SitemapLoader reads sitemap.xml.",
      [table(["url", "metadata", "characters"], [[url, json.dumps(page.metadata), len(page.page_content)]])],
    ),
  ]


# ── Lab 4: PDF text and tables ──────────────────────────────────────────────────
PDF = DATA / "annual_report_2025.pdf"
TABLE_QUESTION = "What was the North region's revenue in 2024? Answer with the number only."


def pdf_pages():
  from langchain_community.document_loaders import PyMuPDFLoader

  pages = PyMuPDFLoader("data/nimbus/annual_report_2025.pdf").load()  # one Document per page
  return pages


def pdf_as_one_document():
  from langchain_community.document_loaders import PyMuPDFLoader

  whole = PyMuPDFLoader("data/nimbus/annual_report_2025.pdf", mode="single").load()  # the whole file as one
  return whole


def remove_repeated_lines(pages):
  from collections import Counter

  def shape(line):  # "Page 2 of 3" and "Page 3 of 3" have the same shape
    return "".join(ch for ch in line if not ch.isdigit()).strip()

  counts = Counter(shape(line) for page in pages for line in set(page.page_content.splitlines()))
  repeated = {line for line, n in counts.items() if n == len(pages) and line}  # on EVERY page: header or footer
  cleaned = ["\n".join(x for x in p.page_content.splitlines() if shape(x) not in repeated) for p in pages]
  return repeated, cleaned


def tables_with_pdfplumber():
  import pdfplumber

  with pdfplumber.open("data/nimbus/annual_report_2025.pdf") as pdf:
    rows = pdf.pages[1].extract_tables()[0]  # found from the table's ruling lines, read as rows
  return rows


def table_as_its_own_document(rows):
  from langchain_core.documents import Document

  header, *body = rows
  markdown = "\n".join(
    ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in body]
  )
  table_doc = Document(
    "Revenue by region (crore rupees), Annual Report 2025:\n\n" + markdown,
    metadata={"source": "annual_report_2025.pdf", "page": 2, "element": "table"},  # filter or format tables apart
  )
  return table_doc


def tables_with_pymupdf():
  import pymupdf

  with pymupdf.open("data/nimbus/annual_report_2025.pdf") as pdf:
    found = pdf[1].find_tables()  # the same idea, built into PyMuPDF
    markdown = found.tables[0].to_markdown()
  return markdown


def ask_about_a_table_chunk(model, flat_chunk, markdown_chunk, question):
  answers = {}
  for label, context in (("flattened chunk", flat_chunk), ("Markdown chunk", markdown_chunk)):
    answers[label] = model.invoke(f"Context:\n{context}\n\nQuestion: {question}").text.strip()
  return answers


@lab(
  "rag.pdf",
  "PDF text and tables: pages, repeated headers, and tables as rows",
  uses_model=True,
  calls="2 calls to Mistral",
)
def pdf_lab():
  """The annual report three ways (plain text per page, plumber tables, PyMuPDF4LLM Markdown), then the same table
  chunk flattened and as Markdown, given to the model."""
  import pymupdf4llm

  pages = pdf_pages()
  whole = pdf_as_one_document()
  repeated, cleaned = remove_repeated_lines(pages)
  rows = tables_with_pdfplumber()
  table_doc = table_as_its_own_document(rows)
  mupdf_table = tables_with_pymupdf()
  llm_pages = pymupdf4llm.to_markdown(str(PDF), page_chunks=True, use_ocr=False)
  variants = {
    "PyMuPDFLoader (plain text)": pages[1].page_content,
    "pdfplumber table → Markdown": table_doc.page_content,
    "PyMuPDF4LLM (production)": llm_pages[1]["text"],
  }
  score_rows = []
  for name, text in variants.items():
    score_rows.append(
      {
        "variant": name,
        "table_rows_kept": "✓" if re.search(r"\|\s*North\s*\|\s*30\s*\|\s*41", text) else "✗",
        "header_footer": sum(
          line.strip() in text for line in ("Confidential - Page 2 of 3", "Nimbus Coffee Co. - Annual Report 2025")
        ),
        "chars": len(text),
      }
    )
  header, *body = rows
  flat_chunk = " ".join(" ".join(r) for r in body[1:])  # the end of the table: its header row is in another chunk
  markdown_chunk = "\n".join(
    ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in body[1:]]
  )
  answers = ask_about_a_table_chunk(get_chat_model(), flat_chunk, markdown_chunk, TABLE_QUESTION)
  verdict = {label: ("30" in a and "41" not in a) for label, a in answers.items()}
  return [
    scorecard(
      score_rows,
      [
        Metric("table_rows_kept", "table kept as rows", "none", "text"),
        Metric("header_footer", "running header / footer lines", "lower", "int", "Before cleaning"),
        Metric("chars", "characters (page 2)", "none", "int"),
      ],
      title="Page 2 of the annual report, three extractions",
    ),
    columns([code(text, "markdown", name) for name, text in variants.items()], title="What each extraction gives"),
    columns(
      [
        answer(a, f"{label} → {'✓ right (30)' if verdict[label] else '✗ wrong'}", meta={"context": _short(ctx, 160)})
        for (label, a), ctx in zip(answers.items(), (flat_chunk, markdown_chunk), strict=True)
      ],
      title=f"The last rows of the table as a chunk. Q: {TABLE_QUESTION}",
    ),
    callout(
      "Plain text extraction turns the table into a list of loose numbers. Once a long table is split, the chunk "
      "with the last rows no longer says which column is 2024. A Markdown table repeats its header row, so "
      "every row stays readable. Production uses PyMuPDF4LLM, which writes tables as Markdown and keeps one "
      "Document per page, and the cleaning step removes lines that repeat on every page.",
      "decision",
      "PDFs",
    ),
  ] + [
    example(
      "rag.l08.1",
      "PyMuPDFLoader: one Document per page",
      pdf_pages,
      "Each page is a Document with its page number (0-based) and the file's metadata. Page numbers are what you cite.",
      [
        table(
          ["page", "total_pages", "title", "characters"],
          [
            [p.metadata["page"], p.metadata["total_pages"], p.metadata.get("title"), len(p.page_content)] for p in pages
          ],
        )
      ],
    ),
    example(
      "rag.l08.2",
      'mode="single": the whole PDF as one Document',
      pdf_as_one_document,
      "Useful when you chunk across page breaks yourself; you lose the page number for citations.",
      [table(["Documents", "characters"], [[len(whole), len(whole[0].page_content)]])],
    ),
    example(
      "rag.l08.3",
      "The mess inside real PDFs",
      pdf_pages,
      "Every page starts with the same running header and a 'Page n of 3' line, in storage order, before the "
      "content. Repeated in every chunk, they make pages look alike.",
      [code("\n".join(pages[2].page_content.splitlines()[:5]), "text", "page 3, first lines")],
    ),
    example(
      "rag.l08.4",
      "Remove lines that repeat on every page",
      remove_repeated_lines,
      "Lines found on every page (comparing their shape, so page numbers don't matter) are headers and footers.",
      [
        table(["Repeated on every page"], [[r] for r in sorted(repeated)]),
        code("\n".join(cleaned[2].splitlines()[:3]), "text", "page 3 now starts with"),
      ],
    ),
    example(
      "rag.l09.1",
      "Plain text flattens a table",
      pdf_pages,
      "Cells come out one per line: which number belongs to which column is lost.",
      [code(" ".join(pages[1].page_content.split())[:320], "text")],
    ),
    example(
      "rag.l09.2",
      "pdfplumber: find the table, read its rows",
      tables_with_pdfplumber,
      "pdfplumber detects the table from its ruling lines and returns a list of rows.",
      [table(rows[0], rows[1:])],
    ),
    example(
      "rag.l09.3",
      "The table as its own Markdown Document",
      table_as_its_own_document,
      "A caption, the Markdown table, and `element: table` in metadata, so tables can be filtered or shown apart.",
      [code(table_doc.page_content, "markdown"), table(["metadata"], [[json.dumps(table_doc.metadata)]])],
    ),
    example(
      "rag.l09.4",
      "After chunking: a flattened table loses its header",
      ask_about_a_table_chunk,
      "The same rows, flattened and as Markdown, given to the model with the same question.",
      [
        table(
          ["Context", "Answer", "Right?"], [[label, a, "✓" if verdict[label] else "✗"] for label, a in answers.items()]
        )
      ],
    ),
    example(
      "rag.l09.5",
      "PyMuPDF's find_tables()",
      tables_with_pymupdf,
      "The same table detection, built into PyMuPDF (and used by PyMuPDF4LLM, production's loader).",
      [code(mupdf_table, "markdown")],
    ),
  ]


# ── Lab 5: Word, Excel, PowerPoint and email ────────────────────────────────────
OFFICE_SOURCES = (
  "supplier_contract.docx",
  "q1_2026_review.pptx",
  "cafe_sales_q1_2026.xlsx",
  "green_valley_price_notice.eml",
)


def docx_flat():
  from langchain_community.document_loaders import Docx2txtLoader

  flat = Docx2txtLoader("data/nimbus/supplier_contract.docx").load()[0]  # quick plain text: structure is lost
  return flat


def docx_sections():
  from pathlib import Path

  from nimbus.rag.ingestion.office import DocxLoader

  sections = DocxLoader(Path("data/nimbus/supplier_contract.docx"), "supplier_contract.docx").load()
  # Inside: walks paragraphs AND tables in order, starts a section at each heading, tables become Markdown
  return sections


def xlsx_rows():
  from pathlib import Path

  from nimbus.rag.ingestion.office import XlsxLoader

  rows = XlsxLoader(Path("data/nimbus/cafe_sales_q1_2026.xlsx"), "cafe_sales_q1_2026.xlsx").load()
  return rows  # one Document per row; the header row is found below any title rows


def xlsx_sheet_as_table():
  from openpyxl import load_workbook

  from nimbus.rag.ingestion.office import markdown_table

  book = load_workbook("data/nimbus/cafe_sales_q1_2026.xlsx", read_only=True, data_only=True)
  try:
    sheets = {}
    for sheet in book.worksheets:
      values = [[str(v) for v in r] for r in sheet.iter_rows(values_only=True) if all(v is not None for v in r)]
      sheets[sheet.title] = markdown_table(values)  # the whole sheet: better for "which café sold most?"
  finally:
    book.close()  # read-only workbooks keep the file open until closed
  return sheets


def pptx_slides():
  from pathlib import Path

  from nimbus.rag.ingestion.office import PptxLoader

  slides = PptxLoader(Path("data/nimbus/q1_2026_review.pptx"), "q1_2026_review.pptx").load()
  return slides  # title, text, tables AND speaker notes per slide


def eml_message():
  from pathlib import Path

  from nimbus.rag.ingestion.office import EmlLoader

  parts = EmlLoader(Path("data/nimbus/green_valley_price_notice.eml"), "green_valley_price_notice.eml").load()
  return parts  # the new text (quoted replies cut) + each attachment, headers in metadata


def _naive_office() -> list[Document]:
  """What a quick loader gives: flat text, slides without notes or tables, cells without headers, the raw email."""
  from openpyxl import load_workbook
  from pptx import Presentation

  docs = [Document(docx_flat().page_content, metadata={"source": "supplier_contract.docx"})]
  for n, slide in enumerate(Presentation(str(DATA / "q1_2026_review.pptx")).slides, start=1):
    text = "\n".join(s.text_frame.text for s in slide.shapes if s.has_text_frame)
    docs.append(Document(text, metadata={"source": "q1_2026_review.pptx", "slide": n}))
  book = load_workbook(DATA / "cafe_sales_q1_2026.xlsx", read_only=True, data_only=True)
  try:
    for sheet in book.worksheets:
      for row in sheet.iter_rows(values_only=True):
        if any(v is not None for v in row):
          docs.append(
            Document(" ".join(str(v) for v in row if v is not None), metadata={"source": "cafe_sales_q1_2026.xlsx"})
          )
  finally:
    book.close()
  docs.append(
    Document((DATA / "green_valley_price_notice.eml").read_text(), metadata={"source": "green_valley_price_notice.eml"})
  )
  return docs


@lab("rag.office", "Word, Excel, PowerPoint and email: keep the structure")
def office_lab():
  """The four office files loaded naively and with structure, chunked the same way, and searched with the 8
  evaluation questions they answer."""
  from nimbus.rag.ingestion.cleaning import clean_file
  from nimbus.rag.ingestion.loaders import loader_for

  structured = [d for name in OFFICE_SOURCES for d in clean_file(loader_for(DATA / name, DATA).load())]
  naive = _naive_office()
  questions = [q for q in corpus.questions() if q.source in OFFICE_SOURCES]
  splitter = build_splitter(500, 80)
  rows, details = [], []
  for name, docs in (("quick loaders (flat text)", naive), ("structure-aware (production)", structured)):
    chunks = split_documents(docs, splitter, 20)
    vectors = np.array(corpus.embeddings().embed_documents([c.page_content for c in chunks]))
    found, answerable = [], []
    for q in questions:
      order = np.argsort(-(vectors @ np.array(corpus.embeddings().embed_query(q.question))))
      rank = next((r for r, i in enumerate(order, start=1) if is_relevant(chunks[i].page_content, q)), None)
      found.append(rank)
      answerable.append(any(is_relevant(c.page_content, q) for c in chunks))
    rows.append(
      {
        "variant": name,
        "answerable": round(sum(answerable) / len(questions), 3),
        "hit3": round(sum(bool(r) and r <= 3 for r in found) / len(questions), 3),
        "mrr": round(sum(1 / r for r in found if r) / len(questions), 3),
        "chunks": len(chunks),
      }
    )
    details.append(found)
  sections = docx_sections()
  flat = docx_flat()
  rows_xlsx = xlsx_rows()
  sheets = xlsx_sheet_as_table()
  slides = pptx_slides()
  email = eml_message()
  return [
    scorecard(
      rows,
      [
        Metric("answerable", "answerable", "higher", "ratio", "Some chunk contains the whole answer"),
        Metric("hit3", "hit@3", "higher", "ratio"),
        Metric("mrr", "MRR", "higher", "ratio"),
        Metric("chunks", "chunks", "none", "int"),
      ],
      title=f"{len(questions)} evaluation questions answered by the docx, pptx, xlsx and eml files",
    ),
    table(
      ["Question", "Source", "Evidence", "Rank: quick", "Rank: structured"],
      [
        [q.question, q.source, " + ".join(q.evidence), details[0][i] or "✗", details[1][i] or "✗"]
        for i, q in enumerate(questions)
      ],
      title="Question by question (rank of the first chunk that contains the whole answer)",
    ),
    callout(
      f"Quick loaders made {rows[0]['answerable']:.0%} of the questions answerable; keeping structure made "
      f"{rows[1]['answerable']:.0%}. The losses are structural: a contract table read as loose cells, slide "
      "speaker notes never extracted, spreadsheet values without their column names, an email with the whole "
      "quoted thread. Each format needs a loader that knows its structure.",
      "decision",
      "Office formats",
    ),
  ] + [
    example(
      "rag.l10.1",
      "Docx2txtLoader: quick, flat text",
      docx_flat,
      "Fast and dependency-light, but headings are ordinary lines and each table cell lands on its own line.",
      [code(flat.page_content.strip()[:700], "text")],
    ),
    example(
      "rag.l10.2",
      "python-docx: paragraphs with styles, tables cell by cell",
      docx_sections,
      "Walking the body in order sees each paragraph's style (`Heading 2`) and each table, so structure survives.",
      [
        table(
          ["section", "characters", "has a table"],
          [[d.metadata["section"], len(d.page_content), "| " in d.page_content] for d in sections],
        )
      ],
    ),
    example(
      "rag.l10.3",
      "One Document per section, heading in metadata",
      docx_sections,
      "Each section starts with its heading in the text and carries it in `metadata['section']` for citations.",
      [code(next(d.page_content for d in sections if "| " in d.page_content), "markdown", "the section with a table")],
    ),
    example(
      "rag.l11.1",
      "Excel: every sheet, the real header row, one Document per row",
      xlsx_rows,
      "`data_only=True` reads formula results. The header is the first fully filled row, so a title row above "
      "the table is skipped. Every value is labelled with its column.",
      [
        table(
          ["sheet", "row", "page_content"],
          [[d.metadata["sheet"], d.metadata["row"], d.page_content] for d in rows_xlsx[:6]],
        )
      ],
    ),
    example(
      "rag.l11.2",
      "A whole sheet as one Markdown table",
      xlsx_sheet_as_table,
      "Rows answer lookups (*who manages Park Street?*); a whole-sheet table answers comparisons (*which café sold "
      "most?*). Choose per sheet, or index both with an `element` field to tell them apart.",
      [code(next(iter(sheets.values())), "markdown", next(iter(sheets)))],
    ),
    example(
      "rag.l11.3",
      "PowerPoint: one Document per slide, speaker notes included",
      pptx_slides,
      "Speaker notes often hold what the presenter said, the details that aren't on the slide.",
      [
        table(
          ["slide", "title", "page_content"],
          [[d.metadata["slide"], d.metadata["title"], _short(d.page_content, 220)] for d in slides],
        )
      ],
    ),
    example(
      "rag.l11.4",
      "Email: headers as metadata, quoted replies cut, attachments kept",
      eml_message,
      "The quoted earlier thread is removed (it's indexed from its own email), and each attachment becomes a "
      "Document with the same headers, linked to the message.",
      [
        table(
          ["part", "metadata", "page_content"],
          [
            [
              d.metadata.get("attachment", "body"),
              json.dumps({k: d.metadata[k] for k in ("from", "subject", "date")}),
              _short(d.page_content, 200),
            ]
            for d in email
          ],
        )
      ],
    ),
    example(
      "rag.l11.5",
      "All of it in one index",
      _naive_office,
      "Both versions searched with the same questions: answers that live in speaker notes (*Could the Dubai opening "
      "be delayed?*) and in an email (*By how much will green bean prices rise?*) are only found with structure.",
      [
        table(
          ["Question", "Rank: quick", "Rank: structured"],
          [[q.question, details[0][i] or "✗", details[1][i] or "✗"] for i, q in enumerate(questions)],
        )
      ],
    ),
  ]


# ── Lab 6: custom loaders ───────────────────────────────────────────────────────
def a_custom_loader():
  from collections.abc import Iterator

  from langchain_core.document_loaders import BaseLoader
  from langchain_core.documents import Document

  class OrdersApiLoader(BaseLoader):
    """Orders from an API, a database or a queue: the same pattern for any source."""

    def __init__(self, count: int):
      self.count = count

    def lazy_load(self) -> Iterator[Document]:  # implement this one method...
      for n in range(1, self.count + 1):
        record = {"order_id": f"NC-{n:06d}", "total": 200 + n % 900, "notes": "x" * 400}  # one page of an API
        yield Document(
          f"Order {record['order_id']}, total {record['total']} rupees. {record['notes']}",
          metadata={"source": "orders-api", "order_id": record["order_id"]},
        )

  docs = OrdersApiLoader(3).load()  # ...and load(), aload() and alazy_load() come for free
  return OrdersApiLoader, docs


def memory_of_load_vs_lazy_load(loader_class, count):
  import tracemalloc

  tracemalloc.start()
  everything = loader_class(count).load()  # every Document in memory at once
  load_peak = tracemalloc.get_traced_memory()[1]
  del everything
  tracemalloc.reset_peak()
  total = 0
  for doc in loader_class(count).lazy_load():  # one at a time: process it (embed, write) and let it go
    total += len(doc.page_content)
  lazy_peak = tracemalloc.get_traced_memory()[1]
  tracemalloc.stop()
  return load_peak, lazy_peak


def one_loader_for_a_folder(folder):
  from pathlib import Path

  from nimbus.rag.ingestion.loaders import discover, loader_for

  files, unsupported = discover(Path(folder))  # every file, in a stable order, hidden files skipped
  per_file = {}
  for path in files:
    docs = loader_for(path, Path(folder)).load()  # the right loader for each file type
    per_file[path.name] = (len(docs), docs[0].metadata.get("file_type") if docs else None)
  return per_file, [p.name for p in unsupported]


@lab(
  "rag.custom-loaders",
  "Custom loaders: your own sources, memory, and one loader for a folder",
  inputs=[Input("count", "Records in the source", "number", default=50_000, min=1_000, max=200_000)],
)
def custom_loaders(count: int):
  """A custom loader for an orders API; load() vs lazy_load() peak memory; production's folder loader."""
  loader_class, docs = a_custom_loader()
  tracemalloc_was_on = tracemalloc.is_tracing()
  started = time.perf_counter()
  load_peak, lazy_peak = memory_of_load_vs_lazy_load(loader_class, count)
  seconds = time.perf_counter() - started
  if tracemalloc_was_on:
    tracemalloc.start()
  per_file, unsupported = one_loader_for_a_folder(str(DATA))
  rows = [
    {"variant": "load(): a list of every Document", "peak_mb": round(load_peak / 1e6, 1)},
    {"variant": "lazy_load(): one at a time", "peak_mb": round(lazy_peak / 1e6, 2)},
  ]
  return [
    scorecard(
      rows,
      [Metric("peak_mb", "peak memory (MB)", "lower", "text", "Measured with tracemalloc")],
      title=f"{count:,} records (~0.5 KB each) from a custom loader · {seconds:.1f} s",
    ),
    bars(
      [{"label": r["variant"], "values": {"MB": r["peak_mb"]}} for r in rows],
      title="Peak memory",
      unit="MB",
    ),
    table(
      ["File", "Documents", "file_type"],
      [[name, n, t] for name, (n, t) in per_file.items()],
      title=f"Production's folder loader over data/nimbus: {len(per_file)} files loaded, "
      f"{len(unsupported)} unsupported ({', '.join(unsupported)})",
    ),
    callout(
      f"load() held all {count:,} Documents at once ({rows[0]['peak_mb']} MB at the peak); lazy_load() kept memory "
      f"flat ({rows[1]['peak_mb']} MB) whatever the size. For a large source, stream: load a batch, embed and "
      "write it, move on. Production's folder loader picks a loader per file type and reports the files it can't "
      "read (images, until OCR in Module 3) instead of skipping them silently.",
      "decision",
      "Custom and streaming loaders",
    ),
  ] + [
    example(
      "rag.l12.1",
      "BaseLoader: implement lazy_load(), get the rest for free",
      a_custom_loader,
      "Subclass `BaseLoader`, yield Documents from `lazy_load()`, and the standard `load()`, `aload()` and "
      "`alazy_load()` work, so it plugs into everything that takes a loader.",
      [table(["page_content", "metadata"], [[_short(d.page_content, 60), json.dumps(d.metadata)] for d in docs])],
    ),
    example(
      "rag.l12.2",
      "lazy_load keeps memory flat",
      memory_of_load_vs_lazy_load,
      f"{count:,} records: `load()` peaked at {rows[0]['peak_mb']} MB, `lazy_load()` at {rows[1]['peak_mb']} MB.",
      [table(["Way", "Peak memory (MB)"], [[r["variant"], r["peak_mb"]] for r in rows])],
    ),
    example(
      "rag.l12.3",
      "One loader for a mixed folder",
      one_loader_for_a_folder,
      "Production's `discover` + `loader_for`: the right loader per extension, uniform metadata (`source`, `page`, "
      "`file_type`), and a list of what it couldn't load.",
      [
        table(["File", "Documents", "file_type"], [[name, n, t] for name, (n, t) in per_file.items()]),
        table(["Unsupported"], [[u] for u in unsupported]),
      ],
    ),
  ]


# ── Production path: loaders v1 vs v2 ───────────────────────────────────────────
@lru_cache
def _retriever(version: str):
  chunks = corpus.chunks(500, 80, version)
  vectors = corpus.chunk_vectors(500, 80, version)

  def retrieve(query: str, k: int):
    q = np.array(corpus.embeddings().embed_query(query))
    return [chunks[i] for i in np.argsort(-(vectors @ q))[:k]]

  return retrieve


def evaluate_loaders(k: int = 5) -> dict:
  reports = {}
  for version in ("v1", "v2"):
    chunks = corpus.chunks(500, 80, version)
    reports[version] = evaluate(
      _retriever(version),
      corpus.questions(),
      [c.page_content for c in chunks],
      k=k,
      indexed_sources=corpus.indexed_sources(500, 80, version),
    )
  return reports


@lab(
  "rag.loaders-eval",
  "Evaluate: loaders v1 vs v2 on all 45 questions",
  inputs=[Input("k", "Top k", "number", default=5, min=1, max=10)],
)
def loaders_eval(k: int):
  """Both loader versions, same cleaning, chunking (500/80) and embeddings, scored on every evaluation question."""
  started = time.perf_counter()
  reports = evaluate_loaders(k)
  seconds = time.perf_counter() - started
  rows = []
  for version, report in reports.items():
    s = report.summary()
    rows.append(
      {
        "variant": f"loaders {version}",
        "files": len(corpus.indexed_sources(500, 80, version)),
        "chunks": len(corpus.chunks(500, 80, version)),
        "coverage": s["coverage"],
        "answerable": s["answerable"],
        "hit": s[f"hit@{k}"] if f"hit@{k}" in s else report.at["hit"][k],
        "mrr": s["mrr"],
        "hit_reachable": s["hit_reachable"],
      }
    )
  v1, v2 = reports["v1"], reports["v2"]
  gained = [r2 for r1, r2 in zip(v1.results, v2.results, strict=True) if not r1.reachable and r2.reachable]
  still = [r.question for r in v2.results if not r.reachable]
  return [
    scorecard(
      rows,
      [
        Metric("files", "files indexed", "higher", "int"),
        Metric("chunks", "chunks", "none", "int"),
        Metric("coverage", "questions whose file is indexed", "higher", "ratio"),
        Metric("answerable", "answerable", "higher", "ratio", "Some chunk contains the whole answer"),
        Metric("hit", f"hit@{k}", "higher", "ratio", "Over all 45 questions"),
        Metric("mrr", "MRR", "higher", "ratio"),
        Metric("hit_reachable", f"hit@{k} (reachable only)", "higher", "ratio", "The retriever's own quality"),
      ],
      title=f"45 questions, k={k} · {seconds:.1f} s",
    ),
    table(
      ["Question", "Source", f"Found in top {k} with v2?"],
      [[r.question.question, r.question.source, "✓" if any(r.relevant) else "✗"] for r in gained],
      title=f"{len(gained)} questions v1 couldn't reach at all",
    ),
    table(["Still unreachable", "Source"], [[q.question, q.source] for q in still], title="Left for Module 3 (OCR)"),
    callout(
      f"v2 indexes {rows[1]['files'] - rows[0]['files']} more files and makes {rows[1]['answerable']:.0%} of the "
      f"questions answerable (v1: {rows[0]['answerable']:.0%}); hit@{k} over all questions goes "
      f"{rows[0]['hit']:.2f} → {rows[1]['hit']:.2f}. "
      + (
        f"The retriever's own hit rate on reachable questions fell ({rows[0]['hit_reachable']:.2f} → "
        f"{rows[1]['hit_reachable']:.2f}): {rows[1]['chunks'] - rows[0]['chunks']} new chunks compete for the top "
        f"{k}, and the new questions (slides, spreadsheet rows) are harder. Chunking (Module 4) and hybrid search "
        "with reranking (Module 6) work on that. "
        if rows[1]["hit_reachable"] < rows[0]["hit_reachable"]
        else ""
      )
      + "What's left unreachable is the scanned PDF and the images, which have no text layer: Module 3 adds OCR. "
      "Production now uses v2 (NIMBUS_LOADER_VERSION=v2).",
      "decision",
      "Ship loaders v2",
    ),
  ]
