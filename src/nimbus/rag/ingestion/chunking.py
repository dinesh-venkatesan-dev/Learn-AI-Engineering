"""Step 3 — Chunk: split documents into pieces small enough to match a question precisely.

Search compares a question with chunks, not whole documents. A small chunk is about one thing, so it matches
sharply; but it must be big enough to hold a complete answer. Overlap repeats a little text between neighbours,
so a sentence cut at a boundary still appears whole in one of them.
"""

import hashlib

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from nimbus.config import PROJECT_DIR

# Try the biggest natural boundary first: Markdown headings, then paragraphs, lines, sentences, words, characters.
SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
  return RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=SEPARATORS,
    keep_separator="start",  # a heading stays at the top of the chunk it introduces
    add_start_index=True,  # metadata["start_index"]: where the chunk starts in its document
    strip_whitespace=True,
  )


def split_documents(
  documents: list[Document], splitter: RecursiveCharacterTextSplitter, min_chars: int
) -> list[Document]:
  """Chunks keep their document's metadata (source, page, file type) plus their position in it."""
  chunks = splitter.split_documents(documents)
  kept = [c for c in chunks if len(c.page_content) >= min_chars]  # a lone heading or page number carries no facts
  for position, c in enumerate(kept):
    c.metadata["chunk_index"] = position
  return kept


# ── Chunking v2 (RAG Module 4): split by section first, and give chunks back the context they lost ─────────────
HEADINGS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
CONTEXT_CACHE = PROJECT_DIR / ".cache" / "chunk_context"


def split_by_sections(
  documents: list[Document], splitter: RecursiveCharacterTextSplitter, min_chars: int
) -> list[Document]:
  """Two stages: at Markdown headings first (the heading path becomes metadata["section"]), then by size inside
  sections that are still too long. A chunk never spans two sections."""
  header_splitter = MarkdownHeaderTextSplitter(HEADINGS, strip_headers=False)
  sections = []
  for doc in documents:
    parts = header_splitter.split_text(doc.page_content) if "#" in doc.page_content else [Document(doc.page_content)]
    for part in parts:
      path = " › ".join(part.metadata[k] for k in ("h1", "h2", "h3") if k in part.metadata)
      sections.append(Document(part.page_content, metadata={**doc.metadata, **({"section": path} if path else {})}))
  return split_documents(sections, splitter, min_chars)


def add_context_headers(chunks: list[Document]) -> list[Document]:
  """Prepend where each chunk comes from ("[handbook.md › Leave policy]"): embedded with the chunk, free to compute."""
  out = []
  for c in chunks:
    where = c.metadata["source"] + (f" › {c.metadata['section']}" if c.metadata.get("section") else "")
    out.append(Document(f"[{where}]\n{c.page_content}", metadata=dict(c.metadata)))
  return out


CONTEXT_PROMPT = (
  "<document>\n{document}\n</document>\n\nHere is a chunk from that document:\n<chunk>\n{chunk}\n</chunk>\n\n"
  "Write ONE short sentence that situates this chunk in the document (what it is about, naming the product, policy "
  "or entity it refers to), to improve search. Reply with the sentence only."
)


def add_llm_context(chunks: list[Document], documents: dict[str, str], model: BaseChatModel) -> list[Document]:
  """Contextual retrieval: a model writes one sentence situating each chunk in its document, prepended before
  embedding. One call per chunk at indexing time, cached by (document, chunk) hash, never per question."""
  CONTEXT_CACHE.mkdir(parents=True, exist_ok=True)
  out = []
  for c in chunks:
    document = documents[c.metadata["source"]][:12_000]  # long documents: the start carries most of the context
    path = CONTEXT_CACHE / (hashlib.sha256(f"{document}\n\n{c.page_content}".encode()).hexdigest()[:24] + ".txt")
    if path.exists():
      context = path.read_text()
    else:
      context = model.invoke(CONTEXT_PROMPT.format(document=document, chunk=c.page_content)).text.strip()
      path.write_text(context)
    out.append(Document(f"{context}\n{c.page_content}", metadata={**c.metadata, "context": context}))
  return out


def make_chunks(
  cleaned_by_file: dict[str, list[Document]],
  chunk_size: int,
  chunk_overlap: int,
  min_chars: int,
  strategy: str = "recursive",
  context: str = "none",
  model: BaseChatModel | None = None,
) -> list[Document]:
  """Every file's chunks with the configured strategy (recursive | sections) and context (none | headers | llm)."""
  splitter = build_splitter(chunk_size, chunk_overlap)
  split = split_by_sections if strategy == "sections" else split_documents
  chunks = [c for docs in cleaned_by_file.values() for c in split(docs, splitter, min_chars)]
  if context == "headers":
    return add_context_headers(chunks)
  if context == "llm":
    if model is None:
      raise ValueError("context='llm' needs a chat model")
    documents = {source: "\n\n".join(d.page_content for d in docs) for source, docs in cleaned_by_file.items()}
    return add_llm_context(chunks, documents, model)
  return chunks
