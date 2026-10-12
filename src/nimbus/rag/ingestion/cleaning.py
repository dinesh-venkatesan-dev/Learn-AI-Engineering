"""Step 2 — Clean: keep the knowledge, drop the noise.

Noise in the index becomes noise in search results: a navigation menu repeated on 1,000 pages makes 1,000 chunks
look alike. Every function here is pure (text in, text out), so each rule is easy to test.
"""

import re
from collections import Counter

from bs4 import BeautifulSoup
from langchain_core.documents import Document

BOILERPLATE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]
PAGE_NUMBER = re.compile(r"^\s*(page\s+\d+(\s+of\s+\d+)?|\d+\s*/\s*\d+)\s*$", re.IGNORECASE)


def html_to_text(html: str) -> str:
  """The readable text of a page: its <main> (or <body>) without menus, scripts and footers."""
  soup = BeautifulSoup(html, "html.parser")
  for tag in soup(BOILERPLATE_TAGS):
    tag.decompose()
  root = soup.find("main") or soup.body or soup
  return root.get_text("\n")


def remove_repeated_lines(pages: list[str], min_share: float = 0.6) -> list[str]:
  """Drop lines that appear on most pages of the same file: running headers and footers, not content."""
  if len(pages) < 2:
    return pages
  seen_on = Counter(line.strip() for page in pages for line in set(page.splitlines()) if line.strip())
  repeated = {line for line, count in seen_on.items() if count / len(pages) >= min_share}
  return [
    "\n".join(
      line
      for line in page.splitlines()
      if line.strip() not in repeated and not PAGE_NUMBER.match(line) and not _is_page_footer(line)
    )
    for page in pages
  ]


def _is_page_footer(line: str) -> bool:
  return bool(re.search(r"\bpage \d+ of \d+\s*$", line, re.IGNORECASE))


def normalize(text: str) -> str:
  """Formatting that carries no meaning: bold markers, runs of spaces, runs of blank lines."""
  text = text.replace("**", "").replace(" ", " ")
  text = re.sub(r"[ \t]+", " ", text)
  text = re.sub(r" *\n *", "\n", text)
  return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_file(documents: list[Document]) -> list[Document]:
  """Clean all Documents of ONE file together (headers can only be spotted across a file's pages).

  Documents that end up empty (a blank page) are dropped.
  """
  if not documents:
    return []
  texts = [d.page_content for d in documents]
  if documents[0].metadata["file_type"] == "html":
    texts = [html_to_text(t) for t in texts]
  if documents[0].metadata["file_type"] == "pdf":
    texts = remove_repeated_lines(texts)
  cleaned = [Document(normalize(t), metadata=dict(d.metadata)) for d, t in zip(documents, texts, strict=True)]
  return [d for d in cleaned if d.page_content]
