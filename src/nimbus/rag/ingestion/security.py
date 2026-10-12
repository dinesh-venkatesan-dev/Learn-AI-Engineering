"""Ingestion-time security: what must not reach the index, or must not reach it as written.

1. Indirect prompt injection: a document that talks to the model ("ignore previous instructions…"). Its chunks are
   quarantined: kept out of the index and listed in the ingestion report for a person to review.
2. Personal data: personal email addresses and phone numbers are redacted before embedding (role addresses like
   orders@ stay: customers are meant to use them).
Access control is not here: it's a filter at retrieval time (Retriever, `audience`), on the registry's `access`.
"""

import re

from langchain_core.documents import Document

INJECTION = re.compile(
  r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules)"
  r"|disregard (the |all )?(system|previous) (prompt|instructions)"
  r"|you are now (a|an|in) |new instructions:|system prompt:|<\s*/?\s*system\s*>"
  r"|(tell|inform) (the )?(customer|user)s? (that )?(every|all) .{0,40}(approved|free|refund)",
  re.IGNORECASE,
)
EMAIL = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[a-z]{2,})\b")
PHONE = re.compile(r"(?<![\w=-])(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?![\w-])")
ROLE_ADDRESSES = {
  "orders",
  "support",
  "help",
  "it-help",
  "privacy",
  "security",
  "procurement",
  "operations",
  "careers",
  "press",
}


def looks_like_injection(text: str) -> str | None:
  """The matched phrase if the text addresses the model with instructions, else None."""
  match = INJECTION.search(text)
  return match.group(0) if match else None


def redact(text: str) -> tuple[str, int]:
  """Personal emails and phone numbers replaced; role addresses kept. Returns (text, how many were redacted)."""
  count = 0

  def email(m: re.Match) -> str:
    nonlocal count
    if m.group(1).lower() in ROLE_ADDRESSES:
      return m.group(0)
    count += 1
    return "[email]"

  def phone(m: re.Match) -> str:
    nonlocal count
    count += 1
    return "[phone]"

  text = EMAIL.sub(email, text)
  text = PHONE.sub(phone, text)
  return text, count


def secure(chunks: list[Document]) -> tuple[list[Document], list[dict], int]:
  """(chunks to index, quarantined chunks with the reason, number of redactions)."""
  kept, quarantined, redactions = [], [], 0
  for c in chunks:
    phrase = looks_like_injection(c.page_content)
    if phrase:
      quarantined.append(
        {
          "source": c.metadata.get("source"),
          "reason": f"instructions to the model: {phrase!r}",
          "text": c.page_content[:200],
        }
      )
      continue
    text, n = redact(c.page_content)
    meta = {k: (redact(v)[0] if isinstance(v, str) else v) for k, v in c.metadata.items()}  # headers too (From: …)
    redactions += n
    kept.append(Document(text, metadata=meta))
  return kept, quarantined, redactions
