"""Safe releases of the support reply prompt: a canary share of callers, shadow answers nobody sees, and rollback
as a configuration change.

  variant(caller_id, percent)   "canary" or "stable", from a HASH of the caller id: stable per caller, so nobody flips
                                between versions from one request to the next
  prompt_ref(settings, variant) the registry ref each variant runs
"""

import hashlib

from nimbus.config import Settings


def variant(caller_id: str, percent: int) -> str:
  bucket = int(hashlib.sha256(caller_id.encode()).hexdigest()[:8], 16) % 100  # 0-99, the same every time
  return "canary" if bucket < percent else "stable"


def prompt_ref(settings: Settings, which: str) -> str:
  return settings.canary_prompt if which == "canary" else settings.support_prompt
