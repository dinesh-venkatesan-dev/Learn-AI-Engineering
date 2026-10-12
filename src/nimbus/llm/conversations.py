"""Conversations: a model remembers nothing between calls, so the app stores the messages and resends them.

Stored as JSON files, one per conversation id (production would use a database table; the format is the same).
"""

import json
import re
from pathlib import Path

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ConversationStore:
  def __init__(self, folder: Path):
    self.folder = folder

  def _path(self, conversation_id: str) -> Path:
    if not SAFE_ID.match(conversation_id):  # the id becomes a file name: never let it climb out of the folder
      raise ValueError("conversation id: 1-64 letters, digits, '-' or '_'")
    return self.folder / f"{conversation_id}.json"

  def load(self, conversation_id: str) -> list[BaseMessage]:
    path = self._path(conversation_id)
    return messages_from_dict(json.loads(path.read_text())) if path.exists() else []

  def save(self, conversation_id: str, messages: list[BaseMessage]) -> None:
    path = self._path(conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(messages_to_dict(messages), ensure_ascii=False, indent=1))
    tmp.replace(path)  # atomic: a crash mid-write never leaves half a conversation

  def append(self, conversation_id: str, *messages: BaseMessage) -> list[BaseMessage]:
    history = self.load(conversation_id) + list(messages)
    self.save(conversation_id, history)
    return history
