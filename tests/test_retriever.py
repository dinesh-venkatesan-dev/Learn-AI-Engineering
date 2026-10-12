from langchain_core.language_models.fake_chat_models import FakeListChatModel

from nimbus.rag import retriever
from nimbus.rag.retriever import needs_translation, prepare


def test_translation_policy():
  assert needs_translation("मुझे साल में कितनी छुट्टी मिलती है?", "non_ascii")
  assert needs_translation("Wie lange dauert die Lieferung in eine Großstadt?", "non_ascii")
  assert not needs_translation("Hotel ka kitna reimburse milta hai?", "non_ascii")  # Hinglish is ASCII
  assert not needs_translation("How many days of leave?", "non_ascii")
  assert needs_translation("How many days of leave?", "always")
  assert not needs_translation("मुझे छुट्टी", "off")


def test_prepare_translates_once_and_caches(tmp_path, monkeypatch):
  monkeypatch.setattr(retriever, "CACHE", tmp_path)
  model = FakeListChatModel(responses=["How many days of leave do I get in a year?"])
  first = prepare("मुझे साल में कितनी छुट्टी मिलती है?", "non_ascii", model)
  assert first.translated and first.query == "How many days of leave do I get in a year?"
  again = prepare("मुझे साल में कितनी छुट्टी मिलती है?", "non_ascii", FakeListChatModel(responses=["unused"]))
  assert again.query == first.query  # served from the cache, not the second model


def test_english_questions_are_searched_as_they_are():
  prepared = prepare("How many days of leave?", "non_ascii", FakeListChatModel(responses=["should not be called"]))
  assert not prepared.translated and prepared.query == "How many days of leave?"
