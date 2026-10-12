from nimbus.rag.quality import EvalItem, _norm, copies_the_text, load_synthetic, save_questions


def test_a_question_that_copies_the_chunk_is_caught():
  chunk = "Points expire 12 months after they are earned."
  assert copies_the_text("Do points expire 12 months after they are earned?", chunk)
  assert not copies_the_text("How long do my loyalty points last?", chunk)


def test_evidence_comparison_ignores_formatting_and_curly_quotes():
  assert _norm("Every employee gets **24 days** of leave.") == _norm("every employee gets 24 days of leave")
  assert _norm("We don’t sell data") == _norm("We don't sell data")
  assert _norm("| On delivery | 60% |") == _norm("On delivery 60%")


def test_the_set_round_trips(tmp_path):
  items = [
    EvalItem("How long do points last?", "12 months", "Points expire 12 months after they are earned.", "faq.html")
  ]
  save_questions(items, tmp_path / "s.json")
  assert load_synthetic(tmp_path / "s.json") == items
