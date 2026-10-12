from nimbus.assistant.evaluation import correct
from nimbus.assistant.skills import load_skills, skill_tools, skills_prompt


def test_skills_load_with_descriptions_only_in_the_prompt():
  skills = load_skills()
  assert set(skills) == {"catering_orders", "complaint_handling", "dietary_questions"}
  prompt = skills_prompt(skills)
  assert "catering_orders" in prompt and "catering_quote" not in prompt  # the full text stays out until loaded


def test_skill_files_stay_inside_the_skill_and_quotes_follow_the_price_list():
  tools = {t.name: t for t in skill_tools(load_skills())}
  assert "Cold Brew" in tools["read_skill_file"].invoke({"name": "dietary_questions", "file": "allergens.md"})
  outside = tools["read_skill_file"].invoke({"name": "dietary_questions", "file": "../catering_orders/SKILL.md"})
  assert "No file" in outside
  assert "₹3600" in tools["catering_quote"].invoke({"people": 30, "date": "Friday"})
  assert "₹5000" in tools["catering_quote"].invoke({"people": 50, "date": "Friday"})
  assert "20 people" in tools["catering_quote"].invoke({"people": 5, "date": "Friday"})


def test_scoring_rejects_negated_answers_and_specialists_on_small_talk():
  vegan = r"cold brew[^.]{0,40}\b(is|are)\b(?![^.]{0,25}\b(not|n't)\b)[^.]{0,25}vegan"
  m1 = {"kind": "multi", "answer_all": ["shipped"], "answer_regex": vegan}
  assert correct(m1, "It shipped. The cold brew is vegan.", ["account", "cafe"])
  assert not correct(m1, "It shipped. The cold brew is not listed, so I can't confirm it's vegan.", ["account"])
  small = {"kind": "small talk"}
  assert correct(small, "You're welcome!", []) and not correct(small, "Here are your orders…", ["account"])
