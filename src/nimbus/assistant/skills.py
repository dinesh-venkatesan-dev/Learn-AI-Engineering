"""Skills: specialised instructions an agent loads only when a question needs them (progressive disclosure).

data/skills/<name>/SKILL.md   front matter (name, description: always in the prompt) + the full instructions
data/skills/<name>/*.md       extra files the instructions point to, read only if needed

  load_skill(name)            the full instructions; recorded in state, which can unlock tools
  read_skill_file(name, file) one extra file of a loaded skill
  unlock_tools                middleware: catering_quote is offered only after catering_orders is loaded
"""

import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from langchain.agents import AgentState
from langchain.agents.middleware import wrap_model_call
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool
from langgraph.types import Command

from nimbus.config import PROJECT_DIR

SKILLS_DIR = PROJECT_DIR / "data" / "skills"
UNLOCKS = {"catering_quote": "catering_orders"}  # tool → the skill that must be loaded first
CATERING_PER_PERSON = {20: 120, 50: 100}  # rupees per person from 20 and from 50 people


@dataclass(frozen=True)
class Skill:
  name: str
  description: str
  instructions: str
  folder: Path


def load_skills(folder: Path = SKILLS_DIR) -> dict[str, Skill]:
  skills = {}
  for path in sorted(folder.glob("*/SKILL.md")):
    text = path.read_text()
    meta, body = re.match(r"---\n(.*?)\n---\n(.*)", text, re.DOTALL).groups()
    fields = dict(line.split(": ", 1) for line in meta.splitlines() if ": " in line)
    skills[fields["name"]] = Skill(fields["name"], fields["description"], body.strip(), path.parent)
  return skills


def skills_prompt(skills: dict[str, Skill]) -> str:
  """Only the one-line descriptions: the full text costs tokens only when a question needs it."""
  lines = "\n".join(f"- {s.name}: {s.description}" for s in skills.values())
  return f"Skills you can load with load_skill before answering questions they cover:\n{lines}"


class SkillState(AgentState):
  skills_loaded: Annotated[list[str], operator.add]


def skill_tools(skills: dict[str, Skill]) -> list:
  @tool
  def load_skill(name: str, runtime: ToolRuntime) -> Command:
    """Load a skill's full instructions. Load the matching skill BEFORE answering a question it covers."""
    skill = skills.get(name)
    if skill is None:
      raise ToolException(f"No skill {name!r}. Skills: {', '.join(skills)}.")
    return Command(
      update={
        "skills_loaded": [name],  # recorded in state: it can unlock tools
        "messages": [ToolMessage(skill.instructions, tool_call_id=runtime.tool_call_id)],
      }
    )

  @tool
  def read_skill_file(name: str, file: str) -> str:
    """Read an extra file that a loaded skill's instructions point to, e.g. ('dietary_questions', 'allergens.md')."""
    skill = skills.get(name)
    path = (skill.folder / file).resolve() if skill else None
    if path is None or path.parent != skill.folder.resolve() or not path.exists():  # only files inside the skill
      raise ToolException(f"No file {file!r} in skill {name!r}.")
    return path.read_text()

  @tool
  def catering_quote(people: int, date: str) -> str:
    """Price coffee catering for an office or event: number of people and the date."""
    if people < 20:
      raise ToolException("Catering starts at 20 people.")
    per_person = CATERING_PER_PERSON[50 if people >= 50 else 20]
    return f"Catering for {people} people on {date}: ₹{per_person} per person, ₹{people * per_person} in total."

  for t in (load_skill, read_skill_file, catering_quote):
    t.handle_tool_error = True
  return [load_skill, read_skill_file, catering_quote]


@wrap_model_call
def unlock_tools(request, handler):
  """A locked tool is hidden from the model until its skill is loaded."""
  loaded = set(request.state.get("skills_loaded", []))
  tools = [t for t in request.tools if UNLOCKS.get(getattr(t, "name", "")) in (None, *loaded)]
  return handler(request.override(tools=tools))
