"""Extract every learning point ("What you'll learn" items) from the old CLI courses into content/syllabus.json, plus
the points that are new in this course (content/additions.json).

Run once:  uv run python scripts/build_syllabus.py <folder holding the old *_learn packages>
Each item gets a stable id (course.lesson.item, e.g. rag.l06.2) and its part (rag.p2), so the curriculum can assign
whole parts or single lessons to the new modules, and a test can prove that nothing was left out.
"""

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "src"
COURSES = {
  "lc": "langchain_learn",
  "rag": "rag_learn",
  "lg": "langgraph_learn",
  "mcp": "mcp_learn",
  "smith": "smith_learn",
  "deep": "deep_learn",
  "api": "api_learn",
  "prod": "prod_learn",
  "data": "data_learn",
  "deploy": "deploy_learn",
}
ITEM = re.compile(r"^\s{2,}(\d+[a-z]?)\.\s+(.*)$")


def items(doc: str) -> list[tuple[str, str]]:
  found, current = [], None
  for line in doc.splitlines():
    match = ITEM.match(line)
    if match:
      current = [match.group(1), match.group(2).strip()]
      found.append(current)
    elif current and line.startswith("     ") and line.strip():
      current[1] += " " + line.strip()  # a wrapped item
    elif not line.strip():
      current = None
  return [(n, t) for n, t in found]


def main() -> None:
  syllabus = []
  for prefix, package in COURSES.items():
    for part_dir in sorted((OLD / package).glob("part*")):
      part = f"{prefix}.p{int(re.match(r'part(\d+)', part_dir.name).group(1))}"
      for path in sorted(part_dir.glob("l*.py")):
        number = int(re.match(r"l(\d+)_", path.name).group(1))
        doc = ast.get_docstring(ast.parse(path.read_text())) or ""
        lesson_id, title = f"{prefix}.l{number:02d}", doc.splitlines()[0].split(" — ", 1)[-1].rstrip(".")
        points = items(doc) or [("0", title)]  # a capstone without a list counts as one point
        for n, text in points:
          syllabus.append(
            {
              "id": f"{lesson_id}.{n}",
              "lesson": lesson_id,
              "part": part,
              "course": prefix,
              "lesson_title": title,
              "text": text,
            }
          )
  additions = json.loads((ROOT / "content" / "additions.json").read_text())
  for lesson in additions["lessons"]:  # learning points that are new in this course
    for n, text in enumerate(lesson["points"], start=1):
      syllabus.append(
        {
          "id": f"{lesson['lesson']}.{n}",
          "lesson": lesson["lesson"],
          "part": lesson["part"],
          "course": "new",
          "lesson_title": lesson["title"],
          "text": text,
        }
      )
  out = ROOT / "content" / "syllabus.json"
  out.write_text(json.dumps(syllabus, ensure_ascii=False, indent=1))
  print(
    f"{len(syllabus)} learning points from {len({s['lesson'] for s in syllabus})} lessons → {out.relative_to(ROOT)}"
  )


if __name__ == "__main__":
  main()
