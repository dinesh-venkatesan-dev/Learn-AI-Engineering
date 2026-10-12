"""The quality gate CI runs on pull requests: a support reply prompt (default: the one the app runs) on the
evaluation dataset; exit 1 if a score is below its floor.

  uv run python scripts/eval_gate.py                      the production prompt
  uv run python scripts/eval_gate.py support-reply:<commit>   a candidate
"""

import sys

from nimbus.config import get_settings
from nimbus.observability.tracing import guard_uploads

guard_uploads(get_settings())  # nothing goes to LangSmith from CI without Nimbus's own key

from nimbus.support import reply_eval as rv  # noqa: E402

FLOORS = {"no_promises": 1.0, "facts": 0.8}


def main(ref: str) -> int:
  examples = rv.load_examples()
  outputs = rv.answer_all(ref, examples)
  averages = rv.averages(rv.scores(rv.run_experiment(outputs, examples, [rv.facts, rv.no_promises])))
  failed = {k: round(averages[k], 2) for k, floor in FLOORS.items() if averages[k] < floor}
  shown = {k: round(v, 2) for k, v in averages.items()}
  print(f"{ref}: {shown}; floors {FLOORS}; {'FAILED ' + str(failed) if failed else 'passed'}")
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else get_settings().support_prompt))
