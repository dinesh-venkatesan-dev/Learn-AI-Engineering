"""The weekly ops review as it ships (Agents Module 10, the capstone): one deep agent with the whole harness.

memory     /memories/AGENTS.md: the house rules (managers, how to email them), in a store, loaded every thread;
           the agent may propose edits, but writes to /memories/ wait for a person
skill      /skills/weekly-report/: the review's procedure, read only when the job needs it (data/ops/skills)
subagents  sales-analyst and voice-of-customer from Module 9, one café at a time, clean contexts
approvals  every send_email pauses for a person (interrupt_on)
result     OpsReviewResult (ToolStrategy): the cafés that need attention and the emails, for code to use
context    OpsContext: who asked for the review, read by tools, never by the model

  reviewer, outbox = build_reviewer(model, store, checkpointer)
"""

from dataclasses import dataclass, field

from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from nimbus.config import PROJECT_DIR
from nimbus.ops.agents import SALES_ANALYST, VOICE_OF_CUSTOMER, list_cafes

OPS_DIR = PROJECT_DIR / "data" / "ops"
MEMORY_NAMESPACE = ("nimbus-ops",)
SYSTEM = (
  "You are Nimbus Coffee's operations lead, writing this week's review (2026-09-24 to 2026-09-30). Follow the "
  "weekly-report skill and the house rules. Delegate every number to sales-analyst and every complaint to "
  "voice-of-customer, one café per task (several tasks per turn when you can). Never compute numbers yourself."
)


@dataclass
class OpsContext:
  requested_by: str = "the ops lead"


class Email(BaseModel):
  to: str
  cafe: str


class OpsReviewResult(BaseModel):
  """The review's outcome, for code: which cafés need attention and who was emailed."""

  needs_attention: list[str] = Field(description="Full café names, e.g. 'Nimbus Colaba'")
  emails: list[Email] = Field(default_factory=list, description="Every email sent")
  summary: str = Field(description="Two sentences for the ops channel")


@dataclass
class Outbox:
  sent: list[dict] = field(default_factory=list)


def house_rules() -> str:
  return (OPS_DIR / "AGENTS.md").read_text()


def seed_memory(store) -> None:
  """The house rules go into the store once; under the /memories/ route the key drops the prefix."""
  from deepagents.backends.utils import create_file_data

  if store.get(MEMORY_NAMESPACE, "/AGENTS.md") is None:
    store.put(MEMORY_NAMESPACE, "/AGENTS.md", create_file_data(house_rules()))


def email_tool(outbox: Outbox):
  @tool
  def send_email(to: str, subject: str, body: str, runtime: ToolRuntime) -> str:
    """Send an email to a Nimbus café manager. A person approves every email before it goes."""
    requested_by = runtime.context.requested_by if runtime.context else "the ops lead"
    outbox.sent.append({"to": to, "subject": subject, "body": body, "requested_by": requested_by})
    return f"Email sent to {to}."

  return send_email


def build_reviewer(model: BaseChatModel, store, checkpointer, outbox: Outbox | None = None, *, harness: bool = True):
  """The capstone reviewer. harness=False is the comparison: the same agent with the rules and the procedure pasted
  into its prompt instead of memory and a skill (still with approvals)."""
  from deepagents import create_deep_agent
  from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
  from deepagents.middleware.permissions import FilesystemPermission
  from langchain.agents.middleware import TodoListMiddleware

  outbox = outbox or Outbox()
  send_email = email_tool(outbox)
  common = {
    "model": model,
    "tools": [list_cafes, send_email],
    "subagents": [SALES_ANALYST, VOICE_OF_CUSTOMER],
    "middleware": [TodoListMiddleware()],
    "interrupt_on": {"send_email": True},  # every email waits for a person
    "response_format": ToolStrategy(OpsReviewResult),
    "context_schema": OpsContext,
    "checkpointer": checkpointer,
  }
  if not harness:
    skill = (OPS_DIR / "skills" / "weekly-report" / "SKILL.md").read_text()
    return create_deep_agent(system_prompt=f"{SYSTEM}\n\n{house_rules()}\n\n{skill}", **common), outbox
  seed_memory(store)
  backend = CompositeBackend(
    default=StateBackend(),
    routes={
      "/memories/": StoreBackend(namespace=lambda rt: MEMORY_NAMESPACE),
      "/skills/": FilesystemBackend(root_dir=OPS_DIR / "skills", virtual_mode=True),  # the repo's skill files
    },
  )
  permissions = [
    FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny"),  # procedures change in code review
    FilesystemPermission(operations=["write"], paths=["/memories/**"], mode="interrupt"),  # a person approves
  ]
  agent = create_deep_agent(
    system_prompt=SYSTEM,
    backend=backend,
    store=store,
    memory=["/memories/AGENTS.md"],
    skills=["/skills/"],
    permissions=permissions,
    **common,
  )
  return agent, outbox
