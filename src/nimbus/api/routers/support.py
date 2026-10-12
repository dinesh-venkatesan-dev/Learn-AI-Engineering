"""Support replies: a validated request in, the model's answer out (the shape of most AI APIs).

The model client is the ONE the lifespan created; per-request settings travel with the call. Model failures become
503 (try again) or 502 (our request was wrong), never a stack trace.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Header, Request, Response
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from nimbus.api.deps import ChatModelDep, SettingsDep, audit
from nimbus.api.errors import ask_model
from nimbus.api.limits import BudgetDep, RateLimited
from nimbus.api.release import prompt_ref, variant
from nimbus.api.schemas import Draft, ReplyPublic, ReplyRequest
from nimbus.llm.models import model_name
from nimbus.observability.metrics import TOKENS

log = logging.getLogger("nimbus.api.support")
router = APIRouter(prefix="/support", tags=["support"], dependencies=[Depends(audit)])  # every call: key + audit


V1 = (
  "You are Nimbus Coffee's support assistant. Reply in a {tone} tone, in at most {max_words} words. "
  "Never promise refunds or dates: say a teammate will check."
)


def system_prompt(body: ReplyRequest, ref: str | None = None) -> str:
  """The system prompt from the prompt registry (the version tagged production), filled in for this request."""
  from nimbus.config import get_settings
  from nimbus.prompts.registry import PromptRegistry

  version = PromptRegistry().pull(ref or get_settings().support_prompt)
  return version.messages[0][1].format(tone=body.tone, max_words=body.max_words)


async def draft_reply(model, body: ReplyRequest, ref: str | None = None) -> Draft:
  history = [(HumanMessage if m.role == "user" else AIMessage)(m.content) for m in body.history]
  prompt = system_prompt(body, ref)
  reply = await ask_model(model, [SystemMessage(prompt), *history, HumanMessage(body.message)])  # 503 / 502
  usage = reply.usage_metadata or {}
  name = model_name(model)
  TOKENS.labels(name, "input").inc(usage.get("input_tokens", 0))
  TOKENS.labels(name, "output").inc(usage.get("output_tokens", 0))
  return Draft(
    answer=reply.text,
    system_prompt=prompt,
    model=name,
    input_tokens=usage.get("input_tokens", 0),
    output_tokens=usage.get("output_tokens", 0),
  )


async def shadow_reply(request: Request, model, body: ReplyRequest, shown: Draft) -> None:
  """After the response: the shadow prompt answers the same message; nobody sees it, the comparison is kept."""
  try:
    shadow = await draft_reply(model, body, request.app.state.settings.shadow_prompt)
  except Exception:  # noqa: BLE001 - a shadow failure must never touch the real request
    log.exception("shadow reply failed")
    return
  request.app.state.shadow.append({"message": body.message, "shown": shown.answer, "shadow": shadow.answer})


@router.post("/reply")
async def reply(
  body: ReplyRequest,
  model: ChatModelDep,
  caller: RateLimited,
  budget: BudgetDep,
  settings: SettingsDep,
  request: Request,
  response: Response,
  tasks: BackgroundTasks,
  x_user_id: Annotated[str | None, Header()] = None,
) -> ReplyPublic:
  """Draft a reply to a customer's message (rate-limited per plan, within the caller's daily token budget). A share
  of users (canary_percent, by a hash of X-User-ID) gets the canary prompt; X-Variant says which answered."""
  which = variant(x_user_id or caller, settings.canary_percent)
  draft = await draft_reply(model, body, prompt_ref(settings, which))  # async: other requests go on meanwhile
  budget.spend(draft.input_tokens + draft.output_tokens)
  response.headers["X-Variant"] = which
  if settings.shadow_prompt:
    tasks.add_task(shadow_reply, request, model, body, draft)
  return ReplyPublic(answer=draft.answer, words=len(draft.answer.split()))
