"""The Nimbus assistant's specialists: each answers one kind of question with its own prompt and tools.

  account    the support desk (Agents Module 4), read-only: orders, subscription, loyalty, refund eligibility
  knowledge  the RAG answerer (RAG Module 7) on PUBLIC documents: shipping, returns, loyalty rules, cafés, products
  cafe       an agent with the café menu and skills (allergens, catering quotes, complaints)

All three take (customer_id, question) and return an answer, so any architecture (one agent, a router, a
supervisor) can use them as functions, as tools, or as graph nodes.
"""

from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from nimbus.assistant.skills import SkillState, load_skills, skill_tools, skills_prompt, unlock_tools
from nimbus.support.agent import CustomerContext
from nimbus.support.tools import SupportStore, build_tools

Name = Literal["account", "knowledge", "cafe"]
DESCRIPTIONS = {
  "account": "the customer's own orders, deliveries, subscription, loyalty points and refund eligibility",
  "knowledge": "Nimbus policies and public information: shipping, returns, subscriptions, loyalty rules, cafés "
  "(addresses, opening hours, Wi-Fi), products and blends",
  "cafe": "the café menu: prices, vegan options, allergens and ingredients, catering quotes, complaints about a "
  "café visit",
}
CAFE_PROMPT = (
  "You are Nimbus Coffee's café assistant. Use cafe_menu_item for menu facts; load the matching skill before "
  "answering a question it covers and follow it. Answer in two or three short sentences."
)


def cafe_tools(store: SupportStore) -> list:
  menu = next(t for t in build_tools(store) if t.name == "cafe_menu_item")
  return [menu, *skill_tools(load_skills())]


def cafe_agent(model: BaseChatModel, store: SupportStore):
  skills = load_skills()
  return create_agent(
    model,
    cafe_tools(store),
    system_prompt=f"{CAFE_PROMPT}\n\n{skills_prompt(skills)}",
    state_schema=SkillState,
    middleware=[unlock_tools],
    name="cafe",
  )


@dataclass
class Specialists:
  """The three specialists, built once and reused for every question."""

  desk: object
  answerer: object
  cafe: object
  store: SupportStore

  def ask(self, name: Name, customer_id: str, question: str, config: dict | None = None) -> str:
    config = config or {}
    if name == "account":
      context = CustomerContext(customer_id, self.store.customers.get(customer_id, "the customer"))
      payload = {"messages": [HumanMessage(question)], "customer_id": customer_id, "actions": []}
      return self.desk.invoke(payload, config, context=context)["ticket"].reply
    if name == "knowledge":
      result = self.answerer.answer(question, audience="public")  # customers only see public documents
      return result.answer if result.answerable else "Nothing in our public documents answers that."
    result = self.cafe.invoke({"messages": [HumanMessage(question)], "skills_loaded": []}, config)
    return result["messages"][-1].text


def build_specialists(settings, model: BaseChatModel, retriever, store: SupportStore | None = None) -> Specialists:
  from nimbus.rag.answer import Answerer
  from nimbus.support.desk import build_support_desk

  store = store or SupportStore()
  desk = build_support_desk(model, store, read_only=True)  # changes to an account go through the desk + approvals
  return Specialists(desk, Answerer(settings, retriever, model), cafe_agent(model, store), store)
