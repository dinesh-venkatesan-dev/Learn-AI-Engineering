"""The Nimbus MCP servers, tested through the real MCP layer (MCP Module 4).

In memory, Client(server): no subprocess, no port, no model, yet validation, errors and schemas are MCP's own.
Over HTTP in process (asgi_client): the account server with real tokens.
The real-model test is marked `integration` and skipped by default:  uv run pytest -m integration
"""

import copy

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.utilities.tests import asgi_client

import nimbus.mcp.server as support
from nimbus.mcp.account import build_account_server
from nimbus.mcp.auth import DevIdentityProvider, verifier

pytestmark = pytest.mark.anyio
IDP = DevIdentityProvider()
CONTRACT = {  # changing a tool's arguments breaks every client that calls it: make it a deliberate decision
  "get_order": ["order_id"],
  "cafe_menu_item": ["name"],
  "check_refund": ["opened", "order_id"],
  "loyalty_points": ["amount_rupees", "order_id"],
  "escalate_to_human": ["reason", "urgency"],
  "cancel_order": ["order_id"],
}


@pytest.fixture
def anyio_backend():
  return "asyncio"


@pytest.fixture(autouse=True)
def fresh_orders():
  """Tools change orders (cancel, refund): every test starts from the data files' state."""
  saved = copy.deepcopy(support.store.orders)
  yield
  support.store.orders.clear()
  support.store.orders.update(saved)


@pytest.fixture
async def client():
  async with Client(support.mcp) as connected:  # the server object itself: nothing is started
    yield connected


def as_user(subject: str, scopes=(), roles=()):
  """The account server over HTTP, in process, with a real signed token."""
  return asgi_client(build_account_server(verifier(IDP.public_key)), auth=IDP.token(subject, scopes, roles))


async def test_get_order_returns_the_public_fields(client):
  result = await client.call_tool("get_order", {"order_id": "NC-12093"})
  assert result.structured_content["status"] == "delivered"
  assert "customer_id" not in result.structured_content  # the return type filters private fields


async def test_unknown_order_is_a_tool_error(client):
  with pytest.raises(ToolError, match="No order NC-99999"):
    await client.call_tool("get_order", {"order_id": "NC-99999"})


async def test_bad_arguments_never_reach_the_tool(client):
  with pytest.raises(ToolError, match="pattern"):
    await client.call_tool("get_order", {"order_id": "12093"})


async def test_tool_contract(client):
  tools = {t.name: sorted(t.input_schema["properties"]) for t in await client.list_tools()}
  assert tools == CONTRACT


async def test_annotations_tell_hosts_what_is_safe(client):
  annotations = {t.name: t.annotations for t in await client.list_tools()}
  assert annotations["check_refund"].readOnlyHint
  assert annotations["cancel_order"].destructiveHint


async def test_policy_resource_and_reply_prompt(client):
  policy = (await client.read_resource("nimbus://policies/refunds"))[0].text
  assert "refund" in policy.lower()
  prompt = await client.get_prompt("support_reply", {"issue": "damaged bag"})
  assert "damaged bag" in prompt.messages[0].content.text


@pytest.mark.parametrize(("accept", "expected"), [(True, "Cancelled NC-55021"), (False, "NOT cancelled")])
async def test_cancel_order_asks_the_customer(accept, expected):
  async def customer(message, response_type, params, context):
    return {"confirm": accept}

  async with Client(support.mcp, elicitation_handler=customer) as connected:
    result = await connected.call_tool("cancel_order", {"order_id": "NC-55021"})
  assert expected in result.data


async def test_a_customer_sees_only_their_own_orders():
  async with as_user("c-1042") as priya:
    mine = (await priya.call_tool("my_orders", {})).structured_content["result"]
    assert {o["order_id"] for o in mine} == {"NC-12093", "NC-20417"}
    with pytest.raises(ToolError, match="No order NC-31877 on your account"):  # Arjun's: looks like it doesn't exist
      await priya.call_tool("get_order", {"order_id": "NC-31877"})


async def test_refunds_need_the_scope():
  async with as_user("c-1042") as customer:
    assert "issue_refund" not in {t.name for t in await customer.list_tools()}
    with pytest.raises(ToolError, match="Unknown tool"):
      await customer.call_tool("issue_refund", {"order_id": "NC-12093", "reason": "test"})
  async with as_user("staff-anita", scopes=("orders:read_all", "orders:refund")) as staff:
    assert "refunded ₹720" in (await staff.call_tool("issue_refund", {"order_id": "NC-12093", "reason": "t"})).data


async def test_weekly_sales_needs_the_manager_role():
  async with as_user("staff-anita", scopes=("orders:read_all",)) as staff:
    assert "weekly_sales" not in {t.name for t in await staff.list_tools()}
  async with as_user("meera", roles=("manager",)) as manager:
    assert len((await manager.call_tool("weekly_sales", {})).data) == 6


@pytest.mark.parametrize(
  "token",
  [
    None,
    DevIdentityProvider().token("c-1042"),
    IDP.token("c-1042", audience="other-api"),
    IDP.token("c-1042", expires_in_seconds=-60),
  ],
  ids=["no token", "someone else's key", "wrong audience", "expired"],
)
async def test_bad_tokens_are_refused_before_any_tool(token):
  import httpx

  app = build_account_server(verifier(IDP.public_key)).http_app()
  headers = {"Accept": "application/json, text/event-stream", **({"Authorization": f"Bearer {token}"} if token else {})}
  async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as http:
    response = await http.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=headers)
  assert response.status_code == 401


async def test_notes_belong_to_the_user_not_the_connection():
  server = build_account_server(verifier(IDP.public_key))
  async with asgi_client(server, auth=IDP.token("c-1042")) as first:
    await first.call_tool("remember_note", {"note": "oat milk"})
  async with asgi_client(server, auth=IDP.token("c-1042")) as again:
    assert (await again.call_tool("my_notes", {})).data == ["oat milk"]
  async with asgi_client(server, auth=IDP.token("c-2210")) as someone_else:
    assert (await someone_else.call_tool("my_notes", {})).data == []


@pytest.mark.integration
async def test_a_real_model_picks_the_right_tool():
  from langchain.mcp import MCPAdapter

  from nimbus.llm import get_chat_model
  from nimbus.llm.models import aclose_clients
  from nimbus.mcp.client_agent import build_mcp_agent

  async with MCPAdapter(support.mcp) as adapter:
    agent = await build_mcp_agent(get_chat_model(), adapter, policy_uri="nimbus://policies/refunds")
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "Where is my order NC-78123?"}]})
  await aclose_clients()
  calls = [(c["name"], c["args"]) for m in result["messages"] for c in getattr(m, "tool_calls", None) or []]
  assert ("get_order", {"order_id": "NC-78123"}) in calls  # structure, not wording
