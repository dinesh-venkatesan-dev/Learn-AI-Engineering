"""The support agent as an MCP CLIENT: it gets its tools, its policy and its approval rules from the Nimbus gateway.

tools        discovered from the gateway (public: no internal tools) through MCPAdapter
policy       nimbus://support/policies/refunds, read fresh into the system prompt on every model call
approvals    tools the SERVER marks destructive (destructiveHint) pause for a person; no tool names hard-coded
questions    a server that asks the user mid-call (elicitation) pauses the agent with an interrupt

  async with MCPAdapter(gateway) as adapter:
    agent = await build_mcp_agent(model, adapter, checkpointer)
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ModelRequest, dynamic_prompt
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langchain_core.language_models import BaseChatModel

SYSTEM = (
  "You are Nimbus Coffee's support assistant. Use the tools to look things up and to act; never guess order "
  "details. Reply in two or three short sentences."
)
POLICY_URI = "nimbus://support/policies/refunds"  # the gateway prefixes the support server's resources


def is_destructive(tool) -> bool:
  annotations = (tool.metadata or {}).get("mcp", {}).get("tool", {}).get("annotations", {})
  return bool(annotations.get("destructive_hint"))


async def build_mcp_agent(
  model: BaseChatModel, adapter, checkpointer=None, policy: bool = True, policy_uri: str = POLICY_URI
):
  """policy_uri: the gateway prefixes the support server's resources; connected to the server directly it's
  nimbus://policies/refunds."""
  tools = await adapter.list_tools()
  client = adapter.client  # the FastMCP client underneath: resources and prompts too
  destructive = {t.name for t in tools if is_destructive(t)}
  gate = InterruptOnConfig(
    allowed_decisions=["approve", "reject"], when=lambda request: request.tool_call["name"] in destructive
  )
  middleware = [HumanInTheLoopMiddleware(interrupt_on={t.name: gate for t in tools})]
  if policy:

    @dynamic_prompt
    async def with_policy(request: ModelRequest) -> str:
      text = (await client.read_resource(policy_uri))[0].text  # fresh on every call: a policy change applies at once
      return f"{SYSTEM}\n\nRefund policy: {text}"

    middleware.insert(0, with_policy)
  return create_agent(
    model, tools, system_prompt=None if policy else SYSTEM, middleware=middleware, checkpointer=checkpointer
  )
