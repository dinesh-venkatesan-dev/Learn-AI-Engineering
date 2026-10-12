"""Stopping an agent run when the client leaves.

Starlette cancels a streaming response through an anyio task group, and anyio cancellation is level-triggered: every
await inside the cancelled scope is cancelled too. LangGraph's own clean-up awaits, so a run cancelled that way
leaves its model call running until it finishes: tokens spent on an answer nobody reads (measured: all 15 words of a
slow stream after the client left at word 2). Iterating the stream in its own asyncio task, and cancelling that task
plainly when the response ends, lets the run clean up and stop at once.
"""

import asyncio
from collections.abc import AsyncIterator

_DONE = object()


async def in_own_task(stream: AsyncIterator) -> AsyncIterator:
  """Yield what `stream` yields; it runs in a separate task that is cancelled when this generator stops."""
  queue: asyncio.Queue = asyncio.Queue()

  async def pump() -> None:
    try:
      async for item in stream:
        queue.put_nowait(("item", item))
    except Exception as error:  # noqa: BLE001 - handed to the consumer below
      queue.put_nowait(("error", error))
    finally:
      queue.put_nowait(("done", _DONE))

  task = asyncio.create_task(pump())  # outside the request's cancel scope
  try:
    while True:
      kind, item = await queue.get()
      if kind == "done":
        return
      if kind == "error":
        raise item
      yield item
  finally:
    task.cancel()  # plain asyncio cancellation: LangGraph cancels the model call and cleans up
