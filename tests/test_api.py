"""The Nimbus API, tested through real HTTP requests without a server, a network or a model.

Every test gets a fresh world: a temporary SQLite database, a test JWT secret and a FAKE chat model (the same answer
every time, no key, no cost). Parts are swapped two ways: create_app(settings, chat_model=...) and
app.dependency_overrides.
"""

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import SecretStr
from starlette.websockets import WebSocketDisconnect

from nimbus.api.app import create_app
from nimbus.api.deps import chat_model
from nimbus.config import Settings

PASSWORD = "test-password-123"


def fake(*answers: str) -> GenericFakeChatModel:
  return GenericFakeChatModel(messages=iter([AIMessage(a) for a in answers or ["A fake answer."] * 50]))


@pytest.fixture
def settings(tmp_path) -> Settings:
  return Settings(api_database_url=f"sqlite:///{tmp_path}/test.db", jwt_secret=SecretStr("t" * 40), api_keys=[])


@pytest.fixture
def client(settings):
  with TestClient(create_app(settings=settings, chat_model=fake())) as test_client:  # `with` runs the lifespan
    yield test_client


def sign_in(client, username: str) -> dict:
  client.post("/auth/register", json={"username": username, "password": PASSWORD})
  token = client.post("/auth/token", data={"username": username, "password": PASSWORD}).json()["access_token"]
  return {"Authorization": f"Bearer {token}"}


def test_register_stores_a_hash_not_the_password(client):
  response = client.post("/auth/register", json={"username": "priya", "password": PASSWORD})
  assert response.status_code == 201
  assert set(response.json()) == {"id", "username"}  # no password, no hash


def test_wrong_user_and_wrong_password_look_the_same(client):
  sign_in(client, "priya")
  unknown = client.post("/auth/token", data={"username": "nobody", "password": PASSWORD})
  wrong = client.post("/auth/token", data={"username": "priya", "password": "not-the-password"})
  assert unknown.status_code == wrong.status_code == 401
  assert unknown.json() == wrong.json()


@pytest.mark.parametrize(
  "header",
  [None, "Bearer not-a-jwt", "Bearer " + jwt.encode({"sub": "priya"}, "x" * 40)],
  ids=["no token", "not a JWT", "signed with another secret"],
)
def test_missing_or_bad_tokens_are_401(client, header):
  sign_in(client, "priya")
  response = client.get("/notes", headers={"Authorization": header} if header else {})
  assert response.status_code == 401
  assert response.json()["error"]["code"] == "unauthorized"


def test_someone_elses_note_looks_missing(client):
  priya, arjun = sign_in(client, "priya"), sign_in(client, "arjun")
  note = client.post("/notes", json={"title": "Private", "content": "Priya's notes."}, headers=priya).json()
  assert client.get(f"/notes/{note['id']}", headers=arjun).status_code == 404
  assert client.get("/notes", headers=arjun).json() == []


def test_duplicate_titles_are_409(client):
  headers = sign_in(client, "priya")
  client.post("/notes", json={"title": "Refunds", "content": "14 days."}, headers=headers)
  response = client.post("/notes", json={"title": "Refunds", "content": "Again."}, headers=headers)
  assert response.status_code == 409 and response.json()["error"]["code"] == "duplicate_title"


def test_patch_changes_only_what_was_sent(client):
  headers = sign_in(client, "priya")
  note = client.post("/notes", json={"title": "Refunds", "content": "14 days."}, headers=headers).json()
  patched = client.patch(f"/notes/{note['id']}", json={"content": "Within 14 days."}, headers=headers).json()
  assert patched["title"] == "Refunds" and patched["content"] == "Within 14 days."


def test_unsupported_uploads_are_415(client):
  headers = sign_in(client, "priya")
  response = client.post("/notes/upload", files=[("files", ("a.png", b"\x89PNG", "image/png"))], headers=headers)
  assert response.status_code == 415


def test_the_summary_is_written_after_the_response(settings):
  with TestClient(create_app(settings=settings, chat_model=fake("One sentence."))) as client:
    headers = sign_in(client, "priya")
    created = client.post("/notes", json={"title": "Oat milk", "content": "40 rupees extra."}, headers=headers)
    assert created.json()["summary"] is None  # the response didn't wait
    assert client.get(f"/notes/{created.json()['id']}", headers=headers).json()["summary"] == "One sentence."


def test_stream_events_arrive_in_order(client):
  headers = sign_in(client, "priya")
  client.post("/notes", json={"title": "Refunds", "content": "Refunds within 14 days."}, headers=headers)
  with client.stream("POST", "/chat/stream", json={"question": "When are refunds possible?"}, headers=headers) as r:
    events = [line.split(":", 1)[1].strip() for line in r.iter_lines() if line.startswith("event:")]
  assert events[0] == "sources" and events[-1] == "done" and "token" in events


def test_links_never_reach_the_user(settings):
  model = fake(*["Here: ![x](https://evil.example/?q=secret) and https://evil.example/leak done."] * 3)
  with TestClient(create_app(settings=settings, chat_model=model)) as client:
    headers = sign_in(client, "priya")
    client.post("/notes", json={"title": "Refunds", "content": "Refunds within 14 days."}, headers=headers)
    answer = client.post("/chat", json={"question": "Refunds?"}, headers=headers).json()["answer"]
  assert "evil.example" not in answer and "[link removed]" in answer


def test_websocket_refuses_a_bad_token(client):
  with pytest.raises(WebSocketDisconnect) as closed, client.websocket_connect("/chat/ws?token=nope") as ws:
    ws.receive_text()
  assert closed.value.code == 1008


def test_dependency_overrides_swap_the_model(settings):
  app = create_app(settings=settings, chat_model=fake("from the app"))
  app.dependency_overrides[chat_model] = lambda: fake("from the override")  # any Depends(...) can be replaced
  with TestClient(app) as client:
    reply = client.post("/support/reply", json={"message": "hello"}).json()
  assert reply["answer"] == "from the override"


@pytest.fixture
def anyio_backend():
  return "asyncio"


@pytest.mark.anyio
async def test_async_client_in_process(settings):
  app = create_app(settings=settings, chat_model=fake())
  async with (
    app.router.lifespan_context(app),  # the lifespan by hand: AsyncClient doesn't run it
    httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
  ):
    response = await client.get("/orders/NC-12093")
  assert response.status_code == 200 and response.json()["status"] == "delivered"


def test_an_answer_can_be_rated_by_its_run_id(client):
  headers = sign_in(client, "priya")
  client.post("/notes", json={"title": "Refunds", "content": "Refunds within 14 days."}, headers=headers)
  answer = client.post("/chat", json={"question": "Refunds?"}, headers=headers).json()
  rated = client.post(
    "/feedback", json={"run_id": answer["run_id"], "score": 0, "comment": "too vague"}, headers=headers
  )
  assert rated.status_code == 201
  mine = client.get("/feedback", params={"run_id": answer["run_id"]}, headers=headers).json()
  assert [f["comment"] for f in mine] == ["too vague"]
  assert client.get("/feedback", headers=sign_in(client, "arjun")).json() == []  # ratings are per user
