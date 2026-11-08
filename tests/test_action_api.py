import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from servicing_agent.api import create_app


@pytest.fixture()
def app(tmp_path: Path) -> FastAPI:
    return create_app(tmp_path / "action-api.db", configure_logs=False)


def request(app: FastAPI, method: str, path: str, **kwargs) -> httpx.Response:
    async def send() -> httpx.Response:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_note_prepare_and_confirm_api(app: FastAPI) -> None:
    headers = {"X-Demo-User": "demo-agent-alex"}
    prepared = request(
        app,
        "POST",
        "/actions/notes/prepare",
        headers=headers,
        json={
            "ticket_id": "CASE-4821",
            "note": "API confirmation test.",
            "operation_id": "op-note-api-001",
        },
    )

    assert prepared.status_code == 200
    assert prepared.json()["state"] == "PENDING"

    confirmed = request(
        app,
        "POST",
        "/actions/notes/op-note-api-001/confirm",
        headers=headers,
        json={
            "customer_confirmed": True,
            "simulate_timeout_after_success": False,
        },
    )

    assert confirmed.status_code == 200
    assert confirmed.json()["state"] == "COMPLETED"


def test_action_api_hides_operation_from_unauthorized_user(app: FastAPI) -> None:
    prepared = request(
        app,
        "POST",
        "/actions/notes/prepare",
        headers={"X-Demo-User": "demo-agent-alex"},
        json={
            "ticket_id": "CASE-4821",
            "note": "Protected operation.",
            "operation_id": "op-note-api-auth-001",
        },
    )
    assert prepared.status_code == 200

    denied = request(
        app,
        "POST",
        "/actions/notes/op-note-api-auth-001/confirm",
        headers={"X-Demo-User": "demo-agent-jordan"},
        json={
            "customer_confirmed": True,
            "simulate_timeout_after_success": False,
        },
    )

    assert denied.status_code == 403
    assert "CASE-4821" not in denied.text
