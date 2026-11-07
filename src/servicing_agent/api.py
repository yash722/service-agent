from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from .auth import DemoUserContext, resolve_demo_user
from .database import Database
from .logging_config import configure_logging
from .schemas import ChatRequest, ChatResponse, HealthResponse
from .service import ChatService
from .settings import Settings


def create_app(
    database_path: str | Path | None = None,
    *,
    configure_logs: bool = True,
) -> FastAPI:
    settings = Settings.from_environment()
    resolved_database_path = Path(database_path or settings.database_path)
    if configure_logs:
        configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = Database(resolved_database_path)
        database.initialize()
        app.state.chat_service = ChatService(database)
        yield

    app = FastAPI(
        title="FDE Servicing Agent",
        version="0.1.0",
        description="Local-only Milestone 1 proof of concept",
        lifespan=lifespan,
    )

    async def get_user_context(
        x_demo_user: str = Header(default="demo-agent-alex"),
    ) -> DemoUserContext:
        user_context = resolve_demo_user(x_demo_user)
        if user_context is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unknown local demo user.",
            )
        return user_context

    async def get_chat_service(request: Request) -> ChatService:
        return request.app.state.chat_service

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(
        request_body: ChatRequest,
        user_context: DemoUserContext = Depends(get_user_context),
        chat_service: ChatService = Depends(get_chat_service),
    ) -> ChatResponse:
        return chat_service.chat(request_body.message, user_context)

    return app


app = create_app()

