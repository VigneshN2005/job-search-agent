from __future__ import annotations

import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent import Agent
from app.config import get_settings
from app.embeddings import GeminiEmbedder
from app.llm import GroqClient, LLMError
from app.models import ChatRequest, ChatResponse
from app.repository import PostgresRepository

PUBLIC = Path(__file__).resolve().parent.parent / "public"

app = FastAPI(title="Conversational Job-Search Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@lru_cache
def repository() -> PostgresRepository:
    return PostgresRepository(get_settings().database_url)


@lru_cache
def agent() -> Agent:
    s = get_settings()
    return Agent(
        llm=GroqClient(s.groq_api_key, s.groq_models),
        repo=repository(),
        embedder=GeminiEmbedder(s.gemini_api_key, s.embedding_model, s.embedding_dim),
        history_limit=s.history_limit,
        max_tool_rounds=s.max_tool_rounds,
    )


def _session(value: str | None) -> str:
    if value:
        try:
            return str(uuid.UUID(value))
        except ValueError:
            raise HTTPException(400, "invalid session_id") from None
    return str(uuid.uuid4())


@app.get("/api/health")
def health() -> dict:
    s = get_settings()
    status = {
        "status": "ok",
        "groq_key": bool(s.groq_api_key),
        "gemini_key": bool(s.gemini_api_key),
        "database": False,
    }
    try:
        repository().stats()
        status["database"] = True
    except Exception:
        status["status"] = "degraded"
    return status


@app.get("/api/stats")
def stats() -> dict:
    try:
        return repository().stats()
    except Exception as exc:
        raise HTTPException(503, f"database unavailable: {str(exc)[:120]}") from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    session_id = _session(body.session_id)
    try:
        return agent().run(session_id, body.message.strip())
    except LLMError as exc:
        raise HTTPException(
            503, "The language model is busy right now (free-tier rate limit). Please try again in a minute."
        ) from exc
    except ValueError as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/api/session/{session_id}")
def session(session_id: str) -> dict:
    sid = _session(session_id)
    repo = repository()
    return {
        "session_id": sid,
        "preferences": repo.load_preferences(sid),
        "history": repo.load_history(sid, 40),
    }


if PUBLIC.exists():

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(PUBLIC / "index.html")

    app.mount("/", StaticFiles(directory=PUBLIC), name="public")
