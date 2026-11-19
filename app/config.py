from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    database_url: str
    groq_api_key: str
    gemini_api_key: str
    groq_models: tuple[str, ...]
    embedding_model: str
    embedding_dim: int
    history_limit: int
    max_tool_rounds: int


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@lru_cache
def get_settings() -> Settings:
    models = os.getenv("GROQ_MODELS", "openai/gpt-oss-120b,llama-3.3-70b-versatile")
    return Settings(
        database_url=os.getenv("DATABASE_URL", ""),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        groq_models=tuple(m.strip() for m in models.split(",") if m.strip()),
        embedding_model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
        embedding_dim=_int("EMBEDDING_DIM", 768),
        history_limit=_int("HISTORY_LIMIT", 12),
        max_tool_rounds=_int("MAX_TOOL_ROUNDS", 4),
    )
