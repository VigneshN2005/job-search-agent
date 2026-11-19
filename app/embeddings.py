from __future__ import annotations

import math
import re
import time
from typing import Literal, Protocol

from app.http import HttpError, request_json

Task = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


class QuotaExhausted(Exception):
    pass


def retry_after(body: str, default: float = 30.0) -> float:
    match = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body)
    return float(match.group(1)) if match else default


class Embedder(Protocol):
    def embed(self, texts: list[str], task: Task) -> list[list[float]]: ...


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class GeminiEmbedder:
    base = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dim: int = 768,
        batch_size: int = 20,
        pause: float = 0.0,
        max_attempts: int = 6,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.pause = pause
        self.max_attempts = max_attempts

    def _batch(self, texts: list[str], task: Task) -> list[list[float]]:
        body = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t[:8000]}]},
                    "taskType": task,
                    "outputDimensionality": self.dim,
                }
                for t in texts
            ]
        }
        url = f"{self.base}/{self.model}:batchEmbedContents"
        for attempt in range(self.max_attempts):
            try:
                data = request_json(url, body, {"x-goog-api-key": self.api_key}, timeout=60)
                return [normalize(e["values"]) for e in data["embeddings"]]
            except HttpError as exc:
                last = attempt == self.max_attempts - 1
                if exc.status == 429:
                    if "perday" in exc.body.lower().replace(" ", "").replace("_", "") or last:
                        raise QuotaExhausted(exc.body[:300]) from exc
                    time.sleep(min(retry_after(exc.body) + 1, 70))
                    continue
                if exc.status in (500, 503) and not last:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise
        return []

    def embed(self, texts: list[str], task: Task) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            if i and self.pause:
                time.sleep(self.pause)
            out.extend(self._batch(texts[i : i + self.batch_size], task))
        return out
