from __future__ import annotations

import math
import time
from typing import Literal, Protocol

from app.http import HttpError, request_json

Task = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


class Embedder(Protocol):
    def embed(self, texts: list[str], task: Task) -> list[list[float]]: ...


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class GeminiEmbedder:
    base = "https://generativelanguage.googleapis.com/v1beta/models"
    batch_size = 100

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dim: int = 768):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.dim = dim

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
        for attempt in range(4):
            try:
                data = request_json(url, body, {"x-goog-api-key": self.api_key}, timeout=60)
                return [normalize(e["values"]) for e in data["embeddings"]]
            except HttpError as exc:
                if exc.status in (429, 500, 503) and attempt < 3:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise
        return []

    def embed(self, texts: list[str], task: Task) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._batch(texts[i : i + self.batch_size], task))
        return out
