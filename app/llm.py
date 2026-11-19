from __future__ import annotations

import re
import time
from typing import Any, Protocol

from app.http import HttpError, request_json


class LLMError(Exception):
    pass


class ChatModel(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
    ) -> dict[str, Any]: ...


RETIRED_HINTS = ("decommissioned", "model_not_found", "does not exist", "not supported")
MAX_WAIT = 12.0


def wait_hint(body: str) -> float | None:
    match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)(ms|s)", body)
    if not match:
        return None
    minutes = float(match.group(1) or 0)
    value = float(match.group(2))
    return minutes * 60 + (value / 1000 if match.group(3) == "ms" else value)


class GroqClient:
    url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, models: tuple[str, ...] | list[str]):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        if not models:
            raise ValueError("at least one Groq model is required")
        self.api_key = api_key
        self.models = list(models)
        self.last_model: str | None = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        errors: list[str] = []
        for model in list(self.models):
            payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.2}
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            for attempt in range(2):
                try:
                    data = request_json(self.url, payload, {"Authorization": f"Bearer {self.api_key}"}, timeout=60)
                except HttpError as exc:
                    body = exc.body.lower()
                    if exc.status == 429 and attempt == 0:
                        delay = wait_hint(exc.body)
                        if delay is not None and delay <= MAX_WAIT:
                            time.sleep(delay + 0.5)
                            continue
                    errors.append(f"{model}: {exc}")
                    if any(h in body for h in RETIRED_HINTS) and model in self.models:
                        self.models.remove(model)
                    if exc.status in (400, 404, 413, 429, 500, 502, 503) or exc.status == 0:
                        break
                    raise LLMError(str(exc)) from exc
                self.last_model = model
                return data["choices"][0]["message"]
        raise LLMError("; ".join(errors) or "no model available")
