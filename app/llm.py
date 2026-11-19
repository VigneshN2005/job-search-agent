from __future__ import annotations

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
            try:
                data = request_json(self.url, payload, {"Authorization": f"Bearer {self.api_key}"}, timeout=60)
            except HttpError as exc:
                errors.append(f"{model}: {exc}")
                body = exc.body.lower()
                if any(h in body for h in RETIRED_HINTS) and model in self.models:
                    self.models.remove(model)
                if exc.status in (400, 404, 413, 429, 500, 502, 503) or exc.status == 0:
                    continue
                raise LLMError(str(exc)) from exc
            self.last_model = model
            return data["choices"][0]["message"]
        raise LLMError("; ".join(errors) or "no model available")
