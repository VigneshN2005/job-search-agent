import pytest

import app.llm as llm
from app.http import HttpError


def test_wait_hint_parses_groq_messages():
    assert llm.wait_hint("Please try again in 7.5s.") == 7.5
    assert llm.wait_hint("Please try again in 340ms.") == 0.34
    assert llm.wait_hint("Please try again in 2m3.5s.") == 123.5
    assert llm.wait_hint("nothing") is None


def test_short_rate_limit_waits_and_retries_same_model(monkeypatch):
    calls, sleeps = [], []

    def fake(url, payload, headers, timeout):
        calls.append(payload["model"])
        if len(calls) == 1:
            raise HttpError(429, "Rate limit reached. Please try again in 2s.")
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(llm, "request_json", fake)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    client = llm.GroqClient("k", ["big", "small"])
    assert client.chat([{"role": "user", "content": "hi"}])["content"] == "ok"
    assert calls == ["big", "big"] and sleeps == [2.5]


def test_long_rate_limit_falls_back_to_next_model(monkeypatch):
    calls = []

    def fake(url, payload, headers, timeout):
        calls.append(payload["model"])
        if payload["model"] == "big":
            raise HttpError(429, "Please try again in 45s.")
        return {"choices": [{"message": {"content": "small answered"}}]}

    monkeypatch.setattr(llm, "request_json", fake)
    client = llm.GroqClient("k", ["big", "small"])
    assert client.chat([])["content"] == "small answered"
    assert client.last_model == "small" and calls == ["big", "small"]


def test_all_models_failing_raises(monkeypatch):
    def fake(url, payload, headers, timeout):
        raise HttpError(503, "down")

    monkeypatch.setattr(llm, "request_json", fake)
    with pytest.raises(llm.LLMError):
        llm.GroqClient("k", ["a", "b"]).chat([])
