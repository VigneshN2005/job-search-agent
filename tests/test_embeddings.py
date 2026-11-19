import pytest

import app.embeddings as emb
from app.http import HttpError


def test_retry_after_reads_google_delay():
    assert emb.retry_after('{"retryDelay": "37s"}') == 37.0
    assert emb.retry_after("no hint", 12) == 12


def test_minute_limit_waits_then_succeeds(monkeypatch):
    calls, sleeps = [], []

    def fake_request(url, body, headers, timeout):
        calls.append(len(body["requests"]))
        if len(calls) == 1:
            raise HttpError(429, '{"retryDelay": "5s", "quotaId": "EmbedContentRequestsPerMinute"}')
        return {"embeddings": [{"values": [3.0, 4.0]} for _ in body["requests"]]}

    monkeypatch.setattr(emb, "request_json", fake_request)
    monkeypatch.setattr(emb.time, "sleep", sleeps.append)
    out = emb.GeminiEmbedder("k", batch_size=2).embed(["a", "b", "c"], "RETRIEVAL_DOCUMENT")
    assert out == [[0.6, 0.8]] * 3
    assert sleeps[0] == 6
    assert calls == [2, 2, 1]


def test_daily_limit_raises_quota_exhausted(monkeypatch):
    def fake_request(url, body, headers, timeout):
        raise HttpError(429, '{"quotaId": "EmbedContentRequestsPerDayPerProject"}')

    monkeypatch.setattr(emb, "request_json", fake_request)
    with pytest.raises(emb.QuotaExhausted):
        emb.GeminiEmbedder("k").embed(["a"], "RETRIEVAL_QUERY")
