from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from app.models import Filters, JobCard
from app.ranking import tokens


def make_job(job_id: int, title: str, company: str = "Acme", **kw: Any) -> JobCard:
    posted = kw.pop("days_ago", 3)
    return JobCard(
        id=job_id,
        title=title,
        company=company,
        url=f"https://example.com/jobs/{job_id}",
        posted_at=(datetime.now(timezone.utc) - timedelta(days=posted)).isoformat(),
        **kw,
    )


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        self.calls.append((texts, task))
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeRepo:
    def __init__(self, jobs: list[JobCard]) -> None:
        self.jobs = {j.id: j for j in jobs}
        self.prefs: dict[str, dict[str, str]] = {}
        self.history: dict[str, list[dict[str, str]]] = {}
        self.last_filters: Filters | None = None

    def _match(self, job: JobCard, f: Filters) -> bool:
        if f.remote is True and not job.remote:
            return False
        if f.remote is False and job.remote:
            return False
        if f.location and f.location.lower() not in job.location.lower():
            if not (job.remote and job.location.lower() in ("worldwide", "anywhere")):
                return False
        if f.skills and not ({s.lower() for s in f.skills} & {t.lower() for t in job.tags}):
            return False
        if f.min_salary and (job.salary_max or job.salary_min or 0) < f.min_salary:
            return False
        if f.job_type and f.job_type.lower() not in job.job_type.lower():
            return False
        return job.id not in f.exclude_ids

    def semantic_candidates(
        self, vector: list[float], filters: Filters, limit: int = 40
    ) -> list[tuple[JobCard, float]]:
        self.last_filters = filters
        query = getattr(filters, "query", "")
        out = []
        for job in self.jobs.values():
            if self._match(job, filters):
                overlap = len(tokens(query) & tokens(job.title + " " + " ".join(job.tags)))
                out.append((job, 0.5 + 0.1 * overlap))
        out.sort(key=lambda p: p[1], reverse=True)
        return out[:limit]

    def filter_jobs(self, filters: Filters, limit: int = 8, sort: str = "newest") -> list[JobCard]:
        self.last_filters = filters
        jobs = [j for j in self.jobs.values() if self._match(j, filters)]
        jobs.sort(key=lambda j: j.posted_at or "", reverse=True)
        return jobs[:limit]

    def get_job(self, job_id: int) -> tuple[JobCard, str] | None:
        job = self.jobs.get(job_id)
        return (job, f"Full description of {job.title}") if job else None

    def existing_ids(self, ids: list[int]) -> set[int]:
        return {i for i in ids if i in self.jobs}

    def ensure_session(self, session_id: str) -> None:
        self.history.setdefault(session_id, [])
        self.prefs.setdefault(session_id, {})

    def load_history(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        return self.history.get(session_id, [])[-limit:]

    def append_message(self, session_id: str, role: str, content: str) -> None:
        self.history.setdefault(session_id, []).append({"role": role, "content": content})

    def load_preferences(self, session_id: str) -> dict[str, str]:
        return dict(self.prefs.get(session_id, {}))

    def save_preference(self, session_id: str, key: str, value: str) -> None:
        self.prefs.setdefault(session_id, {})[key] = value

    def stats(self) -> dict[str, Any]:
        return {"jobs": len(self.jobs), "indexed": len(self.jobs), "remote": 0, "latest": None, "sources": ["fake"]}


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}],
    }


class ScriptedLLM:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.last_model = "scripted"

    def chat(self, messages, tools=None, json_mode=False):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools, "json_mode": json_mode})
        if not self.replies:
            raise AssertionError("LLM called more times than scripted")
        return self.replies.pop(0)
