from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from app.models import JobCard

WEIGHTS = {"similarity": 0.6, "keywords": 0.2, "preferences": 0.12, "recency": 0.08}
STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "for",
    "in",
    "on",
    "at",
    "of",
    "to",
    "with",
    "job",
    "jobs",
    "role",
    "roles",
    "position",
    "looking",
    "want",
    "find",
    "me",
    "i",
    "im",
    "am",
    "work",
    "remote",
    "any",
    "some",
    "that",
    "is",
    "as",
    "or",
    "who",
    "can",
    "show",
    "please",
    "need",
    "like",
    "something",
}
SENIORITY = {
    "intern": ("intern", "internship", "trainee"),
    "junior": ("junior", "jr", "entry", "graduate", "associate", "intern"),
    "mid": ("mid", "ii", "intermediate"),
    "senior": ("senior", "sr", "lead", "staff", "principal"),
}


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9+#.]+", text.lower())
    return {w.strip(".") for w in words if w.strip(".") and w not in STOPWORDS and len(w) > 1}


def keyword_overlap(query: str, job: JobCard) -> float:
    wanted = tokens(query)
    if not wanted:
        return 0.0
    have = tokens(job.title) | {t.lower() for t in job.tags} | tokens(job.company)
    return len(wanted & have) / len(wanted)


def recency(posted_at: str | None, now: datetime | None = None, half_life_days: float = 21.0) -> float:
    if not posted_at:
        return 0.3
    try:
        when = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.3
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    age = max(0.0, ((now or datetime.now(timezone.utc)) - when).total_seconds() / 86400)
    return math.pow(0.5, age / half_life_days)


def preference_score(job: JobCard, prefs: dict[str, str]) -> float:
    if not prefs:
        return 0.0
    title = job.title.lower()
    text = f"{title} {job.company.lower()} {' '.join(job.tags).lower()}"
    score = 0.0
    parts = 0
    skills = [s.strip().lower() for s in prefs.get("skills", "").split(",") if s.strip()]
    if skills:
        parts += 1
        score += sum(1 for s in skills if s in text) / len(skills)
    if "remote" in prefs:
        parts += 1
        wants_remote = prefs["remote"].strip().lower() in ("true", "yes", "remote", "1")
        score += 1.0 if job.remote == wants_remote else 0.0
    level = prefs.get("seniority", "").strip().lower()
    if level in SENIORITY:
        parts += 1
        hits = any(re.search(rf"\b{re.escape(w)}\b", title) for w in SENIORITY[level])
        clash = level in ("intern", "junior") and any(
            re.search(rf"\b{re.escape(w)}\b", title) for w in SENIORITY["senior"]
        )
        score += 1.0 if hits else (0.0 if clash else 0.4)
    if prefs.get("role"):
        parts += 1
        score += keyword_overlap(prefs["role"], job)
    base = score / parts if parts else 0.0
    avoid = [a.strip().lower() for a in prefs.get("avoid", "").split(",") if a.strip()]
    if any(a in text for a in avoid):
        base -= 1.0
    return base


def rerank(
    candidates: list[tuple[JobCard, float]],
    query: str,
    prefs: dict[str, str] | None = None,
    limit: int = 5,
    now: datetime | None = None,
) -> list[JobCard]:
    prefs = prefs or {}
    scored: list[JobCard] = []
    seen: set[tuple[str, str]] = set()
    for job, similarity in candidates:
        key = (job.title.strip().lower(), job.company.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        score = (
            WEIGHTS["similarity"] * similarity
            + WEIGHTS["keywords"] * keyword_overlap(query, job)
            + WEIGHTS["preferences"] * preference_score(job, prefs)
            + WEIGHTS["recency"] * recency(job.posted_at, now)
        )
        scored.append(job.model_copy(update={"score": score}))
    scored.sort(key=lambda j: j.score or 0.0, reverse=True)
    return scored[:limit]
