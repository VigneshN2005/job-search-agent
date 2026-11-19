from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Protocol

from app.models import Filters, JobCard

OPEN_LOCATIONS = ("%worldwide%", "%anywhere%", "%global%")

JOB_COLUMNS = (
    "id, title, company, location, remote, job_type, salary_min, salary_max, "
    "salary_currency, tags, url, posted_at, source, left(description, 400) as snippet"
)


class Repository(Protocol):
    def semantic_candidates(self, vector: list[float], filters: Filters, limit: int) -> list[tuple[JobCard, float]]: ...
    def filter_jobs(self, filters: Filters, limit: int, sort: str) -> list[JobCard]: ...
    def get_job(self, job_id: int) -> tuple[JobCard, str] | None: ...
    def existing_ids(self, ids: list[int]) -> set[int]: ...
    def ensure_session(self, session_id: str) -> None: ...
    def load_history(self, session_id: str, limit: int) -> list[dict[str, str]]: ...
    def append_message(self, session_id: str, role: str, content: str) -> None: ...
    def load_preferences(self, session_id: str) -> dict[str, str]: ...
    def save_preference(self, session_id: str, key: str, value: str) -> None: ...
    def stats(self) -> dict[str, Any]: ...


def build_where(filters: Filters) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if filters.remote is True:
        clauses.append("remote")
    elif filters.remote is False:
        clauses.append("not remote")
    if filters.location:
        like = f"%{filters.location.strip()}%"
        if filters.remote is False:
            clauses.append("location ilike %s")
            params.append(like)
        else:
            clauses.append("(location ilike %s or (remote and location ilike any(%s)))")
            params.extend([like, list(OPEN_LOCATIONS)])
    if filters.job_type:
        clauses.append("replace(lower(job_type), '-', '_') like %s")
        params.append(f"%{filters.job_type.lower().replace('-', '_').replace(' ', '_')}%")
    if filters.min_salary:
        clauses.append("coalesce(salary_max, salary_min) >= %s")
        params.append(filters.min_salary)
    if filters.skills:
        clauses.append("(tags && %s::text[] or title ilike any(%s))")
        params.extend([filters.skills, [f"%{s}%" for s in filters.skills]])
    if filters.posted_within_days:
        clauses.append("posted_at >= now() - make_interval(days => %s)")
        params.append(filters.posted_within_days)
    if filters.keyword:
        clauses.append("(title ilike %s or description ilike %s)")
        params.extend([f"%{filters.keyword}%"] * 2)
    if filters.exclude_ids:
        clauses.append("not (id = any(%s))")
        params.append(filters.exclude_ids)
    return (" and ".join(clauses) or "true"), params


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vector) + "]"


def _card(row: dict[str, Any]) -> JobCard:
    posted = row.get("posted_at")
    return JobCard(
        id=row["id"],
        title=row["title"],
        company=row["company"],
        location=row.get("location") or "",
        remote=bool(row.get("remote")),
        job_type=row.get("job_type") or "",
        salary_min=row.get("salary_min"),
        salary_max=row.get("salary_max"),
        salary_currency=row.get("salary_currency"),
        tags=list(row.get("tags") or []),
        url=row.get("url") or "",
        posted_at=posted.isoformat() if isinstance(posted, datetime) else posted,
        source=row.get("source") or "",
        snippet=(row.get("snippet") or "").strip(),
    )


class PostgresRepository:
    def __init__(self, dsn: str):
        if not dsn:
            raise ValueError("DATABASE_URL is not set")
        self.dsn = dsn

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(self.dsn, autocommit=True, prepare_threshold=None, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                yield cur

    def semantic_candidates(
        self, vector: list[float], filters: Filters, limit: int = 40
    ) -> list[tuple[JobCard, float]]:
        where, params = build_where(filters)
        vec = vector_literal(vector)
        sql = (
            f"select {JOB_COLUMNS}, 1 - (embedding <=> %s::vector) as similarity "
            f"from jobs where embedding is not null and {where} "
            f"order by embedding <=> %s::vector limit %s"
        )
        with self._cursor() as cur:
            cur.execute(sql, [vec, *params, vec, limit])
            return [(_card(r), float(r["similarity"])) for r in cur.fetchall()]

    def filter_jobs(self, filters: Filters, limit: int = 8, sort: str = "newest") -> list[JobCard]:
        where, params = build_where(filters)
        order = (
            "coalesce(salary_max, salary_min) desc nulls last, posted_at desc nulls last"
            if sort == "salary"
            else "posted_at desc nulls last"
        )
        with self._cursor() as cur:
            cur.execute(f"select {JOB_COLUMNS} from jobs where {where} order by {order} limit %s", [*params, limit])
            return [_card(r) for r in cur.fetchall()]

    def get_job(self, job_id: int) -> tuple[JobCard, str] | None:
        with self._cursor() as cur:
            cur.execute(f"select {JOB_COLUMNS}, description from jobs where id = %s", [job_id])
            row = cur.fetchone()
        return (_card(row), row["description"]) if row else None

    def existing_ids(self, ids: list[int]) -> set[int]:
        if not ids:
            return set()
        with self._cursor() as cur:
            cur.execute("select id from jobs where id = any(%s)", [ids])
            return {r["id"] for r in cur.fetchall()}

    def ensure_session(self, session_id: str) -> None:
        with self._cursor() as cur:
            cur.execute("insert into sessions (id) values (%s) on conflict do nothing", [uuid.UUID(session_id)])

    def load_history(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        with self._cursor() as cur:
            cur.execute(
                "select role, content from (select id, role, content from messages where session_id = %s "
                "order by id desc limit %s) m order by id",
                [uuid.UUID(session_id), limit],
            )
            return [{"role": r["role"], "content": r["content"]} for r in cur.fetchall()]

    def append_message(self, session_id: str, role: str, content: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                "insert into messages (session_id, role, content) values (%s, %s, %s)",
                [uuid.UUID(session_id), role, content],
            )

    def load_preferences(self, session_id: str) -> dict[str, str]:
        with self._cursor() as cur:
            cur.execute(
                "select key, value from preferences where session_id = %s order by key", [uuid.UUID(session_id)]
            )
            return {r["key"]: r["value"] for r in cur.fetchall()}

    def save_preference(self, session_id: str, key: str, value: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                "insert into preferences (session_id, key, value) values (%s, %s, %s) "
                "on conflict (session_id, key) do update set value = excluded.value, updated_at = now()",
                [uuid.UUID(session_id), key, value],
            )

    def stats(self) -> dict[str, Any]:
        with self._cursor() as cur:
            cur.execute(
                "select count(*) as jobs, count(embedding) as indexed, count(*) filter (where remote) as remote, "
                "max(posted_at) as latest, array_agg(distinct source) as sources from jobs"
            )
            row = cur.fetchone()
        latest = row["latest"]
        return {
            "jobs": row["jobs"],
            "indexed": row["indexed"],
            "remote": row["remote"],
            "latest": latest.isoformat() if latest else None,
            "sources": [s for s in (row["sources"] or []) if s],
        }

    def upsert_jobs(self, rows: list[dict[str, Any]]) -> int:
        sql = (
            "insert into jobs (source, source_id, title, company, location, remote, job_type, salary_min, "
            "salary_max, salary_currency, tags, description, url, posted_at) values "
            "(%(source)s, %(source_id)s, %(title)s, %(company)s, %(location)s, %(remote)s, %(job_type)s, "
            "%(salary_min)s, %(salary_max)s, %(salary_currency)s, %(tags)s, %(description)s, %(url)s, %(posted_at)s) "
            "on conflict (source, source_id) do update set title = excluded.title, company = excluded.company, "
            "location = excluded.location, remote = excluded.remote, job_type = excluded.job_type, "
            "salary_min = excluded.salary_min, salary_max = excluded.salary_max, "
            "salary_currency = excluded.salary_currency, tags = excluded.tags, url = excluded.url, "
            "posted_at = excluded.posted_at, "
            "embedding = case when jobs.description = excluded.description then jobs.embedding else null end, "
            "description = excluded.description"
        )
        with self._cursor() as cur:
            cur.executemany(sql, rows)
        return len(rows)

    def jobs_missing_embeddings(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "select id, title, company, location, remote, job_type, tags, left(description, 1500) as description "
                "from jobs where embedding is null order by id limit %s",
                [limit],
            )
            return cur.fetchall()

    def set_embeddings(self, pairs: list[tuple[int, list[float]]]) -> None:
        with self._cursor() as cur:
            cur.executemany(
                "update jobs set embedding = %s::vector where id = %s",
                [(vector_literal(v), i) for i, v in pairs],
            )

    def prune_older_than(self, days: int) -> int:
        with self._cursor() as cur:
            cur.execute("delete from jobs where posted_at < now() - make_interval(days => %s)", [days])
            return cur.rowcount
