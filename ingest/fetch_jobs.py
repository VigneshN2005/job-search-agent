from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from app.config import get_settings
from app.embeddings import GeminiEmbedder
from app.http import HttpError, request_json
from app.repository import PostgresRepository
from ingest.normalize import embedding_text, from_arbeitnow, from_remotive, is_valid

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
TECH_CATEGORIES = ("software", "data", "devops", "sysadmin", "qa", "product", "machine learning", "ai")
ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_remotive(limit: int) -> list[dict[str, Any]]:
    try:
        data = request_json(f"{REMOTIVE_URL}?limit={limit}", timeout=90)
    except HttpError as exc:
        log(f"  remotive: skipped ({exc.status})")
        return []
    jobs = data.get("jobs", [])
    rows: dict[str, dict[str, Any]] = {}
    for item in jobs:
        category = (item.get("category") or "").lower()
        if not any(c in category for c in TECH_CATEGORIES):
            continue
        row = from_remotive(item)
        if is_valid(row):
            rows[row["source_id"]] = row
    log(f"  remotive: {len(jobs)} fetched, {len(rows)} tech roles kept")
    return list(rows.values())


def fetch_arbeitnow(pages: int) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for page in range(1, pages + 1):
        try:
            data = request_json(f"{ARBEITNOW_URL}?page={page}", timeout=60)
        except HttpError as exc:
            log(f"  arbeitnow/page {page}: stopped ({exc.status})")
            break
        items = data.get("data", [])
        for item in items:
            row = from_arbeitnow(item)
            if is_valid(row):
                rows[row["source_id"]] = row
        log(f"  arbeitnow/page {page}: {len(items)}")
        if not items or not (data.get("links") or {}).get("next"):
            break
        time.sleep(1)
    return list(rows.values())


def embed_missing(repo: PostgresRepository, embedder: GeminiEmbedder, batch: int = 100) -> int:
    total = 0
    while True:
        pending = repo.jobs_missing_embeddings(batch)
        if not pending:
            return total
        vectors = embedder.embed([embedding_text(r) for r in pending], "RETRIEVAL_DOCUMENT")
        repo.set_embeddings([(r["id"], v) for r, v in zip(pending, vectors, strict=True)])
        total += len(pending)
        log(f"  embedded {total}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch real job postings, store them and embed them.")
    parser.add_argument("--remotive-limit", type=int, default=1000)
    parser.add_argument("--arbeitnow-pages", type=int, default=4)
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--prune-days", type=int, default=0)
    args = parser.parse_args(argv)

    settings = get_settings()
    repo = PostgresRepository(settings.database_url)
    embedder = GeminiEmbedder(settings.gemini_api_key, settings.embedding_model, settings.embedding_dim)

    if not args.embed_only:
        log("Fetching Remotive")
        remotive = fetch_remotive(args.remotive_limit)
        log("Fetching Arbeitnow")
        arbeitnow = fetch_arbeitnow(args.arbeitnow_pages)
        rows = remotive + arbeitnow
        if rows:
            repo.upsert_jobs(rows)
        log(f"Stored {len(rows)} jobs ({len(remotive)} Remotive, {len(arbeitnow)} Arbeitnow)")
        if args.prune_days:
            log(f"Pruned {repo.prune_older_than(args.prune_days)} jobs older than {args.prune_days} days")

    log("Embedding new or changed jobs")
    count = embed_missing(repo, embedder)
    stats = repo.stats()
    log(f"Done. Embedded {count} now; index has {stats['indexed']}/{stats['jobs']} jobs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
