from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from app.embeddings import Embedder
from app.models import FilterJobsArgs, GetJobArgs, JobCard, SavePreferenceArgs, SearchJobsArgs, ToolTrace
from app.ranking import rerank
from app.repository import Repository

TOOL_DESCRIPTIONS = {
    "search_jobs": (
        "Semantic search over the job index. Use it whenever the user describes what they want in words. "
        "Put hard constraints (location, remote, job type, salary, skills) in the filter fields."
    ),
    "filter_jobs": (
        "List jobs by exact filters only, without a text query. Use it for requests like 'newest remote "
        "internships' or 'highest paying python jobs'."
    ),
    "get_job_details": "Fetch the full description of one job by id before answering detailed questions about it.",
    "save_preference": (
        "Remember a stable preference the user stated about themselves (role, location, remote, min_salary, "
        "skills, seniority, job_type, avoid). Call it as soon as the user states one."
    ),
}


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        if "anyOf" in prop:
            options = [o for o in prop.pop("anyOf") if o.get("type") != "null"]
            if len(options) == 1:
                prop.update(options[0])
    return schema


class ToolBox:
    def __init__(self, repo: Repository, embedder: Embedder, session_id: str, prefs: dict[str, str]):
        self.repo = repo
        self.embedder = embedder
        self.session_id = session_id
        self.prefs = dict(prefs)
        self.seen: dict[int, JobCard] = {}
        self.trace: list[ToolTrace] = []
        self.handlers: dict[str, tuple[type[BaseModel], Callable[[Any], dict[str, Any]]]] = {
            "search_jobs": (SearchJobsArgs, self.search_jobs),
            "filter_jobs": (FilterJobsArgs, self.filter_jobs),
            "get_job_details": (GetJobArgs, self.get_job_details),
            "save_preference": (SavePreferenceArgs, self.save_preference),
        }

    def specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {"name": name, "description": TOOL_DESCRIPTIONS[name], "parameters": _schema(model)},
            }
            for name, (model, _) in self.handlers.items()
        ]

    def run(self, name: str, raw_args: str | dict | None) -> dict[str, Any]:
        started = time.perf_counter()
        if name not in self.handlers:
            return {"error": f"unknown tool '{name}'"}
        model, handler = self.handlers[name]
        try:
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
            parsed = model.model_validate(args)
        except (json.JSONDecodeError, ValidationError) as exc:
            self._log(name, {}, 0, False, started)
            return {"error": f"invalid arguments: {str(exc)[:400]}"}
        try:
            result = handler(parsed)
        except Exception as exc:
            self._log(name, parsed.model_dump(exclude_defaults=True), 0, False, started)
            return {"error": f"{name} failed: {str(exc)[:200]}"}
        self._log(name, parsed.model_dump(exclude_defaults=True), len(result.get("jobs", [])), True, started)
        return result

    def _log(self, name: str, args: dict, results: int, ok: bool, started: float) -> None:
        ms = int((time.perf_counter() - started) * 1000)
        self.trace.append(ToolTrace(tool=name, args=args, results=results, ok=ok, ms=ms))

    def _remember(self, jobs: list[JobCard]) -> list[dict[str, Any]]:
        for job in jobs:
            self.seen[job.id] = job
        return [j.brief() for j in jobs]

    def _with_pref_filters(self, args: SearchJobsArgs | FilterJobsArgs) -> SearchJobsArgs | FilterJobsArgs:
        update: dict[str, Any] = {}
        if args.remote is None and self.prefs.get("remote", "").lower() in ("true", "yes", "remote"):
            update["remote"] = True
        if args.min_salary is None and self.prefs.get("min_salary", "").isdigit():
            update["min_salary"] = int(self.prefs["min_salary"])
        return args.model_copy(update=update) if update else args

    def search_jobs(self, args: SearchJobsArgs) -> dict[str, Any]:
        args = self._with_pref_filters(args)
        text = args.query
        if args.skills:
            text += " | skills: " + ", ".join(args.skills)
        vector = self.embedder.embed([text], "RETRIEVAL_QUERY")[0]
        candidates = self.repo.semantic_candidates(vector, args, 40)
        if not candidates and (args.skills or args.min_salary or args.location):
            relaxed = args.model_copy(update={"skills": [], "min_salary": None})
            candidates = self.repo.semantic_candidates(vector, relaxed, 40)
            note = "no exact matches, constraints on skills/salary were relaxed"
        else:
            note = ""
        ranked = rerank(candidates, args.query, self.prefs, args.limit)
        out: dict[str, Any] = {"jobs": self._remember(ranked), "total_candidates": len(candidates)}
        if note:
            out["note"] = note
        return out

    def filter_jobs(self, args: FilterJobsArgs) -> dict[str, Any]:
        args = self._with_pref_filters(args)
        jobs = self.repo.filter_jobs(args, args.limit, args.sort)
        return {"jobs": self._remember(jobs)}

    def get_job_details(self, args: GetJobArgs) -> dict[str, Any]:
        found = self.repo.get_job(args.job_id)
        if not found:
            return {"error": f"job {args.job_id} not found", "jobs": []}
        job, description = found
        self.seen[job.id] = job
        return {"job": job.brief() | {"url": job.url}, "description": description[:2500], "jobs": [job.brief()]}

    def save_preference(self, args: SavePreferenceArgs) -> dict[str, Any]:
        value = args.value.strip()
        if args.key == "skills" and self.prefs.get("skills"):
            merged = {s.strip().lower() for s in (self.prefs["skills"] + "," + value).split(",") if s.strip()}
            value = ", ".join(sorted(merged))
        self.repo.save_preference(self.session_id, args.key, value)
        self.prefs[args.key] = value
        return {"saved": {args.key: value}, "preferences": self.prefs}
