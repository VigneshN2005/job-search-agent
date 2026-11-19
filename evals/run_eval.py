from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent import Agent
from app.config import get_settings
from app.embeddings import GeminiEmbedder
from app.llm import GroqClient, LLMError
from app.models import ChatResponse, JobCard
from app.repository import PostgresRepository

HERE = Path(__file__).resolve().parent


def job_text(job: JobCard) -> str:
    return f"{job.title} {job.company} {' '.join(job.tags)} {job.snippet}".lower()


def check_job(job: JobCard, expect: dict[str, Any]) -> list[str]:
    problems = []
    text = job_text(job)
    if expect.get("remote") is True and not job.remote:
        problems.append(f"#{job.id} is not remote")
    if expect.get("remote") is False and job.remote:
        problems.append(f"#{job.id} is remote")
    if expect.get("location_any") and not any(loc in job.location.lower() for loc in expect["location_any"]):
        problems.append(f"#{job.id} location '{job.location}'")
    if expect.get("any_terms") and not any(t in text for t in expect["any_terms"]):
        problems.append(f"#{job.id} off-topic '{job.title}'")
    if expect.get("none_terms") and any(t in text for t in expect["none_terms"]):
        problems.append(f"#{job.id} contains avoided term")
    return problems


def evaluate(case: dict[str, Any], agent: Agent, repo: PostgresRepository) -> dict[str, Any]:
    session = str(uuid.uuid4())
    expect = case["expect"]
    responses: list[ChatResponse] = []
    started = time.perf_counter()
    try:
        for turn in case["turns"]:
            responses.append(agent.run(session, turn))
    except LLMError as exc:
        return {"id": case["id"], "error": str(exc)[:200], "passed": False}
    latency = (time.perf_counter() - started) / len(case["turns"])

    last = responses[-1]
    tools_used = {t.tool for r in responses for t in r.trace}
    picked_ids = [j.id for j in last.jobs]
    real_ids = repo.existing_ids(picked_ids)
    fabricated = [i for i in picked_ids if i not in real_ids]
    dropped = sum(len(r.dropped_picks) for r in responses)

    checks: dict[str, bool] = {"grounded": not fabricated}
    job_problems: list[str] = []
    for job in last.jobs:
        job_problems += check_job(job, expect)
    relevant = sum(1 for j in last.jobs if not check_job(j, expect))
    precision = relevant / len(last.jobs) if last.jobs else None

    if "min_picks" in expect:
        checks["has_results"] = len(last.jobs) >= expect["min_picks"]
    if "max_picks" in expect:
        checks["no_results"] = len(last.jobs) <= expect["max_picks"]
    if expect.get("follow_up"):
        checks["asks_follow_up"] = bool(last.reply.follow_up) or "?" in last.reply.message
    if expect.get("saved_prefs"):
        checks["saved_prefs"] = all(k in last.preferences for k in expect["saved_prefs"])
    if expect.get("tools_any"):
        checks["used_tool"] = bool(tools_used & set(expect["tools_any"]))
    if precision is not None:
        checks["precision>=0.6"] = precision >= 0.6

    return {
        "id": case["id"],
        "turns": case["turns"],
        "reply": last.reply.message,
        "follow_up": last.reply.follow_up,
        "jobs": [{"id": j.id, "title": j.title, "company": j.company, "location": j.location} for j in last.jobs],
        "tools": sorted(tools_used),
        "preferences": last.preferences,
        "precision": precision,
        "dropped_hallucinations": dropped,
        "fabricated_in_output": fabricated,
        "problems": job_problems,
        "checks": checks,
        "passed": all(checks.values()),
        "latency_s": round(latency, 2),
        "model": last.model,
    }


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    precisions = [r["precision"] for r in ok if r.get("precision") is not None]
    return {
        "cases": len(results),
        "passed": sum(r["passed"] for r in results),
        "errors": len(results) - len(ok),
        "mean_precision": round(statistics.mean(precisions), 3) if precisions else None,
        "fabricated_jobs_shown": sum(len(r["fabricated_in_output"]) for r in ok),
        "hallucinated_picks_blocked": sum(r["dropped_hallucinations"] for r in ok),
        "mean_latency_s": round(statistics.mean(r["latency_s"] for r in ok), 2) if ok else None,
    }


def write_report(summary: dict[str, Any], results: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Evaluation results",
        "",
        f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Cases passed | {summary['passed']} / {summary['cases']} |",
        f"| Mean precision of recommended jobs | {summary['mean_precision']} |",
        f"| Fabricated jobs shown to the user | {summary['fabricated_jobs_shown']} |",
        f"| Hallucinated picks caught and removed | {summary['hallucinated_picks_blocked']} |",
        f"| Mean latency per turn (s) | {summary['mean_latency_s']} |",
        "",
        "| Case | Pass | Precision | Tools | Notes |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['id']} | error | - | - | {r['error'][:80]} |")
            continue
        failed = [k for k, v in r["checks"].items() if not v]
        prec = "-" if r["precision"] is None else f"{r['precision']:.2f}"
        note = ", ".join(failed + r["problems"][:2])[:120] or "ok"
        lines.append(f"| {r['id']} | {'yes' if r['passed'] else 'no'} | {prec} | {', '.join(r['tools'])} | {note} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agent against the evaluation cases.")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--pause", type=float, default=20.0, help="seconds between cases (rate limits)")
    args = parser.parse_args(argv)

    s = get_settings()
    repo = PostgresRepository(s.database_url)
    agent = Agent(
        llm=GroqClient(s.groq_api_key, s.groq_models),
        repo=repo,
        embedder=GeminiEmbedder(s.gemini_api_key, s.embedding_model, s.embedding_dim),
        history_limit=s.history_limit,
        max_tool_rounds=s.max_tool_rounds,
    )
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]

    results = []
    for case in cases:
        result = evaluate(case, agent, repo)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status}  {case['id']:<22} {result.get('latency_s', '-')}s  {result.get('problems') or ''}", flush=True)
        time.sleep(args.pause)

    summary = summarise(results)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    (out_dir / f"run-{stamp}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    write_report(summary, results, HERE / "RESULTS.md")
    print(json.dumps(summary, indent=2))
    return 0 if summary["fabricated_jobs_shown"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
