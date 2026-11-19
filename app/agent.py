from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from pydantic import ValidationError

from app.embeddings import Embedder
from app.llm import ChatModel
from app.models import AgentReply, ChatResponse
from app.repository import Repository
from app.tools import ToolBox

SYSTEM_PROMPT = """You are Scout, a job-search assistant. Today is {today}. Known user preferences: {prefs}.

Rules:
- Recommend only jobs returned by your tools in this conversation. Never invent jobs, companies or links.
- search_jobs for requests in words, filter_jobs for pure filters, get_job_details for questions about one job.
- When the user states something lasting about themselves (skills, location, remote, level, salary floor,
  things to avoid), call save_preference for it in the same step as your search.
- If the request is too vague to search (e.g. "hi"), ask one short question instead.
- One descriptive search is usually enough. Show the closest real matches even if imperfect and name the gap.
  Internships, working-student and trainee roles count as junior.

When you are done calling tools, reply with ONLY this JSON:
{{"message": "1-3 short sentences, no job list", "picks": [{{"job_id": 123, "why": "one sentence"}}],
"follow_up": "one short question or null"}}
picks: up to 5 ids from tool results, best first; [] only if nothing was searched or all results are unrelated."""

FINAL_PROMPT = """Answer the user's request using ONLY the candidate jobs below. Reply with ONLY a JSON object:
{{"message": string, "picks": [{{"job_id": integer, "why": string}}], "follow_up": string or null}}
"message": 1-3 short sentences, no job list. "picks": up to 5 of these ids, best first: {ids}. Include partial
matches and name the gap in "why"; use [] only if every candidate is unrelated. "follow_up": one short question
or null."""


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


class Agent:
    def __init__(
        self,
        llm: ChatModel,
        repo: Repository,
        embedder: Embedder,
        history_limit: int = 12,
        max_tool_rounds: int = 4,
    ):
        self.llm = llm
        self.repo = repo
        self.embedder = embedder
        self.history_limit = history_limit
        self.max_tool_rounds = max_tool_rounds

    def _system(self, prefs: dict[str, str]) -> dict[str, str]:
        pref_text = ", ".join(f"{k}: {v}" for k, v in prefs.items()) or "none yet"
        return {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat(), prefs=pref_text)}

    def run(self, session_id: str, user_message: str) -> ChatResponse:
        self.repo.ensure_session(session_id)
        prefs = self.repo.load_preferences(session_id)
        history = self.repo.load_history(session_id, self.history_limit)
        tools = ToolBox(self.repo, self.embedder, session_id, prefs)

        messages: list[dict[str, Any]] = [self._system(prefs), *history, {"role": "user", "content": user_message}]
        specs = tools.specs()
        final_text = ""

        for _ in range(self.max_tool_rounds):
            reply = self.llm.chat(messages, tools=specs)
            calls = reply.get("tool_calls") or []
            if not calls:
                final_text = reply.get("content") or ""
                break
            messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls})
            for call in calls:
                fn = call.get("function", {})
                result = tools.run(fn.get("name", ""), fn.get("arguments"))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(result, default=str, separators=(",", ":"))[:3000],
                    }
                )

        final, dropped = self._parse(final_text, tools)
        if final is None:
            final, dropped = self._finalize(user_message, tools)
        jobs = [tools.seen[p.job_id] for p in final.picks]

        self.repo.append_message(session_id, "user", user_message)
        summary = final.message
        if jobs:
            summary += "\nShown: " + "; ".join(f"#{j.id} {j.title} at {j.company}" for j in jobs)
        self.repo.append_message(session_id, "assistant", summary)

        return ChatResponse(
            session_id=session_id,
            reply=final,
            jobs=jobs,
            trace=tools.trace,
            preferences=tools.prefs,
            dropped_picks=dropped,
            model=getattr(self.llm, "last_model", None),
        )

    def _parse(self, text: str, tools: ToolBox) -> tuple[AgentReply | None, list[int]]:
        try:
            reply = AgentReply.model_validate(_extract_json(text))
        except (json.JSONDecodeError, ValidationError):
            return None, []
        grounded, dropped = self._ground(reply, tools)
        if tools.seen and not grounded.picks:
            return None, []
        return grounded, dropped

    def _finalize(self, user_message: str, tools: ToolBox) -> tuple[AgentReply, list[int]]:
        ids = sorted(tools.seen)
        candidates = [j.brief() | {"snippet": j.snippet[:160]} for j in tools.seen.values()]
        context = (
            f"User request: {user_message}\nPreferences: {json.dumps(tools.prefs)}\n"
            f"Candidate jobs: {json.dumps(candidates, separators=(',', ':'))}"
        )
        convo = [
            {"role": "system", "content": FINAL_PROMPT.format(ids=ids or "none")},
            {"role": "user", "content": context[:7000]},
        ]
        for _ in range(2):
            raw = self.llm.chat(convo, json_mode=True).get("content") or ""
            try:
                reply = AgentReply.model_validate(_extract_json(raw))
            except (json.JSONDecodeError, ValidationError) as exc:
                convo = [*convo, {"role": "user", "content": f"That output was invalid: {str(exc)[:200]}. Fix it."}]
                continue
            return self._ground(reply, tools)
        fallback = AgentReply(
            message="Sorry, I couldn't put together a proper answer just now. Could you rephrase that?",
            picks=[],
            follow_up=None,
        )
        return fallback, []

    def _ground(self, reply: AgentReply, tools: ToolBox) -> tuple[AgentReply, list[int]]:
        picks, used, dropped = [], set(), []
        for pick in reply.picks:
            if pick.job_id in tools.seen and pick.job_id not in used:
                picks.append(pick)
                used.add(pick.job_id)
            elif pick.job_id not in tools.seen:
                dropped.append(pick.job_id)
        return reply.model_copy(update={"picks": picks}), dropped
