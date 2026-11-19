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

SYSTEM_PROMPT = """You are Scout, a job-search assistant. Today is {today}.
You help the user find real job postings from the job index through conversation.

Rules:
- Only recommend jobs returned by your tools in this conversation. Never invent jobs, companies, salaries or links.
- Use search_jobs for anything described in words, filter_jobs for pure filter requests, get_job_details before
  answering detailed questions about one job.
- When the user states something lasting about themselves (skills, preferred location, remote, seniority,
  salary floor, things to avoid), call save_preference for each item, then continue with the search.
- If the request is too vague to search (for example just "hi"), ask one short question instead of searching.
- If nothing matches, say so plainly and suggest how to widen the search.
- Keep answers short and concrete.

Known preferences for this user: {prefs}"""

FINAL_PROMPT = """Write your final answer to the user's last message as a JSON object with exactly these keys:
{{"message": string, "picks": [{{"job_id": integer, "why": string}}], "follow_up": string or null}}

- "message": 1-3 short sentences addressed to the user. Do not list the jobs in it.
- "picks": up to 5 jobs, best first, chosen ONLY from these ids returned by tools: {ids}.
  "why" is one sentence linking the job to what the user asked. Use [] if nothing fits or no search was needed.
- "follow_up": one short question that would improve the next search, or null.
Return only the JSON object."""


def _flatten(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = c.get("function", {})
                names[c.get("id", "")] = fn.get("name", "tool")
                calls.append(f"{fn.get('name')}({fn.get('arguments')})")
            out.append({"role": "assistant", "content": "Called tools: " + "; ".join(calls)})
        elif m["role"] == "tool":
            name = names.get(m.get("tool_call_id", ""), "tool")
            out.append({"role": "system", "content": f"Result of {name}: {m['content']}"})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


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

        for _ in range(self.max_tool_rounds):
            reply = self.llm.chat(messages, tools=specs)
            calls = reply.get("tool_calls") or []
            if not calls:
                break
            messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls})
            for call in calls:
                fn = call.get("function", {})
                result = tools.run(fn.get("name", ""), fn.get("arguments"))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(result, default=str)[:6000],
                    }
                )

        final, dropped = self._finalize(messages, tools)
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

    def _finalize(self, messages: list[dict[str, Any]], tools: ToolBox) -> tuple[AgentReply, list[int]]:
        ids = sorted(tools.seen)
        prompt = {"role": "system", "content": FINAL_PROMPT.format(ids=ids or "none")}
        convo = [*_flatten(messages), prompt]
        error = ""
        for _ in range(2):
            if error:
                convo = [*convo, {"role": "system", "content": f"Your last output was invalid: {error}. Fix it."}]
            raw = self.llm.chat(convo, json_mode=True).get("content") or ""
            try:
                reply = AgentReply.model_validate(_extract_json(raw))
            except (json.JSONDecodeError, ValidationError) as exc:
                error = str(exc)[:300]
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
