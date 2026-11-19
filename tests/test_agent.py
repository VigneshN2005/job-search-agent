import json
import uuid

from app.agent import Agent
from tests.fakes import FakeEmbedder, FakeRepo, ScriptedLLM, make_job, tool_call

JOBS = [
    make_job(1, "Backend Engineer (Python)", "Orbit", remote=True, location="Worldwide", tags=["python", "fastapi"]),
    make_job(2, "Junior Python Developer", "Lumen", remote=True, location="India", tags=["python", "django"]),
    make_job(3, "Senior Java Engineer", "Granite", remote=False, location="Berlin", tags=["java", "spring"]),
    make_job(4, "Frontend Developer", "Pixel", remote=True, location="Europe", tags=["react", "typescript"]),
]


def final(message: str, picks: list[tuple[int, str]], follow_up=None) -> dict:
    body = {"message": message, "picks": [{"job_id": i, "why": w} for i, w in picks], "follow_up": follow_up}
    return {"content": json.dumps(body)}


def new_agent(replies):
    repo = FakeRepo(JOBS)
    llm = ScriptedLLM(replies)
    return Agent(llm=llm, repo=repo, embedder=FakeEmbedder()), repo, llm


def test_search_then_answer_returns_only_grounded_jobs():
    agent, repo, llm = new_agent(
        [
            tool_call("search_jobs", {"query": "python backend", "remote": True, "skills": ["python"]}),
            {"content": "Here are some options."},
            final("Two remote Python roles fit.", [(1, "Python + FastAPI"), (2, "Python, junior"), (99, "made up")]),
        ]
    )
    sid = str(uuid.uuid4())
    res = agent.run(sid, "remote python backend jobs")
    assert [j.id for j in res.jobs] == [1, 2]
    assert res.dropped_picks == [99]
    assert res.trace[0].tool == "search_jobs" and res.trace[0].ok
    assert repo.last_filters.remote is True
    assert llm.requests[-1]["json_mode"] is True


def test_preferences_are_saved_and_reused_in_next_turn():
    agent, repo, llm = new_agent(
        [
            tool_call("save_preference", {"key": "skills", "value": "python, sql"}, "c1"),
            {"content": "Saved."},
            final("Noted your skills.", []),
            tool_call("search_jobs", {"query": "developer jobs"}, "c2"),
            {"content": "ok"},
            final("Here you go.", [(2, "Matches python")]),
        ]
    )
    sid = str(uuid.uuid4())
    agent.run(sid, "I know python and sql")
    assert repo.prefs[sid]["skills"] == "python, sql"
    res = agent.run(sid, "show me something")
    system = llm.requests[3]["messages"][0]["content"]
    assert "skills: python, sql" in system
    history_roles = [m["role"] for m in llm.requests[3]["messages"][1:3]]
    assert history_roles == ["user", "assistant"]
    assert res.preferences["skills"] == "python, sql"


def test_invalid_json_is_retried_once():
    agent, _, llm = new_agent(
        [
            tool_call("search_jobs", {"query": "frontend react"}),
            {"content": "done"},
            {"content": "not json at all"},
            final("One frontend role.", [(4, "React")]),
        ]
    )
    res = agent.run(str(uuid.uuid4()), "frontend react jobs")
    assert [j.id for j in res.jobs] == [4]
    assert "invalid" in llm.requests[-1]["messages"][-1]["content"]


def test_two_bad_outputs_fall_back_gracefully():
    agent, _, _ = new_agent([{"content": "hi"}, {"content": "{}"}, {"content": '{"picks": 3}'}])
    res = agent.run(str(uuid.uuid4()), "hello")
    assert res.jobs == []
    assert "rephrase" in res.reply.message


def test_bad_tool_arguments_are_reported_not_raised():
    agent, _, llm = new_agent(
        [
            tool_call("search_jobs", {"limit": 3}),
            {"content": "sorry"},
            final("I need a bit more detail.", [], "What role are you after?"),
        ]
    )
    res = agent.run(str(uuid.uuid4()), "jobs")
    tool_msg = next(m for m in llm.requests[1]["messages"] if m["role"] == "tool")
    assert "invalid arguments" in tool_msg["content"]
    assert res.trace[0].ok is False
    assert res.reply.follow_up == "What role are you after?"


def test_tool_loop_is_bounded():
    loops = [tool_call("filter_jobs", {"remote": True}, f"c{i}") for i in range(10)]
    agent, _, llm = new_agent(loops)
    agent.max_tool_rounds = 3
    llm.replies = loops[:3] + [final("Remote roles.", [(1, "remote"), (4, "remote")])]
    res = agent.run(str(uuid.uuid4()), "remote jobs")
    assert len(res.trace) == 3
    assert {j.id for j in res.jobs} == {1, 4}


def test_history_is_written_with_shown_jobs():
    agent, repo, _ = new_agent(
        [
            tool_call("get_job_details", {"job_id": 3}),
            {"content": "details"},
            final("It is an on-site Java role in Berlin.", [(3, "The one you asked about")]),
        ]
    )
    sid = str(uuid.uuid4())
    agent.run(sid, "tell me about job 3")
    assert repo.history[sid][-1]["role"] == "assistant"
    assert "#3 Senior Java Engineer at Granite" in repo.history[sid][-1]["content"]


def test_final_step_sends_no_raw_tool_messages():
    agent, _, llm = new_agent(
        [
            tool_call("search_jobs", {"query": "python"}),
            {"content": "ok"},
            final("Found one.", [(1, "python")]),
        ]
    )
    agent.run(str(uuid.uuid4()), "python jobs")
    last = llm.requests[-1]["messages"]
    assert all(m["role"] != "tool" and "tool_calls" not in m for m in last)
    assert any("Candidate jobs" in m["content"] and "Backend Engineer" in m["content"] for m in last)


def test_answer_inside_tool_loop_skips_extra_call():
    agent, _, llm = new_agent(
        [
            tool_call("search_jobs", {"query": "react frontend"}),
            final("One React role.", [(4, "React and TypeScript")]),
        ]
    )
    res = agent.run(str(uuid.uuid4()), "react jobs")
    assert [j.id for j in res.jobs] == [4]
    assert len(llm.requests) == 2


def test_clarifying_question_without_search_is_accepted():
    agent, _, llm = new_agent([final("Happy to help! What kind of role are you after?", [], "Which role?")])
    res = agent.run(str(uuid.uuid4()), "hi")
    assert res.jobs == [] and res.reply.follow_up == "Which role?"
    assert len(llm.requests) == 1


def test_empty_picks_after_search_triggers_compact_final_step():
    agent, _, llm = new_agent(
        [
            tool_call("search_jobs", {"query": "python"}),
            final("Nothing matched.", []),
            final("Closest match.", [(2, "Python, junior")]),
        ]
    )
    res = agent.run(str(uuid.uuid4()), "junior python jobs")
    assert [j.id for j in res.jobs] == [2]
    assert llm.requests[-1]["json_mode"] is True
