from app.tools import ToolBox
from tests.fakes import FakeEmbedder, FakeRepo, make_job

JOBS = [
    make_job(1, "Python Developer", remote=True, location="Worldwide", tags=["python"], salary_max=90000),
    make_job(2, "Java Developer", remote=False, location="Pune", tags=["java"]),
]


def box(prefs=None):
    return ToolBox(FakeRepo(JOBS), FakeEmbedder(), "s1", prefs or {})


def test_specs_are_valid_openai_tool_definitions():
    specs = box().specs()
    names = {s["function"]["name"] for s in specs}
    assert names == {"search_jobs", "filter_jobs", "get_job_details", "save_preference"}
    search = next(s for s in specs if s["function"]["name"] == "search_jobs")["function"]["parameters"]
    assert search["required"] == ["query"]
    assert search["properties"]["remote"]["type"] == "boolean"
    assert "anyOf" not in search["properties"]["location"]


def test_search_records_seen_jobs_and_trace():
    tb = box()
    out = tb.run("search_jobs", '{"query": "python developer", "remote": true}')
    assert [j["id"] for j in out["jobs"]] == [1]
    assert 1 in tb.seen
    assert tb.trace[0].results == 1 and tb.trace[0].ok


def test_search_relaxes_impossible_constraints():
    tb = box()
    out = tb.run("search_jobs", {"query": "python", "skills": ["rust"]})
    assert out.get("note") and out["jobs"]


def test_saved_remote_preference_becomes_a_filter():
    tb = box({"remote": "true", "min_salary": "80000"})
    tb.run("filter_jobs", {})
    assert tb.repo.last_filters.remote is True
    assert tb.repo.last_filters.min_salary == 80000


def test_save_preference_merges_skills():
    tb = box({"skills": "python"})
    out = tb.run("save_preference", {"key": "skills", "value": "SQL, python"})
    assert out["saved"]["skills"] == "python, sql"


def test_invalid_and_unknown_tools_return_errors():
    tb = box()
    assert "invalid arguments" in tb.run("save_preference", {"key": "salary", "value": "x"})["error"]
    assert "unknown tool" in tb.run("delete_everything", {})["error"]
    assert "invalid arguments" in tb.run("search_jobs", "{not json")["error"]


def test_get_job_details_includes_description():
    out = box().run("get_job_details", {"job_id": 2})
    assert out["description"].startswith("Full description")
    assert box().run("get_job_details", {"job_id": 77})["error"]
