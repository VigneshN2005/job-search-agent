import importlib.util
from pathlib import Path

from tests.fakes import make_job

spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).parent.parent / "evals" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)


def test_check_job_flags_each_violation():
    job = make_job(1, "Crypto Trader", location="Berlin", remote=False, tags=["web3"])
    problems = run_eval.check_job(
        job, {"remote": True, "location_any": ["india"], "any_terms": ["python"], "none_terms": ["crypto"]}
    )
    assert len(problems) == 4


def test_check_job_passes_matching_job():
    job = make_job(2, "Python Backend Engineer", location="Worldwide", remote=True, tags=["python"])
    assert run_eval.check_job(job, {"remote": True, "any_terms": ["python"]}) == []


def test_summary_counts():
    results = [
        {
            "passed": True,
            "precision": 1.0,
            "fabricated_in_output": [],
            "dropped_hallucinations": 1,
            "latency_s": 2.0,
        },
        {"error": "x", "passed": False},
    ]
    s = run_eval.summarise(results)
    assert s["passed"] == 1 and s["errors"] == 1 and s["hallucinated_picks_blocked"] == 1
