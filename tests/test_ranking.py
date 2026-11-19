from datetime import datetime, timedelta, timezone

from app.ranking import keyword_overlap, preference_score, recency, rerank
from tests.fakes import make_job


def test_keyword_overlap_ignores_stopwords():
    job = make_job(1, "Python Backend Engineer", tags=["fastapi"])
    assert keyword_overlap("looking for a python fastapi job", job) == 1.0
    assert keyword_overlap("the and for", job) == 0.0


def test_recency_halves_every_half_life():
    now = datetime(2026, 1, 31, tzinfo=timezone.utc)
    fresh = recency(now.isoformat(), now)
    old = recency((now - timedelta(days=21)).isoformat(), now)
    assert fresh == 1.0
    assert abs(old - 0.5) < 1e-6
    assert recency(None, now) == 0.3


def test_preferences_reward_skills_and_penalise_avoid():
    job = make_job(1, "Python Developer", company="BigBank", tags=["python", "sql"], remote=True)
    assert preference_score(job, {"skills": "python, sql", "remote": "true"}) == 1.0
    assert preference_score(job, {"avoid": "bigbank"}) < 0


def test_seniority_preference_demotes_senior_roles_for_interns():
    intern = make_job(1, "Software Engineering Intern")
    senior = make_job(2, "Senior Software Engineer")
    prefs = {"seniority": "intern"}
    assert preference_score(intern, prefs) > preference_score(senior, prefs)


def test_rerank_dedupes_and_orders_by_combined_score():
    a = make_job(1, "Python Engineer", "Orbit", tags=["python"])
    dup = make_job(2, "Python Engineer", "orbit", tags=["python"])
    b = make_job(3, "Office Manager", "Orbit")
    ranked = rerank([(b, 0.9), (a, 0.8), (dup, 0.8)], "python engineer", limit=5)
    assert [j.id for j in ranked] == [1, 3]
    assert ranked[0].score > ranked[1].score


def test_rerank_respects_limit():
    jobs = [(make_job(i, f"Role {i}", company=f"C{i}"), 0.5) for i in range(10)]
    assert len(rerank(jobs, "role", limit=3)) == 3
