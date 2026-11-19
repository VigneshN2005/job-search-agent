from app.models import Filters
from app.repository import build_where, vector_literal


def test_empty_filters_match_everything():
    assert build_where(Filters()) == ("true", [])


def test_remote_with_location_also_accepts_worldwide_roles():
    where, params = build_where(Filters(location="India", remote=True))
    assert where.startswith("remote and (location ilike %s or (remote and location ilike any(%s)))")
    assert params[0] == "%India%"
    assert "%worldwide%" in params[1]


def test_onsite_location_is_strict():
    where, params = build_where(Filters(location="Berlin", remote=False))
    assert where == "not remote and location ilike %s"
    assert params == ["%Berlin%"]


def test_skills_salary_and_exclusions_are_parameterised():
    where, params = build_where(Filters(skills=["Python", " SQL "], min_salary=50000, exclude_ids=[4, 5]))
    assert "tags && %s::text[]" in where
    assert "coalesce(salary_max, salary_min) >= %s" in where
    assert "not (id = any(%s))" in where
    assert ["python", "sql"] in params
    assert 50000 in params and [4, 5] in params
    assert "Python" not in where


def test_job_type_is_normalised():
    _, params = build_where(Filters(job_type="Full-Time"))
    assert params == ["%full_time%"]


def test_skills_accept_comma_string():
    assert Filters(skills="react, Node").skills == ["react", "node"]


def test_vector_literal_format():
    assert vector_literal([0.5, -0.25]) == "[0.500000,-0.250000]"
