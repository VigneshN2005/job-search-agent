from ingest.normalize import (
    detect_job_type,
    embedding_text,
    from_arbeitnow,
    from_remotive,
    html_to_text,
    is_tech_role,
    is_valid,
    parse_salary,
)


def test_html_is_flattened_to_readable_text():
    text = html_to_text("<p>Hello&nbsp;<b>world</b></p><ul><li>Python</li><li>SQL</li></ul>")
    assert "Hello world" in text
    assert "- Python" in text and "- SQL" in text
    assert "<" not in text


def test_salary_ranges_are_parsed():
    assert parse_salary("$80k - $100k") == (80000, 100000, "USD")
    assert parse_salary("€55,000 – €70,000") == (55000, 70000, "EUR")
    assert parse_salary("$40 - $60 per hour") == (83200, 124800, "USD")
    assert parse_salary("Competitive") == (None, None, None)
    assert parse_salary("") == (None, None, None)


def test_job_type_detection():
    assert detect_job_type("full_time") == "full_time"
    assert detect_job_type("Werkstudent (m/w/d)") == "internship"
    assert detect_job_type("Software Engineering Intern") == "internship"
    assert detect_job_type(None, "") == ""


def test_remotive_mapping():
    row = from_remotive(
        {
            "id": 42,
            "url": "https://remotive.com/x",
            "title": " Data Engineer ",
            "company_name": "Nimbus",
            "category": "Data",
            "tags": ["Python", "python", "Airflow"],
            "job_type": "full_time",
            "publication_date": "2026-11-01T10:00:00",
            "candidate_required_location": "India",
            "salary": "$60k - $90k",
            "description": "<p>Build pipelines</p>",
        }
    )
    assert row["source_id"] == "42" and row["remote"] is True
    assert row["title"] == "Data Engineer"
    assert row["tags"] == ["python", "airflow", "data"]
    assert row["salary_min"] == 60000 and row["posted_at"].startswith("2026-11-01T10:00:00")
    assert is_valid(row)


def test_arbeitnow_mapping_and_remote_detection():
    row = from_arbeitnow(
        {
            "slug": "backend-dev-berlin",
            "company_name": "Kraft",
            "title": "Backend Developer",
            "description": "<p>Go and Postgres</p>",
            "remote": False,
            "url": "https://arbeitnow.com/view/backend-dev-berlin",
            "tags": ["Go"],
            "job_types": ["full time"],
            "location": "Berlin",
            "created_at": 1763500000,
        }
    )
    assert row["remote"] is False and row["location"] == "Berlin"
    assert row["job_type"] == "full_time"
    assert row["posted_at"].startswith("2025")
    assert from_arbeitnow({"slug": "x", "title": "T", "company_name": "C", "url": "u", "location": "Remote"})["remote"]


def test_embedding_text_skips_empty_fields():
    text = embedding_text({"title": "QA Engineer", "company": "Zen", "tags": [], "description": "Test things"})
    assert text.startswith("QA Engineer\nCompany: Zen")
    assert "Tags:" not in text


def test_tech_role_filter():
    assert is_tech_role({"title": "Senior Backend Developer", "tags": []})
    assert is_tech_role({"title": "Werkstudent IT-Support", "tags": []})
    assert is_tech_role({"title": "Sachbearbeiter", "tags": ["data analysis"]})
    assert not is_tech_role({"title": "Verkäufer im Einzelhandel", "tags": ["retail"]})
    assert not is_tech_role({"title": "Pflegefachkraft", "tags": []})
