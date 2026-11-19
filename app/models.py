from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

PreferenceKey = Literal["role", "location", "remote", "min_salary", "skills", "seniority", "job_type", "avoid"]


class JobCard(BaseModel):
    id: int
    title: str
    company: str
    location: str = ""
    remote: bool = False
    job_type: str = ""
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    salary_currency: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    url: str = ""
    posted_at: Optional[str] = None
    source: str = ""
    snippet: str = ""
    score: Optional[float] = None

    def brief(self) -> dict:
        data = {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "location": self.location or ("Remote" if self.remote else ""),
            "remote": self.remote,
            "job_type": self.job_type,
            "tags": self.tags[:6],
            "posted_at": (self.posted_at or "")[:10],
            "snippet": self.snippet[:180],
        }
        if self.salary_min or self.salary_max:
            data["salary"] = {
                "min": self.salary_min,
                "max": self.salary_max,
                "currency": self.salary_currency,
            }
        if self.score is not None:
            data["match_score"] = round(self.score, 3)
        return data


class Filters(BaseModel):
    model_config = ConfigDict(extra="ignore")

    location: Optional[str] = Field(None, description="City, country or region, e.g. 'India', 'Berlin', 'Europe'")
    remote: Optional[bool] = Field(None, description="true for remote-only jobs, false for on-site only")
    job_type: Optional[str] = Field(None, description="full_time, part_time, contract or internship")
    min_salary: Optional[int] = Field(None, ge=0, description="Minimum yearly salary")
    skills: list[str] = Field(default_factory=list, description="Skills or technologies, e.g. ['python', 'react']")
    posted_within_days: Optional[int] = Field(None, ge=1, le=365, description="Only jobs posted in the last N days")
    keyword: Optional[str] = Field(None, description="Exact word that must appear in the title or description")
    exclude_ids: list[int] = Field(default_factory=list, description="Job ids already shown that should be skipped")

    @field_validator("skills", mode="before")
    @classmethod
    def _split_skills(cls, value):
        if isinstance(value, str):
            value = [s for s in value.replace(";", ",").split(",")]
        return [str(s).strip().lower() for s in (value or []) if str(s).strip()]

    @field_validator("job_type", "location", "keyword", mode="before")
    @classmethod
    def _blank_to_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value


class SearchJobsArgs(Filters):
    query: str = Field(..., min_length=2, description="What the user is looking for, in natural language")
    limit: int = Field(5, ge=1, le=10)


class FilterJobsArgs(Filters):
    limit: int = Field(8, ge=1, le=15)
    sort: Literal["newest", "salary"] = "newest"


class SavePreferenceArgs(BaseModel):
    key: PreferenceKey
    value: str = Field(..., min_length=1, max_length=200)


class GetJobArgs(BaseModel):
    job_id: int


class Pick(BaseModel):
    job_id: int
    why: str = Field(..., min_length=3, max_length=300)


class AgentReply(BaseModel):
    message: str = Field(..., min_length=1, max_length=1500)
    picks: list[Pick] = Field(default_factory=list, max_length=6)
    follow_up: Optional[str] = Field(None, max_length=300)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None


class ToolTrace(BaseModel):
    tool: str
    args: dict
    results: int
    ok: bool
    ms: int


class ChatResponse(BaseModel):
    session_id: str
    reply: AgentReply
    jobs: list[JobCard]
    trace: list[ToolTrace]
    preferences: dict[str, str]
    dropped_picks: list[int] = Field(default_factory=list)
    model: Optional[str] = None
