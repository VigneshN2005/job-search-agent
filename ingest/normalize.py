from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section"}
REMOTE_WORDS = ("remote", "worldwide", "anywhere", "work from home", "wfh")
JOB_TYPES = {
    "full_time": ("full_time", "full-time", "full time", "permanent", "berufserfahren"),
    "part_time": ("part_time", "part-time", "part time", "teilzeit"),
    "contract": ("contract", "freelance", "contractor", "temporary"),
    "internship": ("internship", "intern", "praktikum", "werkstudent", "working student"),
}
CURRENCIES = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}
TECH_WORDS = (
    "develop",
    "engineer",
    "software",
    "programm",
    "data",
    "devops",
    "sre",
    "cloud",
    "backend",
    "back-end",
    "frontend",
    "front-end",
    "full stack",
    "fullstack",
    "full-stack",
    "web",
    "mobile",
    "ios",
    "android",
    "machine learning",
    "ml ",
    "ai ",
    " ai",
    "llm",
    "python",
    "java",
    "javascript",
    "typescript",
    "react",
    "node",
    "golang",
    " go ",
    "rust",
    "php",
    ".net",
    "c#",
    "c++",
    "qa",
    "test",
    "security",
    "cyber",
    "infrastructure",
    "platform",
    "architect",
    "analyst",
    "scientist",
    "it-",
    "it ",
    "informatik",
    "entwickler",
    "administrator",
    "sysadmin",
    "database",
    "sql",
    "product manager",
    "ux",
    "ui ",
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(raw: str | None) -> str:
    if not raw:
        return ""
    parser = _TextExtractor()
    parser.feed(raw)
    text = html.unescape("".join(parser.parts)).replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def normalize_tags(values: Any) -> list[str]:
    out: list[str] = []
    for value in values or []:
        tag = re.sub(r"\s+", " ", str(value)).strip().lower()
        if tag and tag not in out and len(tag) <= 40:
            out.append(tag)
    return out[:20]


def detect_job_type(*values: Any) -> str:
    text = " ".join(str(v) for v in values if v).lower()
    for name, words in JOB_TYPES.items():
        if any(w in text for w in words):
            return name
    return ""


def is_remote(location: str, flag: Any = None) -> bool:
    if flag is True:
        return True
    return any(w in (location or "").lower() for w in REMOTE_WORDS)


def _amount(number: str, suffix: str) -> int | None:
    try:
        value = float(number.replace(",", ""))
    except ValueError:
        return None
    if suffix.lower() == "k":
        value *= 1000
    elif suffix.lower() in ("l", "lpa", "lakh", "lakhs"):
        value *= 100000
    return int(value)


def parse_salary(text: str | None) -> tuple[int | None, int | None, str | None]:
    if not text:
        return None, None, None
    raw = text.strip()
    currency = next((code for sym, code in CURRENCIES.items() if sym in raw), None)
    if currency is None:
        match = re.search(r"\b(USD|EUR|GBP|INR|CAD|AUD)\b", raw, re.I)
        currency = match.group(1).upper() if match else None
    found = re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*(k|K|l|L|lpa|LPA|lakhs?)?", raw)
    amounts = [a for a in (_amount(n, s or "") for n, s in found) if a]
    if not amounts:
        return None, None, currency
    if re.search(r"(per|/)\s*(hour|hr)", raw, re.I):
        amounts = [a * 2080 for a in amounts]
    elif re.search(r"(per|/)\s*month", raw, re.I):
        amounts = [a * 12 for a in amounts]
    amounts = [a for a in amounts if a >= 1000]
    if not amounts:
        return None, None, currency
    return min(amounts), max(amounts), currency


def parse_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def from_remotive(item: dict[str, Any]) -> dict[str, Any]:
    location = item.get("candidate_required_location") or "Worldwide"
    smin, smax, cur = parse_salary(item.get("salary"))
    return {
        "source": "remotive",
        "source_id": str(item["id"]),
        "title": (item.get("title") or "").strip(),
        "company": (item.get("company_name") or "").strip(),
        "location": location.strip(),
        "remote": True,
        "job_type": detect_job_type(item.get("job_type"), item.get("title")),
        "salary_min": smin,
        "salary_max": smax,
        "salary_currency": cur,
        "tags": normalize_tags([*(item.get("tags") or []), item.get("category")]),
        "description": html_to_text(item.get("description")),
        "url": item.get("url") or "",
        "posted_at": parse_date(item.get("publication_date")),
    }


def from_arbeitnow(item: dict[str, Any]) -> dict[str, Any]:
    location = (item.get("location") or "").strip()
    remote = is_remote(location, item.get("remote"))
    return {
        "source": "arbeitnow",
        "source_id": str(item.get("slug") or item.get("url")),
        "title": (item.get("title") or "").strip(),
        "company": (item.get("company_name") or "").strip(),
        "location": location or ("Remote" if remote else ""),
        "remote": remote,
        "job_type": detect_job_type(*(item.get("job_types") or []), item.get("title")),
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "tags": normalize_tags(item.get("tags")),
        "description": html_to_text(item.get("description")),
        "url": item.get("url") or "",
        "posted_at": parse_date(item.get("created_at")),
    }


def is_tech_role(row: dict[str, Any]) -> bool:
    text = f" {row.get('title', '')} {' '.join(row.get('tags') or [])} ".lower()
    return any(word in text for word in TECH_WORDS)


def is_valid(row: dict[str, Any]) -> bool:
    return bool(row["title"] and row["company"] and row["url"] and row["source_id"])


def embedding_text(row: dict[str, Any]) -> str:
    tags = ", ".join(row.get("tags") or [])
    where = "Remote" if row.get("remote") else ""
    parts = [
        row.get("title", ""),
        f"Company: {row.get('company', '')}",
        f"Location: {row.get('location', '')} {where}".strip(),
        f"Type: {row.get('job_type', '')}",
        f"Tags: {tags}",
        (row.get("description") or "")[:1500],
    ]
    return "\n".join(p for p in parts if p and not p.endswith(": "))
