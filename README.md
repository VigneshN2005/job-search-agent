# Scout · Conversational Job-Search Agent

Describe the job you want in plain words and Scout finds it. It is an LLM agent that calls tools, remembers the
conversation and your preferences, searches real job postings with embeddings and SQL filters, re-ranks the results
for you, and only ever recommends jobs that actually came back from a search.

**Stack:** Python · FastAPI · PostgreSQL + pgvector (Supabase) · Groq (gpt-oss-120b, with fallback) · Gemini embeddings · vanilla JS · GitHub Actions · Vercel

---

**Live:** https://job-search-agent-teal-tau.vercel.app

## Results

Live evaluation over 20 scripted conversations, run twice against the deployed system ([details](evals/RESULTS.md)):

| | |
|---|---|
| Completed conversations that passed | **35 / 35** |
| Precision of recommended jobs | **1.00** |
| Fabricated jobs shown to users | **0** |
| Mean latency per turn | ~9 s on Groq's free tier |

## What it does

| | |
|---|---|
| **Conversational search** | "Remote backend jobs where I can use Python, nothing in crypto" works as a request. No filter forms. |
| **Tool-using agent** | The LLM decides when to call `search_jobs`, `filter_jobs`, `get_job_details` and `save_preference`. Every argument is validated with Pydantic before it runs. |
| **Memory** | Conversation history and stated preferences (skills, location, remote, level, salary floor, things to avoid) are stored per session in Postgres and fed back into every turn. |
| **Hybrid retrieval** | pgvector cosine search over Gemini embeddings, combined with SQL filters for location, remote, job type, salary, skills and freshness. |
| **Personalised ranking** | Candidates are re-ranked on vector similarity, keyword overlap, saved preferences (skills, seniority, avoid list) and recency, then de-duplicated. |
| **Structured, grounded output** | The final answer is JSON validated against a Pydantic schema. Any recommended job id that was not returned by a tool in that turn is dropped before the user sees it. Invalid JSON is retried once. |
| **Evaluation** | `evals/` runs 20 scripted conversations against the live system and reports precision, grounding, memory and tool use. |
| **Real data** | Postings are pulled from the public Remotive and Arbeitnow job APIs, filtered to tech roles, cleaned, de-duplicated and embedded. A scheduled GitHub Action refreshes the index every day. |
| **Resilience** | Rate-limit waits, same-model retry on malformed tool calls, fallback across four Groq models, and an answer built from the jobs already found if every model is busy. |

## How a turn works

```
user message
   │
   ▼
load session ─── history + saved preferences (Postgres)
   │
   ▼
LLM with tools ──► search_jobs ─► embed query (Gemini) ─► pgvector + SQL filters ─► re-rank
   │    ▲          filter_jobs ─► SQL filters
   │    │          get_job_details
   │    └───────── save_preference ─► preferences table
   │  (up to 4 tool rounds)
   ▼
structured answer (JSON mode) ─► Pydantic validation ─► grounding check (ids must come from tools)
   │
   ▼
reply + job cards + tool trace ─► saved to history
```

## Project layout

```
app/
  agent.py        agent loop, final JSON step, grounding check
  tools.py        tool definitions, argument validation, tool trace
  ranking.py      hybrid re-ranking (similarity, keywords, preferences, recency)
  repository.py   Postgres + pgvector queries, sessions, memory
  embeddings.py   Gemini embedding client (batched, retries)
  llm.py          Groq client with model fallback
  models.py       Pydantic schemas for tools, answers and API
  main.py         FastAPI app
ingest/           fetch, clean and embed postings from Remotive and Arbeitnow
db/schema.sql     tables, pgvector HNSW index
evals/            evaluation cases and runner
public/           chat interface
tests/            offline test suite (no network, no database)
api/index.py      Vercel entry point
```

## Run it locally

**1. Database (Supabase, free tier)**

Create a project at supabase.com, open the SQL editor and run `db/schema.sql`. Copy the connection string from
*Connect → Transaction pooler* (port 6543).

**2. Keys**

- Groq: console.groq.com → API keys
- Gemini: aistudio.google.com → Get API key

```bash
cp .env.example .env      # then fill in DATABASE_URL, GROQ_API_KEY, GEMINI_API_KEY
```

**3. Install, load jobs, run**

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

python -m ingest.fetch_jobs       # fetch, store and embed real postings (paced for the Gemini free tier)
uvicorn app.main:app --reload     # open http://localhost:8000
```

Re-run `python -m ingest.fetch_jobs` any time to pull new postings. Only new or changed jobs are re-embedded.
`--prune-days 60` removes stale postings.

## Tests and evaluation

```bash
pytest -q                         # offline suite: agent loop, tools, ranking, SQL builder, parsing
python -m evals.run_eval          # live run of 20 conversations, writes evals/RESULTS.md
```

The evaluation checks, for every case:

- **grounding**: no job shown to the user is missing from the database, and how many invented picks were caught
- **precision**: share of recommended jobs that satisfy the request (remote, location, topic, avoided terms)
- **memory**: preferences stated in an earlier turn are saved and applied later
- **behaviour**: vague messages get a clarifying question instead of a random search, follow-ups use `get_job_details`

## Automation

| Workflow | Trigger | What it does |
|---|---|---|
| `CI/CD` | push, pull request | ruff, pytest, pip-audit, Gitleaks; deploys to Vercel when deploy secrets are set |
| `Ingest jobs` | daily 02:30 UTC, manual | fetches new postings, prunes old ones, embeds only what changed |
| `Evaluate agent` | manual | runs the 20 live conversations and publishes the report |

The three workflows read `DATABASE_URL`, `GROQ_API_KEY` and `GEMINI_API_KEY` from repository secrets.

## Deploy (Vercel + GitHub Actions)

1. Import the repository in Vercel. Add `DATABASE_URL`, `GROQ_API_KEY` and `GEMINI_API_KEY` as environment variables.
2. In the GitHub repository, add the secrets `VERCEL_TOKEN`, `VERCEL_ORG_ID` and `VERCEL_PROJECT_ID`.
3. Push to `main`. The workflow runs lint, tests and security checks (pip-audit, Gitleaks); only if all pass does it
   build and deploy to Vercel and confirm `/api/health` reports a working database.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/api/chat` | `{ "message": "...", "session_id": "optional uuid" }` → reply, job cards, tool trace, preferences |
| `GET` | `/api/session/{id}` | saved preferences and history |
| `GET` | `/api/stats` | size and freshness of the job index |
| `GET` | `/api/health` | key and database status |

## Data sources

Job postings come from the public APIs of [Remotive](https://remotive.com) and [Arbeitnow](https://www.arbeitnow.com).
Every card links back to the original posting.
