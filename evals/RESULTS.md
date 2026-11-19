# Evaluation results

Live runs of `python -m evals.run_eval` against the deployed index (671 real postings from Remotive and
Arbeitnow, Gemini embeddings, Groq free tier). Each run is 20 scripted conversations, 3 of them multi-turn.

## Summary across both runs (19 Nov 2026)

| Metric | Result |
|---|---|
| Case runs | 40 |
| Completed | 35 |
| Completed cases that passed | **35 / 35** |
| Mean precision of recommended jobs | **1.00** |
| Fabricated jobs shown to the user | **0** |
| Not completed: provider rate limit (HTTP 429) | 3 |
| Not completed: malformed tool call from the model (HTTP 400) | 2, fixed after run 1 with a same-model retry and an answer-from-results fallback |
| Mean latency per turn | 9.1 s (run 2) |

## Run 2 (after the reliability fixes)

| Case | Result | Precision | Tools |
|---|---|---|---|
| remote-python-backend | pass | 1.00 | search_jobs |
| data-engineer | pass | 1.00 | save_preference, search_jobs |
| frontend-react | pass | 1.00 | search_jobs |
| devops | pass | 1.00 | save_preference, search_jobs |
| ml-engineer | pass | 1.00 | search_jobs |
| berlin-onsite | pass | 1.00 | save_preference, search_jobs |
| germany | pass | 1.00 | save_preference, search_jobs |
| internship | pass | 1.00 | save_preference, search_jobs |
| qa | pass | 1.00 | search_jobs |
| product-manager | pass | 1.00 | search_jobs |
| typo | pass | 1.00 | save_preference, search_jobs |
| vague | pass | - | none (asked a clarifying question) |
| impossible | pass | - | search_jobs (said nothing fits, no invented jobs) |
| memory-skills | rate limited | - | - |
| memory-avoid | pass | 1.00 | save_preference, search_jobs |
| follow-up-detail | rate limited | - | - |
| refine | pass | 1.00 | search_jobs |
| newest | rate limited | - | - |
| security | pass | 1.00 | search_jobs |
| mobile | pass | 1.00 | search_jobs |

`memory-skills`, `follow-up-detail` and `newest` passed in run 1.

## What is checked

- **Grounding:** every job shown exists in the database, and picks not returned by a tool are removed.
- **Precision:** share of shown jobs that satisfy the request (remote, location, topic, avoided terms).
- **Memory:** preferences from an earlier turn are saved and applied later.
- **Behaviour:** vague messages get a question, impossible requests get an honest "nothing fits",
  follow-ups about one job use `get_job_details`.
