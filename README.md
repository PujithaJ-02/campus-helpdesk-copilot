# Campus Help Desk Copilot

A multi-agent help desk assistant for a university. It answers policy questions
from the Stanford Administrative Guide with cited sources, and opens or triages
ServiceNow incidents, with guardrails around every action.

Built with **Google ADK**, **Gemini and embeddings on Vertex AI**, deployed to
**Vertex AI Agent Engine**, with a **FAISS + BM25** hybrid index and the
**ServiceNow Table API**.

## Why

Large universities run thousands of help desk tickets a month, and many of them
are questions already answered in written policy. This copilot answers those
from the source, and when it can't, it files a clean ticket instead of guessing.

## Architecture

```
                 user (role in session state: requester | it_staff)
                                   |
                         helpdesk_router (Gemini)
                     /                              \
          policy_agent                          ticket_agent
          search_policy                         create_incident, get_incident,
               |                                list_open_incidents, update_incident
   hybrid retriever (FAISS dense +                     |
   BM25, RRF fusion, chapter filter)          ServiceNow Table API
               |
   Admin Guide index (Vertex AI embeddings)

   every tool call: before_tool_callback (role check + audit log)
                    tool (PII redaction, field policy, dry run)
                    after_tool_callback (audit log)
```

| Piece | File |
|---|---|
| Router and specialist agents | `helpdesk_agent/agent.py` |
| Versioned prompts | `helpdesk_agent/prompts.py` |
| Policy search tool | `helpdesk_agent/policy_tools.py` |
| ServiceNow tools | `helpdesk_agent/ticket_tools.py`, `servicenow.py` |
| Guardrails (PII, field policy, roles) | `helpdesk_agent/guardrails.py` |
| Role check and audit callbacks | `helpdesk_agent/callbacks.py` |
| Latency and status logs | `helpdesk_agent/telemetry.py` |
| Crawl, chunk, embed, index | `scripts/crawl_adminguide.py`, `scripts/build_index.py`, `helpdesk_agent/rag/` |
| Evals | `scripts/eval_retrieval.py`, `eval_answers.py`, `eval_triage.py` |

## Guardrails

- **PII redaction**: emails, phones, SSNs and card numbers are masked in ticket text, policy passages, new tickets, and audit logs.
- **Role-based access**: `requester` can search policy and open tickets. `it_staff` can also read and triage. Unknown or missing role gets the least privilege. Enforced in an ADK `before_tool_callback`, so the model cannot talk its way past it.
- **Write policy**: only allowed fields and values can be written (`POLICY` in `guardrails.py`).
- **Human approval**: agents must ask before any write, and `DRY_RUN=true` blocks writes entirely.
- **Audit log**: every tool request, decision and result is a JSON line in Cloud Logging.

## Setup

1. **ServiceNow**: request a free Personal Developer Instance at developer.servicenow.com. It comes with demo incidents. (Idle instances go to sleep; wake them from the developer portal.)
2. **Google Cloud**: create a project, enable the Vertex AI API and Cloud Resource Manager API, install gcloud, then:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```
3. **Local**:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements-dev.txt
   cp helpdesk_agent/.env.example helpdesk_agent/.env    # fill it in
   pytest -q                                            # 25 offline tests
   ```

## Build the index

```bash
python -m scripts.crawl_adminguide --max-pages 300     # public pages, 1s between requests
python -m scripts.build_index
```

The index lands in `helpdesk_agent/index/` and ships with the agent when you deploy.
Commit it (it is built from public pages) so deploys are reproducible.

## Run locally

```bash
adk web          # localhost:8000, choose helpdesk_agent
```

Try:
- "Can I put a personal expense on my travel card?" (policy answer with source)
- "My laptop was stolen and it had student data on it" (policy answer, then offer to open a ticket)
- Set `DEFAULT_ROLE=it_staff`, restart, then "Triage INC0010001"

## Evaluate

```bash
python -m scripts.eval_retrieval     # hit@4 and MRR: BM25 vs dense vs hybrid vs hybrid+filter
python -m scripts.eval_answers       # LLM judge: faithfulness, citation, answered
python -m scripts.eval_triage --limit 50   # category accuracy vs human labels in ServiceNow
```

`eval/policy_golden.jsonl` has 30 hand-written questions across all 6 chapters,
each checked against the page that answers it. About a quarter use exact terms
(policy numbers, PCard, PTA, SU-17, Form 1) to give keyword search a fair chance.

**Why dense is the default:** hybrid search was the original plan, but BM25
matched surface words over meaning (a "stolen laptop" question pulled insurance
and vehicle-records pages), and even the exact-term questions didn't need it.
Lowering BM25's weight helped, but dense alone was best, so the agent uses dense.

### Results (fill in from `eval/results/`)

| Eval | Metric | Value |
|---|---|---|
| Retrieval, BM25 only (n=30) | hit@4 / MRR | 0.83 / 0.75 |
| Retrieval, hybrid, BM25 weight 1.0 | hit@4 / MRR | 0.90 / 0.84 |
| Retrieval, hybrid, BM25 weight 0.25 | hit@4 / MRR | 0.93 / 0.87 |
| **Retrieval, dense only (default)** | hit@4 / MRR | **1.00 / 0.92** |
| Retrieval, dense + chapter filter | hit@4 / MRR | 1.00 / 0.93 |
| Answers (n=) | faithful / cited | |
| Triage (n=) | category accuracy | |
| Latency | p50 answer ms | |

## Deploy to Vertex AI Agent Engine

```bash
adk deploy agent_engine \
  --project=YOUR_PROJECT_ID \
  --region=us-central1 \
  --display_name="Campus Help Desk Copilot" \
  helpdesk_agent

python scripts/query_deployed.py \
  projects/PROJECT_NUMBER/locations/us-central1/reasoningEngines/RESOURCE_ID \
  "Can I share my SUNet password with my assistant?" --role requester
```

Google's docs now also call Agent Engine "Agent Runtime"; the CLI command is unchanged.
The `.env` file is packaged with the deploy. That is fine for a dev instance; for
anything real, move the ServiceNow password to Secret Manager.
**Delete the Agent Engine instance when you are done** so it stops billing.

## Known limits

- The golden set is small and self-written; numbers are a baseline, not a benchmark.
- Regex PII redaction misses names and free-form identifiers. A production system would add a service like Cloud DLP.
- Roles come from session state set by the caller. A real deployment would derive them from SSO claims.
- Demo ServiceNow data is synthetic.
