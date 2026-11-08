# Salesforce FDE Servicing Agent PoC — Milestone 4

A local, evaluation-driven proof of concept for evidence-grounded customer
servicing and controlled, simulated CRM actions. Milestone 4 adds a measured
evaluation dashboard, end-to-end golden scenarios, reproducible demo reset, and a
polished technical-demo interface while preserving all prior capabilities.

All customers, cases, entitlements, policies, product documents, notes, and call
transcripts are synthetic. There are no Salesforce, Snowflake, or production
credentials.

## What works

- FastAPI `GET /health` and `POST /chat` endpoints plus Streamlit chat UI
- Deterministic ticket-status routing that does not invoke LangGraph or RAG
- LangGraph orchestration for authorized complex investigations
- `lexical`, `semantic`, and `hybrid` retrieval modes selected with
  `RETRIEVAL_MODE`
- CPU Sentence Transformers using `sentence-transformers/all-MiniLM-L6-v2`
- Fingerprinted local document-embedding cache that refreshes when model or
  document text changes
- Deterministic RRF with configurable `k` (default `60`)
- Authorization and tenant filtering before either ranker runs
- Effective-date, approval-state, product, and subscription-tier policy filters
- Lexical fallback when semantic dependencies or the model are unavailable
- Inline citations and structured evidence snippets
- Optional OpenAI-compatible answer generation with citation validation
- Retrieval mode, rankings, latency, selected evidence, cache state, and fallback
  reason in JSON logs, API debug metadata, and the Streamlit trace panel
- Ticket-note preview followed by explicit Confirm or Cancel
- Authorization and current ticket-version revalidation immediately before writes
- Durable `PENDING`, `COMPLETED`, `FAILED`, and `UNKNOWN` operation states
- Database-enforced operation/note uniqueness and unknown-outcome reconciliation
- Human handoffs with `CREATED`, `ACCEPTED`, and `RESOLVED` transitions
- Deterministic SLA checks, a one-shot CLI sweep, and simulated backup routing
- Repeatable end-to-end evaluation across chat, retrieval, policy, authorization,
  actions, escalation, and recovery
- Measured lexical/semantic/hybrid quality and latency dashboard
- Five reproducible read scenarios plus controlled-action/SLA demonstrations
- Atomic synthetic demo-data reset
- 56 pytest tests

## Architecture overview

```text
Streamlit demo + evaluation dashboard
              │ HTTP / trusted demo-user header
              ▼
FastAPI ── deterministic intent router ── ticket-status fast path
   │
   ├─ LangGraph read workflow
   │    ├─ authorized SQLite CRM + entitlement tools
   │    ├─ lexical / semantic / RRF hybrid retrieval
   │    └─ deterministic or optional citation-validated LLM generation
   │
   ├─ LangGraph controlled-action workflow
   │    └─ preview → confirmation → version check → idempotent SQLite write
   │
   └─ deterministic SLA + authorized human handoff services

Golden scenarios → isolated/reset SQLite runs → measured JSON report → dashboard
```

The main interview design points are explicit trust boundaries, deterministic
rules for high-risk decisions, evidence before generation, database-enforced
idempotency, and evaluation artifacts that distinguish measured behavior from
illustrative business narratives.

## Retrieval flow

```text
Complex request
  └─> authorized case and entitlement reads
       └─> record / tenant / policy applicability filtering
            ├─> lexical ranking
            ├─> semantic cosine ranking
            └─> hybrid: RRF(lexical, semantic), k=60
                 └─> evidence sufficiency and conflict checks
                      └─> deterministic or optional LLM answer
                           └─> validated citations
```

Protected documents are filtered before lexical scoring or semantic similarity.
Consequently, unauthorized document IDs cannot appear in rankings, logs, prompts,
citations, or the UI.

## Controlled-action flow

```text
Prepare note
  └─> authorize case + snapshot ticket record_version
       └─> durable PENDING operation + preview
            ├─> Cancel: FAILED / CANCELLED
            └─> Confirm
                 └─> reauthorize + compare current record_version
                      ├─> changed: FAILED / STALE_CONFIRMATION
                      └─> unchanged: atomic note insert + version increment
                           ├─> acknowledged: COMPLETED
                           └─> simulated lost response: UNKNOWN
                                └─> reconcile by unique operation_id: COMPLETED
```

SQLite `BEGIN IMMEDIATE`, operation primary keys, and a unique note
`operation_id` make retries safe within this local database. This PoC does not
claim exactly-once behavior for a future external CRM API.

Important files:

```text
.
├── evaluations/
│   ├── benchmark_results.json
│   ├── golden_questions.json
│   └── latest_results.json
├── src/servicing_agent/
│   ├── corpus/knowledge.json
│   ├── actions.py
│   ├── action_workflow.py
│   ├── demo_reset.py
│   ├── demo_scenarios.py
│   ├── evaluation.py
│   ├── evaluation_dashboard.py
│   ├── generation.py
│   ├── handoffs.py
│   ├── index_embeddings.py
│   ├── retrieval.py
│   ├── sla.py
│   ├── sla_sweep.py
│   ├── service.py
│   ├── streamlit_app.py
│   ├── tools.py
│   └── workflow.py
├── tests/
└── pyproject.toml
```

## Windows setup (PowerShell)

Prerequisite: Python 3.11 or newer with the Windows `py` launcher.

### Lexical-only setup

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Semantic and hybrid setup on CPU

Install the official CPU PyTorch wheel first so a CUDA build is not selected,
then install the semantic extra:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,semantic]"
```

This configuration runs the embedding model on CPU and works without an NVIDIA
GPU.

## Initialize the local database

The API and SLA CLI initialize and migrate SQLite automatically. To initialize it
explicitly from PowerShell:

```powershell
New-Item -ItemType Directory -Force data | Out-Null
python -c "from servicing_agent.database import Database; Database('data/service_agent.db').initialize()"
```

The database contains synthetic records only. Milestone 3 adds `operations`,
`ticket_notes`, and `case_handoffs` tables plus ticket version and SLA fields.
Existing local databases from earlier milestones are migrated additively on
startup.

Reset all synthetic demo mutations and restore canonical records before a live
demonstration:

```powershell
python -m servicing_agent.demo_reset --database data\service_agent.db
```

The reset runs in one SQLite transaction and affects only the explicitly selected
local database. Running it repeatedly produces the same canonical state.

## Initialize or refresh embeddings

The first command downloads MiniLM and creates a fingerprinted JSON embedding
index under `data\embedding_cache`. The directory is ignored by Git.

```powershell
.\.venv\Scripts\Activate.ps1
$env:RETRIEVAL_CACHE_DIR = "data\embedding_cache"
python -m servicing_agent.index_embeddings
```

Force a recomputation:

```powershell
python -m servicing_agent.index_embeddings --force
```

Changing the configured model or any document's embedding text automatically
changes the fingerprint, so stale vectors are not reused. No vector database or
index server is required.

## Retrieval configuration

Set variables in the API terminal before starting Uvicorn:

```powershell
$env:RETRIEVAL_MODE = "hybrid" # lexical, semantic, or hybrid
$env:RETRIEVAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
$env:RETRIEVAL_CACHE_DIR = "data\embedding_cache"
$env:RETRIEVAL_RRF_K = "60"
$env:RETRIEVAL_SEMANTIC_MIN_SCORE = "0.25"
$env:RETRIEVAL_SEMANTIC_MIN_TOP_SCORE = "0.45"
$env:SLA_BACKUP_AFTER_MINUTES = "15"
```

The top-score confidence gate was added after the golden evaluation showed that
an unsupported hardware-replacement question had a cosine score around `0.417`
against a case note. The `0.45` default rejected that false positive while
retaining the evaluated relevant paraphrases. It is a small-corpus demo threshold,
not a production calibration.

If semantic initialization fails, `semantic` and `hybrid` requests use lexical
retrieval and report the reason in `retrieval_debug.fallback_reason`.

## Run locally

Terminal 1 — API:

```powershell
.\.venv\Scripts\Activate.ps1
$env:RETRIEVAL_MODE = "hybrid"
$env:SERVICE_AGENT_LLM_PROVIDER = "none"
python -m uvicorn servicing_agent.api:app --reload --port 8000
```

Terminal 2 — UI:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run src\servicing_agent\streamlit_app.py --server.port 8501
```

Open <http://localhost:8501>. API documentation is at
<http://localhost:8000/docs>.

The UI has three focused areas:

- **Customer servicing** — selectable reproducible scenarios, chat, citations,
  and execution traces.
- **Controlled actions** — simulated note confirmation, reconciliation, human
  escalation, and deterministic SLA status.
- **Evaluation dashboard** — measured quality, latency, scenario outcomes, and
  failure categories loaded from `evaluations\latest_results.json`.

The default answer generator is deterministic and explicitly says it is not an
LLM. Retrieval and policy applicability are real local operations in every mode.
CRM writes, handoff queues, and SLA escalations are explicitly simulated in the
UI and persist only in local SQLite.

## Optional OpenAI-compatible generation

LLM use remains opt-in and happens only after authorization and retrieval:

```powershell
$env:SERVICE_AGENT_LLM_PROVIDER = "openai_compatible"
$env:SERVICE_AGENT_LLM_MODEL = "gpt-4.1-mini"
$env:SERVICE_AGENT_LLM_API_BASE = "https://api.openai.com/v1"
$secureKey = Read-Host "API key" -AsSecureString
$env:SERVICE_AGENT_LLM_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
python -m uvicorn servicing_agent.api:app --reload --port 8000
```

The key is read only from the environment. Retrieval, authorization, policy
selection, and the ticket-status path are never delegated to the LLM. Missing or
invalid citations cause deterministic fallback.

The standard evaluation command always uses the deterministic generator. To run
a genuine configured-LLM evaluation, explicitly add `--use-configured-llm`. The
report records requested mode, credential availability, and actual generation
mode counts. If the key is absent, it labels the run as deterministic fallback;
it never reports fallback output as LLM output.

## Five-minute technical demonstration

Before the audience joins:

```powershell
python -m servicing_agent.demo_reset --database data\service_agent.db
```

Then use the Streamlit scenario selector and controls:

1. **0:00–0:30 — Frame the architecture.** State that all data and CRM actions are
   synthetic. Point to FastAPI, LangGraph, SQLite, the authorization boundary, and
   local hybrid retrieval.
2. **0:30–1:00 — Deterministic status.** Run **Deterministic ticket status**.
   Expected: `CASE-4821 is In Progress`, authoritative owner/priority, no RAG or
   LLM, plus deterministic overdue-SLA status.
3. **1:00–1:45 — Multi-source investigation.** Run **Multi-source investigation**.
   Expected in hybrid mode: case-note and call-transcript citations, source
   snippets, retrieval rankings, and the execution trace.
4. **1:45–2:30 — Customer applicability.** Run Premium and Standard policy
   scenarios. Expected: CASE-4821 uses Premium approved policy v2.0/two clock
   hours; CASE-4824 uses Standard approved policy v1.0/eight business hours.
5. **2:30–3:00 — Authorization.** Run **Unauthorized customer access**. Expected:
   no protected citations, snippets, rankings, or prompt context.
6. **3:00–4:00 — Controlled write and recovery.** Prepare a CASE-4821 note, attest
   customer confirmation, and confirm it. A duplicate confirmation returns the
   same operation/note. Repeat with timeout simulation and reconcile `UNKNOWN →
   COMPLETED` without another note.
7. **4:00–4:30 — Mandatory escalation.** Run the SLA check, then show the two-pass
   CLI sweep below. Expected: one SLA handoff and backup routing after 15 minutes,
   without LLM reasoning.
8. **4:30–5:00 — Evaluation dashboard.** Show actual lexical versus semantic versus
   hybrid results, latency, scenario outcomes, and lexical failure categories.
   Explicitly note that CSAT, CPRR, ROI, and production impact are not measured.

## Detailed manual scenarios

Use Alex Morgan:

1. `What is the status of CASE-4821?` — unchanged deterministic fast path.
2. `Why was CASE-4821 escalated?` — exact-match case-note evidence.
3. `Which events led support to send CASE-4821 to Tier 2?` — paraphrased query.
4. `What did the client say was wrong during the CASE-4821 phone conversation?`
   — semantic call-transcript retrieval.
5. `What initial response target applies to CASE-4821?` — Premium policy v2.0.
6. `What initial response target applies to CASE-4824?` — Standard policy v1.0.
7. `Is on-site hardware replacement included for CASE-4821?` — insufficient
   evidence, not a fabricated answer.

Switch to Jordan Lee and repeat a `CASE-4821` question. The trace contains no
retrieval rankings and the response contains no protected snippets or citations.

### Controlled ticket note

1. Open **Controlled actions (simulated)** in Streamlit.
2. Enter `CASE-4821` and a note, then select **Prepare note preview**.
3. Verify the case snapshot and note text. SQLite still contains no note at this
   point.
4. Check the explicit customer-confirmation attestation, then select **Confirm
   simulated save**. Repeating Confirm or retrying with the same operation ID does
   not create another note.
5. Prepare another note and select **Cancel pending note** to produce a terminal
   `FAILED / CANCELLED` operation.
6. Select **Simulate timeout after a successful save**, confirm, then use
   **Reconcile unknown outcome**. The note is found by operation ID and the state
   changes from `UNKNOWN` to `COMPLETED` without a second write.

API clients that need retry idempotency should generate a stable operation ID and
send it in `operation_id` when preparing the note. Reusing it with different
content returns a conflict.

Equivalent PowerShell API flow:

```powershell
$headers = @{ "X-Demo-User" = "demo-agent-alex" }
$prepare = @{
    ticket_id = "CASE-4821"
    note = "Customer confirmed telemetry is current."
    operation_id = "op-demo-note-0001"
} | ConvertTo-Json
$preview = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/actions/notes/prepare" -Headers $headers -ContentType "application/json" -Body $prepare
$confirm = @{
    customer_confirmed = $true
    simulate_timeout_after_success = $false
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/actions/notes/$($preview.operation_id)/confirm" -Headers $headers -ContentType "application/json" -Body $confirm
```

### Human handoff

1. Create a handoff for `CASE-4821` with troubleshooting steps.
2. Inspect the permitted case context and retrieval-derived source references.
3. Select **Accept simulated handoff**, then **Resolve simulated handoff**.
4. Switch to Jordan and verify that the same case/handoff cannot be accessed.

### Deterministic SLA and backup queue

The synthetic deadline for high-priority `CASE-4821` is overdue. Use the
Streamlit SLA button or ask `What is the status of CASE-4821?`; either request-time
path creates at most one SLA handoff without LLM reasoning.

Run a background-independent sweep from PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:SERVICE_AGENT_DB_PATH = "data\service_agent.db"
$env:SLA_BACKUP_AFTER_MINUTES = "15"
python -m servicing_agent.sla_sweep --as-of "2026-11-08T12:00:00Z"
python -m servicing_agent.sla_sweep --as-of "2026-11-08T12:20:00Z"
```

The second command reconciles with the existing SLA handoff instead of creating a
duplicate and routes the still-`CREATED`, twenty-minute-old handoff to **Backup
Escalation Queue (simulated)**. Accepted/resolved handoffs are not rerouted.

## Tests

Tests use mocked embeddings and do not download a model:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
```

The 56-test suite preserves every Milestone 1–3 regression and adds demo-reset,
end-to-end evaluator, deterministic-versus-LLM labeling, dashboard parsing,
scenario-catalog, and major Streamlit-path coverage. Semantic unit tests use
mocked embeddings and do not download the model.

## Run the end-to-end evaluation

Initialize the model/index first, then run each mode five times:

```powershell
.\.venv\Scripts\Activate.ps1
$env:RETRIEVAL_CACHE_DIR = "data\embedding_cache"
python -m servicing_agent.index_embeddings
python -m servicing_agent.evaluation --modes lexical semantic hybrid --runs 5 --output evaluations\latest_results.json
```

For a fully cached/offline run:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m servicing_agent.evaluation --modes lexical semantic hybrid --runs 5
```

Optional genuine configured-LLM run:

```powershell
$env:SERVICE_AGENT_LLM_PROVIDER = "openai_compatible"
$secureKey = Read-Host "Temporary evaluation API key" -AsSecureString
$env:SERVICE_AGENT_LLM_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
python -m servicing_agent.evaluation --modes hybrid --runs 1 --use-configured-llm --output evaluations\results\llm-latest.json
```

The default evaluator uses the deterministic answer generator so LLM variability
and API cost do not affect regression metrics. Every scenario starts from reset
synthetic data in an isolated temporary database. Model download and embedding
index initialization are excluded from warm query latency.

### Actual Milestone 4 evaluation results

Measured on November 8, 2026 using 18 synthetic scenarios, five runs per scenario,
the deterministic generator, CPU MiniLM, and warm model/document caches:

| Mode | Task success | Recall@5 | Precision@3 | Groundedness | Policy | Authorization | Action safety | Escalation | Recovery | Failed runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lexical | 0.8889 | 0.7143 | 0.2381 | 0.8182 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10/90 |
| Semantic | 1.0000 | 1.0000 | 0.3810 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0/90 |
| Hybrid | 1.0000 | 1.0000 | 0.3810 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0/90 |

| Mode | Retrieval P50/P95 | Tool P50/P95 | End-to-end P50/P95 |
|---|---:|---:|---:|
| Lexical | 0.093 / 0.207 ms | 4.321 / 10.757 ms | 6.236 / 23.606 ms |
| Semantic | 15.000 / 24.646 ms | 8.753 / 22.125 ms | 17.945 / 33.805 ms |
| Hybrid | 15.142 / 29.505 ms | 7.045 / 21.412 ms | 18.047 / 29.969 ms |

Lexical failed the paraphrased call-transcript and multi-source investigation
scenarios on every run. Semantic and hybrid retrieved the expected evidence for
this small corpus. Precision@3 can exceed `0.3333` in the expanded suite because
the multi-source scenario declares two relevant documents. The complete measured
report, including per-scenario outcomes and generation-mode counts, is in
`evaluations\latest_results.json`.

These results are regression measurements over synthetic data, not evidence of
production superiority or statistical significance. No CSAT, CPRR, case
deflection, ROI, or other business outcome was measured or inferred.

## Troubleshooting

- **Dashboard says no report:** run the end-to-end evaluation command above or set
  `EVALUATION_RESULTS_PATH` to a schema-v2 report.
- **Semantic mode falls back to lexical:** install `.[semantic]`, initialize the
  embedding index, and remove offline variables for the first model download.
- **Offline model load fails:** confirm `RETRIEVAL_CACHE_DIR` points to the cache
  populated by `python -m servicing_agent.index_embeddings`.
- **Missing LLM key:** expected behavior is the explicitly labeled deterministic
  fallback. Retrieval, actions, and SLA rules continue to work.
- **A demo starts in an unexpected state:** stop the API, run `demo_reset`, then
  restart Uvicorn so long-lived service objects reopen canonical data.
- **Port already in use:** choose different Uvicorn/Streamlit ports and set
  `SERVICE_AGENT_API_URL` in the UI terminal.

## Milestone 4 boundaries

This remains a local demonstration: no real Salesforce/Snowflake integration,
production authentication, financial transactions, refunds, cloud deployment,
vector database, or reranker. CRM mutations and queue routing are SQLite-only
simulations. Customer confirmation is a demo-user attestation, not a customer
signature or production consent record. There is no continuously running
scheduler; SLA sweeps must be invoked by a request or CLI. The corpus and
evaluation set are small and synthetic, local database/embedding files are
unencrypted, and optional LLM
generation has not been production-evaluated for safety, latency, or cost.
The dashboard reads a generated local JSON snapshot rather than a live production
telemetry system. Timing is machine-specific and excludes cold model download.
This is demo-ready for a technical walkthrough, not production-ready software.
