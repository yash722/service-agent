# Salesforce FDE Servicing Agent PoC — Milestone 4 + Snowflake History

A local, evaluation-driven proof of concept for evidence-grounded customer
servicing and controlled, simulated CRM actions. Milestone 4 adds a measured
evaluation dashboard, end-to-end golden scenarios, reproducible demo reset, and a
polished technical-demo interface while preserving all prior capabilities.
The Milestone 3.5 extension adds an opt-in, read-only Snowflake customer-history
source to the existing investigation workflow.

All customers, cases, entitlements, policies, product documents, notes, and call
transcripts are synthetic. The optional Snowflake connection reads synthetic
history only. There is no real Salesforce integration, and no production
credentials or private keys are stored in this repository.

## What works

- FastAPI `GET /health` and `POST /chat` endpoints plus Streamlit chat UI
- Deterministic ticket-status routing that does not invoke LangGraph or RAG
- LangGraph orchestration for authorized complex investigations
- `HISTORY_BACKEND=mock|snowflake`, defaulting to network-free `mock`
- Read-only Snowflake history from `CUSTOMER_ACTIVITY`, `CASE_HISTORY`, and
  `SUPPORT_INTERACTIONS` using RSA key-pair authentication
- Parameterized case/customer filters, verified Snowflake session context,
  query timeouts, safe cleanup, and explicit no-fallback failure reporting
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
- Seven reproducible read scenarios plus controlled-action/SLA demonstrations
- Atomic synthetic demo-data reset
- 118 passing offline pytest tests plus five opt-in integration tests

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
   │    ├─ optional authorized Snowflake historical evidence
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

SQLite CRM remains authoritative for current ticket status. Snowflake is queried
only after the server resolves and authorizes the CRM case, and contributes
historical evidence to complex investigations; it never overrides the status
fast path.

## Retrieval flow

```text
Complex request
  └─> authorized case and entitlement reads
       ├─> authorized Snowflake history read (when configured)
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
│   ├── latest_results.json
│   ├── qwen3_4b_baseline_live.json
│   └── qwen3_4b_instruct_live.json
├── src/servicing_agent/
│   ├── corpus/knowledge.json
│   ├── actions.py
│   ├── action_workflow.py
│   ├── benchmark_ollama.py
│   ├── demo_reset.py
│   ├── demo_scenarios.py
│   ├── evaluation.py
│   ├── evaluation_dashboard.py
│   ├── generation.py
│   ├── handoffs.py
│   ├── history.py
│   ├── index_embeddings.py
│   ├── retrieval.py
│   ├── run_streamlit.py
│   ├── sla.py
│   ├── sla_sweep.py
│   ├── service.py
│   ├── streamlit_app.py
│   ├── tools.py
│   └── workflow.py
├── tests/
├── snowflake/
│   ├── seed_correct_case_4821_customer.sql
│   └── seed_synthetic_history.sql
├── .env.example
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

### Snowflake-enabled setup

Install the connector extra in the same virtual environment. Include `semantic`
as shown when hybrid retrieval is also required:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,semantic,snowflake]"
```

The connector is imported lazily, so the default mock backend does not require
Snowflake packages, credentials, or network access.

## One root `.env` file

All local application configuration is loaded from `.env` in the project root by
`servicing_agent.settings`. The path is derived from the installed module, not
the current directory, so Uvicorn, Streamlit, pytest, and the Python CLIs find the
same file even when launched elsewhere. Explicit process environment variables
take precedence, which keeps deployment and test overrides possible without code
changes.

Create the local file without replacing one that already exists:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Both `.env` and private-key extensions are ignored by Git. `.env.example` is the
tracked, secret-free inventory. Configuration is grouped as follows:

| Area | Variables |
|---|---|
| SQLite | `SERVICE_AGENT_DB_PATH`, `SERVICE_AGENT_SQLITE_TIMEOUT_SECONDS`, `SERVICE_AGENT_SQLITE_BUSY_TIMEOUT_MS` |
| Logging/tracing | `SERVICE_AGENT_LOG_LEVEL`, `SERVICE_AGENT_LOG_JSON`, `SERVICE_AGENT_TRACE_ENABLED` |
| FastAPI/client | `SERVICE_AGENT_API_HOST`, `SERVICE_AGENT_API_PORT`, `SERVICE_AGENT_API_RELOAD`, `SERVICE_AGENT_API_URL`, `SERVICE_AGENT_API_REQUEST_TIMEOUT_SECONDS` |
| Streamlit | `SERVICE_AGENT_UI_HOST`, `SERVICE_AGENT_UI_PORT`, `SERVICE_AGENT_UI_API_READ_TIMEOUT_SECONDS`, `SERVICE_AGENT_UI_HEADLESS`, `SERVICE_AGENT_UI_PAGE_TITLE`, `SERVICE_AGENT_UI_PAGE_ICON`, `SERVICE_AGENT_UI_LAYOUT`, `SERVICE_AGENT_UI_GATHER_USAGE_STATS` |
| LLM | `SERVICE_AGENT_LLM_PROVIDER`, `SERVICE_AGENT_LLM_API_KEY`, `SERVICE_AGENT_LLM_MODEL`, `SERVICE_AGENT_LLM_API_BASE`, `SERVICE_AGENT_LLM_TIMEOUT_SECONDS`, `SERVICE_AGENT_LLM_TEMPERATURE`, `SERVICE_AGENT_LLM_MAX_TOKENS`, `SERVICE_AGENT_LLM_REASONING_EFFORT` |
| Retrieval | `RETRIEVAL_MODE`, `RETRIEVAL_MODEL`, `RETRIEVAL_CACHE_DIR`, `RETRIEVAL_TOP_K`, `RETRIEVAL_RRF_K`, `RETRIEVAL_SEMANTIC_MIN_SCORE`, `RETRIEVAL_SEMANTIC_MIN_TOP_SCORE` |
| History/Snowflake | `HISTORY_BACKEND`, history evidence limits, all `SNOWFLAKE_*` identity/key/timeout/query variables |
| SLA/evaluation | `SLA_BACKUP_AFTER_MINUTES`, all `EVALUATION_*` paths, modes, runs, generation, log level, and fixed-clock values |
| Offline/testing | `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `RUN_SNOWFLAKE_INTEGRATION`, `RUN_OLLAMA_INTEGRATION` |

Relative paths in `.env` are resolved against the project root. Boolean values
accept `true`/`false`, `1`/`0`, `yes`/`no`, or `on`/`off`. Invalid modes, ports,
URLs, dates, ranges, and required Snowflake fields fail with the relevant variable
name; secret values are never included in those errors.

## Initialize the local database

The API and SLA CLI initialize and migrate SQLite automatically. To initialize it
explicitly from PowerShell:

```powershell
python -m servicing_agent.demo_reset
```

The database contains synthetic records only. Milestone 3 adds `operations`,
`ticket_notes`, and `case_handoffs` tables plus ticket version and SLA fields.
Existing local databases from earlier milestones are migrated additively on
startup.

Reset all synthetic demo mutations and restore canonical records before a live
demonstration:

```powershell
python -m servicing_agent.demo_reset
```

The reset runs in one SQLite transaction and affects only the explicitly selected
local database. Running it repeatedly produces the same canonical state: five
cases, three customers, four authoritative entitlements, and no demo-created
notes, operations, or handoffs. It never connects to or modifies Snowflake.

## Initialize or refresh embeddings

The first command downloads MiniLM and creates a fingerprinted JSON embedding
index under `data\embedding_cache`. The directory is ignored by Git.

```powershell
.\.venv\Scripts\Activate.ps1
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

Edit the retrieval block in `.env`, for example:

```dotenv
RETRIEVAL_MODE=hybrid
RETRIEVAL_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_CACHE_DIR=data/embedding_cache
RETRIEVAL_TOP_K=5
RETRIEVAL_RRF_K=60
RETRIEVAL_SEMANTIC_MIN_SCORE=0.25
RETRIEVAL_SEMANTIC_MIN_TOP_SCORE=0.45
```

The top-score confidence gate was added after the golden evaluation showed that
an unsupported hardware-replacement question had a cosine score around `0.417`
against a case note. The `0.45` default rejected that false positive while
retaining the evaluated relevant paraphrases. It is a small-corpus demo threshold,
not a production calibration.

If semantic initialization fails, `semantic` and `hybrid` requests use lexical
retrieval and report the reason in `retrieval_debug.fallback_reason`.

## Snowflake history configuration

Mock mode is the safe default and performs no Snowflake connection:

```dotenv
HISTORY_BACKEND=mock
```

For the verified synthetic Snowflake environment, edit only the ignored `.env`
file. Use forward slashes in the Windows key path and quote values containing `#`
or leading/trailing spaces:

```dotenv
HISTORY_BACKEND=snowflake
SNOWFLAKE_ACCOUNT=your-account-identifier
SNOWFLAKE_USER=SERVICING_AGENT_SVC
SNOWFLAKE_ROLE=SERVICING_HISTORY_READER
SNOWFLAKE_WAREHOUSE=SERVICING_POC_WH
SNOWFLAKE_DATABASE=SERVICING_AGENT_DEMO
SNOWFLAKE_SCHEMA=HISTORY
SNOWFLAKE_PRIVATE_KEY_FILE=C:/Users/your-user/.snowflake/keys/servicing_agent_key.p8
SNOWFLAKE_PRIVATE_KEY_PASSPHRASE="your-encrypted-key-passphrase"
SNOWFLAKE_LOGIN_TIMEOUT_SECONDS=20
SNOWFLAKE_NETWORK_TIMEOUT_SECONDS=30
SNOWFLAKE_QUERY_TIMEOUT_SECONDS=20
SNOWFLAKE_HISTORY_ROW_LIMIT=25
SNOWFLAKE_QUERY_TAG=fde_servicing_agent_history_read
SNOWFLAKE_APPLICATION=FDE_SERVICING_AGENT_POC
```

The private-key contents remain in the external encrypted `.p8` file; only its
path and passphrase are read from `.env`. The connector receives the passphrase
directly, so startup and tests never use `getpass` or another interactive prompt.
Restrict local `.env` permissions because it contains plaintext development
secrets, and never paste the file into logs or support messages.

The adapter requests `SNOWFLAKE_JWT`, verifies that Snowflake actually selected
the configured role, database, and schema, discovers only approved evidence
columns, and issues only `SELECT` statements. `CASE_HISTORY` and
`SUPPORT_INTERACTIONS` are filtered by both server-resolved customer and case;
`CUSTOMER_ACTIVITY` is filtered by the authorized customer. Credentials and row
contents are never written to application logs.

If Snowflake authentication, authorization, or a query fails, the complex answer
continues only with independently authorized CRM/RAG evidence and returns an
explicit `history_debug.status = unavailable` warning. It does not silently use
mock history.

### Seed the synthetic Snowflake history

[`snowflake\seed_synthetic_history.sql`](snowflake/seed_synthetic_history.sql)
contains ID-keyed `MERGE` statements for the three approved history tables. Run
it with a SnowSQL connection whose selected role owns the synthetic demo tables
or has equivalently scoped `MERGE` privileges:

```powershell
Set-Location C:\path\to\serviceagent
snowsql -c your_servicing_demo_owner_connection `
  -f .\snowflake\seed_synthetic_history.sql
```

The connection alias is local SnowSQL configuration, not an application
credential. The script is repeatable, updates/inserts only named synthetic IDs,
deletes nothing, and ends with three expected-count verification queries. It
also corrects legacy `HIST-001` to the authorized `CUST-1001` identifier. Do not
run it as `SERVICING_HISTORY_READER`, and do not grant that application role
`INSERT`, `UPDATE`, `DELETE`, `MERGE`, or ownership. The application connector
continues to issue parameterized `SELECT` statements only.

The canonical scenarios are:

- `CASE-4821`: Premium telemetry investigation with observed delays and retries,
  scheduled-but-unconfirmed maintenance, requested diagnostics of unknown receipt,
  and no confirmed root cause or resolution date.
- `CASE-4822`: Standard billing request with administrator verification, a
  completed update, customer confirmation, and a recorded resolution.
- `CASE-4823`: Standard dashboard investigation awaiting customer diagnostics;
  a failed reproduction attempt is not treated as proof of non-occurrence.
- `CASE-4824`: Standard telemetry monitoring where one retry recovered and the
  escalation threshold was not reached.
- `CASE-4825`: Premium telemetry incident with a confirmed certificate cause,
  completed certificate rotation and queued replay, verified backlog clearance,
  and a recorded resolution.

## Run locally

Terminal 1 — API:

```powershell
.\.venv\Scripts\Activate.ps1
python -m servicing_agent.api
```

Terminal 2 — UI:

```powershell
.\.venv\Scripts\Activate.ps1
python -m servicing_agent.run_streamlit
```

Open <http://localhost:8501>. API documentation is at
<http://localhost:8000/docs>.

The UI has three focused areas:

- **Customer servicing** — selectable reproducible scenarios, chat, citations,
  execution traces, and Snowflake history freshness/latency when enabled.
- **Controlled actions** — simulated note confirmation, reconciliation, human
  escalation, and deterministic SLA status.
- **Evaluation dashboard** — measured quality, latency, scenario outcomes, and
  failure categories loaded from `evaluations\latest_results.json`.

The default answer generator is deterministic and explicitly says it is not an
LLM. Retrieval and policy applicability are real local operations in every mode.
CRM writes, handoff queues, and SLA escalations are explicitly simulated in the
UI and persist only in local SQLite.

## Local Ollama generation

The existing OpenAI-compatible generator works directly with Ollama; no separate
client is required. Install Ollama, start it, and download the configured model:

```powershell
ollama serve
ollama pull qwen3:4b-instruct
```

The root `.env` uses:

```dotenv
SERVICE_AGENT_LLM_PROVIDER=openai_compatible
SERVICE_AGENT_LLM_API_KEY=ollama
SERVICE_AGENT_LLM_MODEL=qwen3:4b-instruct
SERVICE_AGENT_LLM_API_BASE=http://localhost:11434/v1
SERVICE_AGENT_LLM_TIMEOUT_SECONDS=120
SERVICE_AGENT_LLM_TEMPERATURE=0
SERVICE_AGENT_LLM_MAX_TOKENS=512
SERVICE_AGENT_LLM_REASONING_EFFORT=none
```

Restart `python -m servicing_agent.api` after changing `.env`. The key is read
only through centralized settings; `ollama` is a non-secret placeholder required
by the OpenAI-compatible request format. The configurable 120-second timeout
allows for CPU model startup and generation. Qwen3 receives its documented
`/no_think` switch on the user turn, and the OpenAI-compatible request uses
`reasoning_effort=none` to request no thinking output. Ollama documents this
control for `/v1/chat/completions` in its
[OpenAI compatibility reference](https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx#v1chatcompletions).
The 512-token output budget remains bounded for concise servicing answers.
Responses that end with `finish_reason=length` are rejected and use the
intent-aware deterministic fallback.
Retrieval, authorization, policy
selection, and the ticket-status path are never delegated to the LLM. Missing or
invalid citations cause deterministic fallback.

Customer-facing LLM output is also rejected when it contains planning or
thinking text, ends mid-sentence, does not answer the requested answer type,
makes an unsupported future-resolution promise, or attaches a citation to a
claim with no substantive overlap with that source. Guarantee and controlled
action requests are handled deterministically without an LLM call. The fallback
answers status, explanation, timeline, guarantee, unknown-information,
escalation, and action requests directly before adding only material evidence.

Snowflake `EVENT_TIME` is treated as the historical event timestamp. Retrieval
time remains separate in `history_debug.retrieved_at` and is never substituted
as a citation's event date; a missing event timestamp is labeled unavailable.

For complex requests, the configured generator receives only the already
authorized evidence selected by the application, grouped as current CRM facts,
verified entitlements, approved RAG evidence, and read-only Snowflake historical
context. Its output must cite the supplied source labels and distinguish missing
information from confirmed facts. The Streamlit response shows the actual mode
(`Configured LLM` or `Deterministic evidence fallback`) and every returned source
ID. LLM failure never changes authorization, deterministic SLA decisions, or
controlled-write confirmation rules.

Run the real local smoke test after Ollama is serving the model:

```powershell
$env:RUN_OLLAMA_INTEGRATION = "true"
python -m pytest -m ollama_integration tests\test_ollama_connection.py -q
Remove-Item Env:RUN_OLLAMA_INTEGRATION
```

The default suite keeps network integrations disabled. The local-only smoke test
verifies invocation, grounded output or an explicit validated fallback, and the
deterministic guarantee path. The combined test below is the strict LLM-success
gate.

Streamlit uses separate bounded HTTP budgets: the general API timeout covers
connect/write/pool operations, while `SERVICE_AGENT_UI_API_READ_TIMEOUT_SECONDS`
defaults to 240 seconds for Snowflake plus a cold local model invocation. A read
timeout is reported separately from connection failures, HTTP rejections, and
invalid JSON responses. Successful complex responses expose Snowflake/history,
retrieval, generation, and total request latency in the debug panel.

To validate the complete live path with Snowflake, hybrid retrieval, and Ollama:

```powershell
$env:RUN_SNOWFLAKE_INTEGRATION = "true"
$env:RUN_OLLAMA_INTEGRATION = "true"
python -m pytest -s -m "snowflake_integration and ollama_integration" `
  tests\test_live_snowflake_ollama.py -q
Remove-Item Env:RUN_SNOWFLAKE_INTEGRATION
Remove-Item Env:RUN_OLLAMA_INTEGRATION
```

The combined live test requires both investigation questions to return
`generation_mode=llm`; a safe deterministic fallback fails the test. It also
asserts a normal `stop` finish reason, a bounded nonzero output-token count,
authorized Snowflake history, effective hybrid retrieval, and inline citations.

To repeat the matched three-run benchmark (one excluded warm-up, then each
question three times), keep Snowflake and hybrid retrieval enabled in `.env` and
run:

```powershell
python -m servicing_agent.benchmark_ollama --runs 3 `
  --output evaluations\qwen3_4b_instruct_live.json

# Optional comparison, if the old model is still installed locally:
python -m servicing_agent.benchmark_ollama --runs 3 --model qwen3:4b `
  --output evaluations\qwen3_4b_baseline_live.json
```

The JSON artifacts contain aggregate diagnostics only: generation and total
latency, finish reason, output-token count, citation/relevance checks, fallback
state/reason, cited source IDs, and authorized history record count. They do not
store generated answer text, source snippets, credentials, or private-key data.
Measured on the local Windows CPU demo environment on 2026-11-08, with a
512-token limit:

| Model / question | Mean generation | Mean total | Finish reasons | Mean output tokens | Citation valid | Relevant | Fallback |
|---|---:|---:|---|---:|---:|---:|---:|
| `qwen3:4b` / still investigating | 113.83 s | 117.69 s | `length` 3/3 | 512 | 3/3 | 3/3 | 3/3 |
| `qwen3:4b-instruct` / still investigating | 25.47 s | 28.52 s | `stop` 3/3 | 208 | 3/3 | 3/3 | 0/3 |
| `qwen3:4b` / escalation reason | 97.45 s | 101.27 s | `length` 3/3 | 512 | 3/3 | 3/3 | 3/3 |
| `qwen3:4b-instruct` / escalation reason | 19.32 s | 22.75 s | `stop` 3/3 | 152 | 3/3 | 3/3 | 0/3 |

These are local measurements, not production claims. The benchmark's citation
validity metric checks that every returned citation label appears inline;
`generation_mode=llm` additionally proves the existing substantive citation and
answer-quality validators accepted the model response. Relevance is the
benchmark's question-specific deterministic check. Since every instruct response
completed normally well below 512 tokens, the configured limit was not increased.

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
   snippets, retrieval rankings, and the execution trace. With Snowflake enabled,
   the answer also shows `SF-*` history citations and history freshness.
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
The authorization denial occurs before any Snowflake connection is opened.

### Snowflake history demonstration

After setting the secure Snowflake variables and starting the API, submit:

```text
Using customer activity, case changes, and support interactions, summarize the history of CASE-4821.
```

Expected: the response remains a LangGraph `complex_request`; current status is
still sourced from SQLite CRM; authorized history appears as `SF-*` citations;
and the debug panel reports `backend=snowflake`, per-table counts, retrieval time,
and the newest source timestamp. Switch to `demo-agent-jordan` and repeat the
request: no Snowflake query or history citation is allowed.

The local CRM authoritatively links `CASE-4821` to `CUST-1001`. If an older
synthetic Snowflake seed links `HIST-001` to legacy customer `CUST-42`, an
authorized `CASE-4821` lookup correctly returns no history because both the
customer and case predicates are enforced in Snowflake. Run
`snowflake\seed_correct_case_4821_customer.sql` once with the synthetic-data
owner role to correct that seed. The script is idempotent and narrowly scoped;
do not run it as, or grant writes to, `SERVICING_HISTORY_READER`.

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
python -m servicing_agent.sla_sweep --as-of "2026-11-08T12:00:00Z"
python -m servicing_agent.sla_sweep --as-of "2026-11-08T12:20:00Z"
```

The database path and backup delay come from `SERVICE_AGENT_DB_PATH` and
`SLA_BACKUP_AFTER_MINUTES` in `.env`.

The second command reconciles with the existing SLA handoff instead of creating a
duplicate and routes the still-`CREATED`, twenty-minute-old handoff to **Backup
Escalation Queue (simulated)**. Accepted/resolved handoffs are not rerouted.

## Tests

Tests use mocked embeddings and do not download a model:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
```

The test suite preserves every Milestone 1–4 regression and adds demo-reset,
end-to-end evaluator, deterministic-versus-LLM labeling, dashboard parsing,
scenario-catalog, major Streamlit-path coverage, mocked Snowflake connector
queries, authorization boundaries, cleanup, and no-fallback handling. Semantic
unit tests use mocked embeddings and do not download the model. Response-quality
regressions cover intent-aware answers, reasoning/partial-output rejection,
unsupported promises and root causes, claim-level citation relevance, evidence
polarity, planned-versus-completed actions, and historical event-time accuracy.
The full suite collects 123 tests: 118 pass offline and the five network tests
are skipped unless explicitly enabled.

The real Snowflake test is opt-in and runs the query through FastAPI and the
LangGraph workflow. It never prompts interactively:

```powershell
# Configure Snowflake in .env first; this process override opts into the live test.
$env:RUN_SNOWFLAKE_INTEGRATION = "1"
python -m pytest -m snowflake_integration tests\test_snowflake_connection.py -q
Remove-Item Env:RUN_SNOWFLAKE_INTEGRATION
```

Without `RUN_SNOWFLAKE_INTEGRATION=1`, this one network test is reported as
skipped while all mocked tests still run.

## Run the end-to-end evaluation

Initialize the model/index first, then run each mode three times:

```powershell
.\.venv\Scripts\Activate.ps1
python -m servicing_agent.index_embeddings
python -m servicing_agent.evaluation
```

The default dataset, output, modes, run count, fixed clock, and generation mode
come from the `EVALUATION_*` block in `.env`. Command-line flags remain available
as explicit one-off overrides.

For a fully cached/offline run:

```dotenv
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Then run `python -m servicing_agent.evaluation`.

Optional genuine configured-LLM run:

```dotenv
EVALUATION_MODES=hybrid
EVALUATION_RUNS=1
EVALUATION_USE_CONFIGURED_LLM=true
EVALUATION_RESULTS_PATH=evaluations/results/llm-latest.json
```

Configure the LLM block, then run `python -m servicing_agent.evaluation`.

The default evaluator uses the deterministic answer generator so LLM variability
and API cost do not affect regression metrics. Every scenario starts from reset
synthetic data in an isolated temporary database. Model download and embedding
index initialization are excluded from warm query latency.

### Current synthetic evaluation results

Measured on November 9, 2026 using 24 synthetic scenarios, three runs per
scenario, the deterministic generator, CPU MiniLM, and warm model/document
caches. The evaluator forces network-free mock history even when the application
`.env` enables Snowflake; live history has separate opt-in integration tests.

| Mode | Task success | Recall@5 | Precision@3 | Grounded | Answer quality | Claim/citation relevance | Unsupported claims avoided | Required evidence | Failed runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lexical | 0.9167 | 0.8750 | 0.3056 | 0.8824 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 6/72 |
| Semantic | 0.8750 | 0.8333 | 0.2778 | 0.8235 | 0.9412 | 1.0000 | 1.0000 | 0.9412 | 9/72 |
| Hybrid | 1.0000 | 1.0000 | 0.3611 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0/72 |

| Mode | Policy | Authorization | Action safety | Escalation | Recovery |
|---|---:|---:|---:|---:|---:|
| Lexical | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Semantic | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

| Mode | Retrieval P50/P95 | Tool P50/P95 | End-to-end P50/P95 |
|---|---:|---:|---:|
| Lexical | 0.117 / 0.155 ms | 4.591 / 41.148 ms | 8.762 / 107.793 ms |
| Semantic | 13.452 / 18.142 ms | 16.322 / 49.213 ms | 24.235 / 119.496 ms |
| Hybrid | 16.039 / 22.528 ms | 22.398 / 49.395 ms | 29.846 / 122.702 ms |

Lexical still misses the paraphrased call transcript and therefore cannot satisfy
the two-document investigation contract. Semantic-only misses the exact
escalation record, the two-document investigation contract, and one dashboard
policy contract. Hybrid retrieved the required evidence and satisfied all 24
synthetic scenario contracts in this run. No evaluation assertion or seed fact
was weakened. Precision@3 can exceed `0.3333` because the multi-source scenario
declares two relevant documents. The complete measured report, including
per-scenario outcomes, claim/citation relevance, claim-safety fields, and
generation-mode counts, is in `evaluations\latest_results.json`.

These results are regression measurements over synthetic data, not evidence of
production superiority or statistical significance. No CSAT, CPRR, case
deflection, ROI, or other business outcome was measured or inferred.

## Troubleshooting

- **Dashboard says no report:** run the end-to-end evaluation command above or set
  `EVALUATION_RESULTS_PATH` in `.env` to a schema-v2 report.
- **Semantic mode falls back to lexical:** install `.[semantic]`, initialize the
  embedding index, remove offline variables for the first model download, then
  restart FastAPI so the long-lived process imports the semantic dependency.
- **Offline model load fails:** confirm `RETRIEVAL_CACHE_DIR` points to the cache
  populated by `python -m servicing_agent.index_embeddings`.
- **Missing LLM key:** expected behavior is the explicitly labeled deterministic
  fallback. Retrieval, actions, and SLA rules continue to work.
- **Snowflake configuration error at startup:** confirm all eight required
  `SNOWFLAKE_*` identity/key entries are set in root `.env` and that the key path
  is readable.
- **Snowflake history is unavailable:** inspect `history_debug.failure_category`
  (`authentication`, `authorization`, or `query`) and Snowflake query history for
  the fixed query tag `fde_servicing_agent_history_read`. The application does
  not log connector details, passphrases, private-key contents, or returned rows.
- **Snowflake returns no history:** confirm the synthetic Snowflake customer and
  case identifiers match the CRM `customer_id`/`ticket_id`. The adapter will not
  weaken the filter or substitute another customer's history.
- **A demo starts in an unexpected state:** stop the API, run `demo_reset`, then
  restart Uvicorn so long-lived service objects reopen canonical data.
- **Port already in use:** change `SERVICE_AGENT_API_PORT`,
  `SERVICE_AGENT_API_URL`, and/or `SERVICE_AGENT_UI_PORT` in `.env`, then restart
  both launchers.

## Milestone 4 boundaries

This remains a local demonstration: there is no real Salesforce integration,
Snowflake write access, production authentication, financial transaction,
refund, cloud deployment, vector database, or reranker. The Snowflake adapter is
limited by application code and the `SERVICING_HISTORY_READER` role to synthetic
read-only history; this is not a production security certification. CRM mutations
and queue routing are SQLite-only simulations. Customer confirmation is a
demo-user attestation, not a customer signature or production consent record.
There is no continuously running scheduler; SLA sweeps must be invoked by a
request or CLI. The corpus and evaluation set are small and synthetic, local
database/embedding files are unencrypted, and optional LLM
generation has not been production-evaluated for safety, latency, or cost.
The dashboard reads a generated local JSON snapshot rather than a live production
telemetry system. Timing is machine-specific and excludes cold model download.
This is demo-ready for a technical walkthrough, not production-ready software.
