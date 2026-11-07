# Salesforce FDE Servicing Agent PoC — Milestone 1

A local proof of concept for service-agent request routing and orchestration. It
uses synthetic CRM records only; it does **not** connect to Salesforce or require
enterprise credentials.

## What Milestone 1 includes

- FastAPI backend with `GET /health`, `POST /chat`, and generated API docs
- Streamlit chat UI with two deliberately different demo-user permission scopes
- SQLite data seeded automatically with synthetic customers and cases, including
  `CASE-4821`
- Deterministic routing for explicit ticket-status requests
- A small LangGraph workflow for other requests, backed by mock CRM, knowledge,
  and policy tools
- CRM authorization based on an immutable, server-resolved demo-user context
- One-line JSON execution logs with an `execution_id`
- pytest coverage for health, routing, workflow use, and authorization boundaries

## Project layout

```text
.
├── data/                         # Local SQLite database is created here
├── src/servicing_agent/
│   ├── api.py                    # FastAPI app and demo-user dependency
│   ├── auth.py                   # Server-side demo identities and scopes
│   ├── database.py               # SQLite schema and synthetic seed data
│   ├── intent_router.py          # Deterministic router
│   ├── logging_config.py         # JSON log formatter
│   ├── schemas.py                # API contracts
│   ├── service.py                # Chat orchestration
│   ├── streamlit_app.py          # Local UI
│   ├── tools.py                  # Authorized CRM and other mock tools
│   └── workflow.py               # LangGraph workflow
├── tests/
└── pyproject.toml
```

## Windows setup (PowerShell)

Prerequisite: Python 3.11 or newer installed with the `py` launcher. From the
repository root in VS Code's PowerShell terminal, run exactly:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The database needs no separate setup. The API creates `data\service_agent.db`
and idempotently seeds the synthetic records on startup.

## Run locally on Windows

Open two VS Code PowerShell terminals from the repository root.

Terminal 1 — API:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn servicing_agent.api:app --reload --port 8000
```

Terminal 2 — UI:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run src\servicing_agent\streamlit_app.py --server.port 8501
```

Open <http://localhost:8501>. FastAPI documentation is at
<http://localhost:8000/docs>.

Try `What is the status of CASE-4821?` as Alex Morgan. Then switch to Jordan
Lee and repeat it to see the CRM authorization boundary.

## API smoke checks (PowerShell)

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"

$headers = @{ "X-Demo-User" = "demo-agent-alex" }
$body = @{ message = "What is the status of CASE-4821?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" -Headers $headers -ContentType "application/json" -Body $body
```

`X-Demo-User` is a local demo selector, not authentication. The API maps it to
an allowlisted identity and permission scope. In a real deployment, a verified
identity provider would create this trusted context instead.

## Run tests

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
```

## Milestone boundary

This milestone intentionally has no real Salesforce integration, LLM calls,
advanced RAG, refund handling, data mutation, production authentication, or
production deployment infrastructure. The knowledge and policy responses are
clearly marked mock behavior so the orchestration and security boundary can be
demonstrated before later integrations are designed.
