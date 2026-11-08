from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_path: Path
    log_level: str
    llm_provider: str
    llm_model: str
    llm_api_base: str
    llm_api_key: str | None
    llm_timeout_seconds: float
    retrieval_mode: str
    retrieval_model: str
    retrieval_cache_dir: Path
    retrieval_rrf_k: int
    retrieval_semantic_min_score: float
    retrieval_semantic_min_top_score: float
    history_backend: str
    sla_backup_after_minutes: int

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            database_path=Path(
                os.getenv("SERVICE_AGENT_DB_PATH", "data/service_agent.db")
            ),
            log_level=os.getenv("SERVICE_AGENT_LOG_LEVEL", "INFO").upper(),
            llm_provider=os.getenv("SERVICE_AGENT_LLM_PROVIDER", "none").lower(),
            llm_model=os.getenv("SERVICE_AGENT_LLM_MODEL", "gpt-4.1-mini"),
            llm_api_base=os.getenv(
                "SERVICE_AGENT_LLM_API_BASE", "https://api.openai.com/v1"
            ),
            llm_api_key=os.getenv("SERVICE_AGENT_LLM_API_KEY"),
            llm_timeout_seconds=float(
                os.getenv("SERVICE_AGENT_LLM_TIMEOUT_SECONDS", "20")
            ),
            retrieval_mode=os.getenv("RETRIEVAL_MODE", "lexical").lower(),
            retrieval_model=os.getenv(
                "RETRIEVAL_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            retrieval_cache_dir=Path(
                os.getenv("RETRIEVAL_CACHE_DIR", "data/embedding_cache")
            ),
            retrieval_rrf_k=int(os.getenv("RETRIEVAL_RRF_K", "60")),
            retrieval_semantic_min_score=float(
                os.getenv("RETRIEVAL_SEMANTIC_MIN_SCORE", "0.25")
            ),
            retrieval_semantic_min_top_score=float(
                os.getenv("RETRIEVAL_SEMANTIC_MIN_TOP_SCORE", "0.45")
            ),
            history_backend=os.getenv("HISTORY_BACKEND", "mock").lower(),
            sla_backup_after_minutes=int(
                os.getenv("SLA_BACKUP_AFTER_MINUTES", "15")
            ),
        )

