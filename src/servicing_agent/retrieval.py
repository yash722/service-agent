from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, Sequence

EvidenceStatus = Literal["sufficient", "insufficient", "conflicting"]
RetrievalMode = Literal["lexical", "semantic", "hybrid"]

_CASE_REFERENCE = re.compile(r"\bcase-\d+\b", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "case",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "why",
    "with",
}


def tokenize(value: str) -> list[str]:
    value = _CASE_REFERENCE.sub(" ", value.lower())
    return [token for token in _TOKEN.findall(value) if token not in _STOP_WORDS]


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    source_id: str
    title: str
    document_type: str
    version: str
    approval_status: str
    effective_date: date
    expiration_date: date | None
    product_ids: tuple[str, ...]
    subscription_tiers: tuple[str, ...]
    customer_id: str | None
    case_id: str | None
    access_scope: str
    conflicts_with: tuple[str, ...]
    text: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KnowledgeDocument":
        return cls(
            document_id=value["document_id"],
            source_id=value["source_id"],
            title=value["title"],
            document_type=value["document_type"],
            version=value["version"],
            approval_status=value["approval_status"],
            effective_date=date.fromisoformat(value["effective_date"]),
            expiration_date=(
                date.fromisoformat(value["expiration_date"])
                if value.get("expiration_date")
                else None
            ),
            product_ids=tuple(value.get("product_ids", [])),
            subscription_tiers=tuple(value.get("subscription_tiers", [])),
            customer_id=value.get("customer_id"),
            case_id=value.get("case_id"),
            access_scope=value["access_scope"],
            conflicts_with=tuple(value.get("conflicts_with", [])),
            text=value["text"],
        )

    @property
    def citation_label(self) -> str:
        return f"[{self.source_id} v{self.version}]"

    @property
    def embedding_text(self) -> str:
        return (
            f"{self.title}. Type: {self.document_type}. "
            f"Products: {', '.join(self.product_ids)}. {self.text}"
        )


@dataclass(frozen=True)
class RetrievedEvidence:
    document: KnowledgeDocument
    score: float


@dataclass(frozen=True)
class RetrievalDiagnostics:
    requested_mode: str = "lexical"
    effective_mode: str = "lexical"
    latency_ms: float = 0.0
    rankings: dict[str, tuple[str, ...]] = field(default_factory=dict)
    selected_document_ids: tuple[str, ...] = ()
    fallback_reason: str | None = None
    embedding_cache_status: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    evidence: tuple[RetrievedEvidence, ...]
    status: EvidenceStatus
    warnings: tuple[str, ...]
    diagnostics: RetrievalDiagnostics = field(default_factory=RetrievalDiagnostics)


@dataclass(frozen=True)
class RankedDocument:
    document: KnowledgeDocument
    score: float


class EmbeddingProvider(Protocol):
    model_id: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class EmbeddingUnavailableError(RuntimeError):
    pass


class SentenceTransformerEmbeddingProvider:
    """Lazy CPU Sentence Transformers adapter."""

    def __init__(self, model_id: str, cache_dir: str | Path):
        self.model_id = model_id
        self.cache_dir = Path(cache_dir)
        self._model: Any | None = None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        try:
            vectors = model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # library/runtime boundary
            raise EmbeddingUnavailableError(
                f"sentence-transformer encoding failed: {type(exc).__name__}"
            ) from exc
        return [[float(value) for value in vector] for vector in vectors]

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingUnavailableError(
                "sentence-transformers is not installed"
            ) from exc
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = SentenceTransformer(
                self.model_id,
                device="cpu",
                cache_folder=str(self.cache_dir),
            )
        except Exception as exc:  # model download/load boundary
            raise EmbeddingUnavailableError(
                f"sentence-transformer model unavailable: {type(exc).__name__}"
            ) from exc
        return self._model


class DocumentEmbeddingIndex:
    """Fingerprint-keyed JSON cache for normalized document embeddings."""

    def __init__(
        self,
        documents: Sequence[KnowledgeDocument],
        provider: EmbeddingProvider,
        cache_dir: str | Path,
    ):
        self.documents = tuple(documents)
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.fingerprint = self._fingerprint()
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "-", provider.model_id)
        self.cache_path = (
            self.cache_dir
            / "indexes"
            / f"{safe_model}-{self.fingerprint[:16]}.json"
        )
        self._vectors: dict[str, list[float]] | None = None
        self.last_cache_status: str | None = None

    def ensure(self, *, force: bool = False) -> dict[str, list[float]]:
        if self._vectors is not None and not force:
            self.last_cache_status = "memory"
            return self._vectors
        if not force:
            cached = self._read_cache()
            if cached is not None:
                self._vectors = cached
                self.last_cache_status = "disk"
                return cached

        vectors = self.provider.encode(
            [document.embedding_text for document in self.documents]
        )
        if len(vectors) != len(self.documents) or any(not vector for vector in vectors):
            raise EmbeddingUnavailableError("embedding provider returned invalid vectors")
        self._vectors = {
            document.document_id: vector
            for document, vector in zip(self.documents, vectors, strict=True)
        }
        self.last_cache_status = "refreshed" if force else "created"
        self._write_cache(self._vectors)
        return self._vectors

    def refresh(self) -> str:
        self.ensure(force=True)
        return self.last_cache_status or "refreshed"

    def _fingerprint(self) -> str:
        payload = {
            "model_id": self.provider.model_id,
            "documents": [
                {
                    "document_id": document.document_id,
                    "embedding_text": document.embedding_text,
                }
                for document in self.documents
            ],
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _read_cache(self) -> dict[str, list[float]] | None:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        if (
            payload.get("model_id") != self.provider.model_id
            or payload.get("fingerprint") != self.fingerprint
            or set(payload.get("vectors", {}))
            != {document.document_id for document in self.documents}
        ):
            return None
        vectors = payload["vectors"]
        if any(not isinstance(vector, list) or not vector for vector in vectors.values()):
            return None
        return {
            document_id: [float(value) for value in vector]
            for document_id, vector in vectors.items()
        }

    def _write_cache(self, vectors: dict[str, list[float]]) -> None:
        payload = {
            "model_id": self.provider.model_id,
            "fingerprint": self.fingerprint,
            "vectors": vectors,
        }
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.cache_path.with_suffix(".tmp")
            temporary_path.write_text(
                json.dumps(payload, separators=(",", ":")), encoding="utf-8"
            )
            temporary_path.replace(self.cache_path)
        except OSError:
            # Retrieval can continue from memory if the local cache is read-only.
            self.last_cache_status = "memory_only"


def load_default_corpus() -> tuple[KnowledgeDocument, ...]:
    corpus_path = files("servicing_agent").joinpath("corpus/knowledge.json")
    raw_documents = json.loads(corpus_path.read_text(encoding="utf-8"))
    return tuple(KnowledgeDocument.from_dict(item) for item in raw_documents)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise EmbeddingUnavailableError("embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def reciprocal_rank_fusion(
    rankings: dict[str, Sequence[RankedDocument]], *, k: int = 60
) -> list[RankedDocument]:
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: dict[str, float] = {}
    documents: dict[str, KnowledgeDocument] = {}
    for ranking in rankings.values():
        for rank, item in enumerate(ranking, start=1):
            document_id = item.document.document_id
            documents[document_id] = item.document
            scores[document_id] = scores.get(document_id, 0.0) + 1 / (k + rank)
    return [
        RankedDocument(documents[document_id], round(score, 8))
        for document_id, score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )
    ]


class HybridRetrievalService:
    """Authorization-safe lexical, semantic, and RRF hybrid retrieval."""

    VALID_MODES = {"lexical", "semantic", "hybrid"}

    def __init__(
        self,
        documents: Iterable[KnowledgeDocument] | None = None,
        *,
        mode: RetrievalMode | str = "lexical",
        embedding_provider: EmbeddingProvider | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_cache_dir: str | Path = "data/embedding_cache",
        rrf_k: int = 60,
        semantic_min_score: float = 0.25,
        semantic_min_top_score: float = 0.45,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported retrieval mode {mode!r}; expected lexical, semantic, or hybrid"
            )
        if rrf_k <= 0:
            raise ValueError("RRF k must be positive")
        self.documents = tuple(
            documents if documents is not None else load_default_corpus()
        )
        self.mode = str(mode)
        self.embedding_model = embedding_model
        self.embedding_cache_dir = Path(embedding_cache_dir)
        self.rrf_k = rrf_k
        self.semantic_min_score = semantic_min_score
        self.semantic_min_top_score = semantic_min_top_score
        self._embedding_provider = embedding_provider
        self._embedding_index: DocumentEmbeddingIndex | None = None
        self._document_terms = {
            document.document_id: Counter(
                tokenize(f"{document.title} {document.text}")
            )
            for document in self.documents
        }
        document_frequency: Counter[str] = Counter()
        for terms in self._document_terms.values():
            document_frequency.update(terms.keys())
        corpus_size = len(self.documents)
        self._idf = {
            term: math.log((corpus_size + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        *,
        as_of: date,
        customer_id: str | None,
        case_id: str | None,
        entitlements: list[dict[str, Any]],
        top_k: int = 5,
    ) -> RetrievalResult:
        started = time.perf_counter()
        eligible, warnings = self._eligible_documents(
            query,
            as_of=as_of,
            customer_id=customer_id,
            case_id=case_id,
            entitlements=entitlements,
        )
        rankings: dict[str, list[RankedDocument]] = {}
        fallback_reason: str | None = None
        cache_status: str | None = None
        effective_mode = self.mode

        if self.mode in {"lexical", "hybrid"}:
            rankings["lexical"] = self._lexical_rank(query, eligible, case_id)

        if self.mode in {"semantic", "hybrid"}:
            try:
                rankings["semantic"] = self._semantic_rank(query, eligible)
                if self._embedding_index:
                    cache_status = self._embedding_index.last_cache_status
            except EmbeddingUnavailableError as exc:
                fallback_reason = str(exc)
                effective_mode = "lexical"
                if "lexical" not in rankings:
                    rankings["lexical"] = self._lexical_rank(
                        query, eligible, case_id
                    )
                warnings.append(
                    "Semantic retrieval was unavailable; lexical fallback was used."
                )

        if effective_mode == "hybrid":
            final_ranking = reciprocal_rank_fusion(rankings, k=self.rrf_k)
        else:
            final_ranking = rankings.get(effective_mode, [])

        selected_items = tuple(final_ranking[:top_k])
        selected = tuple(
            RetrievedEvidence(item.document, round(item.score, 8))
            for item in selected_items
        )
        conflicts = self._has_conflicts(selected)
        if conflicts:
            warnings.append("The retrieved evidence contains an explicit conflict.")
            status: EvidenceStatus = "conflicting"
        elif selected:
            status = "sufficient"
        else:
            status = "insufficient"
            warnings.append("No relevant, current, authorized evidence was found.")

        diagnostics = RetrievalDiagnostics(
            requested_mode=self.mode,
            effective_mode=effective_mode,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            rankings={
                name: tuple(item.document.document_id for item in ranking)
                for name, ranking in rankings.items()
            },
            selected_document_ids=tuple(
                item.document.document_id for item in selected
            ),
            fallback_reason=fallback_reason,
            embedding_cache_status=cache_status,
        )
        return RetrievalResult(selected, status, tuple(warnings), diagnostics)

    def refresh_embeddings(self, *, force: bool = True) -> str:
        index = self._ensure_embedding_index()
        if force:
            return index.refresh()
        index.ensure()
        return index.last_cache_status or "memory"

    def _eligible_documents(
        self,
        query: str,
        *,
        as_of: date,
        customer_id: str | None,
        case_id: str | None,
        entitlements: list[dict[str, Any]],
    ) -> tuple[list[KnowledgeDocument], list[str]]:
        eligible: list[KnowledgeDocument] = []
        rejected_reasons: Counter[str] = Counter()
        query_terms = Counter(tokenize(query))
        for document in self.documents:
            # Tenant/record filtering precedes lexical or semantic ranking.
            if document.access_scope == "customer_record" and (
                customer_id is None
                or case_id is None
                or document.customer_id != customer_id
                or document.case_id != case_id
            ):
                continue
            rejection = self._rejection_reason(
                document,
                as_of=as_of,
                entitlements=entitlements,
            )
            if rejection:
                # Eligibility diagnostics are limited to public metadata and use
                # lexical overlap; protected records never influence warnings.
                if document.access_scope == "public" and self._has_lexical_overlap(
                    query_terms, document
                ):
                    rejected_reasons[rejection] += 1
                continue
            eligible.append(document)

        eligible = self._latest_applicable_policy_versions(eligible)
        warnings: list[str] = []
        if rejected_reasons.get("outdated"):
            warnings.append(
                "An older matching version was excluded because it was no longer effective."
            )
        if rejected_reasons.get("unapproved"):
            warnings.append("A matching version was excluded because it was not approved.")
        if rejected_reasons.get("inapplicable"):
            warnings.append(
                "Other matching policy versions were excluded because they did not "
                "apply to the verified entitlement."
            )
        return eligible, warnings

    def _lexical_rank(
        self,
        query: str,
        documents: Sequence[KnowledgeDocument],
        case_id: str | None,
    ) -> list[RankedDocument]:
        query_terms = Counter(tokenize(query))
        if not query_terms:
            return []
        ranking: list[RankedDocument] = []
        for document in documents:
            if not self._has_lexical_overlap(query_terms, document):
                continue
            score = self._lexical_score(query_terms, document)
            if case_id and document.case_id == case_id:
                score += 2.0
            if score > 0:
                ranking.append(RankedDocument(document, round(score, 8)))
        ranking.sort(key=lambda item: (-item.score, item.document.document_id))
        return ranking

    def _semantic_rank(
        self, query: str, documents: Sequence[KnowledgeDocument]
    ) -> list[RankedDocument]:
        if not documents:
            return []
        index = self._ensure_embedding_index()
        vectors = index.ensure()
        query_vector = index.provider.encode([query])[0]
        ranking = [
            RankedDocument(
                document,
                round(cosine_similarity(query_vector, vectors[document.document_id]), 8),
            )
            for document in documents
        ]
        ranking = [
            item for item in ranking if item.score >= self.semantic_min_score
        ]
        ranking.sort(key=lambda item: (-item.score, item.document.document_id))
        if not ranking or ranking[0].score < self.semantic_min_top_score:
            return []
        return ranking

    def _ensure_embedding_index(self) -> DocumentEmbeddingIndex:
        if self._embedding_index is None:
            provider = self._embedding_provider or SentenceTransformerEmbeddingProvider(
                self.embedding_model, self.embedding_cache_dir / "models"
            )
            self._embedding_index = DocumentEmbeddingIndex(
                self.documents, provider, self.embedding_cache_dir
            )
        return self._embedding_index

    def _has_lexical_overlap(
        self, query_terms: Counter[str], document: KnowledgeDocument
    ) -> bool:
        document_terms = self._document_terms[document.document_id]
        overlap_count = sum(1 for term in query_terms if document_terms.get(term))
        required_overlap = 2 if len(query_terms) >= 3 else 1
        return overlap_count >= required_overlap

    def _lexical_score(
        self, query_terms: Counter[str], document: KnowledgeDocument
    ) -> float:
        document_terms = self._document_terms[document.document_id]
        score = 0.0
        title_terms = Counter(tokenize(document.title))
        for term, query_count in query_terms.items():
            term_count = document_terms.get(term, 0)
            if not term_count:
                continue
            score += (
                self._idf.get(term, 1.0)
                * (1 + math.log(term_count))
                * query_count
            )
            if title_terms.get(term):
                score += 0.75
        return score

    @staticmethod
    def _rejection_reason(
        document: KnowledgeDocument,
        *,
        as_of: date,
        entitlements: list[dict[str, Any]],
    ) -> str | None:
        if document.approval_status == "superseded" or (
            document.expiration_date and document.expiration_date < as_of
        ):
            return "outdated"
        if document.approval_status != "approved":
            return "unapproved"
        if document.effective_date > as_of:
            return "outdated"
        if document.document_type == "support_policy" and not any(
            entitlement["product_id"] in document.product_ids
            and entitlement["subscription_tier"] in document.subscription_tiers
            for entitlement in entitlements
        ):
            return "inapplicable"
        return None

    @staticmethod
    def _latest_applicable_policy_versions(
        documents: Sequence[KnowledgeDocument],
    ) -> list[KnowledgeDocument]:
        latest_by_source: dict[str, KnowledgeDocument] = {}
        non_policy: list[KnowledgeDocument] = []
        for document in documents:
            if document.document_type != "support_policy":
                non_policy.append(document)
                continue
            existing = latest_by_source.get(document.source_id)
            if existing is None or (
                document.effective_date,
                document.version,
            ) > (existing.effective_date, existing.version):
                latest_by_source[document.source_id] = document
        return [*non_policy, *latest_by_source.values()]

    @staticmethod
    def _has_conflicts(evidence: tuple[RetrievedEvidence, ...]) -> bool:
        selected_ids = {item.document.document_id for item in evidence}
        return any(
            selected_ids.intersection(item.document.conflicts_with)
            for item in evidence
        )


class LexicalRetrievalService(HybridRetrievalService):
    """Backwards-compatible lexical-only service."""

    def __init__(self, documents: Iterable[KnowledgeDocument] | None = None):
        super().__init__(documents, mode="lexical")
