"""Exact, character n-gram and semantic contamination scan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data import normalize_text
from .embeddings import cosine


@dataclass(frozen=True)
class Match:
    document_id: str
    exact: bool
    ngram_score: float
    semantic_score: float
    source: str
    lexical_match: dict[str, Any] | None = None
    semantic_match: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "exact": self.exact,
            "ngram_score": self.ngram_score,
            "semantic_score": self.semantic_score,
            "source": self.source,
            "lexical_match": self.lexical_match,
            "semantic_match": self.semantic_match,
        }

    def lexical_evidence(self) -> dict[str, Any]:
        return self.lexical_match or {
            "document_id": self.document_id,
            "exact": self.exact,
            "ngram_score": self.ngram_score,
            "semantic_score": self.semantic_score,
            "source": self.source,
        }

    def semantic_evidence(self) -> dict[str, Any]:
        return self.semantic_match or {
            "document_id": self.document_id,
            "exact": self.exact,
            "ngram_score": self.ngram_score,
            "semantic_score": self.semantic_score,
            "source": self.source,
        }


def char_ngrams(text: str, n: int = 13) -> set[str]:
    normalized = normalize_text(text)
    if len(normalized) <= n:
        return {normalized} if normalized else set()
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def ngram_containment(task_text: str, document_text: str, n: int = 13) -> float:
    task_grams = char_ngrams(task_text, n)
    if not task_grams:
        return 0.0
    document_grams = char_ngrams(document_text, n)
    return len(task_grams & document_grams) / len(task_grams)


def best_match(
    *,
    task_text: str,
    task_vector: list[float],
    documents: list[dict[str, Any]],
    document_vectors: list[list[float]],
    ngram_n: int,
) -> Match:
    normalized_task = normalize_text(task_text)
    best_lexical: dict[str, Any] | None = None
    best_semantic: dict[str, Any] | None = None
    for document, vector in zip(documents, document_vectors, strict=True):
        document_text = str(document["text"])
        exact = normalized_task in normalize_text(document_text)
        ngram_score = ngram_containment(task_text, document_text, ngram_n)
        semantic_score = cosine(task_vector, vector)
        candidate = {
            "document_id": str(document["id"]),
            "exact": exact,
            "ngram_score": ngram_score,
            "semantic_score": semantic_score,
            "source": str(document.get("source", "")),
        }
        lexical_rank = (exact, ngram_score)
        current_lexical_rank = (
            bool(best_lexical["exact"]),
            float(best_lexical["ngram_score"]),
        ) if best_lexical is not None else (False, -1.0)
        if best_lexical is None or lexical_rank > current_lexical_rank:
            best_lexical = candidate
        if best_semantic is None or semantic_score > float(best_semantic["semantic_score"]):
            best_semantic = candidate
    if best_lexical is None or best_semantic is None:
        return Match("", False, 0.0, 0.0, "")
    primary = best_lexical if best_lexical["exact"] or best_lexical["ngram_score"] > 0 else best_semantic
    return Match(
        document_id=str(primary["document_id"]),
        exact=bool(primary["exact"]),
        ngram_score=float(primary["ngram_score"]),
        semantic_score=float(primary["semantic_score"]),
        source=str(primary["source"]),
        lexical_match=best_lexical,
        semantic_match=best_semantic,
    )


def is_suspect(
    match: Match,
    *,
    ngram_threshold: float,
    semantic_threshold: float,
) -> bool:
    lexical = match.lexical_evidence()
    semantic = match.semantic_evidence()
    return bool(
        lexical["exact"]
        or float(lexical["ngram_score"]) >= ngram_threshold
        or float(semantic["semantic_score"]) >= semantic_threshold
    )
