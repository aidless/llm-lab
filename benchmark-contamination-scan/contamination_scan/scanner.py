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

    def to_dict(self) -> dict[str, Any]:
        return {
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
    best = Match("", False, 0.0, 0.0, "")
    for document, vector in zip(documents, document_vectors, strict=True):
        document_text = str(document["text"])
        exact = normalized_task in normalize_text(document_text)
        ngram_score = ngram_containment(task_text, document_text, ngram_n)
        semantic_score = cosine(task_vector, vector)
        if (exact and not best.exact) or ngram_score > best.ngram_score or semantic_score > best.semantic_score:
            best = Match(
                document_id=str(document["id"]),
                exact=exact,
                ngram_score=ngram_score,
                semantic_score=semantic_score,
                source=str(document.get("source", "")),
            )
    return best


def is_suspect(
    match: Match,
    *,
    ngram_threshold: float,
    semantic_threshold: float,
) -> bool:
    return match.exact or match.ngram_score >= ngram_threshold or match.semantic_score >= semantic_threshold
