"""BM25 over Arabic-normalised text.

Written out rather than taken from a library because the tokeniser has to be the
Arabic-normalising one from :mod:`shurta.arabic`, on both the passages and the
query. BM25 Okapi with the usual defaults: ``k1`` bounds what a repeated term
adds, ``b`` how strongly a long passage is penalised.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from shurta.arabic import tokenise

K1 = 1.5
B = 0.75


@dataclass
class Bm25Index:
    """An in-memory BM25 index over a fixed list of documents."""

    document_frequency: dict[str, int] = field(default_factory=dict)
    term_frequencies: list[Counter[str]] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    average_length: float = 0.0

    @property
    def size(self) -> int:
        return len(self.term_frequencies)

    @classmethod
    def build(cls, documents: list[str]) -> Bm25Index:
        """Index a list of documents, tokenising each with Arabic normalisation."""
        term_frequencies = [Counter(tokenise(document)) for document in documents]
        lengths = [sum(counts.values()) for counts in term_frequencies]

        document_frequency: dict[str, int] = {}
        for counts in term_frequencies:
            for term in counts:
                document_frequency[term] = document_frequency.get(term, 0) + 1

        return cls(
            document_frequency=document_frequency,
            term_frequencies=term_frequencies,
            lengths=lengths,
            average_length=(sum(lengths) / len(lengths)) if lengths else 0.0,
        )

    def _inverse_document_frequency(self, term: str) -> float:
        """How much a term's presence tells us, in the BM25 Okapi form.

        A term in every document carries almost no information, and the shape of
        this expression drives its weight towards zero without going negative.
        """
        frequency = self.document_frequency.get(term, 0)
        if frequency == 0:
            return 0.0
        total = self.size
        return math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))

    def scores(self, query: str) -> list[float]:
        """Score every document against the query. Index order is preserved."""
        terms = tokenise(query)
        results = [0.0] * self.size
        if not terms or not self.average_length:
            return results

        weights = {term: self._inverse_document_frequency(term) for term in set(terms)}

        for position, counts in enumerate(self.term_frequencies):
            length = self.lengths[position]
            normaliser = K1 * (1 - B + B * length / self.average_length)
            total = 0.0
            for term in set(terms):
                weight = weights[term]
                if weight == 0.0:
                    continue
                occurrences = counts.get(term, 0)
                if occurrences == 0:
                    continue
                total += weight * (occurrences * (K1 + 1)) / (occurrences + normaliser)
            results[position] = total
        return results

    def rank(self, query: str, limit: int) -> list[tuple[int, float]]:
        """The best ``limit`` documents as ``(index, score)``, best first.

        Documents that match nothing are left out rather than returned with a
        zero score, so that fusion is not handed a ranking of non-matches.
        """
        scored = [
            (position, score) for position, score in enumerate(self.scores(query)) if score > 0
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:limit]

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable form, for persisting the index to disk."""
        return {
            "document_frequency": self.document_frequency,
            "term_frequencies": [dict(counts) for counts in self.term_frequencies],
            "lengths": self.lengths,
            "average_length": self.average_length,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Bm25Index:
        """Rebuild an index from its stored form."""
        return cls(
            document_frequency={
                str(term): int(count) for term, count in payload["document_frequency"].items()
            },
            term_frequencies=[
                Counter({str(term): int(count) for term, count in counts.items()})
                for counts in payload["term_frequencies"]
            ],
            lengths=[int(length) for length in payload["lengths"]],
            average_length=float(payload["average_length"]),
        )
