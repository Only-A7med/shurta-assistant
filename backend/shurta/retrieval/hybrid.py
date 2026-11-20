"""The retriever: keyword search and vector search, fused.

BM25 is exact about a figure or a name but cannot cross languages. The ``bge-m3``
embeddings cross languages and paraphrase but are vaguer about a specific string.
Fused, vector search alone scores better overall (recall@1 0.910 against 0.890)
yet misses one code-switched question in five, which fusion gets right.

Both languages are always searched, with a small preference for passages in the
reader's own. See docs/architecture.md and eval/retrieval-results.md.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from shurta.chunking import Chunk
from shurta.models import Language
from shurta.retrieval.fusion import reciprocal_rank_fusion
from shurta.retrieval.store import SearchIndex

#: How many candidates each method contributes before fusion, wider than the
#: final number so fusion has something to disagree about.
CANDIDATES = 20

#: Weight on each ranking during fusion. Equal: a sweep favouring vector search
#: gained nothing overall and lost the code-switched questions.
LEXICAL_WEIGHT = 1.0
DENSE_WEIGHT = 1.0

#: Multiplier favouring a passage in the reader's own language. Small on
#: purpose: it settles a near-tie rather than overriding a better passage.
SAME_LANGUAGE_BOOST = 1.15

EmbedQuery = Callable[[str], Awaitable[Sequence[float]]]


@dataclass(frozen=True)
class Retrieved:
    """One retrieved chunk with the evidence for why it was retrieved."""

    chunk: Chunk
    score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None

    @property
    def found_by(self) -> str:
        if self.lexical_rank is not None and self.dense_rank is not None:
            return "both"
        if self.lexical_rank is not None:
            return "keyword"
        return "vector"


@dataclass(frozen=True)
class RetrievalResult:
    """What a search returned, and how long each half of it took."""

    chunks: tuple[Retrieved, ...] = ()
    best_similarity: float = 0.0
    """Cosine similarity of the closest passage to the question.

    Reported separately from the fused score, which is built from rank positions
    and so scores the top passage highly whether or not it is relevant. Only an
    absolute measure can say that nothing here is close.
    """

    embedding_ms: float = 0.0
    search_ms: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return self.embedding_ms + self.search_ms


class HybridRetriever:
    """Searches an index with both methods and fuses the results."""

    def __init__(self, index: SearchIndex, embed_query: EmbedQuery) -> None:
        self.index = index
        self._embed_query = embed_query

    def lexical_ranking(self, query: str, limit: int = CANDIDATES) -> list[int]:
        return [position for position, _score in self.index.lexical.rank(query, limit)]

    def dense_ranking(self, query_vector: Sequence[float], limit: int = CANDIDATES) -> list[int]:
        scores = self.index.dense_scores(np.asarray(query_vector, dtype=np.float32))
        if scores.size == 0:
            return []
        take = min(limit, scores.size)
        # argpartition finds the top rows without sorting the whole array.
        candidates = np.argpartition(-scores, take - 1)[:take]
        return [int(position) for position in candidates[np.argsort(-scores[candidates])]]

    async def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        language: Language | None = None,
        candidates: int = CANDIDATES,
    ) -> RetrievalResult:
        """Retrieve the best ``top_k`` chunks for a query."""
        if not self.index.chunks:
            return RetrievalResult()

        started = time.perf_counter()
        query_vector = await self._embed_query(query)
        embedding_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        similarities = self.index.dense_scores(np.asarray(query_vector, dtype=np.float32))
        best_similarity = float(similarities.max()) if similarities.size else 0.0
        lexical = self.lexical_ranking(query, candidates)
        dense = self.dense_ranking(query_vector, candidates)

        lexical_positions = {document: rank for rank, document in enumerate(lexical)}
        dense_positions = {document: rank for rank, document in enumerate(dense)}

        fused = reciprocal_rank_fusion([lexical, dense], weights=[LEXICAL_WEIGHT, DENSE_WEIGHT])
        if language is not None:
            fused = sorted(
                (
                    (
                        document,
                        score
                        * (
                            SAME_LANGUAGE_BOOST
                            if self.index.chunks[document].language is language
                            else 1.0
                        ),
                    )
                    for document, score in fused
                ),
                key=lambda pair: (-pair[1], pair[0]),
            )

        retrieved = tuple(
            Retrieved(
                chunk=self.index.chunks[document],
                score=score,
                lexical_rank=lexical_positions.get(document),
                dense_rank=dense_positions.get(document),
            )
            for document, score in fused[:top_k]
        )
        search_ms = (time.perf_counter() - started) * 1000

        return RetrievalResult(
            chunks=retrieved,
            best_similarity=best_similarity,
            embedding_ms=embedding_ms,
            search_ms=search_ms,
            timings={
                "embedding_ms": round(embedding_ms, 2),
                "search_ms": round(search_ms, 2),
                "total_ms": round(embedding_ms + search_ms, 2),
            },
        )

    async def search_lexical_only(self, query: str, *, top_k: int = 4) -> list[Chunk]:
        """Keyword search alone, for the retrieval comparison."""
        return [self.index.chunks[position] for position in self.lexical_ranking(query, top_k)]

    async def search_dense_only(self, query: str, *, top_k: int = 4) -> list[Chunk]:
        """Vector search alone, for the retrieval comparison."""
        vector = await self._embed_query(query)
        return [self.index.chunks[position] for position in self.dense_ranking(vector, top_k)]
