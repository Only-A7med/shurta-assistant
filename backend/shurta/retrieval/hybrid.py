"""The retriever: keyword search and vector search, fused.

Each method fails in a way the other covers.

BM25 matches the words that were actually typed. It is exact about a fee figure,
a service name or an identifier, and it is the only one of the two that reliably
finds a rare term. But it cannot match a question to a passage that says the
same thing in different words, and it cannot cross languages at all: the Arabic
``الرسوم`` shares no token with the English "fees".

The embedding model crosses both gaps, because ``bge-m3`` places text with the
same meaning near each other regardless of the language it is written in. What
it is weaker at is precision on a specific string — it will happily rank a
passage about a different certificate close to one about the right one.

So both are run and their rankings fused. Measured on the evaluation set, vector
search alone has the better overall recall@1 — 0.91 against the fused 0.89, two
questions in a hundred — but it is the fused ranking that answers every
code-switched question correctly, where vector search alone misses one in five.
Two questions of overall accuracy is a fair price for not failing a whole class
of question, which is why the fused ranking is the one the assistant uses.

The retriever also takes the
language it should prefer: both languages are searched, since a question in one
may be answered by a passage in the other, but when two passages say the same
thing the one in the reader's language is the better citation.
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

#: How many candidates each method contributes before fusion. Wider than the
#: final number of chunks, so fusion has something to disagree about.
CANDIDATES = 20

#: Weight on each ranking during fusion, equal after a sweep showed that
#: favouring vector search gained nothing overall and lost the code-switched
#: questions. See :mod:`shurta.retrieval.fusion` and ``eval/retrieval-results.md``.
LEXICAL_WEIGHT = 1.0
DENSE_WEIGHT = 1.0

#: How much a passage in the reader's own language is favoured, as a multiplier
#: on its fused score. Small on purpose: it should settle a near-tie, not
#: override a clearly better passage in the other language.
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
        # argpartition finds the top rows without sorting the whole array, which
        # matters not at all at this corpus size but costs nothing to get right.
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
