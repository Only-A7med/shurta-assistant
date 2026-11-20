"""The on-disk search index: a vector array, a lexical index, and metadata.

The corpus is a few hundred passages, so the dense side is a single NumPy array
of about a megabyte. A vector database server would add a process to run, a
protocol to speak and a failure mode to handle, in exchange for nothing a
corpus this size needs: an exhaustive cosine similarity over 500 vectors is a
matrix multiply that finishes in well under a millisecond.

Three files make up an index:

* ``vectors.npy`` — one row per chunk, L2-normalised so that cosine similarity
  is a dot product.
* ``chunks.json`` — the chunks themselves, in the same order as the rows.
* ``index.json`` — the lexical index and the metadata describing the build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from shurta.chunking import Chunk
from shurta.retrieval.bm25 import Bm25Index

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"
METADATA_FILE = "index.json"

#: Bumped when the stored layout changes, so a stale index is reported rather
#: than loaded and silently misread.
FORMAT_VERSION = 1


class IndexError_(RuntimeError):
    """Raised when an index cannot be loaded."""


@dataclass(frozen=True)
class IndexMetadata:
    """What a stored index records about how it was built."""

    format_version: int
    embedding_model: str
    built_at: datetime
    chunk_count: int
    dimensions: int
    corpus_collected_at: datetime | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "embedding_model": self.embedding_model,
            "built_at": self.built_at.isoformat(),
            "chunk_count": self.chunk_count,
            "dimensions": self.dimensions,
            "corpus_collected_at": (
                self.corpus_collected_at.isoformat() if self.corpus_collected_at else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> IndexMetadata:
        collected = payload.get("corpus_collected_at")
        return cls(
            format_version=_as_int(payload, "format_version"),
            embedding_model=str(payload["embedding_model"]),
            built_at=datetime.fromisoformat(str(payload["built_at"])),
            chunk_count=_as_int(payload, "chunk_count"),
            dimensions=_as_int(payload, "dimensions"),
            corpus_collected_at=(
                datetime.fromisoformat(str(collected)) if isinstance(collected, str) else None
            ),
        )


def _as_int(payload: dict[str, object], key: str) -> int:
    """Read a whole number from stored metadata, or say which field is wrong."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise IndexError_(f"the index metadata field {key!r} is not a number: {value!r}")
    try:
        return int(value)
    except ValueError as error:
        raise IndexError_(f"the index metadata field {key!r} is not a number: {value!r}") from error


def normalise_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, so cosine similarity is a dot product.

    A zero row would divide by zero; it is left as zeros instead, which scores
    zero against every query rather than producing a NaN that poisons the
    ranking.
    """
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise IndexError_(f"expected a two-dimensional array of vectors, got {array.ndim}")
    magnitudes = np.linalg.norm(array, axis=1, keepdims=True)
    magnitudes[magnitudes == 0] = 1.0
    unit: np.ndarray = array / magnitudes
    return unit


@dataclass
class SearchIndex:
    """A loaded index, ready to search."""

    chunks: list[Chunk]
    vectors: np.ndarray
    lexical: Bm25Index
    metadata: IndexMetadata

    def __post_init__(self) -> None:
        if len(self.chunks) != self.vectors.shape[0]:
            raise IndexError_(
                f"the index holds {len(self.chunks)} chunks but "
                f"{self.vectors.shape[0]} vectors, so they cannot be matched up"
            )

    @classmethod
    def create(
        cls,
        chunks: list[Chunk],
        vectors: np.ndarray,
        *,
        embedding_model: str,
        corpus_collected_at: datetime | None = None,
    ) -> SearchIndex:
        """Build an index from chunks and their embeddings."""
        unit = normalise_rows(vectors)
        return cls(
            chunks=chunks,
            vectors=unit,
            lexical=Bm25Index.build([chunk.text for chunk in chunks]),
            metadata=IndexMetadata(
                format_version=FORMAT_VERSION,
                embedding_model=embedding_model,
                built_at=datetime.now(UTC),
                chunk_count=len(chunks),
                dimensions=int(unit.shape[1]) if unit.size else 0,
                corpus_collected_at=corpus_collected_at,
            ),
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / VECTORS_FILE, self.vectors)
        (directory / CHUNKS_FILE).write_text(
            json.dumps(
                [chunk.model_dump(mode="json") for chunk in self.chunks],
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        (directory / METADATA_FILE).write_text(
            json.dumps(
                {"metadata": self.metadata.to_dict(), "lexical": self.lexical.to_dict()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> SearchIndex:
        """Load an index, or explain precisely what is missing."""
        for name in (VECTORS_FILE, CHUNKS_FILE, METADATA_FILE):
            if not (directory / name).exists():
                raise IndexError_(
                    f"no search index at {directory}: {name} is missing. "
                    "Run `python -m shurta index` to build it."
                )

        payload = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
        metadata = IndexMetadata.from_dict(payload["metadata"])
        if metadata.format_version != FORMAT_VERSION:
            raise IndexError_(
                f"the index at {directory} was written in format {metadata.format_version}, "
                f"but this version reads format {FORMAT_VERSION}. Rebuild it with "
                "`python -m shurta index`."
            )

        chunks = [
            Chunk.model_validate(record)
            for record in json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))
        ]
        return cls(
            chunks=chunks,
            vectors=np.load(directory / VECTORS_FILE),
            lexical=Bm25Index.from_dict(payload["lexical"]),
            metadata=metadata,
        )

    def dense_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query against every chunk."""
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        magnitude = float(np.linalg.norm(query))
        if magnitude == 0:
            return np.zeros(len(self.chunks), dtype=np.float32)
        similarities: np.ndarray = self.vectors @ (query / magnitude)
        return similarities
