"""Building the search index: chunk the corpus, embed it, write it out."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from shurta.chunking import chunk_corpus
from shurta.config import Settings
from shurta.corpus import load_corpus
from shurta.llm.ollama import OllamaClient
from shurta.models import Corpus
from shurta.retrieval.store import SearchIndex


async def build_index_async(
    corpus: Corpus,
    *,
    directory: Path,
    client: OllamaClient,
) -> SearchIndex:
    """Chunk, embed and persist a corpus."""
    chunks = chunk_corpus(corpus)
    if not chunks:
        raise ValueError("the corpus produced no chunks, so there is nothing to index")

    vectors = await client.embed([chunk.text for chunk in chunks])
    index = SearchIndex.create(
        chunks,
        np.asarray(vectors, dtype=np.float32),
        embedding_model=client.embedding_model,
        corpus_collected_at=corpus.collected_at,
    )
    index.save(directory)
    return index


def build_index(settings: Settings) -> str:
    """Build the index the settings describe, returning a one-line summary."""
    corpus = load_corpus(settings.services_file)

    async def run() -> SearchIndex:
        client = OllamaClient(
            settings.ollama_base_url,
            generation_model=settings.generation_model,
            embedding_model=settings.embedding_model,
            embedding_timeout=settings.embedding_timeout,
            generation_timeout=settings.generation_timeout,
        )
        try:
            return await build_index_async(corpus, directory=settings.index_path, client=client)
        finally:
            await client.close()

    index = asyncio.run(run())
    languages = {chunk.language.value for chunk in index.chunks}
    return (
        f"Indexed {len(index.chunks)} chunks from {len(corpus.services)} services "
        f"({', '.join(sorted(languages))}) with {index.metadata.embedding_model}, "
        f"{index.metadata.dimensions} dimensions -> {settings.index_path}"
    )
