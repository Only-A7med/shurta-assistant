"""What the API holds open for the life of the process.

The index and the model client are expensive to set up and cheap to keep, so
they are created once at start-up rather than per request.

Start-up does not fail when the index or Ollama is missing: a fresh clone should
get an API that starts and a health endpoint that names what is missing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum

from shurta.answering import AnswerPipeline
from shurta.config import Settings
from shurta.corpus import CorpusMissing, load_configured_corpus
from shurta.llm.base import ModelClient
from shurta.llm.ollama import OllamaClient
from shurta.models import Corpus
from shurta.retrieval.hybrid import HybridRetriever
from shurta.retrieval.store import SearchIndex


class WarmState(StrEnum):
    """Where the generation model is in loading."""

    OFF = "off"
    WARMING = "warming"
    READY = "ready"
    FAILED = "failed"


def _default_client(settings: Settings) -> ModelClient:
    return OllamaClient(
        settings.ollama_base_url,
        generation_model=settings.generation_model,
        embedding_model=settings.embedding_model,
        embedding_timeout=settings.embedding_timeout,
        generation_timeout=settings.generation_timeout,
    )


@dataclass
class ApplicationState:
    """The long-lived objects behind the endpoints."""

    settings: Settings
    client: ModelClient
    index: SearchIndex | None = None
    corpus: Corpus | None = None
    index_error: str = ""
    corpus_error: str = ""
    warm_state: WarmState = WarmState.OFF
    warm_error: str = ""
    _pipeline: AnswerPipeline | None = field(default=None, init=False, repr=False)
    _warming: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    @classmethod
    def create(cls, settings: Settings, client: ModelClient | None = None) -> ApplicationState:
        """Build the state, optionally against a supplied model client.

        The client is injectable because the provider is an interface: it is how
        the tests run the API against a stand-in, and how a second provider
        would arrive.
        """
        state = cls(settings=settings, client=client or _default_client(settings))
        state.load()
        return state

    def load(self) -> None:
        """Read the index and the catalogue, recording why if either is absent."""
        try:
            self.index = SearchIndex.load(self.settings.index_path)
            self.index_error = ""
        except RuntimeError as error:
            self.index = None
            self.index_error = str(error)

        try:
            self.corpus = load_configured_corpus(self.settings)
            self.corpus_error = ""
        except (CorpusMissing, ValueError) as error:
            self.corpus = None
            self.corpus_error = str(error)

    @property
    def ready(self) -> bool:
        return self.index is not None

    @property
    def warmed(self) -> bool:
        return self.warm_state is WarmState.READY

    def pipeline(self) -> AnswerPipeline:
        """The answer pipeline, built once."""
        if self.index is None:
            raise RuntimeError(self.index_error or "the search index is not loaded")

        if self._pipeline is None:

            async def embed(query: str) -> list[float]:
                return (await self.client.embed([query]))[0]

            self._pipeline = AnswerPipeline(
                HybridRetriever(self.index, embed),
                self.client,
                top_k=self.settings.top_k,
                options=self.settings.generation_options(),
            )
        return self._pipeline

    def start_warming(self) -> None:
        """Begin loading the generation model without blocking start-up.

        Loading takes seconds, and awaiting it would stop uvicorn accepting
        connections for that whole time. The health endpoint reports ``warming``
        meanwhile, which is the state the interface shows.
        """
        if not self.settings.warm_model_on_startup:
            return
        self.warm_state = WarmState.WARMING
        self._warming = asyncio.create_task(self.warm())

    async def warm(self) -> None:
        """Load the generation model, recording whether it worked."""
        try:
            await self.client.warm(keep_alive=self.settings.keep_alive)
            self.warm_state = WarmState.READY
            self.warm_error = ""
        # A missing model or a stopped Ollama is reported by the health
        # endpoint; neither is a reason to refuse to start.
        except Exception as error:
            self.warm_state = WarmState.FAILED
            self.warm_error = str(error)

    async def close(self) -> None:
        if self._warming is not None and not self._warming.done():
            self._warming.cancel()
        await self.client.close()
