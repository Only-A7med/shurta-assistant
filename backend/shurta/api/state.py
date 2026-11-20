"""What the API holds open for the life of the process.

The index and the model client are both expensive to set up and cheap to keep,
so they are created once at start-up rather than per request. The index in
particular is a NumPy array that would otherwise be read from disk on every
question.

Start-up deliberately does not fail when the index or Ollama is missing. A
reader who has cloned the repository and not yet run ``ingest`` and ``index``
should get an API that starts and a health endpoint that says exactly what is
missing — not a stack trace at boot with no explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shurta.answering import AnswerPipeline
from shurta.config import Settings
from shurta.corpus import CorpusMissing, load_configured_corpus
from shurta.llm.base import GenerationOptions, ModelClient
from shurta.llm.ollama import OllamaClient
from shurta.models import Corpus
from shurta.retrieval.hybrid import HybridRetriever
from shurta.retrieval.store import SearchIndex


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
    warmed: bool = False
    warm_error: str = ""
    _pipeline: AnswerPipeline | None = field(default=None, init=False, repr=False)

    @classmethod
    def create(cls, settings: Settings, client: ModelClient | None = None) -> ApplicationState:
        """Build the state, optionally against a supplied model client.

        The client is injectable because the provider is an interface: it lets
        the tests run the whole API against a stand-in, and it is the seam a
        second provider would arrive through.
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
                options=GenerationOptions(
                    num_ctx=self.settings.num_ctx,
                    max_tokens=self.settings.max_answer_tokens,
                    keep_alive=self.settings.keep_alive,
                ),
            )
        return self._pipeline

    async def warm(self) -> None:
        """Load the generation model so the first question does not wait for it."""
        if not self.settings.warm_model_on_startup:
            return
        try:
            await self.client.warm()
            self.warmed = True
            self.warm_error = ""
        # A missing model or a stopped Ollama is reported by the health endpoint;
        # neither is a reason to refuse to start.
        except Exception as error:
            self.warmed = False
            self.warm_error = str(error)

    async def close(self) -> None:
        await self.client.close()
