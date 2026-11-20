"""What the rest of the project needs from a model provider.

Only this interface is used outside :mod:`shurta.llm`, so a provider can be
swapped without touching retrieval or the API, and the tests can run against a
stand-in.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class GenerationOptions:
    """The generation budget: a short context and a capped answer.

    There is no option to switch a model's reasoning off. Whether reasoning can
    be disabled is a property of the model, not of the request; on this provider
    asking for it off only stops the reasoning being separated from the answer.
    """

    num_ctx: int = 4096
    max_tokens: int = 512
    temperature: float = 0.1
    keep_alive: str = "30m"

    num_thread: int | None = None
    """Generation threads, or ``None`` to leave the provider's own choice."""

    stop: tuple[str, ...] = field(default=())
    """Sequences that end generation early, if the provider supports them."""


@dataclass(frozen=True)
class GenerationStats:
    """What the provider reports about one completed generation."""

    prompt_tokens: int = 0
    prompt_ms: float = 0.0
    output_tokens: int = 0
    output_ms: float = 0.0
    load_ms: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.output_ms <= 0 or self.output_tokens <= 0:
            return 0.0
        return self.output_tokens / (self.output_ms / 1000)


@dataclass(frozen=True)
class GenerationChunk:
    """One streamed piece of an answer; the last one carries the statistics."""

    text: str = ""
    stats: GenerationStats | None = None


@dataclass(frozen=True)
class ModelStatus:
    """What a provider can say about itself, for the health check and doctor."""

    reachable: bool
    available_models: tuple[str, ...] = ()
    detail: str = ""


class ModelError(RuntimeError):
    """Raised when the provider cannot answer."""


@runtime_checkable
class ModelClient(Protocol):
    """A provider of embeddings and streamed generation."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per text, in order."""
        ...

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        """Stream an answer piece by piece."""
        ...

    async def status(self) -> ModelStatus:
        """Whether the provider is reachable, and which models it holds."""
        ...

    async def warm(self, model: str | None = None, keep_alive: str | None = None) -> None:
        """Load a model into memory so the first real question does not wait."""
        ...

    async def close(self) -> None:
        """Release any held connections."""
        ...
