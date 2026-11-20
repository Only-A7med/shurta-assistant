"""What the rest of the project needs from a model provider.

Only this interface is used anywhere outside :mod:`shurta.llm`, which keeps two
things possible: swapping Ollama for another provider without touching the
retrieval or the API, and running the whole test suite against a stand-in
instead of a real model.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class GenerationOptions:
    """The generation budget, which on processor-only inference is the design.

    Every field here exists to keep a small machine responsive: a short context
    and a capped answer.

    There is deliberately no option to switch a model's reasoning off. Whether
    reasoning can be disabled is a property of the model rather than of the
    request — on the provider used here, asking for it to be off merely stops
    the reasoning being separated from the answer. Choosing a model whose
    reasoning does not have to be paid for is the real lever, and that is what
    the answer evaluation measures.
    """

    num_ctx: int = 4096
    max_tokens: int = 512
    temperature: float = 0.1
    keep_alive: str = "30m"

    stop: tuple[str, ...] = field(default=())
    """Sequences that end generation early, if the provider supports them."""


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
    ) -> AsyncIterator[str]:
        """Stream an answer token by token."""
        ...

    async def status(self) -> ModelStatus:
        """Whether the provider is reachable, and which models it holds."""
        ...

    async def warm(self, model: str | None = None) -> None:
        """Load a model into memory so the first real question does not wait."""
        ...

    async def close(self) -> None:
        """Release any held connections."""
        ...
