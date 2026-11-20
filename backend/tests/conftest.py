"""Shared fixtures. Everything here is invented or synthetic, so the suite
needs no network access and no collected content."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from shurta.config import Settings
from shurta.llm.base import GenerationChunk, GenerationStats, ModelStatus
from shurta.models import Corpus

FIXTURE_CORPUS = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "services.json"


@pytest.fixture
def corpus() -> Corpus:
    """The invented fixture catalogue."""
    return Corpus.model_validate_json(FIXTURE_CORPUS.read_text(encoding="utf-8"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a scratch directory, so no test touches real data."""
    return Settings(
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "data" / "index",
        warm_model_on_startup=False,
        _env_file=None,  # type: ignore[call-arg]
    )


def service_record(
    *,
    name: str,
    slug: str,
    document_id: str = "doc-1",
    components: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A service record in the shape the source publishes, with invented content."""
    return {
        "id": 1,
        "documentId": document_id,
        "name": name,
        "url": slug,
        "locale": "en",
        "dPServiceOptions": components or [],
    }


def component(kind: str, **fields: Any) -> dict[str, Any]:
    """One content component of the given kind."""
    return {"__component": f"dp-service-options.{kind}", "id": 1, **fields}


class FakeModelClient:
    """A stand-in for Ollama that never leaves the process.

    Satisfies :class:`~shurta.llm.base.ModelClient`. The answer it streams is
    fixed per instance and split into pieces, so streaming and the first-token
    timing are exercised rather than bypassed.
    """

    def __init__(
        self,
        answer: str = "The fee is 75 dirhams [1].",
        *,
        dimensions: int = 8,
        available: tuple[str, ...] = ("qwen3:4b", "bge-m3:latest"),
        fail_after: int | None = None,
        reachable: bool = True,
        stats: GenerationStats | None = None,
    ) -> None:
        self.answer = answer
        self.dimensions = dimensions
        self.available = available
        self.fail_after = fail_after
        self.reachable = reachable
        self.stats = stats
        self.prompts: list[str] = []
        self.embedded: list[str] = []
        self.options: list[object] = []
        self.warm_calls = 0
        self.closed = False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [self.vector(text) for text in texts]

    def vector(self, text: str) -> list[float]:
        """A crude but deterministic embedding: character counts per bucket."""
        values = [0.0] * self.dimensions
        for position, character in enumerate(text):
            values[position % self.dimensions] += ord(character) % 7
        total = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / total for value in values]

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        options: object | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        from shurta.llm.base import ModelError

        self.prompts.append(prompt)
        self.options.append(options)
        for position, piece in enumerate(self.answer.split(" ")):
            if self.fail_after is not None and position >= self.fail_after:
                raise ModelError("the model stopped part way")
            yield GenerationChunk(text=piece if position == 0 else f" {piece}")
        yield GenerationChunk(stats=self.stats or GenerationStats())

    async def status(self) -> ModelStatus:
        if not self.reachable:
            return ModelStatus(reachable=False, detail="connection refused")
        return ModelStatus(reachable=True, available_models=self.available)

    async def warm(self, model: str | None = None, keep_alive: str | None = None) -> None:
        self.warm_calls += 1

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_client() -> FakeModelClient:
    return FakeModelClient()
