"""The Ollama implementation of :class:`~shurta.llm.base.ModelClient`.

Ollama runs on the same machine and serves both the embedding model and the
instruction model. Nothing here reaches outside that machine.

This client never sends ``think``. That parameter does not disable a model's
reasoning; it disables Ollama's separation of it, after which the reasoning
arrives in ``response`` and cannot be told apart from the answer.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from shurta.llm.base import (
    GenerationChunk,
    GenerationOptions,
    GenerationStats,
    ModelError,
    ModelStatus,
)

#: Embedding batch size, to keep one request's memory use modest.
EMBED_BATCH_SIZE = 16


class OllamaClient:
    """Talks to an Ollama instance over its HTTP interface."""

    def __init__(
        self,
        base_url: str,
        *,
        generation_model: str,
        embedding_model: str,
        embedding_timeout: float = 120.0,
        generation_timeout: float = 300.0,
        keep_alive: str = "30m",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.keep_alive = keep_alive
        self._embedding_timeout = embedding_timeout
        self._generation_timeout = generation_timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=generation_timeout,
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    # --- embeddings ----------------------------------------------------------

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches, preserving the input order."""
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            payload = await self._post_json(
                "/api/embed",
                {
                    "model": self.embedding_model,
                    "input": batch,
                    # Held resident, or every question pays to load it again.
                    "keep_alive": self.keep_alive,
                },
                timeout=self._embedding_timeout,
            )
            embeddings = payload.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise ModelError(
                    f"{self.embedding_model} returned {type(embeddings).__name__} for a batch of "
                    f"{len(batch)} texts rather than one vector each"
                )
            vectors.extend(embeddings)
        return vectors

    # --- generation ----------------------------------------------------------

    def _generation_body(
        self, prompt: str, model: str | None, settings: GenerationOptions
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.generation_model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": settings.keep_alive,
            "options": {
                "num_ctx": settings.num_ctx,
                "num_predict": settings.max_tokens,
                "temperature": settings.temperature,
            },
        }
        if settings.num_thread is not None:
            body["options"]["num_thread"] = settings.num_thread
        if settings.stop:
            body["options"]["stop"] = list(settings.stop)
        return body

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        """Stream the answer, discarding any reasoning the model emits."""
        body = self._generation_body(prompt, model, options or GenerationOptions())

        try:
            async with self._client.stream(
                "POST", "/api/generate", json=body, timeout=self._generation_timeout
            ) as response:
                if response.status_code != httpx.codes.OK:
                    detail = (await response.aread()).decode("utf-8", "replace")[:300]
                    raise ModelError(f"Ollama returned HTTP {response.status_code}: {detail}")
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    event = _event(line)
                    if event is None:
                        continue
                    piece = event.get("response")
                    text = piece if isinstance(piece, str) else ""
                    stats = generation_stats(event)
                    if text or stats is not None:
                        yield GenerationChunk(text=text, stats=stats)
        except httpx.HTTPError as error:
            raise ModelError(f"Ollama at {self.base_url} could not be reached: {error}") from error

    # --- health --------------------------------------------------------------

    async def status(self) -> ModelStatus:
        """Whether Ollama answers, and which models it holds."""
        try:
            payload = await self._post_json("/api/tags", None, timeout=10.0, method="GET")
        except ModelError as error:
            return ModelStatus(reachable=False, detail=str(error))

        models = payload.get("models")
        names: list[str] = []
        if isinstance(models, list):
            for entry in models:
                if isinstance(entry, dict) and isinstance(entry.get("model"), str):
                    names.append(entry["model"])
        return ModelStatus(reachable=True, available_models=tuple(names))

    async def warm(self, model: str | None = None, keep_alive: str | None = None) -> None:
        """Load the models now, so the first question does not pay for them.

        A zero-length generation loads the instruction model without producing
        tokens; a one-word embedding loads the embedding model, which retrieval
        needs before the generation model is reached at all.
        """
        residency = keep_alive or self.keep_alive
        await self._post_json(
            "/api/generate",
            {
                "model": model or self.generation_model,
                "prompt": "",
                "stream": False,
                "keep_alive": residency,
            },
            timeout=self._generation_timeout,
        )
        await self.embed(["warm"])

    # --- plumbing ------------------------------------------------------------

    async def _post_json(
        self,
        path: str,
        body: dict[str, Any] | None,
        *,
        timeout: float,
        method: str = "POST",
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, json=body, timeout=timeout)
        except httpx.HTTPError as error:
            raise ModelError(f"Ollama at {self.base_url} could not be reached: {error}") from error

        if response.status_code != httpx.codes.OK:
            raise ModelError(f"Ollama returned HTTP {response.status_code} for {path}")

        try:
            payload = response.json()
        except ValueError as error:
            raise ModelError(f"Ollama returned malformed JSON for {path}: {error}") from error

        if not isinstance(payload, dict):
            raise ModelError(f"Ollama returned {type(payload).__name__} for {path}, not an object")
        return payload


def _event(line: str) -> dict[str, Any] | None:
    """One streamed line as an object, or ``None`` if it cannot be read.

    A malformed line is skipped: losing one fragment beats failing an answer
    that is nearly complete.
    """
    try:
        event = json.loads(line)
    except ValueError:
        return None
    if not isinstance(event, dict):
        return None
    if error := event.get("error"):
        raise ModelError(str(error))
    return event


def answer_text(line: str) -> str:
    """The answer text carried by one streamed line, excluding any reasoning."""
    event = _event(line)
    if event is None:
        return ""
    piece = event.get("response")
    return piece if isinstance(piece, str) else ""


def _milliseconds(event: dict[str, Any], key: str) -> float:
    """One nanosecond duration from a streamed event, in milliseconds."""
    value = event.get(key)
    return value / 1_000_000 if isinstance(value, int | float) else 0.0


def _count(event: dict[str, Any], key: str) -> int:
    value = event.get(key)
    return int(value) if isinstance(value, int | float) else 0


def generation_stats(event: dict[str, Any]) -> GenerationStats | None:
    """The statistics on Ollama's final streamed event, if it carries any.

    Prompt processing dominates time to first token on processor-only
    inference, so these counts are what the latency work is measured against.
    """
    if not event.get("done"):
        return None
    if "prompt_eval_count" not in event and "eval_count" not in event:
        return None
    return GenerationStats(
        prompt_tokens=_count(event, "prompt_eval_count"),
        prompt_ms=_milliseconds(event, "prompt_eval_duration"),
        output_tokens=_count(event, "eval_count"),
        output_ms=_milliseconds(event, "eval_duration"),
        load_ms=_milliseconds(event, "load_duration"),
    )


def model_matches(available: tuple[str, ...], wanted: str) -> bool:
    """Whether a wanted model is present, allowing for the implied ``:latest``.

    ``bge-m3`` and ``bge-m3:latest`` are the same model, so a health check must
    not report one missing over the tag alone.
    """
    candidates = {wanted, f"{wanted}:latest", wanted.removesuffix(":latest")}
    return any(name in candidates for name in available)
