"""The Ollama implementation of :class:`~shurta.llm.base.ModelClient`.

Ollama runs on the host machine and serves both halves of the pipeline: the
embedding model for retrieval and the instruction model for the answer. Nothing
here reaches outside the machine it runs on.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from shurta.llm.base import GenerationOptions, ModelError, ModelStatus

#: Embedding batch size. The corpus is a few hundred passages, so this exists to
#: keep a single request's memory use modest rather than to go faster.
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
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.generation_model = generation_model
        self.embedding_model = embedding_model
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
                {"model": self.embedding_model, "input": batch},
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

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[str]:
        """Stream an answer, yielding each piece of text as it arrives."""
        settings = options or GenerationOptions()
        body: dict[str, Any] = {
            "model": model or self.generation_model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": settings.keep_alive,
            "think": settings.think,
            "options": {
                "num_ctx": settings.num_ctx,
                "num_predict": settings.max_tokens,
                "temperature": settings.temperature,
            },
        }
        if settings.stop:
            body["options"]["stop"] = list(settings.stop)

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
                    piece = _token(line)
                    if piece:
                        yield piece
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

    async def warm(self, model: str | None = None) -> None:
        """Load a model now, so the first question does not pay for it.

        A zero-length generation is enough: Ollama loads the model and keeps it
        resident for the configured period without producing any tokens.
        """
        await self._post_json(
            "/api/generate",
            {
                "model": model or self.generation_model,
                "prompt": "",
                "stream": False,
                "keep_alive": GenerationOptions().keep_alive,
            },
            timeout=self._generation_timeout,
        )

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


def _token(line: str) -> str:
    """The text carried by one streamed line, if any.

    Ollama streams newline-delimited JSON objects. A malformed line is skipped
    rather than ending the answer: losing one fragment is better than failing a
    response that is most of the way done.
    """
    try:
        event = json.loads(line)
    except ValueError:
        return ""
    if not isinstance(event, dict):
        return ""
    if error := event.get("error"):
        raise ModelError(str(error))
    piece = event.get("response")
    return piece if isinstance(piece, str) else ""


def model_matches(available: tuple[str, ...], wanted: str) -> bool:
    """Whether a wanted model is present, allowing for the implied ``:latest``.

    ``bge-m3`` and ``bge-m3:latest`` are the same model, and the two spellings
    appear in different places, so a health check must not report a model
    missing merely because of the tag.
    """
    candidates = {wanted, f"{wanted}:latest", wanted.removesuffix(":latest")}
    return any(name in candidates for name in available)
