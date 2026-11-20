"""The HTTP interface: a streaming chat endpoint, a health check, a service list.

Nothing a reader types is stored. There is no session, no identifier and no
history: a question arrives, an answer streams back, and nothing remains.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from shurta.answering import ResponseKind
from shurta.api.sse import event
from shurta.api.state import ApplicationState, WarmState
from shurta.config import Settings, get_settings
from shurta.llm.base import ModelClient, ModelError
from shurta.llm.ollama import model_matches
from shurta.models import Language

STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-store",
    "Connection": "keep-alive",
    # Tells nginx and similar not to buffer the stream, which would otherwise
    # hold every token back until the answer was complete.
    "X-Accel-Buffering": "no",
}


class ChatRequest(BaseModel):
    """A question. Deliberately the only thing the endpoint accepts."""

    question: str = Field(min_length=1, max_length=2000)
    language: Language | None = None
    """Override the detected language, which the interface's toggle uses."""


class ServiceSummary(BaseModel):
    """One service as the interface lists it."""

    id: str
    name_en: str
    name_ar: str
    category_en: str
    category_ar: str
    url_en: str
    url_ar: str
    collected_at: str
    has_fees: bool
    has_required_documents: bool


def create_app(settings: Settings | None = None, client: ModelClient | None = None) -> FastAPI:
    """Build the application.

    Both arguments are seams: the settings let a test or a container point the
    application at another directory without touching the process environment,
    and the client is how a provider other than Ollama would be supplied.
    """
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = ApplicationState.create(resolved, client)
        app.state.shurta = state
        # Warming runs in the background so the port is accepting connections
        # at once; the health endpoint reports "warming" until it finishes.
        state.start_warming()
        try:
            yield
        finally:
            await state.close()

    app = FastAPI(
        title="Shurta Assistant",
        version="0.1.0",
        summary=(
            "Bilingual question answering over public Dubai Police service information, "
            "running entirely on the machine it is installed on."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    _register(app)
    return app


def _state(request: Request) -> ApplicationState:
    state: ApplicationState = request.app.state.shurta
    return state


def _register(app: FastAPI) -> None:
    @app.post("/api/chat")
    async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
        """Answer a question, streaming the text and then the citations.

        The event sequence is fixed: ``start`` with the classification and
        language, then ``token`` repeatedly, then ``done`` with the citations and
        timings — or ``error`` instead of ``done`` if generation failed part way.
        """
        state = _state(request)
        if not state.ready:
            return StreamingResponse(
                _single_error(state.index_error),
                media_type="text/event-stream",
                headers=STREAM_HEADERS,
            )

        return StreamingResponse(
            _answer_stream(state, body),
            media_type="text/event-stream",
            headers=STREAM_HEADERS,
        )

    @app.get("/api/health")
    async def health(request: Request) -> JSONResponse:
        """Whether every part the assistant needs is actually in place."""
        state = _state(request)
        settings = state.settings
        status = await state.client.status()

        models = {
            "generation": {
                "name": settings.generation_model,
                "present": model_matches(status.available_models, settings.generation_model),
            },
            "embedding": {
                "name": settings.embedding_model,
                "present": model_matches(status.available_models, settings.embedding_model),
            },
        }

        index: dict[str, Any] = {"loaded": state.index is not None}
        if state.index is not None:
            index |= {
                "chunks": state.index.metadata.chunk_count,
                "embedding_model": state.index.metadata.embedding_model,
                "built_at": state.index.metadata.built_at.isoformat(),
            }
        else:
            index["detail"] = state.index_error

        corpus: dict[str, Any] = {"loaded": state.corpus is not None}
        if state.corpus is not None:
            corpus |= {
                "services": len(state.corpus.services),
                "collected_at": state.corpus.collected_at.isoformat(),
            }
        else:
            corpus["detail"] = state.corpus_error

        healthy = (
            status.reachable
            and all(model["present"] for model in models.values())
            and state.index is not None
        )
        warming = healthy and state.warm_state is WarmState.WARMING

        payload = {
            "status": "warming" if warming else "ok" if healthy else "degraded",
            "ollama": {
                "base_url": settings.ollama_base_url,
                "reachable": status.reachable,
                "detail": status.detail,
            },
            "models": models,
            "index": index,
            "corpus": corpus,
            "model_warmed": state.warmed,
            "warm_state": state.warm_state.value,
            "warm_detail": state.warm_error,
        }
        # 503 lets an orchestrator and a browser agree on whether this is usable
        # without reading the body. Warming is usable, only slower, so it is 200.
        return JSONResponse(payload, status_code=200 if healthy else 503)

    @app.get("/api/services")
    async def services(request: Request) -> JSONResponse:
        """The catalogue the answers are drawn from."""
        state = _state(request)
        if state.corpus is None:
            return JSONResponse({"detail": state.corpus_error, "services": []}, status_code=503)

        listed = [
            ServiceSummary(
                id=service.id,
                name_en=service.name_en,
                name_ar=service.name_ar,
                category_en=service.category.en,
                category_ar=service.category.ar,
                url_en=service.source_url(Language.EN),
                url_ar=service.source_url(Language.AR),
                collected_at=service.collected_at.date().isoformat(),
                has_fees=bool(service.fees),
                has_required_documents=bool(service.required_documents),
            ).model_dump()
            for service in state.corpus.services
        ]
        return JSONResponse(
            {
                "collected_at": state.corpus.collected_at.isoformat(),
                "source": state.corpus.source,
                "count": len(listed),
                "services": listed,
            }
        )


async def _single_error(detail: str) -> AsyncIterator[str]:
    yield event("error", {"detail": detail})


async def _answer_stream(state: ApplicationState, body: ChatRequest) -> AsyncIterator[str]:
    """Run the pipeline and frame it as server-sent events."""
    pipeline = state.pipeline()
    context = await pipeline.prepare(body.question)
    language = body.language or context.language

    yield event(
        "start",
        {
            "kind": context.kind.value,
            "language": language.value,
            "detected_language": context.language.value,
            "citations": [asdict(citation) for citation in context.citations],
            "retrieval": context.retrieval.timings,
        },
    )

    pieces: list[str] = []
    try:
        async for piece in pipeline.stream(context):
            pieces.append(piece)
            yield event("token", {"text": piece})
    except ModelError as error:
        yield event(
            "error",
            {
                "detail": str(error),
                "partial": "".join(pieces),
            },
        )
        return

    yield event(
        "done",
        {
            "kind": context.kind.value,
            "language": language.value,
            "text": "".join(pieces),
            "citations": [asdict(citation) for citation in context.citations],
            "emergency_numbers": context.emergency_numbers,
            "retrieval": context.retrieval.timings,
            "best_similarity": round(context.retrieval.best_similarity, 4),
            "prompt_tokens": context.stats.prompt_tokens if context.stats else 0,
            "prompt_ms": round(context.stats.prompt_ms, 1) if context.stats else 0.0,
            "model": state.settings.generation_model
            if context.kind is ResponseKind.ANSWER
            else None,
        },
    )
