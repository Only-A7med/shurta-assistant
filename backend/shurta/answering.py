"""The answer pipeline: guard, retrieve, generate, cite.

The whole flow in one place, independent of how it is delivered, so the API and
the evaluation exercise the same behaviour.

The order matters. The guard runs before retrieval and before any model call, so
an emergency costs nothing. When nothing retrieved is relevant enough the
pipeline answers from a fixed response rather than letting a small model work
from an empty context.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum

from shurta.arabic import detect_language
from shurta.guardrails import Intent, classify, emergency_response
from shurta.llm.base import GenerationOptions, GenerationStats, ModelClient
from shurta.models import Language
from shurta.prompting import (
    Citation,
    build_prompt,
    citations_for,
    no_context_response,
    personal_case_response,
)
from shurta.retrieval.hybrid import HybridRetriever, RetrievalResult

#: Cosine similarity below which nothing retrieved is worth a model call, and
#: the fixed "not in my information" response is used instead.
#:
#: Measured, not chosen: answerable questions' closest passage scored 0.526 to
#: 0.804, out-of-scope ones 0.355 to 0.566. The floor sits below every
#: answerable question, so it rejects nothing the corpus can answer and still
#: catches four of six out-of-scope questions before any model runs. The fused
#: score cannot do this job — it encodes rank alone. See docs/architecture.md.
SIMILARITY_FLOOR = 0.50


class ResponseKind(StrEnum):
    """How a question was handled, which the interface shows and the evaluation scores."""

    ANSWER = "answer"
    EMERGENCY = "emergency"
    PERSONAL_CASE = "personal_case"
    NO_CONTEXT = "no_context"


@dataclass
class AnswerContext:
    """Everything decided about a question before the answer is produced."""

    question: str
    language: Language
    kind: ResponseKind
    citations: list[Citation] = field(default_factory=list)
    retrieval: RetrievalResult = field(default_factory=RetrievalResult)
    prompt: str = ""
    fixed_text: str = ""
    emergency_numbers: dict[str, str] = field(default_factory=dict)
    stats: GenerationStats | None = None
    """Set once generation finishes, from what the provider reported."""

    @property
    def needs_model(self) -> bool:
        return self.kind is ResponseKind.ANSWER


@dataclass
class Answer:
    """A complete answer, once the stream has finished."""

    text: str
    context: AnswerContext
    first_token_ms: float = 0.0
    total_ms: float = 0.0
    output_tokens: int = 0

    @property
    def stats(self) -> GenerationStats | None:
        return self.context.stats

    @property
    def prompt_tokens(self) -> int:
        return self.context.stats.prompt_tokens if self.context.stats else 0

    @property
    def prompt_ms(self) -> float:
        """How long the provider spent reading the prompt, which is most of the wait."""
        return self.context.stats.prompt_ms if self.context.stats else 0.0

    @property
    def load_ms(self) -> float:
        return self.context.stats.load_ms if self.context.stats else 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.context.stats and self.context.stats.tokens_per_second:
            return self.context.stats.tokens_per_second
        generating = self.total_ms - self.first_token_ms
        if generating <= 0 or self.output_tokens <= 1:
            return 0.0
        return (self.output_tokens - 1) / (generating / 1000)


class AnswerPipeline:
    """Turns a question into a streamed, cited answer."""

    def __init__(
        self,
        retriever: HybridRetriever,
        client: ModelClient,
        *,
        top_k: int = 4,
        options: GenerationOptions | None = None,
        model: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.client = client
        self.top_k = top_k
        self.options = options or GenerationOptions()
        self.model = model

    async def prepare(self, question: str) -> AnswerContext:
        """Classify, retrieve, and decide how the question will be answered."""
        language = Language(detect_language(question))
        intent = classify(question)

        if intent is Intent.EMERGENCY:
            response = emergency_response(language.value)
            return AnswerContext(
                question=question,
                language=language,
                kind=ResponseKind.EMERGENCY,
                fixed_text=response.text,
                emergency_numbers=response.numbers,
            )

        if intent is Intent.PERSONAL_CASE:
            return AnswerContext(
                question=question,
                language=language,
                kind=ResponseKind.PERSONAL_CASE,
                fixed_text=personal_case_response(language),
            )

        retrieval = await self.retriever.search(question, top_k=self.top_k, language=language)
        if not retrieval.chunks or retrieval.best_similarity < SIMILARITY_FLOOR:
            return AnswerContext(
                question=question,
                language=language,
                kind=ResponseKind.NO_CONTEXT,
                retrieval=retrieval,
                fixed_text=no_context_response(language),
            )

        return AnswerContext(
            question=question,
            language=language,
            kind=ResponseKind.ANSWER,
            citations=citations_for(retrieval.chunks, language),
            retrieval=retrieval,
            prompt=build_prompt(question, retrieval.chunks, language),
        )

    async def stream(self, context: AnswerContext) -> AsyncIterator[str]:
        """Yield the answer text for a prepared question.

        A fixed response is yielded whole: there is nothing to wait for, and
        splitting it into fake tokens would only make it arrive later.
        """
        if not context.needs_model:
            yield context.fixed_text
            return

        async for chunk in self.client.generate(
            context.prompt, model=self.model, options=self.options
        ):
            if chunk.stats is not None:
                context.stats = chunk.stats
            if chunk.text:
                yield chunk.text

    async def answer(self, question: str) -> Answer:
        """Run the whole pipeline and collect the answer, with timings."""
        context = await self.prepare(question)

        started = time.perf_counter()
        first_token_ms = 0.0
        pieces: list[str] = []

        async for piece in self.stream(context):
            if not pieces:
                first_token_ms = (time.perf_counter() - started) * 1000
            pieces.append(piece)

        total_ms = (time.perf_counter() - started) * 1000
        return Answer(
            text="".join(pieces),
            context=context,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            output_tokens=len(pieces),
        )
