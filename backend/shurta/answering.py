"""The answer pipeline: guard, retrieve, generate, cite.

This is the whole flow in one place, independent of how it is delivered, so the
API and the evaluation exercise the same behaviour rather than two similar
versions of it.

The order is the point. The guard runs before retrieval and before any model
call, so an emergency costs nothing and cannot be mishandled. Retrieval runs
next, and when nothing it finds is relevant enough the pipeline answers from a
fixed response rather than asking a model to work from an empty context, which
is where a model of this size would start inventing.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum

from shurta.arabic import detect_language
from shurta.guardrails import Intent, classify, emergency_response
from shurta.llm.base import GenerationOptions, ModelClient
from shurta.models import Language
from shurta.prompting import (
    Citation,
    build_prompt,
    citations_for,
    no_context_response,
    personal_case_response,
)
from shurta.retrieval.hybrid import HybridRetriever, RetrievalResult

#: Cosine similarity below which nothing retrieved is close enough to be worth
#: a model call, and the fixed "not in my information" response is used instead.
#:
#: Measured on the evaluation set rather than chosen. Over the 100 answerable
#: questions the closest passage scored 0.526 at worst and 0.697 at the median;
#: over the out-of-scope questions it ranged from 0.355 to 0.566. The two
#: distributions overlap, and honestly so — "how do I renew my driving licence"
#: really is close to the traffic services in the corpus.
#:
#: So the floor is set below every answerable question rather than between the
#: two groups. At 0.50 it rejects no question the corpus can answer and catches
#: four of the six out-of-scope ones before any model runs. The remaining two
#: are the genuinely adjacent government services, and those are left to the
#: prompt, which can see the passages and say they cover something else. A
#: higher floor would start refusing questions the corpus does answer, which is
#: the worse failure.
#:
#: The fused retrieval score is deliberately not used for this: it is built from
#: rank positions, so the best passage scores near the maximum regardless of
#: relevance. Measured, out-of-scope questions scored *higher* on it than most
#: answerable ones.
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
    def tokens_per_second(self) -> float:
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

        async for piece in self.client.generate(
            context.prompt, model=self.model, options=self.options
        ):
            yield piece

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
