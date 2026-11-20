"""Measuring generated answers, and choosing the default model from the numbers.

What gets scored, and why each one is here:

**Key-fact correctness.** Does the answer contain the figures, durations and
documents the cited service actually states? This is the measure that matters:
an answer that reads well and gives the wrong fee is worse than no answer.
Checked by rule, on Arabic-normalised text, so a fee written in Arabic-Indic
digits matches one written in Western digits.

**Citation correctness.** Every bracketed number in the answer must be one of
the passages that was actually supplied. A model that cites ``[7]`` when four
passages were given has invented a source, which is the failure most likely to
be believed.

**Answer language.** An Arabic question must get an Arabic answer. A small model
will otherwise answer in the language of the retrieved passages.

**Behaviour.** Does an emergency get the emergency numbers, a personal-case
question a refusal, an out-of-scope question an admission of not knowing?

**Speed.** Time to first token and tokens per second, at the median and the
95th percentile. On processor-only inference these decide whether the thing is
usable at all, so they carry as much weight as accuracy in the final choice.
"""

from __future__ import annotations

import asyncio
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from shurta.answering import AnswerPipeline, ResponseKind
from shurta.arabic import arabic_ratio
from shurta.config import REPOSITORY_ROOT, Settings
from shurta.evaluation.dataset import (
    Behaviour,
    KeyFacts,
    Question,
    load_key_facts,
    load_questions,
)
from shurta.evaluation.facts import contains_amount, contains_document, contains_duration
from shurta.llm.base import GenerationOptions, ModelError
from shurta.llm.ollama import OllamaClient
from shurta.models import Language
from shurta.retrieval.hybrid import HybridRetriever
from shurta.retrieval.store import SearchIndex

RESULTS_DIR = REPOSITORY_ROOT / "eval"
RESULTS_MARKDOWN = RESULTS_DIR / "results.md"
RESULTS_JSON = RESULTS_DIR / "answer-results.json"

#: The expected response kind for each expected behaviour.
EXPECTED_KIND = {
    Behaviour.ANSWER: ResponseKind.ANSWER,
    Behaviour.EMERGENCY: ResponseKind.EMERGENCY,
    Behaviour.REFUSE: ResponseKind.PERSONAL_CASE,
    Behaviour.OUT_OF_SCOPE: ResponseKind.NO_CONTEXT,
}

#: An out-of-scope question may legitimately be routed to the model, which is
#: then expected to say the passages do not cover it. So both outcomes count as
#: correct behaviour for those items.
ALSO_ACCEPTABLE = {Behaviour.OUT_OF_SCOPE: {ResponseKind.ANSWER}}

_CITATION_PATTERN = r"\[(\d{1,2})\]"

#: Phrases that count as admitting the answer is not in the passages. Checked
#: only for out-of-scope questions that reached the model.
_DECLINING_EN = (
    "could not find",
    "not in",
    "does not cover",
    "do not cover",
    "no information",
    "not covered",
    "cannot find",
    "i don't have",
    "i do not have",
    "unable to",
    "not available in",
)
_DECLINING_AR = (
    "لم أجد",
    "لا تتضمن",
    "لا تغطي",
    "لا توجد معلومات",
    "غير متوفرة",
    "لا أملك",
    "ليست لدي",
    "لا تحتوي",
)


@dataclass
class ItemResult:
    """How one model did on one question."""

    id: str
    language: str
    group: str
    behaviour: str
    kind: str
    behaviour_correct: bool
    language_correct: bool
    citations_valid: bool
    cited_count: int
    facts_expected: int
    facts_found: int
    first_token_ms: float
    total_ms: float
    tokens_per_second: float
    used_model: bool
    error: str = ""

    @property
    def facts_correct(self) -> bool:
        """Whether every checkable fact appeared. Vacuously true with none."""
        return self.facts_found == self.facts_expected


@dataclass
class ModelResult:
    """Everything measured for one model."""

    model: str
    items: list[ItemResult] = field(default_factory=list)

    def _generated(self) -> list[ItemResult]:
        """Items that actually ran the model, which is what timings describe."""
        return [item for item in self.items if item.used_model and not item.error]

    def _with_facts(self) -> list[ItemResult]:
        return [item for item in self.items if item.facts_expected > 0 and not item.error]

    def _answerable(self) -> list[ItemResult]:
        return [item for item in self.items if item.behaviour == Behaviour.ANSWER.value]

    @property
    def fact_recall(self) -> float:
        """Share of all checkable facts that appeared in an answer."""
        expected = sum(item.facts_expected for item in self._with_facts())
        found = sum(item.facts_found for item in self._with_facts())
        return found / expected if expected else 0.0

    @property
    def fact_complete(self) -> float:
        """Share of questions where *every* checkable fact appeared."""
        items = self._with_facts()
        return sum(1 for item in items if item.facts_correct) / len(items) if items else 0.0

    @property
    def citation_validity(self) -> float:
        cited = [item for item in self.items if item.cited_count > 0]
        return sum(1 for item in cited if item.citations_valid) / len(cited) if cited else 0.0

    @property
    def citation_presence(self) -> float:
        """Share of answered questions that carried any citation at all."""
        items = [item for item in self._answerable() if not item.error]
        return sum(1 for item in items if item.cited_count > 0) / len(items) if items else 0.0

    @property
    def language_match(self) -> float:
        items = [item for item in self.items if not item.error]
        return sum(1 for item in items if item.language_correct) / len(items) if items else 0.0

    @property
    def behaviour_accuracy(self) -> float:
        return (
            sum(1 for item in self.items if item.behaviour_correct) / len(self.items)
            if self.items
            else 0.0
        )

    @property
    def failures(self) -> int:
        return sum(1 for item in self.items if item.error)

    def _percentile(self, values: Sequence[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = min(int(fraction * len(ordered)), len(ordered) - 1)
        return ordered[position]

    @property
    def first_token_p50(self) -> float:
        return self._percentile([item.first_token_ms for item in self._generated()], 0.5)

    @property
    def first_token_p95(self) -> float:
        return self._percentile([item.first_token_ms for item in self._generated()], 0.95)

    @property
    def total_p50(self) -> float:
        return self._percentile([item.total_ms for item in self._generated()], 0.5)

    @property
    def total_p95(self) -> float:
        return self._percentile([item.total_ms for item in self._generated()], 0.95)

    @property
    def tokens_per_second(self) -> float:
        rates = [item.tokens_per_second for item in self._generated() if item.tokens_per_second]
        return statistics.median(rates) if rates else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "questions": len(self.items),
            "generated": len(self._generated()),
            "failures": self.failures,
            "key_fact_recall": round(self.fact_recall, 4),
            "key_fact_complete": round(self.fact_complete, 4),
            "citation_validity": round(self.citation_validity, 4),
            "citation_presence": round(self.citation_presence, 4),
            "language_match": round(self.language_match, 4),
            "behaviour_accuracy": round(self.behaviour_accuracy, 4),
            "first_token_ms_p50": round(self.first_token_p50, 1),
            "first_token_ms_p95": round(self.first_token_p95, 1),
            "total_ms_p50": round(self.total_p50, 1),
            "total_ms_p95": round(self.total_p95, 1),
            "tokens_per_second_median": round(self.tokens_per_second, 2),
        }


# --- scoring one answer ------------------------------------------------------


def _cited_numbers(text: str) -> list[int]:
    import re

    return [int(match) for match in re.findall(_CITATION_PATTERN, text)]


def check_citations(text: str, available: int) -> tuple[bool, int]:
    """Whether every cited number exists, and how many were cited.

    A citation outside the supplied range is an invented source, which is the
    failure a reader is least able to detect.
    """
    numbers = _cited_numbers(text)
    if not numbers:
        return True, 0
    valid = all(1 <= number <= available for number in numbers)
    return valid, len(numbers)


def check_language(text: str, expected: Language) -> bool:
    """Whether the answer is written in the language that was asked for.

    An Arabic answer always contains Arabic script. An English answer is allowed
    some Arabic, because a correct English answer may quote an Arabic service
    name, so the test is that it is not *predominantly* Arabic.
    """
    stripped = text.strip()
    if not stripped:
        return False
    ratio = arabic_ratio(stripped)
    return ratio >= 0.25 if expected is Language.AR else ratio < 0.5


def check_declines(text: str, language: Language) -> bool:
    """Whether the answer admits the passages do not cover the question."""
    lowered = text.casefold()
    phrases = _DECLINING_AR if language is Language.AR else _DECLINING_EN
    if any(phrase.casefold() in lowered for phrase in phrases):
        return True
    # A model answering in the other language still counts if it declines there.
    other = _DECLINING_EN if language is Language.AR else _DECLINING_AR
    return any(phrase.casefold() in lowered for phrase in other)


def count_facts(text: str, facts: KeyFacts) -> tuple[int, int]:
    """How many of the expected facts the answer states, and how many there were."""
    expected = len(facts.amounts) + len(facts.durations) + len(facts.documents)
    found = sum(
        [
            sum(1 for amount in facts.amounts if contains_amount(text, amount)),
            sum(1 for duration in facts.durations if contains_duration(text, duration)),
            sum(1 for document in facts.documents if contains_document(text, document)),
        ]
    )
    return found, expected


def _behaviour_correct(question: Question, kind: ResponseKind, text: str) -> bool:
    expected = EXPECTED_KIND[question.expected_behaviour]
    if kind is expected:
        # An out-of-scope question routed to the fixed response is correct by
        # construction; one that reached the model must actually decline.
        return True
    if kind in ALSO_ACCEPTABLE.get(question.expected_behaviour, set()):
        return check_declines(text, Language(question.language))
    return False


async def score_question(
    pipeline: AnswerPipeline, question: Question, facts: KeyFacts | None
) -> ItemResult:
    """Run one question and score the answer."""
    group = question.dialect or question.language
    try:
        answer = await pipeline.answer(question.question)
    except ModelError as error:
        return ItemResult(
            id=question.id,
            language=question.language.value,
            group=str(group),
            behaviour=question.expected_behaviour.value,
            kind="error",
            behaviour_correct=False,
            language_correct=False,
            citations_valid=False,
            cited_count=0,
            facts_expected=0,
            facts_found=0,
            first_token_ms=0.0,
            total_ms=0.0,
            tokens_per_second=0.0,
            used_model=True,
            error=str(error),
        )

    context = answer.context
    valid, cited = check_citations(answer.text, len(context.citations))
    found, expected_facts = count_facts(answer.text, facts) if facts else (0, 0)

    return ItemResult(
        id=question.id,
        language=question.language.value,
        group=str(group),
        behaviour=question.expected_behaviour.value,
        kind=context.kind.value,
        behaviour_correct=_behaviour_correct(question, context.kind, answer.text),
        language_correct=check_language(answer.text, Language(question.language)),
        citations_valid=valid,
        cited_count=cited,
        facts_expected=expected_facts,
        facts_found=found,
        first_token_ms=answer.first_token_ms,
        total_ms=answer.total_ms,
        tokens_per_second=answer.tokens_per_second,
        used_model=context.needs_model,
    )


async def evaluate_model(
    index: SearchIndex,
    client: OllamaClient,
    model: str,
    questions: Sequence[Question],
    key_facts: dict[str, KeyFacts],
    settings: Settings,
    *,
    progress: bool = True,
) -> ModelResult:
    """Score one model over the whole question set."""

    async def embed(query: str) -> list[float]:
        return (await client.embed([query]))[0]

    pipeline = AnswerPipeline(
        HybridRetriever(index, embed),
        client,
        top_k=settings.top_k,
        model=model,
        options=GenerationOptions(
            num_ctx=settings.num_ctx,
            max_tokens=settings.max_answer_tokens,
            keep_alive=settings.keep_alive,
        ),
    )

    # Warming first keeps the model-load cost out of the first question's
    # time-to-first-token, which would otherwise dominate the percentiles.
    await client.warm(model)

    result = ModelResult(model=model)
    for position, question in enumerate(questions, start=1):
        result.items.append(await score_question(pipeline, question, key_facts.get(question.id)))
        if progress and position % 10 == 0:
            print(f"  {model}: {position}/{len(questions)}")
    return result


# --- choosing the default ----------------------------------------------------

#: Weights for the composite score that picks the default model. Accuracy
#: dominates, but speed is not a tie-breaker here: on processor-only inference a
#: model that is right and unusably slow is not the better choice.
WEIGHTS = {
    "key_fact_recall": 0.30,
    "citation_validity": 0.15,
    "language_match": 0.15,
    "behaviour_accuracy": 0.15,
    "speed": 0.25,
}

#: Time to first token, in milliseconds, at which the speed component reaches
#: zero. Chosen as the point where a reader watching a blank panel gives up.
SLOW_FIRST_TOKEN_MS = 20_000


def speed_score(result: ModelResult) -> float:
    """Speed as a number between 0 and 1, from the 95th-percentile first token."""
    if not result.first_token_p95:
        return 0.0
    return max(0.0, 1.0 - result.first_token_p95 / SLOW_FIRST_TOKEN_MS)


def composite(result: ModelResult) -> float:
    """One number per model, so the choice is reproducible rather than a feeling."""
    return (
        WEIGHTS["key_fact_recall"] * result.fact_recall
        + WEIGHTS["citation_validity"] * result.citation_validity
        + WEIGHTS["language_match"] * result.language_match
        + WEIGHTS["behaviour_accuracy"] * result.behaviour_accuracy
        + WEIGHTS["speed"] * speed_score(result)
    )


def choose_default(results: Sequence[ModelResult]) -> ModelResult:
    """The model with the best composite score."""
    return max(results, key=composite)


# --- reporting ---------------------------------------------------------------


def as_markdown(results: Sequence[ModelResult], *, index: SearchIndex) -> str:
    chosen = choose_default(results) if results else None

    lines = [
        "# Answer evaluation",
        "",
        f"Measured on {datetime.now(UTC):%Y-%m-%d}, over "
        f"{len(results[0].items) if results else 0} questions and "
        f"{index.metadata.chunk_count} indexed passages, on processor-only inference.",
        "",
        "Every figure below is measured by rule, not judged by a model. Monetary",
        "amounts, durations and document phrases are compared against what the cited",
        "service actually states, on Arabic-normalised text so that a fee written in",
        "Arabic-Indic digits matches one written in Western digits.",
        "",
        "## Accuracy",
        "",
        "| Model | Key facts recalled | All facts present | Citations valid | Citations present "
        "| Answer language | Behaviour |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| `{result.model}` | {result.fact_recall:.3f} | {result.fact_complete:.3f} "
            f"| {result.citation_validity:.3f} | {result.citation_presence:.3f} "
            f"| {result.language_match:.3f} | {result.behaviour_accuracy:.3f} |"
        )

    lines += [
        "",
        "## Speed",
        "",
        "| Model | First token p50 | First token p95 | Full answer p50 | Full answer p95 "
        "| Tokens per second |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| `{result.model}` | {result.first_token_p50 / 1000:.1f} s "
            f"| {result.first_token_p95 / 1000:.1f} s "
            f"| {result.total_p50 / 1000:.1f} s | {result.total_p95 / 1000:.1f} s "
            f"| {result.tokens_per_second:.1f} |"
        )

    lines += [
        "",
        "## Composite score",
        "",
        "One number per model so the choice is reproducible rather than a matter of",
        "taste. Weights: "
        + ", ".join(f"{name} {weight:.0%}" for name, weight in WEIGHTS.items())
        + ".",
        "",
        "Speed is a quarter of the score rather than a tie-breaker. A model that is",
        "accurate and takes half a minute to begin answering is not the better choice",
        "for something a reader is waiting on.",
        "",
        "| Model | Composite | Speed component |",
        "|---|---:|---:|",
    ]
    for result in sorted(results, key=composite, reverse=True):
        lines.append(f"| `{result.model}` | {composite(result):.3f} | {speed_score(result):.3f} |")

    if chosen is not None:
        lines += [
            "",
            "## Default",
            "",
            f"**`{chosen.model}`** is the configured default, on the numbers above.",
            "",
            "Change it by setting `GENERATION_MODEL` in `.env`. Every model compared here",
            "is a roughly four-billion-parameter instruction model at four-bit",
            "quantisation; a larger model is a supported configuration rather than the",
            "default, because the default has to work on an ordinary machine.",
        ]

    failures = sum(result.failures for result in results)
    if failures:
        lines += ["", f"{failures} question(s) failed outright during generation."]

    return "\n".join(lines) + "\n"


def main(settings: Settings, models: list[str] | None = None) -> int:
    """Run the answer evaluation for each candidate model and write the results."""
    try:
        index = SearchIndex.load(settings.index_path)
    except RuntimeError as error:
        print(f"Cannot evaluate answers: {error}")
        return 1

    key_facts = load_key_facts()
    if not key_facts:
        print(
            "No key facts derived yet, so answers cannot be scored for correctness.\n"
            "Run `python -m shurta eval prepare` first."
        )
        return 1

    questions = load_questions()
    candidates = models or [settings.generation_model]

    async def run() -> list[ModelResult]:
        client = OllamaClient(
            settings.ollama_base_url,
            generation_model=settings.generation_model,
            embedding_model=settings.embedding_model,
            embedding_timeout=settings.embedding_timeout,
            generation_timeout=settings.generation_timeout,
        )
        try:
            collected: list[ModelResult] = []
            for model in candidates:
                print(f"Evaluating {model} over {len(questions)} questions...")
                collected.append(
                    await evaluate_model(index, client, model, questions, key_facts, settings)
                )
            return collected
        finally:
            await client.close()

    results = asyncio.run(run())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "measured_at": datetime.now(UTC).isoformat(),
                "chunks_indexed": index.metadata.chunk_count,
                "questions": len(questions),
                "weights": WEIGHTS,
                "models": [result.to_dict() for result in results],
                "chosen": choose_default(results).model if results else None,
                "items": {
                    result.model: [item.__dict__ for item in result.items] for result in results
                },
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    RESULTS_MARKDOWN.write_text(as_markdown(results, index=index), encoding="utf-8")

    print()
    for result in results:
        print(
            f"  {result.model:14} facts {result.fact_recall:.3f}  "
            f"citations {result.citation_validity:.3f}  "
            f"language {result.language_match:.3f}  "
            f"behaviour {result.behaviour_accuracy:.3f}  "
            f"first token p95 {result.first_token_p95 / 1000:.1f}s  "
            f"composite {composite(result):.3f}"
        )
    if results:
        print(f"\nBest on the numbers: {choose_default(results).model}")
    print(f"Written to {RESULTS_MARKDOWN}")
    return 0
