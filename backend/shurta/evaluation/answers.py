"""Measuring generated answers, and choosing the default model from the numbers.

Scored by rule, not by a model: key facts, citation validity, answer language,
guardrail behaviour and speed. Comparisons run on Arabic-normalised text, so a
fee in Arabic-Indic digits matches one in Western digits.

A run is resumable: each model's scorecard is written as that model finishes and
reused next time unless ``--force`` is given.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

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
from shurta.llm.base import ModelError
from shurta.llm.ollama import OllamaClient
from shurta.models import Language
from shurta.retrieval.hybrid import HybridRetriever
from shurta.retrieval.store import SearchIndex

RESULTS_DIR = REPOSITORY_ROOT / "eval"
RESULTS_MARKDOWN = RESULTS_DIR / "results.md"
RESULTS_JSON = RESULTS_DIR / "answer-results.json"
SCORECARD_DIR = RESULTS_DIR / "scorecards"
SUBSET_DIR = RESULTS_DIR / "subsets"

#: The selection name for a run over the whole question set.
FULL_SELECTION = "all"

#: The expected response kind for each expected behaviour.
EXPECTED_KIND = {
    Behaviour.ANSWER: ResponseKind.ANSWER,
    Behaviour.EMERGENCY: ResponseKind.EMERGENCY,
    Behaviour.REFUSE: ResponseKind.PERSONAL_CASE,
    Behaviour.OUT_OF_SCOPE: ResponseKind.NO_CONTEXT,
}

#: An out-of-scope question may legitimately reach the model, which is then
#: expected to say the passages do not cover it. Both outcomes count as correct.
ALSO_ACCEPTABLE = {Behaviour.OUT_OF_SCOPE: {ResponseKind.ANSWER}}

_CITATION_PATTERN = re.compile(r"\[(\d{1,2})\]")

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
    prompt_tokens: int = 0
    prompt_ms: float = 0.0
    load_ms: float = 0.0
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
    selection: str = FULL_SELECTION
    note: str = ""

    @property
    def comparable(self) -> bool:
        """Whether this run covered the whole question set."""
        return self.selection == FULL_SELECTION

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
    def prompt_tokens_p50(self) -> float:
        """Median prompt length, which is what the latency work is aimed at."""
        counts = [float(item.prompt_tokens) for item in self._generated() if item.prompt_tokens]
        return statistics.median(counts) if counts else 0.0

    @property
    def prompt_ms_p50(self) -> float:
        return self._percentile([item.prompt_ms for item in self._generated()], 0.5)

    @property
    def prompt_ms_p95(self) -> float:
        return self._percentile([item.prompt_ms for item in self._generated()], 0.95)

    @property
    def tokens_per_second(self) -> float:
        rates = [item.tokens_per_second for item in self._generated() if item.tokens_per_second]
        return statistics.median(rates) if rates else 0.0

    def to_dict(self) -> dict[str, object]:
        """The aggregate scores, which is all that gets published."""
        return {
            "model": self.model,
            "selection": self.selection,
            "note": self.note,
            "questions": len(self.items),
            "generated": len(self._generated()),
            "failures": self.failures,
            "key_fact_recall": round(self.fact_recall, 4),
            "key_fact_complete": round(self.fact_complete, 4),
            "citation_validity": round(self.citation_validity, 4),
            "citation_presence": round(self.citation_presence, 4),
            "language_match": round(self.language_match, 4),
            "behaviour_accuracy": round(self.behaviour_accuracy, 4),
            "prompt_tokens_p50": round(self.prompt_tokens_p50, 1),
            "prompt_ms_p50": round(self.prompt_ms_p50, 1),
            "prompt_ms_p95": round(self.prompt_ms_p95, 1),
            "first_token_ms_p50": round(self.first_token_p50, 1),
            "first_token_ms_p95": round(self.first_token_p95, 1),
            "total_ms_p50": round(self.total_p50, 1),
            "total_ms_p95": round(self.total_p95, 1),
            "tokens_per_second_median": round(self.tokens_per_second, 2),
        }


# --- choosing which questions to run -----------------------------------------


def group_of(question: Question) -> str:
    """The group a question is counted in: its dialect, or else its language."""
    return question.dialect or question.language.value


def first_per_group(questions: Sequence[Question], limit: int) -> list[Question]:
    """The first ``limit`` questions of each group, in file order."""
    seen: dict[str, int] = {}
    kept: list[Question] = []
    for question in questions:
        group = group_of(question)
        if seen.get(group, 0) >= limit:
            continue
        seen[group] = seen.get(group, 0) + 1
        kept.append(question)
    return kept


def subset_file(name: str) -> Path:
    """The file a ``--subset`` value names: a path, or a name under eval/subsets."""
    candidate = Path(name)
    if candidate.suffix or candidate.parent != Path("."):
        return candidate
    return SUBSET_DIR / f"{name}.txt"


def read_subset_ids(path: Path) -> list[str]:
    """Question identifiers from a subset file, ignoring blanks and comments."""
    if not path.exists():
        raise FileNotFoundError(f"no subset file at {path}")
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            ids.append(stripped)
    return ids


def by_ids(questions: Sequence[Question], ids: Iterable[str]) -> list[Question]:
    """The named questions, in the order the names were given."""
    known = {question.id: question for question in questions}
    missing = [identifier for identifier in ids if identifier not in known]
    if missing:
        raise KeyError(f"unknown question id(s): {', '.join(missing)}")
    return [known[identifier] for identifier in ids]


def select(
    questions: Sequence[Question],
    *,
    limit: int | None = None,
    subset: str | None = None,
) -> tuple[list[Question], str]:
    """The questions to run, and the name that selection is recorded under."""
    if subset is not None:
        path = subset_file(subset)
        return by_ids(questions, read_subset_ids(path)), f"subset:{path.stem}"
    if limit is not None:
        return first_per_group(questions, limit), f"limit:{limit}"
    return list(questions), FULL_SELECTION


# --- scoring one answer ------------------------------------------------------


def _cited_numbers(text: str) -> list[int]:
    return [int(match) for match in _CITATION_PATTERN.findall(text)]


def check_citations(text: str, available: int) -> tuple[bool, int]:
    """Whether every cited number exists, and how many were cited.

    A citation outside the supplied range is an invented source.
    """
    numbers = _cited_numbers(text)
    if not numbers:
        return True, 0
    valid = all(1 <= number <= available for number in numbers)
    return valid, len(numbers)


def check_language(text: str, expected: Language) -> bool:
    """Whether the answer is written in the language that was asked for.

    An English answer is allowed some Arabic, because it may quote an Arabic
    service name, so the test is that it is not predominantly Arabic.
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
        return True
    if kind in ALSO_ACCEPTABLE.get(question.expected_behaviour, set()):
        return check_declines(text, Language(question.language))
    return False


async def score_question(
    pipeline: AnswerPipeline, question: Question, facts: KeyFacts | None
) -> ItemResult:
    """Run one question and score the answer."""
    group = group_of(question)
    try:
        answer = await pipeline.answer(question.question)
    except ModelError as error:
        return ItemResult(
            id=question.id,
            language=question.language.value,
            group=group,
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
        group=group,
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
        prompt_tokens=answer.prompt_tokens,
        prompt_ms=answer.prompt_ms,
        load_ms=answer.load_ms,
    )


async def evaluate_model(
    index: SearchIndex,
    client: OllamaClient,
    model: str,
    questions: Sequence[Question],
    key_facts: dict[str, KeyFacts],
    settings: Settings,
    *,
    selection: str = FULL_SELECTION,
    note: str = "",
    progress: bool = True,
) -> ModelResult:
    """Score one model over the selected questions."""

    async def embed(query: str) -> list[float]:
        return (await client.embed([query]))[0]

    pipeline = AnswerPipeline(
        HybridRetriever(index, embed),
        client,
        top_k=settings.top_k,
        model=model,
        options=settings.generation_options(),
        max_passage_chars=settings.max_passage_chars,
    )

    # Warming keeps the model-load cost out of the first question's time to
    # first token, which would otherwise dominate the percentiles.
    await client.warm(model, keep_alive=settings.keep_alive)

    result = ModelResult(model=model, selection=selection, note=note)
    for position, question in enumerate(questions, start=1):
        result.items.append(await score_question(pipeline, question, key_facts.get(question.id)))
        if progress and position % 10 == 0:
            print(f"  {model}: {position}/{len(questions)}")
    return result


# --- scorecards, so a long run is resumable ----------------------------------


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", value.casefold())


def scorecard_path(model: str, selection: str = FULL_SELECTION) -> Path:
    """Where one model's scorecard for one selection is kept."""
    return SCORECARD_DIR / f"{_slug(model)}--{_slug(selection)}.json"


def write_scorecard(result: ModelResult, *, settings: Settings) -> Path:
    """Write one model's scorecard as soon as that model finishes."""
    path = scorecard_path(result.model, result.selection)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "measured_at": datetime.now(UTC).isoformat(),
                "settings": {
                    "top_k": settings.top_k,
                    "num_ctx": settings.num_ctx,
                    "max_answer_tokens": settings.max_answer_tokens,
                    "max_passage_chars": settings.max_passage_chars,
                    "num_thread": settings.num_thread,
                    "keep_alive": settings.keep_alive,
                },
                "scores": result.to_dict(),
                "items": [item.__dict__ for item in result.items],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return path


def _read_scorecard_file(path: Path, model: str) -> ModelResult | None:
    """One scorecard file, or ``None`` if it cannot be read."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = [ItemResult(**item) for item in payload["items"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    scores = payload.get("scores", {})
    return ModelResult(
        model=model,
        items=items,
        selection=str(scores.get("selection", FULL_SELECTION)),
        note=str(scores.get("note", "")),
    )


def read_scorecard(model: str, selection: str = FULL_SELECTION) -> ModelResult | None:
    """A previously written scorecard for this model and selection."""
    path = scorecard_path(model, selection)
    if not path.exists():
        return None
    result = _read_scorecard_file(path, model)
    return result if result is not None and result.selection == selection else None


def reuse_scorecard(model: str, selection: str = FULL_SELECTION) -> ModelResult | None:
    """Any scorecard for this model, preferring the selection asked for.

    A model measured on a subset is still worth reporting beside the others, so
    one report can be assembled from runs made separately. Its own selection
    comes with it, which is what keeps it out of the default-model choice.
    """
    exact = read_scorecard(model, selection)
    if exact is not None:
        return exact
    others = sorted(
        SCORECARD_DIR.glob(f"{_slug(model)}--*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in others:
        result = _read_scorecard_file(path, model)
        if result is not None:
            return result
    return None


# --- choosing the default ----------------------------------------------------

#: Weights for the composite score that picks the default model. Accuracy
#: dominates, but speed is not a tie-breaker: on processor-only inference a
#: model that is right and unusably slow is not the better choice.
WEIGHTS = {
    "key_fact_recall": 0.30,
    "citation_validity": 0.15,
    "language_match": 0.15,
    "behaviour_accuracy": 0.15,
    "speed": 0.25,
}

#: Time to first token at which the speed component reaches zero, in
#: milliseconds: the point where a reader watching a blank panel gives up.
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


def choose_default(results: Sequence[ModelResult]) -> ModelResult | None:
    """The best composite score among the models run over the whole set."""
    comparable = [result for result in results if result.comparable and result.items]
    return max(comparable, key=composite) if comparable else None


# --- reporting ---------------------------------------------------------------


def _questions_cell(result: ModelResult) -> str:
    if result.comparable:
        return str(len(result.items))
    return f"{len(result.items)} (subset)"


def as_markdown(results: Sequence[ModelResult], *, chunks_indexed: int) -> str:
    chosen = choose_default(results)

    lines = [
        "# Answer evaluation",
        "",
        f"Measured on {datetime.now(UTC):%Y-%m-%d} over {chunks_indexed} indexed passages, "
        "on processor-only inference.",
        "",
        "Every figure is measured by rule, not judged by a model. Amounts, durations and",
        "document phrases are compared against what the cited service states, on",
        "Arabic-normalised text so that a fee in Arabic-Indic digits matches one in",
        "Western digits.",
        "",
        "## Accuracy",
        "",
        "| Model | Questions | Key facts recalled | All facts present | Citations valid "
        "| Citations present | Answer language | Behaviour |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| `{result.model}` | {_questions_cell(result)} | {result.fact_recall:.3f} "
            f"| {result.fact_complete:.3f} | {result.citation_validity:.3f} "
            f"| {result.citation_presence:.3f} | {result.language_match:.3f} "
            f"| {result.behaviour_accuracy:.3f} |"
        )

    lines += [
        "",
        "## Speed",
        "",
        "| Model | Prompt tokens p50 | Prompt processing p50 | First token p50 "
        "| First token p95 | Full answer p50 | Full answer p95 | Tokens per second |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| `{result.model}` | {result.prompt_tokens_p50:.0f} "
            f"| {result.prompt_ms_p50 / 1000:.1f} s "
            f"| {result.first_token_p50 / 1000:.1f} s "
            f"| {result.first_token_p95 / 1000:.1f} s "
            f"| {result.total_p50 / 1000:.1f} s | {result.total_p95 / 1000:.1f} s "
            f"| {result.tokens_per_second:.1f} |"
        )

    notes = [result for result in results if result.note]
    if notes:
        lines += [""]
        lines += [f"- `{result.model}`: {result.note}" for result in notes]

    lines += [
        "",
        "## Composite score",
        "",
        "One number per model so the choice is reproducible. Weights: "
        + ", ".join(f"{name} {weight:.0%}" for name, weight in WEIGHTS.items())
        + ".",
        "",
        "Speed is a quarter of the score rather than a tie-breaker: a model that is",
        "accurate and takes half a minute to begin answering is not the better choice",
        "for something a reader is waiting on.",
        "",
        "| Model | Composite | Speed component |",
        "|---|---:|---:|",
    ]
    for result in sorted(results, key=composite, reverse=True):
        suffix = "" if result.comparable else " *(subset, not comparable)*"
        lines.append(
            f"| `{result.model}`{suffix} | {composite(result):.3f} | {speed_score(result):.3f} |"
        )

    if chosen is not None:
        lines += [
            "",
            "## Default",
            "",
            f"**`{chosen.model}`** is the configured default, on the numbers above.",
            "",
            "Change it by setting `GENERATION_MODEL` in `.env`. Every model compared here is",
            "a roughly four-billion-parameter instruction model at four-bit quantisation; a",
            "larger model is a supported configuration rather than the default, because the",
            "default has to work on an ordinary machine.",
        ]

    failures = sum(result.failures for result in results)
    if failures:
        lines += ["", f"{failures} question(s) failed outright during generation."]

    return "\n".join(lines) + "\n"


def write_reports(results: Sequence[ModelResult], *, chunks_indexed: int, questions: int) -> None:
    """Write the published report and aggregate scores from what has been run."""
    chosen = choose_default(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "measured_at": datetime.now(UTC).isoformat(),
                "chunks_indexed": chunks_indexed,
                "questions": questions,
                "weights": WEIGHTS,
                "models": [result.to_dict() for result in results],
                "chosen": chosen.model if chosen else None,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    RESULTS_MARKDOWN.write_text(
        as_markdown(results, chunks_indexed=chunks_indexed), encoding="utf-8"
    )


def main(
    settings: Settings,
    models: list[str] | None = None,
    *,
    force: bool = False,
    limit: int | None = None,
    subset: str | None = None,
    notes: dict[str, str] | None = None,
) -> int:
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

    try:
        questions, selection = select(load_questions(), limit=limit, subset=subset)
    except (FileNotFoundError, KeyError) as error:
        print(f"Cannot evaluate answers: {error}")
        return 1

    candidates = models or [settings.generation_model]
    labels = notes or {}

    async def run() -> list[ModelResult]:
        client = OllamaClient(
            settings.ollama_base_url,
            generation_model=settings.generation_model,
            embedding_model=settings.embedding_model,
            embedding_timeout=settings.embedding_timeout,
            generation_timeout=settings.generation_timeout,
            keep_alive=settings.keep_alive,
        )
        try:
            collected: list[ModelResult] = []
            for model in candidates:
                existing = None if force else reuse_scorecard(model, selection)
                if existing is not None:
                    print(
                        f"Reusing the scorecard for {model} ({existing.selection}); "
                        "--force re-runs it."
                    )
                    existing.note = labels.get(model, existing.note)
                    collected.append(existing)
                    continue

                print(f"Evaluating {model} over {len(questions)} questions ({selection})...")
                result = await evaluate_model(
                    index,
                    client,
                    model,
                    questions,
                    key_facts,
                    settings,
                    selection=selection,
                    note=labels.get(model, ""),
                )
                collected.append(result)
                print(f"  wrote {write_scorecard(result, settings=settings)}")
                write_reports(
                    collected, chunks_indexed=index.metadata.chunk_count, questions=len(questions)
                )
            return collected
        finally:
            await client.close()

    results = asyncio.run(run())
    write_reports(results, chunks_indexed=index.metadata.chunk_count, questions=len(questions))

    print()
    for result in results:
        print(
            f"  {result.model:28} facts {result.fact_recall:.3f}  "
            f"citations {result.citation_validity:.3f}  "
            f"language {result.language_match:.3f}  "
            f"behaviour {result.behaviour_accuracy:.3f}  "
            f"prompt {result.prompt_tokens_p50:.0f} tok / {result.prompt_ms_p50 / 1000:.1f}s  "
            f"first token p50 {result.first_token_p50 / 1000:.1f}s  "
            f"composite {composite(result):.3f}"
        )
    chosen = choose_default(results)
    if chosen is not None:
        print(f"\nBest on the numbers: {chosen.model}")
    print(f"Written to {RESULTS_MARKDOWN}")
    return 0
