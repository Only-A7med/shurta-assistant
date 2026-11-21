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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from shurta.answering import AnswerPipeline, ResponseKind
from shurta.arabic import arabic_ratio, normalise
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

#: Sections of the report that are written by hand rather than measured, kept
#: beside the generator that inlines them so that ``eval/`` holds only results.
NOTE_DIR = Path(__file__).parent / "notes"
LATENCY_NOTE = NOTE_DIR / "latency-note.md"
FAILURE_NOTE = NOTE_DIR / "failure-causes.md"

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
    # "The passages do not contain information on how to renew a passport" is a
    # correct refusal, and was being scored as a wrong answer for want of this.
    "not contain",
    "no information",
    "not covered",
    "cannot find",
    "i don't have",
    "i do not have",
    "unable to",
    "not available in",
)
#: Matched against *normalised* text, so each phrase is written in normalised
#: spelling or it silently never matches. Each is an equivalent of one on the
#: English list, so both languages are held to the same standard.
_DECLINING_AR = (
    "لم اجد",  # could not find
    "لا تتضمن",  # does not include
    "لا تحتوي",  # does not contain
    "لا تشمل",  # does not include
    "لا تغطي",  # does not cover
    "لا تذكر",  # does not mention
    "لم ترد",  # is not mentioned
    "لا توجد معلومات",  # there is no information
    "غير متوفر",  # not available (also matches غير متوفره)
    "لا تتوفر",  # is not available
    "لا املك",  # I do not have
    "ليست لدي",  # I do not have
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

    text: str = ""
    """The answer as written, kept so a scorecard can be re-scored.

    Without it a change to the grader means regenerating every answer, which
    is an hour of processor-only inference to re-check a rule. It stays out of
    the published aggregate: scorecards are local, like the key facts they are
    scored against.
    """

    available_citations: int = 0
    """How many passages the answer could have cited, which is what makes a
    stored answer re-checkable for invented citations."""

    facts_expected_all: int = 0
    facts_found_all: int = 0
    """The same count against every figure the question's fields state, rather
    than only the variant it asks about. Reported beside the scoped figure so
    the effect of scoping the questions is visible."""

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
    def fact_recall_all_figures(self) -> float:
        """Share recalled when every figure a field states is required.

        What the same answers scored before the questions recorded which variant
        they ask about: a question asking a resident's fee was graded against
        the citizen and overseas figures too. Kept as a second column so the
        change is visible rather than only the improved number.
        """
        items = [item for item in self.items if item.facts_expected_all > 0 and not item.error]
        expected = sum(item.facts_expected_all for item in items)
        found = sum(item.facts_found_all for item in items)
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

    @property
    def answers_stored(self) -> int:
        """How many questions produced an answer at all."""
        return sum(1 for item in self.items if item.text)

    @property
    def accuracy_reported(self) -> bool:
        """Whether every question produced an answer to grade.

        A model that returns nothing for half its questions has accuracy
        figures covering only the half it answered, which read as the model's
        score and are not. The published report blanks those columns and says
        how many answers it has, rather than printing a number over a set of
        questions nobody can see.
        """
        return bool(self.items) and self.answers_stored == len(self.items)

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
            "answers_stored": self.answers_stored,
            # The accuracy figures below cover only the questions that produced
            # an answer, so a reader taking them from here needs telling when
            # that is not every question. The published report prints no number
            # at all in that case.
            "accuracy_reported": self.accuracy_reported,
            "key_fact_recall": round(self.fact_recall, 4),
            "key_fact_recall_all_figures": round(self.fact_recall_all_figures, 4),
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
    """Whether the answer admits the passages do not cover the question.

    Matched on normalised text, so an Arabic answer is not judged on which of
    several equivalent spellings it happened to use.
    """
    normalised = normalise(text)
    phrases = _DECLINING_AR if language is Language.AR else _DECLINING_EN
    if any(phrase in normalised for phrase in phrases):
        return True
    # A model answering in the other language still counts if it declines there.
    other = _DECLINING_EN if language is Language.AR else _DECLINING_AR
    return any(phrase in normalised for phrase in other)


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


def count_facts_both_ways(text: str, facts: KeyFacts | None) -> tuple[int, int, int, int]:
    """The scoped fact count, and the count against every figure the fields state."""
    if facts is None:
        return 0, 0, 0, 0
    found, expected = count_facts(text, facts)
    found_all, expected_all = count_facts(text, facts.every_figure())
    return found, expected, found_all, expected_all


def behaviour_correct_for(
    expected_behaviour: Behaviour, kind: ResponseKind, text: str, language: Language
) -> bool:
    """Whether a response of this kind is right for this expected behaviour.

    Takes plain values rather than a question, so a stored scorecard item can be
    re-graded with it: a change to the declining phrases is a grader change, and
    behaviour is the score it moves.
    """
    expected = EXPECTED_KIND[expected_behaviour]
    if kind is expected:
        return True
    if kind in ALSO_ACCEPTABLE.get(expected_behaviour, set()):
        return check_declines(text, language)
    return False


def _behaviour_correct(question: Question, kind: ResponseKind, text: str) -> bool:
    return behaviour_correct_for(
        question.expected_behaviour, kind, text, Language(question.language)
    )


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
    found, expected_facts, found_all, expected_all = count_facts_both_ways(answer.text, facts)

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
        facts_expected_all=expected_all,
        facts_found_all=found_all,
        first_token_ms=answer.first_token_ms,
        total_ms=answer.total_ms,
        tokens_per_second=answer.tokens_per_second,
        used_model=context.needs_model,
        prompt_tokens=answer.prompt_tokens,
        prompt_ms=answer.prompt_ms,
        load_ms=answer.load_ms,
        text=answer.text,
        available_citations=len(context.citations),
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
    scored: Mapping[str, ItemResult] | None = None,
    after_each: Callable[[ModelResult], None] | None = None,
) -> ModelResult:
    """Score one model over the selected questions.

    Questions already in ``scored`` are carried over rather than run again, and
    ``after_each`` is called as every question finishes, so an interrupted run
    resumes where it stopped rather than from the beginning.
    """
    already = dict(scored or {})
    result = ModelResult(model=model, selection=selection, note=note)
    pending = [question for question in questions if question.id not in already]
    if not pending:
        result.items = [already[question.id] for question in questions if question.id in already]
        return result

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

    for position, question in enumerate(questions, start=1):
        if question.id in already:
            result.items.append(already[question.id])
            continue
        result.items.append(await score_question(pipeline, question, key_facts.get(question.id)))
        if after_each is not None:
            after_each(result)
        if progress and position % 10 == 0:
            print(f"  {model}: {position}/{len(questions)}", flush=True)
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


# --- re-scoring stored answers ------------------------------------------------


def rescore_item(item: ItemResult, facts: KeyFacts | None) -> ItemResult:
    """Re-apply the grading rules to an answer already generated.

    Covers every rule a grader fix can change: key facts, citations, answer
    language, and behaviour — an out-of-scope question that reached the model is
    judged on whether its answer declines, which is a phrase list. The response
    *kind* is not recomputed, because that was the guardrail's decision and not
    the text's. An item stored before answers were kept has no text and is
    returned untouched.
    """
    if not item.text:
        return item
    found, expected, found_all, expected_all = count_facts_both_ways(item.text, facts)
    valid, cited = check_citations(item.text, item.available_citations)
    return replace(
        item,
        facts_found=found,
        facts_expected=expected,
        facts_found_all=found_all,
        facts_expected_all=expected_all,
        citations_valid=valid,
        cited_count=cited,
        language_correct=check_language(item.text, Language(item.language)),
        behaviour_correct=behaviour_correct_for(
            Behaviour(item.behaviour),
            ResponseKind(item.kind),
            item.text,
            Language(item.language),
        ),
    )


def rescore(result: ModelResult, key_facts: Mapping[str, KeyFacts]) -> ModelResult:
    """One model's scorecard re-graded from its stored answers."""
    return ModelResult(
        model=result.model,
        items=[rescore_item(item, key_facts.get(item.id)) for item in result.items],
        selection=result.selection,
        note=result.note,
    )


def rescorable(result: ModelResult) -> int:
    """How many of a scorecard's items carry the answer text to re-grade."""
    return result.answers_stored


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

#: What each weight is called in the published report, so the reader sees the
#: thing measured rather than the attribute it is stored in.
WEIGHT_LABELS = {
    "key_fact_recall": "key facts",
    "citation_validity": "citation validity",
    "language_match": "answer language",
    "behaviour_accuracy": "behaviour",
    "speed": "speed",
}

#: Bounds of the speed component, in milliseconds of median time to first token:
#: linear between them, clamped outside. They describe what processor-only
#: inference can do, not what any candidate scores, and were fixed before the
#: models were compared. An earlier scale zeroing at 20 s on the 95th percentile
#: scored every model that can run here exactly 0.000. See eval/results.md.
FAST_FIRST_TOKEN_MS = 5_000
SLOW_FIRST_TOKEN_MS = 60_000


def speed_score(result: ModelResult, *, first_token_p50_ms: float | None = None) -> float:
    """Speed as a number between 0 and 1, from the median time to first token.

    The median rather than the 95th percentile: run-to-run variance on
    processor-only inference is large, and the tail measures the longest prompt
    rather than how the model feels to use.

    ``first_token_p50_ms`` replaces the run's own median. Latency is the
    noisiest thing measured here and the only score that anything else running
    on the machine can move, so it is worth taking from a run made on a quiet
    machine over a fixed set of questions while accuracy comes from the whole
    set.
    """
    first_token = result.first_token_p50 if first_token_p50_ms is None else first_token_p50_ms
    if not first_token:
        return 0.0
    span = SLOW_FIRST_TOKEN_MS - FAST_FIRST_TOKEN_MS
    return min(1.0, max(0.0, (SLOW_FIRST_TOKEN_MS - first_token) / span))


def composite(result: ModelResult, *, speed_reference: Mapping[str, float] | None = None) -> float:
    """One number per model, so the choice is reproducible rather than a feeling.

    ``speed_reference`` maps a model name to a median time to first token
    measured elsewhere, which then supplies the speed component.
    """
    reference = (speed_reference or {}).get(result.model)
    return (
        WEIGHTS["key_fact_recall"] * result.fact_recall
        + WEIGHTS["citation_validity"] * result.citation_validity
        + WEIGHTS["language_match"] * result.language_match
        + WEIGHTS["behaviour_accuracy"] * result.behaviour_accuracy
        + WEIGHTS["speed"] * speed_score(result, first_token_p50_ms=reference)
    )


def comparable_on_the_same_questions(results: Sequence[ModelResult]) -> bool:
    """Whether every run covered exactly the same questions.

    Two models are only comparable on identical question sets. A run that
    stopped part way through still records ``selection: all``, and comparing its
    score with a finished run's reads as a difference between the models when it
    is a difference between the question sets: an interrupted run's questions
    are the *first* N, which are not a representative sample of the whole.
    """
    scored = [result for result in results if result.items]
    if len(scored) < 2:
        return True
    first = {item.id for item in scored[0].items}
    return all({item.id for item in result.items} == first for result in scored[1:])


def choose_default(
    results: Sequence[ModelResult], *, speed_reference: Mapping[str, float] | None = None
) -> ModelResult | None:
    """The best composite score among the models run over the whole set.

    Returns nothing unless the candidates were scored on identical questions,
    so a half-finished run can never win or lose the comparison on its length.
    """
    candidates = [result for result in results if result.comparable and result.items]
    if not candidates or not comparable_on_the_same_questions(candidates):
        return None
    return max(candidates, key=lambda result: composite(result, speed_reference=speed_reference))


def speed_reference_from_subset(models: Sequence[str], subset: str) -> dict[str, float]:
    """Each model's median time to first token from its subset scorecard.

    Latency is the one score another process on the machine can move, so it is
    taken from a short run over a fixed set of questions made with nothing else
    running, while accuracy comes from the whole set. A model with no such
    scorecard is left out and keeps its own run's median.
    """
    selection = f"subset:{subset}"
    reference: dict[str, float] = {}
    for model in models:
        card = read_scorecard(model, selection)
        if card is not None and card.first_token_p50:
            reference[model] = card.first_token_p50
    return reference


# --- reporting ---------------------------------------------------------------


def _questions_cell(result: ModelResult) -> str:
    if result.comparable:
        return str(len(result.items))
    return f"{len(result.items)} (subset)"


def _subset_clause(results: Sequence[ModelResult]) -> str:
    """How to refer to the rows scored on fewer questions, if there are any."""
    subsets = sum(1 for result in results if result.items and not result.comparable)
    if not subsets:
        return ""
    if subsets == 1:
        return " The subset row is not."
    return f" The {subsets} subset rows are not."


def _comparability_line(results: Sequence[ModelResult]) -> str:
    """State plainly which questions each model was scored on.

    Without this the table invites exactly the wrong comparison: a model scored
    on part of the set read beside one scored on all of it.
    """
    full = [result for result in results if result.comparable and result.items]
    sizes = {len(result.items) for result in full}
    if len(full) > 1 and comparable_on_the_same_questions(full):
        counted = "Both" if len(full) == 2 else f"All {len(full)}"
        return (
            f"{counted} instruction models were scored on the same {sizes.pop()} questions, "
            "so their figures are directly comparable." + _subset_clause(results)
        )
    if len(full) > 1:
        listed = ", ".join(f"`{result.model}` {len(result.items)}" for result in full)
        return (
            f"These models were not scored on the same questions ({listed}), so the figures "
            "below are not comparable with each other and no default is chosen from them. "
            "Re-run the shorter ones over the whole set."
        )
    return "One model is reported, so there is no model-to-model comparison." + (
        _subset_clause(results) or " Every row covers the whole question set."
    )


def _model_note(result: ModelResult) -> str:
    """The footnote for one model: what was passed in, plus what the data says.

    The missing-answers sentence is derived rather than written by hand,
    because the columns it explains are blanked from the same count.
    """
    parts: list[str] = []
    if result.note:
        parts.append(result.note.rstrip("."))
    if result.items and not result.accuracy_reported:
        parts.append(
            f"only {result.answers_stored} of its {len(result.items)} answers were "
            "non-empty, so accuracy is not reported"
        )
    if not parts:
        return ""
    body = ". ".join([parts[0]] + [part[0].upper() + part[1:] for part in parts[1:]])
    return f"- `{result.model}`: {body}."


def as_markdown(
    results: Sequence[ModelResult],
    *,
    chunks_indexed: int,
    speed_reference: Mapping[str, float] | None = None,
    speed_source: str = "",
) -> str:
    chosen = choose_default(results, speed_reference=speed_reference)
    reference = speed_reference or {}

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
        _comparability_line(results),
        "",
        "Key facts recalled is scored on the variant of a field each question asks about,",
        "so a question asking a resident's fee is graded on the resident figure. The next",
        "column is the same answers graded against every figure the field states, which is",
        "what earlier runs reported. Both are shown so the change is visible. The composite",
        "score uses the scoped figure.",
        "",
        "| Model | Questions | Key facts recalled | Recalled, all figures | All facts present "
        "| Citations valid | Citations present | Answer language | Behaviour |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        if result.accuracy_reported:
            scores = [
                result.fact_recall,
                result.fact_recall_all_figures,
                result.fact_complete,
                result.citation_validity,
                result.citation_presence,
                result.language_match,
                result.behaviour_accuracy,
            ]
            cells = " ".join(f"| {score:.3f}" for score in scores)
        else:
            cells = "| — " * 7
        lines.append(f"| `{result.model}` | {_questions_cell(result)} {cells.strip()} |")

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

    footnotes = [_model_note(result) for result in results]
    written = [note for note in footnotes if note]
    if written:
        lines += [""]
        lines += written

    # The record of the latency work is written by hand and inlined here, so
    # this file stays the one place the speed figures live even though it is
    # regenerated on every run. The same goes for the causes of the wrong
    # answers, which are read off the answers and the passages, neither of
    # which is committed. Both are read without an existence check: they ship
    # beside this module, so a missing one is a packaging fault and should say
    # so rather than quietly drop a section of the report.
    lines += ["", LATENCY_NOTE.read_text(encoding="utf-8").strip()]
    lines += ["", FAILURE_NOTE.read_text(encoding="utf-8").strip()]

    lines += [
        "",
        "## Composite score",
        "",
        "One number per model so the choice is reproducible. Weights: "
        + ", ".join(
            f"{WEIGHT_LABELS[name]} {weight:.0%}"
            for name, weight in sorted(WEIGHTS.items(), key=lambda pair: -pair[1])
        )
        + ".",
        "",
        "Speed is a quarter of the score rather than a tie-breaker: a model that is",
        "accurate and takes half a minute to begin answering is not the better choice",
        "for something a reader is waiting on. It is a linear scale on the median time",
        "to first token.",
        "",
        "```",
        "speed = clamp((SLOW - first_token_p50) / (SLOW - FAST), 0, 1)",
        f"  FAST = {FAST_FIRST_TOKEN_MS / 1000:.0f} s or less -> 1.0",
        f"  SLOW = {SLOW_FIRST_TOKEN_MS / 1000:.0f} s or more -> 0.0",
        "```",
        "",
        "The bounds describe what processor-only inference can do, not what any candidate",
        "scores, and were fixed before the models were compared: five seconds is about the",
        "floor for a four-billion-parameter model that must read several hundred tokens of",
        "prompt before it can begin, and a minute is long past the point a reader gives up.",
        "An earlier version zeroed at twenty seconds on the 95th percentile, which put",
        "every realistic candidate at exactly 0.000 and so removed a quarter of the score.",
        "The median rather than the tail, because run-to-run variance is large and the tail",
        "measures the longest prompt rather than how the model feels to use.",
        "",
        "| Model | Composite | Speed component | First token p50 used |",
        "|---|---:|---:|---:|",
    ]
    ordered = sorted(
        results, key=lambda result: composite(result, speed_reference=reference), reverse=True
    )
    for result in ordered:
        suffix = "" if result.comparable else " *(subset, not comparable)*"
        used = reference.get(result.model, result.first_token_p50)
        marker = " *(clean)*" if result.model in reference else ""
        # A composite over accuracy columns that are not reported would be a
        # number standing for a measurement that was not made.
        total = (
            f"{composite(result, speed_reference=reference):.3f}"
            if result.accuracy_reported
            else "—"
        )
        lines.append(
            f"| `{result.model}`{suffix} | {total} "
            f"| {speed_score(result, first_token_p50_ms=reference.get(result.model)):.3f} "
            f"| {used / 1000:.1f} s{marker} |"
        )
    if speed_source:
        lines += ["", speed_source]

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


def write_reports(
    results: Sequence[ModelResult],
    *,
    chunks_indexed: int,
    questions: int,
    speed_reference: Mapping[str, float] | None = None,
    speed_source: str = "",
) -> None:
    """Write the published report and aggregate scores from what has been run."""
    chosen = choose_default(results, speed_reference=speed_reference)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "measured_at": datetime.now(UTC).isoformat(),
                "chunks_indexed": chunks_indexed,
                "questions": questions,
                "weights": WEIGHTS,
                "speed_first_token_ms_p50": dict(speed_reference or {}),
                "models": [result.to_dict() for result in results],
                "chosen": chosen.model if chosen else None,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    RESULTS_MARKDOWN.write_text(
        as_markdown(
            results,
            chunks_indexed=chunks_indexed,
            speed_reference=speed_reference,
            speed_source=speed_source,
        ),
        encoding="utf-8",
    )


def _rescore_only(
    candidates: Sequence[str],
    selection: str,
    key_facts: Mapping[str, KeyFacts],
    questions: int,
    chunks_indexed: int,
    *,
    settings: Settings,
    notes: Mapping[str, str],
    speed_reference: Mapping[str, float] | None = None,
    speed_source: str = "",
) -> int:
    """Re-grade stored answers and rewrite the reports, calling no model."""
    collected: list[ModelResult] = []
    for model in candidates:
        existing = reuse_scorecard(model, selection)
        if existing is None:
            print(f"No scorecard for {model}; nothing to re-score.")
            return 1
        have = rescorable(existing)
        if have < len(existing.items):
            print(
                f"{model}: only {have} of {len(existing.items)} items carry the answer "
                "text, so the rest cannot be re-graded without generating again."
            )
        regraded = rescore(existing, key_facts)
        regraded.note = notes.get(model, existing.note)
        print(
            f"Re-scored {model} ({regraded.selection}) from {have} stored answers: "
            f"key facts {existing.fact_recall:.4f} -> {regraded.fact_recall:.4f}, "
            f"citations present {existing.citation_presence:.4f} -> "
            f"{regraded.citation_presence:.4f}"
        )
        write_scorecard(regraded, settings=settings)
        collected.append(regraded)
    write_reports(
        collected,
        chunks_indexed=chunks_indexed,
        questions=questions,
        speed_reference=speed_reference,
        speed_source=speed_source,
    )
    chosen = choose_default(collected, speed_reference=speed_reference)
    if chosen is not None:
        print(f"Best on the numbers: {chosen.model}")
    print(f"Written to {RESULTS_MARKDOWN}")
    return 0


def main(
    settings: Settings,
    models: list[str] | None = None,
    *,
    force: bool = False,
    limit: int | None = None,
    subset: str | None = None,
    notes: dict[str, str] | None = None,
    rescore_only: bool = False,
    speed_from_subset: str | None = None,
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

    speed_reference: dict[str, float] = {}
    speed_source = ""
    if speed_from_subset:
        speed_reference = speed_reference_from_subset(candidates, speed_from_subset)
        if speed_reference:
            speed_source = (
                "Each model marked *(clean)* above takes its speed component from the "
                f"`{speed_from_subset}` subset, run with nothing else on the machine, "
                "because latency is the only score another process can move, while its "
                "accuracy comes from the whole question set."
            )
        else:
            print(f"No {speed_from_subset} subset scorecard found; using each run's own median.")

    if rescore_only:
        return _rescore_only(
            candidates,
            selection,
            key_facts,
            len(questions),
            index.metadata.chunk_count,
            settings=settings,
            notes=labels,
            speed_reference=speed_reference,
            speed_source=speed_source,
        )

    async def run() -> list[ModelResult]:
        client = OllamaClient(
            settings.ollama_base_url,
            generation_model=settings.generation_model,
            embedding_model=settings.embedding_model,
            embedding_timeout=settings.embedding_timeout,
            generation_timeout=settings.generation_timeout,
            keep_alive=settings.keep_alive,
        )

        def save(partial: ModelResult) -> None:
            write_scorecard(partial, settings=settings)

        try:
            collected: list[ModelResult] = []
            for model in candidates:
                existing = None if force else reuse_scorecard(model, selection)
                wanted = {question.id for question in questions}
                scored: dict[str, ItemResult] = {}
                if existing is not None:
                    if existing.selection != selection or wanted <= {
                        item.id for item in existing.items
                    }:
                        print(
                            f"Reusing the scorecard for {model} ({existing.selection}); "
                            "--force re-runs it."
                        )
                        existing.note = labels.get(model, existing.note)
                        collected.append(existing)
                        continue
                    # A partial scorecard: carry over what is scored already.
                    scored = {item.id: item for item in existing.items if item.id in wanted}
                    print(f"Resuming {model}: {len(scored)} of {len(questions)} already scored.")

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
                    scored=scored,
                    # Written after every question, so an interrupted run loses
                    # one question rather than the whole model.
                    after_each=save,
                )
                collected.append(result)
                print(f"  wrote {write_scorecard(result, settings=settings)}")
                write_reports(
                    collected,
                    chunks_indexed=index.metadata.chunk_count,
                    questions=len(questions),
                    speed_reference=speed_reference,
                    speed_source=speed_source,
                )
            return collected
        finally:
            await client.close()

    results = asyncio.run(run())
    write_reports(
        results,
        chunks_indexed=index.metadata.chunk_count,
        questions=len(questions),
        speed_reference=speed_reference,
        speed_source=speed_source,
    )

    print()
    for result in results:
        print(
            f"  {result.model:28} facts {result.fact_recall:.3f}  "
            f"citations {result.citation_validity:.3f}  "
            f"language {result.language_match:.3f}  "
            f"behaviour {result.behaviour_accuracy:.3f}  "
            f"prompt {result.prompt_tokens_p50:.0f} tok / {result.prompt_ms_p50 / 1000:.1f}s  "
            f"first token p50 {result.first_token_p50 / 1000:.1f}s  "
            f"composite {composite(result, speed_reference=speed_reference):.3f}"
        )
    chosen = choose_default(results, speed_reference=speed_reference)
    if chosen is not None:
        print(f"\nBest on the numbers: {chosen.model}")
    print(f"Written to {RESULTS_MARKDOWN}")
    return 0
