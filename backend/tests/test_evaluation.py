"""Tests for the evaluation harness, and an end-to-end run over the fixtures.

The scoring is tested because the default model was chosen on it: a citation
check that accepts an invented reference, or a fact comparison that misses an
Arabic-Indic figure, would quietly change that choice.

The end-to-end run uses the invented fixture catalogue and a stand-in model, so
continuous integration exercises the whole path — chunk, index, retrieve,
score — with no provider running.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import FakeModelClient

from shurta.answering import AnswerPipeline, ResponseKind
from shurta.chunking import chunk_corpus
from shurta.config import REPOSITORY_ROOT, Settings
from shurta.evaluation.answers import (
    FULL_SELECTION,
    ItemResult,
    ModelResult,
    check_citations,
    check_declines,
    check_language,
    choose_default,
    composite,
    count_facts,
    first_per_group,
    read_scorecard,
    score_question,
    select,
    speed_score,
    subset_file,
    write_scorecard,
)
from shurta.evaluation.dataset import Behaviour, KeyFacts, Question, load_questions, summarise
from shurta.evaluation.facts import (
    amounts_in,
    contains_amount,
    contains_document,
    contains_duration,
    durations_in,
    extract,
)
from shurta.evaluation.prepare import build_key_facts
from shurta.evaluation.retrieval import evaluate_retrieval
from shurta.models import Corpus, Language
from shurta.retrieval.hybrid import HybridRetriever
from shurta.retrieval.store import SearchIndex

FIXTURE_QUESTIONS = REPOSITORY_ROOT / "eval" / "fixtures" / "questions.jsonl"


# --------------------------------------------------------------------------
# Fact extraction and comparison
# --------------------------------------------------------------------------
class TestAmounts:
    def test_a_figure_before_the_currency_is_found(self) -> None:
        assert amounts_in("The fee is 200 AED") == ["200"]

    def test_a_figure_after_the_currency_is_found(self) -> None:
        assert amounts_in("The fee is AED 200") == ["200"]

    def test_arabic_figures_are_found(self) -> None:
        assert amounts_in("الرسوم 200 درهم") == ["200"]

    def test_thousands_separators_are_removed_so_forms_compare(self) -> None:
        assert amounts_in("AED 2,000") == ["2000"]

    def test_an_arabic_indic_figure_matches_a_western_one(self) -> None:
        """A model may answer with either numeral form."""
        assert contains_amount("الرسوم ٢٠٠ درهم", "200")

    def test_a_number_without_a_currency_is_not_an_amount(self) -> None:
        assert amounts_in("within 5 working days") == []

    def test_a_wrong_figure_does_not_match(self) -> None:
        assert not contains_amount("The fee is AED 300", "200")


class TestDurations:
    def test_working_days_are_found(self) -> None:
        assert durations_in("Processing time: 5 working days") == ["5 working_day"]

    def test_arabic_working_days_are_found(self) -> None:
        assert durations_in("مدة الإنجاز: 5 أيام عمل") == ["5 working_day"]

    def test_an_immediate_service_is_recognised_in_both_languages(self) -> None:
        assert "instant" in durations_in("Processing time: Instant")
        assert "instant" in durations_in("مدة الإنجاز: فوري")

    def test_an_english_answer_satisfies_an_arabic_duration(self) -> None:
        """Both languages reduce to the same token so either can be checked."""
        assert contains_duration("It takes 5 working days", "5 working_day")
        assert contains_duration("تستغرق 5 أيام عمل", "5 working_day")

    def test_plain_days_satisfy_working_days(self) -> None:
        assert contains_duration("It takes about 4 days", "4 working_day")

    def test_the_wrong_number_does_not_match(self) -> None:
        assert not contains_duration("It takes 9 working days", "5 working_day")


class TestDocuments:
    def test_a_named_document_matches_a_paraphrased_condition(self) -> None:
        assert contains_document(
            "You need a copy of your passport if you apply from outside the country",
            "Applying from abroad — copy of the passport",
        )

    def test_an_unrelated_answer_does_not_match(self) -> None:
        assert not contains_document("Pay the fee online", "Noor identity card, valid or expired")

    def test_extraction_skips_phrases_too_short_to_be_evidence(self, corpus: Corpus) -> None:
        """A one-word requirement would match almost any answer."""
        service = corpus.by_id()["home-security-while-travelling"]
        _amounts, _durations, documents = extract(service, ["required_documents"], Language.EN)
        assert all(len(phrase.split()) >= 2 for phrase in documents)


class TestExtraction:
    def test_fees_yield_amounts_and_processing_time_yields_durations(self, corpus: Corpus) -> None:
        service = corpus.by_id()["good-conduct-certificate"]
        amounts, _durations, _documents = extract(service, ["fees"], Language.EN)
        _amounts, durations, _documents2 = extract(service, ["processing_time"], Language.EN)
        assert "75" in amounts
        assert "4 working_day" in durations

    def test_the_arabic_side_yields_the_same_figures(self, corpus: Corpus) -> None:
        service = corpus.by_id()["good-conduct-certificate"]
        english, _d1, _x1 = extract(service, ["fees"], Language.EN)
        arabic, _d2, _x2 = extract(service, ["fees"], Language.AR)
        assert set(english) == set(arabic)

    def test_a_field_the_source_left_empty_yields_nothing(self, corpus: Corpus) -> None:
        service = corpus.by_id()["lost-item-report"]
        amounts, _durations, documents = extract(
            service, ["fees", "required_documents"], Language.EN
        )
        assert amounts == []
        assert documents == []


# --------------------------------------------------------------------------
# Answer scoring
# --------------------------------------------------------------------------
class TestCitationChecking:
    def test_a_citation_within_range_is_valid(self) -> None:
        assert check_citations("The fee is AED 200 [1].", 4) == (True, 1)

    def test_several_citations_are_counted(self) -> None:
        valid, count = check_citations("Fee [1] and time [3].", 4)
        assert valid is True
        assert count == 2

    def test_a_citation_beyond_the_supplied_passages_is_invented(self) -> None:
        """This is the failure a reader is least able to detect."""
        assert check_citations("The fee is AED 200 [7].", 4) == (False, 1)

    def test_a_zero_citation_is_invalid(self) -> None:
        assert check_citations("See [0].", 4) == (False, 1)

    def test_no_citation_is_not_an_invalid_citation(self) -> None:
        assert check_citations("The fee is AED 200.", 4) == (True, 0)


class TestLanguageChecking:
    def test_an_arabic_answer_passes_for_an_arabic_question(self) -> None:
        assert check_language("الرسوم 200 درهم", Language.AR)

    def test_an_english_answer_fails_for_an_arabic_question(self) -> None:
        assert not check_language("The fee is AED 200", Language.AR)

    def test_an_english_answer_passes_for_an_english_question(self) -> None:
        assert check_language("The fee is AED 200", Language.EN)

    def test_an_english_answer_may_quote_an_arabic_service_name(self) -> None:
        assert check_language(
            "The Good Conduct Certificate (شهادة حسن السيرة) costs AED 200 for residents [1].",
            Language.EN,
        )

    def test_an_empty_answer_fails(self) -> None:
        assert not check_language("   ", Language.EN)


class TestDeclineChecking:
    def test_an_english_admission_is_recognised(self) -> None:
        assert check_declines("I could not find this in the passages provided.", Language.EN)

    def test_an_arabic_admission_is_recognised(self) -> None:
        assert check_declines("لم أجد هذا في المقاطع المتوفرة.", Language.AR)

    def test_an_admission_in_the_other_language_still_counts(self) -> None:
        assert check_declines("I could not find this.", Language.AR)

    def test_a_confident_wrong_answer_is_not_an_admission(self) -> None:
        assert not check_declines("The capital is Ulaanbaatar.", Language.EN)


class TestFactCounting:
    def test_every_fact_present_counts_fully(self) -> None:
        facts = KeyFacts(id="x", amounts=["75", "150"], durations=["4 working_day"])
        found, expected = count_facts(
            "Citizens pay AED 75 and residents AED 150, within 4 working days [1].", facts
        )
        assert (found, expected) == (3, 3)

    def test_a_missing_fact_is_counted_as_missing(self) -> None:
        facts = KeyFacts(id="x", amounts=["75", "150"])
        found, expected = count_facts("Citizens pay AED 75 [1].", facts)
        assert (found, expected) == (1, 2)

    def test_no_expected_facts_gives_nothing_to_find(self) -> None:
        assert count_facts("anything at all", KeyFacts(id="x")) == (0, 0)


def item(**overrides: object) -> ItemResult:
    """One scored item, with everything not under test held fixed."""
    defaults: dict[str, object] = {
        "id": "x",
        "language": "en",
        "group": "en",
        "behaviour": "answer",
        "kind": "answer",
        "behaviour_correct": True,
        "language_correct": True,
        "citations_valid": True,
        "cited_count": 1,
        "facts_expected": 2,
        "facts_found": 2,
        "first_token_ms": 5000.0,
        "total_ms": 8000.0,
        "tokens_per_second": 9.0,
        "used_model": True,
        "prompt_tokens": 900,
        "prompt_ms": 4200.0,
    }
    defaults.update(overrides)
    return ItemResult(**defaults)  # type: ignore[arg-type]


class TestComposite:
    def _result(self, **overrides: object) -> ModelResult:
        defaults: dict[str, object] = {
            "id": "x",
            "language": "en",
            "group": "en",
            "behaviour": "answer",
            "kind": "answer",
            "behaviour_correct": True,
            "language_correct": True,
            "citations_valid": True,
            "cited_count": 1,
            "facts_expected": 2,
            "facts_found": 2,
            "first_token_ms": 5000.0,
            "total_ms": 8000.0,
            "tokens_per_second": 9.0,
            "used_model": True,
        }
        defaults.update(overrides)
        return ModelResult(model="test", items=[ItemResult(**defaults)])  # type: ignore[arg-type]

    def test_a_perfect_fast_model_scores_near_one(self) -> None:
        assert composite(self._result()) > 0.9

    def test_a_slower_model_scores_lower_on_otherwise_equal_accuracy(self) -> None:
        fast = self._result(first_token_ms=5000.0)
        slow = self._result(first_token_ms=18000.0)
        assert composite(fast) > composite(slow)

    def test_wrong_facts_cost_more_than_being_slow(self) -> None:
        """Accuracy has to dominate, or the comparison picks the wrong model."""
        slow_but_right = self._result(first_token_ms=15000.0)
        fast_but_wrong = self._result(first_token_ms=1000.0, facts_found=0)
        assert composite(slow_but_right) > composite(fast_but_wrong)

    def test_speed_reaches_zero_for_an_unusably_slow_model(self) -> None:
        assert speed_score(self._result(first_token_ms=60000.0)) == 0.0

    def test_metrics_are_bounded(self) -> None:
        result = self._result()
        assert 0.0 <= result.fact_recall <= 1.0
        assert 0.0 <= result.citation_validity <= 1.0
        assert 0.0 <= composite(result) <= 1.0


class TestPromptStatistics:
    def test_the_median_prompt_length_is_reported(self) -> None:
        """The latency work is aimed at prompt length, so it has to be measured."""
        result = ModelResult(
            model="test",
            items=[
                item(prompt_tokens=600, prompt_ms=3000.0),
                item(prompt_tokens=900, prompt_ms=4000.0),
                item(prompt_tokens=1200, prompt_ms=9000.0),
            ],
        )
        assert result.prompt_tokens_p50 == 900
        assert result.prompt_ms_p50 == 4000.0

    def test_items_that_never_reached_the_model_are_left_out(self) -> None:
        result = ModelResult(
            model="test",
            items=[item(prompt_tokens=900), item(used_model=False, prompt_tokens=0)],
        )
        assert result.prompt_tokens_p50 == 900


# --------------------------------------------------------------------------
# Choosing which questions to run, and resuming a long run
# --------------------------------------------------------------------------
class TestSelection:
    @pytest.fixture
    def questions(self) -> list[Question]:
        return load_questions()

    def test_everything_is_selected_by_default(self, questions: list[Question]) -> None:
        selected, name = select(questions)
        assert len(selected) == len(questions)
        assert name == FULL_SELECTION

    def test_a_limit_takes_the_first_items_of_each_group(self, questions: list[Question]) -> None:
        selected, name = select(questions, limit=2)
        assert name == "limit:2"
        groups: dict[str, int] = {}
        for question in selected:
            key = question.dialect or question.language.value
            groups[key] = groups.get(key, 0) + 1
        assert set(groups.values()) == {2}
        assert selected[0].id == questions[0].id

    def test_a_limit_wider_than_a_group_keeps_what_there_is(
        self, questions: list[Question]
    ) -> None:
        assert len(first_per_group(questions, 10_000)) == len(questions)

    def test_the_committed_latency_subset_is_selectable(self, questions: list[Question]) -> None:
        """Every change in the latency work is measured on this same subset."""
        selected, name = select(questions, subset="latency")
        assert name == "subset:latency"
        assert len(selected) == 20
        groups = {question.dialect or question.language.value for question in selected}
        assert groups == {"en", "ar", "gulf", "code_switched"}

    def test_a_subset_name_resolves_under_the_evaluation_directory(self) -> None:
        assert subset_file("latency").name == "latency.txt"
        assert subset_file("eval/subsets/latency.txt").name == "latency.txt"

    def test_an_unknown_question_in_a_subset_is_refused(
        self, questions: list[Question], tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.txt"
        bad.write_text("en-001\nno-such-question\n", encoding="utf-8")
        with pytest.raises(KeyError, match="no-such-question"):
            select(questions, subset=str(bad))


class TestScorecards:
    """A long run must not be all-or-nothing: the first attempt was lost that way."""

    @pytest.fixture
    def scorecards(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        import shurta.evaluation.answers as answers

        monkeypatch.setattr(answers, "SCORECARD_DIR", tmp_path / "scorecards")
        return tmp_path / "scorecards"

    def test_a_scorecard_is_written_and_read_back(
        self, scorecards: Path, settings: Settings
    ) -> None:
        result = ModelResult(model="gemma3:4b", items=[item(), item(facts_found=1)])
        written = write_scorecard(result, settings=settings)
        assert written.parent == scorecards

        again = read_scorecard("gemma3:4b")
        assert again is not None
        assert len(again.items) == 2
        assert again.fact_recall == result.fact_recall
        assert again.prompt_tokens_p50 == result.prompt_tokens_p50

    def test_a_subset_scorecard_does_not_stand_in_for_a_full_run(
        self, scorecards: Path, settings: Settings
    ) -> None:
        write_scorecard(
            ModelResult(model="gemma3:4b", items=[item()], selection="subset:latency"),
            settings=settings,
        )
        assert read_scorecard("gemma3:4b", "subset:latency") is not None
        assert read_scorecard("gemma3:4b", FULL_SELECTION) is None

    def test_nothing_written_reads_back_as_nothing(self, scorecards: Path) -> None:
        assert read_scorecard("never-run") is None

    def test_a_damaged_scorecard_is_ignored_rather_than_raised(
        self, scorecards: Path, settings: Settings
    ) -> None:
        """A half-written file should cost a re-run, not the whole command."""
        write_scorecard(ModelResult(model="gemma3:4b", items=[item()]), settings=settings)
        from shurta.evaluation.answers import scorecard_path

        scorecard_path("gemma3:4b").write_text("{ not json", encoding="utf-8")
        assert read_scorecard("gemma3:4b") is None

    def test_a_model_name_with_a_colon_becomes_a_usable_file_name(
        self, scorecards: Path, settings: Settings
    ) -> None:
        written = write_scorecard(
            ModelResult(model="qwen3:4b-instruct-2507-q4_K_M", items=[item()]),
            settings=settings,
        )
        assert ":" not in written.name


class TestChoosingTheDefault:
    def test_a_subset_run_cannot_win_the_comparison(self) -> None:
        """A model measured on 20 questions is not comparable with one on 120."""
        full = ModelResult(model="full", items=[item(first_token_ms=9000.0)])
        subset = ModelResult(
            model="subset", items=[item(first_token_ms=1000.0)], selection="subset:latency"
        )
        assert composite(subset) > composite(full)
        chosen = choose_default([full, subset])
        assert chosen is not None
        assert chosen.model == "full"

    def test_nothing_comparable_chooses_nothing(self) -> None:
        subset = ModelResult(model="subset", items=[item()], selection="subset:latency")
        assert choose_default([subset]) is None


# --------------------------------------------------------------------------
# The committed question set
# --------------------------------------------------------------------------
class TestQuestionSet:
    @pytest.fixture
    def questions(self) -> list[Question]:
        return load_questions()

    def test_it_is_the_intended_size_and_composition(self, questions: list[Question]) -> None:
        assert len(questions) >= 120
        english = [q for q in questions if q.language is Language.EN and not q.dialect]
        arabic = [q for q in questions if q.language is Language.AR and not q.dialect]
        dialect = [q for q in questions if q.dialect]
        assert len(english) >= 45
        assert len(arabic) >= 45
        assert len(dialect) >= 10

    def test_every_expected_behaviour_is_represented(self, questions: list[Question]) -> None:
        present = {question.expected_behaviour for question in questions}
        assert present == set(Behaviour)

    def test_identifiers_are_unique(self, questions: list[Question]) -> None:
        identifiers = [question.id for question in questions]
        assert len(set(identifiers)) == len(identifiers)

    def test_answerable_questions_name_a_service_and_a_field(
        self, questions: list[Question]
    ) -> None:
        for question in questions:
            if question.expected_behaviour is Behaviour.ANSWER:
                assert question.expected_service_ids, question.id
                assert question.expected_fields, question.id

    def test_questions_that_need_no_retrieval_name_no_service(
        self, questions: list[Question]
    ) -> None:
        for question in questions:
            if question.expected_behaviour is not Behaviour.ANSWER:
                assert question.expected_service_ids == [], question.id

    def test_every_named_field_exists_on_the_schema(self, questions: list[Question]) -> None:
        from shurta.models import Service

        for question in questions:
            for field in question.expected_fields:
                assert field in Service.model_fields, f"{question.id}: {field}"

    def test_the_summary_describes_the_set(self, questions: list[Question]) -> None:
        assert "questions" in summarise(questions)


# --------------------------------------------------------------------------
# End to end over the fixtures, with no provider running
# --------------------------------------------------------------------------
class TestEndToEndOverFixtures:
    """The whole evaluation path, which is what continuous integration runs."""

    @pytest.fixture
    def fixture_questions(self) -> list[Question]:
        return load_questions(FIXTURE_QUESTIONS)

    @pytest.fixture
    def index(self, corpus: Corpus, fake_client: FakeModelClient, tmp_path: Path) -> SearchIndex:
        chunks = chunk_corpus(corpus)
        vectors = np.asarray([fake_client.vector(chunk.text) for chunk in chunks], dtype=np.float32)
        return SearchIndex.create(chunks, vectors, embedding_model="stand-in")

    def test_the_fixture_questions_reference_only_fixture_services(
        self, fixture_questions: list[Question], corpus: Corpus
    ) -> None:
        known = set(corpus.by_id())
        for question in fixture_questions:
            for service_id in question.expected_service_ids:
                assert service_id in known, f"{question.id} names an unknown service"

    def test_key_facts_can_be_derived_for_every_answerable_fixture_question(
        self, corpus: Corpus, fixture_questions: list[Question], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import shurta.evaluation.prepare as prepare

        monkeypatch.setattr(prepare, "load_questions", lambda: fixture_questions)
        facts = build_key_facts(corpus)
        assert facts
        for entry in facts:
            assert entry.amounts or entry.durations or entry.documents

    async def test_retrieval_finds_the_right_service_for_the_fixture_questions(
        self, index: SearchIndex, fake_client: FakeModelClient, fixture_questions: list[Question]
    ) -> None:
        async def embed(query: str) -> list[float]:
            return (await fake_client.embed([query]))[0]

        scores = await evaluate_retrieval(HybridRetriever(index, embed), fixture_questions)
        hybrid = scores["hybrid"]["overall"]
        assert hybrid.total > 0
        # The stand-in embedder is not a language model, so this asserts the
        # harness computes sane metrics rather than a quality threshold.
        assert 0.0 <= hybrid.recall(5) <= 1.0
        assert hybrid.mrr >= hybrid.recall(1) * 0.5

    async def test_each_method_is_scored_separately(
        self, index: SearchIndex, fake_client: FakeModelClient, fixture_questions: list[Question]
    ) -> None:
        async def embed(query: str) -> list[float]:
            return (await fake_client.embed([query]))[0]

        scores = await evaluate_retrieval(HybridRetriever(index, embed), fixture_questions)
        assert set(scores) == {"bm25", "dense", "hybrid"}
        for method in scores:
            assert scores[method]["overall"].total == scores["hybrid"]["overall"].total

    async def test_an_answer_containing_the_facts_scores_full_marks(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        """The scoring end of the harness, with an answer engineered to be right."""
        service = corpus.by_id()["good-conduct-certificate"]
        answering = FakeModelClient("Citizens pay 75 dirhams and residents pay 150 dirhams [1].")

        chunks = chunk_corpus(corpus)
        vectors = np.asarray([answering.vector(chunk.text) for chunk in chunks], dtype=np.float32)
        index = SearchIndex.create(chunks, vectors, embedding_model="stand-in")

        async def embed(query: str) -> list[float]:
            return (await answering.embed([query]))[0]

        pipeline = AnswerPipeline(HybridRetriever(index, embed), answering, top_k=4)
        amounts, _durations, _documents = extract(service, ["fees"], Language.EN)
        question = Question(
            id="fx-score",
            question="How much does a good conduct certificate cost?",
            language=Language.EN,
            expected_service_ids=["good-conduct-certificate"],
            expected_fields=["fees"],
        )

        result = await score_question(
            pipeline, question, KeyFacts(id="fx-score", amounts=["75", "150"])
        )
        assert "75" in amounts
        assert result.facts_found == 2
        assert result.facts_correct
        assert result.citations_valid
        assert result.language_correct
        assert result.behaviour_correct

    async def test_an_answer_citing_a_passage_that_was_not_supplied_is_caught(
        self, corpus: Corpus
    ) -> None:
        inventing = FakeModelClient("The fee is 75 dirhams [9].")
        chunks = chunk_corpus(corpus)
        vectors = np.asarray([inventing.vector(chunk.text) for chunk in chunks], dtype=np.float32)
        index = SearchIndex.create(chunks, vectors, embedding_model="stand-in")

        async def embed(query: str) -> list[float]:
            return (await inventing.embed([query]))[0]

        pipeline = AnswerPipeline(HybridRetriever(index, embed), inventing, top_k=4)
        question = Question(
            id="fx-invented",
            question="How much does a good conduct certificate cost?",
            language=Language.EN,
            expected_service_ids=["good-conduct-certificate"],
            expected_fields=["fees"],
        )
        result = await score_question(pipeline, question, KeyFacts(id="fx-invented"))
        assert result.citations_valid is False

    async def test_emergency_and_personal_fixture_questions_score_as_correct_behaviour(
        self, corpus: Corpus, fake_client: FakeModelClient, fixture_questions: list[Question]
    ) -> None:
        chunks = chunk_corpus(corpus)
        vectors = np.asarray([fake_client.vector(chunk.text) for chunk in chunks], dtype=np.float32)
        index = SearchIndex.create(chunks, vectors, embedding_model="stand-in")

        async def embed(query: str) -> list[float]:
            return (await fake_client.embed([query]))[0]

        pipeline = AnswerPipeline(HybridRetriever(index, embed), fake_client, top_k=4)

        guarded = [
            question
            for question in fixture_questions
            if question.expected_behaviour in {Behaviour.EMERGENCY, Behaviour.REFUSE}
        ]
        assert guarded

        for question in guarded:
            result = await score_question(pipeline, question, None)
            assert result.behaviour_correct, question.id
            assert result.used_model is False, question.id
            assert result.kind in {
                ResponseKind.EMERGENCY.value,
                ResponseKind.PERSONAL_CASE.value,
            }
