"""Tests for the evaluation harness, and an end-to-end run over the fixtures.

The scoring is tested because the default model was chosen on it. The
end-to-end run uses the fixture catalogue and a stand-in model, so continuous
integration exercises chunking, indexing, retrieval and scoring with no
provider running.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeModelClient
from pydantic import ValidationError

from shurta.answering import AnswerPipeline, ResponseKind
from shurta.chunking import chunk_corpus
from shurta.config import REPOSITORY_ROOT, Settings
from shurta.evaluation.answers import (
    FAST_FIRST_TOKEN_MS,
    FULL_SELECTION,
    SLOW_FIRST_TOKEN_MS,
    ItemResult,
    ModelResult,
    check_citations,
    check_declines,
    check_language,
    choose_default,
    comparable_on_the_same_questions,
    composite,
    count_facts,
    evaluate_model,
    first_per_group,
    read_scorecard,
    rescorable,
    rescore,
    rescore_item,
    reuse_scorecard,
    score_question,
    select,
    speed_reference_from_subset,
    speed_score,
    subset_file,
    write_scorecard,
)
from shurta.evaluation.dataset import Behaviour, KeyFacts, Question, load_questions, summarise
from shurta.evaluation.facts import (
    FIELD_VARIANTS,
    amounts_in,
    contains_amount,
    contains_document,
    contains_duration,
    document_name,
    durations_in,
    extract,
    narrow,
    narrow_items,
)
from shurta.evaluation.prepare import build_key_facts
from shurta.evaluation.retrieval import evaluate_retrieval
from shurta.models import Corpus, Language, Service
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


class TestQualifiedFees:
    """A fee named after what it funds puts a word between number and currency."""

    def test_a_named_dirham_component_is_found(self) -> None:
        """`en-018`: the answer stated this fee correctly and scored nothing."""
        answer = (
            "The fee for a good conduct certificate is 75 Noor Dirham, 15 Records Dirham, "
            "and an additional 150 dirhams if submitted in person [2]."
        )
        assert contains_amount(answer, "15")
        assert contains_amount(answer, "150")

    def test_the_plain_adjacent_forms_still_work(self) -> None:
        assert amounts_in("Resident: 150 dirhams") == ["150"]
        assert amounts_in("Records fee: 15 dirhams") == ["15"]

    def test_a_number_far_from_a_currency_is_not_a_fee(self) -> None:
        """Only one intervening word, or any sentence would yield amounts."""
        assert amounts_in("It takes 4 working days and costs 150 dirhams") == ["150"]

    def test_a_duration_is_not_read_as_a_fee(self) -> None:
        assert amounts_in("Processing takes 2 working days") == []


class TestDocuments:
    def test_a_named_document_matches_a_paraphrased_condition(self) -> None:
        assert contains_document(
            "You need a copy of your passport if you apply from outside the country",
            "Applying from abroad — copy of the passport",
        )

    def test_an_unrelated_answer_does_not_match(self) -> None:
        assert not contains_document("Pay the fee online", "Noor identity card, valid or expired")

    def test_a_grouped_document_is_graded_on_the_document_not_the_condition(self) -> None:
        """`en-004`: the same three documents scored 3/3 and 0/3 for two models.

        A flattened entry is "From outside the UAE - Personal photo". One model
        happened to mention the condition in a later sentence and was credited
        for it; the other listed all three documents, said nothing about the
        condition, and scored nothing. The document is the fact being checked.
        """
        grouped = "Applying from abroad — copy of the previous residence permit"
        listed_without_the_condition = (
            "- Noor identity card\n- Copy of the previous residence permit [1]"
        )
        assert contains_document(listed_without_the_condition, grouped)

    def test_naming_only_the_condition_is_not_a_document(self) -> None:
        """The same bug the other way round: this used to count as a hit.

        "From outside the UAE you must apply early" shares four of the six
        tokens of "From outside the UAE - Personal photo" and names no document
        at all. The fix has to make this stricter, not looser.
        """
        assert not contains_document(
            "Applying from abroad, you must apply early.",
            "Applying from abroad — copy of the previous residence permit",
        )

    def test_a_grouped_entry_still_needs_the_right_document(self) -> None:
        assert not contains_document(
            "- Driving licence of each driver\n- Vehicle registration card [1]",
            "Applying from abroad — copy of the previous residence permit",
        )

    def test_an_ungrouped_document_is_still_graded_whole(self) -> None:
        assert contains_document(
            "Bring your Noor identity card, valid or expired [1].",
            "Noor identity card, valid or expired",
        )

    def test_a_one_word_document_half_keeps_the_whole_phrase(self) -> None:
        """Otherwise "X - photo" would match any answer saying "photo"."""
        assert document_name("Applying from abroad — photo") == ("Applying from abroad — photo")

    def test_the_arabic_conjunction_does_not_hide_a_document(self) -> None:
        """`dia-014`: "وجدول الفعاليات" names the same document as "جدول الفعاليات".

        The conjunction attaches to the word it joins, so a document listed
        after "and" became a different token and the answer was scored wrong.
        """
        assert contains_document(
            "بطاقة هوية نور وصورة شخصية بخلفية سادة [2]", "صورة شخصية بخلفية سادة"
        )

    def test_the_conjunction_is_allowed_not_stripped(self) -> None:
        """Stripping it would break words whose first letter is a waw.

        "وثيقة" (document) begins with the same letter, so the conjunction is
        matched on the answer's side rather than removed from every token.
        """
        assert not contains_document("وثيقة تأمين سارية [1]", "صورة شخصية بخلفية سادة")

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
# Scoping a question to one variant of a field
# --------------------------------------------------------------------------
#: A fee field in the shape the source writes one: a labelled figure per line.
FEE_FIELD = "Citizen: 75 dirhams\nResident: 150 dirhams"


class TestFieldVariants:
    """A fee field states a figure per applicant category, one per line.

    A question asking a resident's fee was graded against every figure in the
    field — the citizen's, the overseas one, the surcharge — so an answer that
    gave exactly the figure asked for scored as incomplete. A question now
    records which variant it asks about, and only that variant's facts are
    derived. A question that asks about the whole field records none and keeps
    every figure, which is what makes the fix a question-set change rather than
    a looser grader.
    """

    @pytest.fixture
    def service(self, corpus: Corpus) -> Service:
        return corpus.by_id()["good-conduct-certificate"]

    def scoped_question(self, variant: str) -> Question:
        return Question(
            id="q",
            question="How much does a good conduct certificate cost for a resident?",
            language=Language.EN,
            expected_service_ids=["good-conduct-certificate"],
            expected_fields=["fees"],
            expected_variant=variant,
        )

    def test_a_scoped_question_derives_only_its_variants_facts(
        self, corpus: Corpus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import shurta.evaluation.prepare as prepare

        monkeypatch.setattr(prepare, "load_questions", lambda: [self.scoped_question("resident")])
        entry = build_key_facts(corpus)[0]
        assert entry.amounts == ["150"]
        assert entry.narrowed

    def test_an_unscoped_question_still_derives_every_figure(
        self, corpus: Corpus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import shurta.evaluation.prepare as prepare

        monkeypatch.setattr(prepare, "load_questions", lambda: [self.scoped_question("")])
        entry = build_key_facts(corpus)[0]
        assert entry.amounts == ["75", "150", "240", "15"]
        assert not entry.narrowed

    def test_every_figure_is_kept_beside_the_scoped_ones(
        self, corpus: Corpus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """So the report can show the old recall beside the new one."""
        import shurta.evaluation.prepare as prepare

        monkeypatch.setattr(prepare, "load_questions", lambda: [self.scoped_question("resident")])
        entry = build_key_facts(corpus)[0]
        assert entry.all_amounts == ["75", "150", "240", "15"]
        assert entry.every_figure().amounts == entry.all_amounts

    def test_each_variant_picks_out_its_own_figure(self, service: Service) -> None:
        for variant, expected in (("citizen", "75"), ("resident", "150")):
            amounts, _durations, _documents = extract(service, ["fees"], Language.EN, variant)
            assert amounts == [expected], variant

    def test_the_arabic_side_scopes_to_the_same_figure(self, service: Service) -> None:
        amounts, _durations, _documents = extract(service, ["fees"], Language.AR, "resident")
        assert amounts == ["150"]

    def test_a_scoped_document_list_keeps_only_its_condition(self, service: Service) -> None:
        _amounts, _durations, documents = extract(
            service, ["required_documents"], Language.EN, "outside_country"
        )
        assert documents
        assert all("abroad" in phrase for phrase in documents)

    def test_a_document_is_matched_on_its_condition_not_its_name(self) -> None:
        """Or a document whose name carries a category word joins the wrong group."""
        items = [
            "Applying from abroad — copy of the passport",
            "Individuals — proof of an address outside the emirate",
        ]
        assert narrow_items(items, "outside_country") == [items[0]]

    def test_narrowing_to_nothing_yields_nothing_rather_than_everything(self) -> None:
        """A variant that matches no line must not silently fall back."""
        assert narrow(FEE_FIELD, "in_person") == ""

    def test_an_unscoped_field_is_left_whole(self) -> None:
        assert narrow(FEE_FIELD, "") == FEE_FIELD

    def test_an_unknown_variant_label_is_refused(self) -> None:
        """A typo in the question set would otherwise silently derive no facts."""
        with pytest.raises(ValidationError):
            self.scoped_question("tourists")

    def test_the_committed_questions_use_the_scoping(self) -> None:
        questions = load_questions()
        scoped = [question for question in questions if question.expected_variant]
        assert scoped
        for question in scoped:
            assert question.expected_variant in FIELD_VARIANTS
            assert question.expected_fields, question.id

    def test_both_recall_figures_are_reported(self) -> None:
        """The scoped figure and the all-figures one, so the change is visible."""
        result = ModelResult(
            model="m",
            items=[item(facts_expected=1, facts_found=1, facts_expected_all=5, facts_found_all=1)],
        )
        assert result.fact_recall == 1.0
        assert result.fact_recall_all_figures == 0.2

    def test_rescoring_counts_a_stored_answer_both_ways(self) -> None:
        facts = KeyFacts(
            id="x",
            amounts=["200"],
            all_amounts=["100", "200", "300"],
        )
        regraded = rescore_item(item(text="The fee is AED 200 [1].", available_citations=2), facts)
        assert (regraded.facts_found, regraded.facts_expected) == (1, 1)
        assert (regraded.facts_found_all, regraded.facts_expected_all) == (1, 3)


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


class TestGraderFalseNegatives:
    """Each case here was a correct answer the rules scored as wrong.

    Every phrase was taken from an answer a model actually produced, and each
    one is pinned so the fix cannot regress. These loosen nothing else: a wrong
    answer still fails.
    """

    def test_an_answer_that_says_the_passages_do_not_contain_it_declines(self) -> None:
        """`oos-001`: a correct refusal scored as a wrong answer."""
        assert check_declines(
            "The passages do not contain information on how to renew a passport [2]. "
            "Please refer to the Dubai Police app, or 901.",
            Language.EN,
        )

    def test_the_singular_form_declines_too(self) -> None:
        assert check_declines("This passage does not contain that fee.", Language.EN)

    def test_arabic_declines_are_matched_whichever_spelling_is_used(self) -> None:
        """The phrases are matched on normalised text, so hamzas cannot hide one.

        "لم أجد" and "لم اجد" are the same refusal. Matching raw text meant the
        grading depended on which spelling the model happened to produce.
        """
        for written in ("لم أجد هذه المعلومة في المقاطع.", "لم اجد هذه المعلومه في المقاطع."):
            assert check_declines(written, Language.AR)

    def test_arabic_has_an_equivalent_of_each_english_decline(self) -> None:
        """English and Arabic answers must be held to the same standard."""
        equivalents = {
            "does not contain": "المقاطع لا تحتوي على هذه المعلومة.",
            "does not include": "المقاطع لا تتضمن رسوم هذه الخدمة.",
            "does not cover": "معلوماتي لا تغطي هذا الموضوع.",
            "does not mention": "المقاطع لا تذكر هذه الخدمة.",
            "is not mentioned": "لم ترد هذه الخدمة في المقاطع.",
            "no information": "لا توجد معلومات عن هذه الخدمة.",
            "not available": "هذه المعلومة غير متوفرة في المقاطع.",
            "is not available": "لا تتوفر لدي معلومات عن ذلك.",
            "could not find": "لم أجد ذلك في المقاطع.",
            "i do not have": "لا أملك معلومات عن هذه الخدمة.",
        }
        for english, arabic in equivalents.items():
            assert check_declines(arabic, Language.AR), f"missing the Arabic form of {english!r}"

    def test_an_ordinary_arabic_answer_does_not_count_as_declining(self) -> None:
        """The added phrases must not turn a real answer into a refusal."""
        for answered in (
            "رسوم شهادة حسن السيرة والسلوك 150 درهماً نورياً [1] وتستغرق 4 أيام عمل [1].",
            "المستندات المطلوبة هي بطاقة هوية نور وصورة شخصية بخلفية سادة [2].",
            "تبلغ المدة 4 أيام عمل [1].",
        ):
            assert not check_declines(answered, Language.AR)

    def test_an_answer_that_states_a_fact_still_does_not_count_as_declining(self) -> None:
        """The fix must not turn an ordinary answer into a refusal."""
        assert not check_declines(
            "A good conduct certificate costs 150 dirhams [1] and takes 4 working days [1].",
            Language.EN,
        )

    def test_an_answer_from_an_adjacent_service_does_not_count_as_declining(self) -> None:
        """`oos-003`: answering from a related service is a real failure."""
        assert not check_declines(
            "متاح للأفراد، وأن تكون رخصة القيادة صادرة من إمارة دبي [1].",
            Language.AR,
        )


class TestRescoringStoredAnswers:
    """A grader fix must be re-checkable without an hour of inference."""

    def _item(self, **overrides: object) -> ItemResult:
        return item(**overrides)

    def test_a_stored_answer_is_regraded_without_a_model(self) -> None:
        stored = self._item(
            id="q1",
            text="The fee is 150 dirhams [1].",
            available_citations=2,
            facts_expected=1,
            facts_found=0,
            cited_count=0,
        )
        facts = KeyFacts(id="q1", amounts=["150"])
        regraded = rescore_item(stored, facts)
        assert regraded.facts_found == 1
        assert regraded.facts_expected == 1
        assert regraded.cited_count == 1
        assert regraded.citations_valid

    def test_an_invented_citation_is_still_caught_on_a_stored_answer(self) -> None:
        stored = self._item(id="q1", text="The fee is 150 dirhams [7].", available_citations=2)
        regraded = rescore_item(stored, KeyFacts(id="q1", amounts=["150"]))
        assert not regraded.citations_valid

    def test_an_item_with_no_stored_answer_is_left_alone(self) -> None:
        """Scorecards written before answers were kept must not be corrupted."""
        stored = self._item(id="q1", text="", facts_found=2, facts_expected=2)
        regraded = rescore_item(stored, KeyFacts(id="q1", amounts=["150"]))
        assert regraded == stored

    def test_rescoring_a_whole_scorecard_reports_how_much_it_could_regrade(self) -> None:
        result = ModelResult(
            model="test",
            items=[
                self._item(id="a", text="150 dirhams [1]", available_citations=1),
                self._item(id="b", text=""),
            ],
        )
        assert rescorable(result) == 1
        regraded = rescore(result, {"a": KeyFacts(id="a", amounts=["150"])})
        assert regraded.model == "test"
        assert regraded.selection == result.selection
        assert len(regraded.items) == 2

    def test_a_behaviour_fix_is_regradeable_from_a_stored_answer(self) -> None:
        """The grader fix for `oos-001` had to be checkable without re-running."""
        stored = self._item(
            id="oos-001",
            behaviour="out_of_scope",
            kind="answer",
            behaviour_correct=False,
            text="The passages do not contain information on how to renew a passport [2].",
            available_citations=4,
        )
        assert rescore_item(stored, None).behaviour_correct

    def test_regrading_does_not_excuse_an_answer_from_a_related_service(self) -> None:
        stored = self._item(
            id="oos-003",
            behaviour="out_of_scope",
            kind="answer",
            behaviour_correct=False,
            language="ar",
            text="متاح للأفراد، وأن تكون رخصة القيادة صادرة من إمارة دبي [1].",
            available_citations=4,
        )
        assert not rescore_item(stored, None).behaviour_correct

    def test_the_published_aggregate_never_carries_the_answers(self) -> None:
        """Answers are corpus-derived, so they stay in the local scorecard."""
        result = ModelResult(
            model="test",
            items=[self._item(id="a", text="150 dirhams [1]", available_citations=1)],
        )
        assert "text" not in result.to_dict()
        assert "150 dirhams" not in json.dumps(result.to_dict())


class TestSpeedReference:
    """Latency is the one score another process on the machine can move."""

    def _run(self, model: str, first_token_ms: float) -> ModelResult:
        return ModelResult(
            model=model,
            items=[item(first_token_ms=first_token_ms, facts_expected=2, facts_found=2)],
        )

    def test_a_reference_median_replaces_the_runs_own(self) -> None:
        contaminated = self._run("a", 9_200.0)
        assert speed_score(contaminated) > speed_score(contaminated, first_token_p50_ms=10_800.0)

    def test_the_composite_uses_the_reference_when_one_is_given(self) -> None:
        run = self._run("a", 9_200.0)
        assert composite(run, speed_reference={"a": 10_800.0}) < composite(run)

    def test_a_model_without_a_reference_keeps_its_own_median(self) -> None:
        run = self._run("a", 9_200.0)
        assert composite(run, speed_reference={"other": 30_000.0}) == composite(run)

    def test_the_reference_can_change_which_model_is_chosen(self) -> None:
        """If it could not, taking the clean figures would be pointless."""
        fast_but_contaminated = ModelResult(
            model="a",
            items=[item(id="q1", first_token_ms=6_000.0, facts_expected=10, facts_found=5)],
        )
        slower_looking = ModelResult(
            model="b",
            items=[item(id="q1", first_token_ms=7_000.0, facts_expected=10, facts_found=5)],
        )
        both = [fast_but_contaminated, slower_looking]
        assert choose_default(both) is not None
        assert choose_default(both).model == "a"  # type: ignore[union-attr]
        # Measured cleanly, "a" is in fact the slower of the two.
        clean = {"a": 40_000.0, "b": 7_000.0}
        assert choose_default(both, speed_reference=clean).model == "b"  # type: ignore[union-attr]

    def test_a_reference_is_read_from_the_subset_scorecard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from shurta.evaluation import answers as module

        monkeypatch.setattr(module, "SCORECARD_DIR", tmp_path)
        card = ModelResult(
            model="gemma3:4b",
            items=[item(first_token_ms=10_800.0)],
            selection="subset:latency",
        )
        write_scorecard(card, settings=Settings())
        reference = speed_reference_from_subset(["gemma3:4b", "absent:4b"], "latency")
        assert reference == {"gemma3:4b": 10_800.0}


class TestComparability:
    """Every model-to-model comparison has to be on the same questions."""

    def _run(self, model: str, ids: list[str], found: int) -> ModelResult:
        return ModelResult(
            model=model,
            items=[item(id=identifier, facts_expected=1, facts_found=found) for identifier in ids],
        )

    def test_a_partial_run_is_not_comparable_with_a_full_one(self) -> None:
        """The trap this guards: a model scored on 36 questions read as if it
        had been scored on all 130."""
        full = self._run("a", ["q1", "q2", "q3"], 1)
        partial = ModelResult(
            model="b",
            items=[item(id="q1", facts_expected=1, facts_found=0)],
            selection="all",
        )
        assert not comparable_on_the_same_questions([full, partial])

    def test_runs_over_the_same_questions_are_comparable(self) -> None:
        left = self._run("a", ["q1", "q2"], 1)
        right = self._run("b", ["q1", "q2"], 0)
        assert comparable_on_the_same_questions([left, right])

    def test_the_default_is_only_chosen_from_comparable_runs(self) -> None:
        full = self._run("a", ["q1", "q2", "q3"], 0)
        partial = ModelResult(
            model="b",
            items=[item(id="q1", facts_expected=1, facts_found=1)],
            selection="all",
        )
        # "b" scores better, but on one question out of three.
        assert choose_default([full, partial]) is None


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
        assert speed_score(self._result(first_token_ms=SLOW_FIRST_TOKEN_MS)) == 0.0

    def test_speed_reaches_one_for_a_model_that_answers_immediately(self) -> None:
        assert speed_score(self._result(first_token_ms=FAST_FIRST_TOKEN_MS)) == 1.0

    def test_speed_is_clamped_outside_the_bounds(self) -> None:
        assert speed_score(self._result(first_token_ms=500.0)) == 1.0
        assert speed_score(self._result(first_token_ms=120_000.0)) == 0.0

    def test_speed_is_linear_between_the_bounds(self) -> None:
        """The midpoint has to score a half, or the scale is not the stated one."""
        midpoint = (FAST_FIRST_TOKEN_MS + SLOW_FIRST_TOKEN_MS) / 2
        assert speed_score(self._result(first_token_ms=midpoint)) == pytest.approx(0.5)
        quarter = FAST_FIRST_TOKEN_MS + (SLOW_FIRST_TOKEN_MS - FAST_FIRST_TOKEN_MS) / 4
        assert speed_score(self._result(first_token_ms=quarter)) == pytest.approx(0.75)

    def test_a_realistic_processor_only_model_scores_above_zero(self) -> None:
        """The point of the bounds: a measurable speed must be a usable number.

        The previous scale zeroed at twenty seconds on the 95th percentile, so
        every model that can actually run here scored exactly 0.000 and a
        quarter of the composite carried no weight at all.
        """
        realistic = speed_score(self._result(first_token_ms=13_600.0))
        assert 0.0 < realistic < 1.0

    def test_speed_is_measured_on_the_median_not_the_tail(self) -> None:
        fast_median = ModelResult(
            model="test",
            items=[item(first_token_ms=6_000.0) for _ in range(9)]
            + [item(first_token_ms=55_000.0)],
        )
        assert speed_score(fast_median) > 0.8

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

    def test_a_subset_scorecard_can_still_be_reused_in_a_full_report(
        self, scorecards: Path, settings: Settings
    ) -> None:
        """One report is assembled from runs made separately."""
        write_scorecard(
            ModelResult(model="qwen3:4b", items=[item()], selection="subset:latency"),
            settings=settings,
        )
        reused = reuse_scorecard("qwen3:4b", FULL_SELECTION)
        assert reused is not None
        assert reused.selection == "subset:latency"
        assert reused.comparable is False

    def test_reuse_prefers_the_selection_that_was_asked_for(
        self, scorecards: Path, settings: Settings
    ) -> None:
        write_scorecard(
            ModelResult(model="gemma3:4b", items=[item()], selection="subset:latency"),
            settings=settings,
        )
        write_scorecard(ModelResult(model="gemma3:4b", items=[item(), item()]), settings=settings)
        reused = reuse_scorecard("gemma3:4b", FULL_SELECTION)
        assert reused is not None
        assert reused.selection == FULL_SELECTION
        assert len(reused.items) == 2

    def test_reuse_finds_nothing_for_a_model_never_run(self, scorecards: Path) -> None:
        scorecards.mkdir(parents=True, exist_ok=True)
        assert reuse_scorecard("never-run") is None

    def test_a_model_name_with_a_colon_becomes_a_usable_file_name(
        self, scorecards: Path, settings: Settings
    ) -> None:
        written = write_scorecard(
            ModelResult(model="qwen3:4b-instruct-2507-q4_K_M", items=[item()]),
            settings=settings,
        )
        assert ":" not in written.name


class TestResumingPartway:
    """An interrupted run has to resume where it stopped, not from the start."""

    @pytest.fixture
    def questions(self) -> list[Question]:
        return load_questions()[:6]

    async def _run(
        self,
        corpus: Corpus,
        settings: Settings,
        questions: list[Question],
        scored: dict[str, ItemResult],
        saves: list[int],
    ) -> ModelResult:
        client = FakeModelClient("The fee is 75 dirhams [1].")
        chunks = chunk_corpus(corpus)
        vectors = np.asarray([client.vector(chunk.text) for chunk in chunks], dtype=np.float32)
        index = SearchIndex.create(chunks, vectors, embedding_model="stand-in")
        return await evaluate_model(
            index,
            client,  # type: ignore[arg-type]
            "stand-in",
            questions,
            {},
            settings,
            progress=False,
            scored=scored,
            after_each=lambda partial: saves.append(len(partial.items)),
        )

    async def test_already_scored_questions_are_not_run_again(
        self, corpus: Corpus, settings: Settings, questions: list[Question]
    ) -> None:
        first = questions[0]
        carried = item(id=first.id, facts_found=1, facts_expected=1)
        saves: list[int] = []
        result = await self._run(corpus, settings, questions, {first.id: carried}, saves)

        assert [entry.id for entry in result.items] == [question.id for question in questions]
        assert result.items[0] is carried
        # One save per question actually run, and the carried one is not re-run.
        assert saves == list(range(2, len(questions) + 1))

    async def test_nothing_left_to_run_needs_no_model_at_all(
        self, corpus: Corpus, settings: Settings, questions: list[Question]
    ) -> None:
        scored = {question.id: item(id=question.id) for question in questions}
        saves: list[int] = []
        result = await self._run(corpus, settings, questions, scored, saves)
        assert len(result.items) == len(questions)
        assert saves == []


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
