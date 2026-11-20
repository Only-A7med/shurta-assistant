"""Tests for the checks that run before any model call.

A missed emergency gives somebody service fees when they needed a telephone
number; an answered personal-case question invites a model to invent a fine.
"""

from __future__ import annotations

import pytest

from shurta.guardrails import (
    AMBULANCE_NUMBER,
    FIRE_NUMBER,
    NON_EMERGENCY_NUMBER,
    POLICE_NUMBER,
    Intent,
    classify,
    emergency_response,
    is_emergency,
    is_personal_case,
)


class TestEmergencyDetection:
    @pytest.mark.parametrize(
        "question",
        [
            "Someone is breaking into my house right now",
            "There has been a serious accident and someone is injured",
            "Help, there is a fire in my building",
            "I need an ambulance immediately",
            "My neighbour is being attacked",
            "He is not breathing",
            "I am being robbed",
        ],
    )
    def test_english_emergencies_are_caught(self, question: str) -> None:
        assert is_emergency(question)

    @pytest.mark.parametrize(
        "question",
        [
            "النجدة، هناك شخص يحاول دخول منزلي الآن",
            "وقع حادث خطير وهناك مصاب، ماذا أفعل؟",
            "حريق في المبنى، أرجو المساعدة فوراً",
            "أحتاج إسعاف بسرعة",
            "هناك لص في البيت",
            "شخص فاقد الوعي",
        ],
    )
    def test_arabic_emergencies_are_caught(self, question: str) -> None:
        assert is_emergency(question)

    @pytest.mark.parametrize(
        "question",
        [
            "How do I report a robbery?",
            "What documents do I need to report a theft?",
            "How much does it cost to report a lost item?",
            "How long does a cybercrime report take?",
            "كيف أبلغ عن سرقة؟",
            "ما المستندات المطلوبة للإبلاغ عن حريق؟",
            "كم تستغرق معالجة بلاغ الحادث؟",
        ],
    )
    def test_asking_about_a_service_is_not_an_emergency(self, question: str) -> None:
        """Otherwise every reporting service would answer with telephone numbers."""
        assert not is_emergency(question)

    def test_an_arabic_word_inside_a_longer_word_does_not_fire(self) -> None:
        """``لص`` sits inside ``الصلاحية``; matching substrings caught that."""
        assert not is_emergency(
            "هل يمكنني طلب شهادة بحث الحالة الجنائية إذا كانت هويتي الإماراتية منتهية الصلاحية؟"
        )

    @pytest.mark.parametrize(
        "question",
        ["What is the fee for a police clearance certificate?", "كم رسوم الشهادة؟"],
    )
    def test_ordinary_questions_are_not_emergencies(self, question: str) -> None:
        assert not is_emergency(question)


class TestEmergencyResponse:
    def test_the_english_response_names_all_three_numbers(self) -> None:
        text = emergency_response("en").text
        assert POLICE_NUMBER in text
        assert AMBULANCE_NUMBER in text
        assert FIRE_NUMBER in text

    def test_the_arabic_response_names_all_three_numbers(self) -> None:
        text = emergency_response("ar").text
        assert POLICE_NUMBER in text
        assert AMBULANCE_NUMBER in text
        assert FIRE_NUMBER in text

    def test_the_arabic_response_is_in_arabic(self) -> None:
        assert "الشرطة" in emergency_response("ar").text

    def test_it_also_gives_the_non_emergency_number(self) -> None:
        assert NON_EMERGENCY_NUMBER in emergency_response("en").text

    def test_it_says_it_cannot_contact_anybody(self) -> None:
        """A reader must not wait for the page to call somebody for them."""
        assert "cannot contact" in emergency_response("en").text

    def test_the_numbers_are_returned_as_data_too(self) -> None:
        numbers = emergency_response("en").numbers
        assert numbers["police"] == POLICE_NUMBER
        assert numbers["ambulance"] == AMBULANCE_NUMBER
        assert numbers["fire"] == FIRE_NUMBER


class TestPersonalCaseDetection:
    @pytest.mark.parametrize(
        "question",
        [
            "What is the status of my traffic fine?",
            "How much do I personally owe in fines right now?",
            "Has my police clearance certificate been issued yet?",
            "Can you tell me if there is a travel ban on my name?",
            "ما هي حالة مخالفتي المرورية؟",
            "كم المبلغ المستحق علي من المخالفات؟",
            "هل صدرت شهادتي الجنائية؟",
        ],
    )
    def test_record_lookups_are_caught(self, question: str) -> None:
        assert is_personal_case(question)

    @pytest.mark.parametrize(
        "question",
        [
            "Can I pay my traffic fines in instalments?",
            "How do I check the status of an application I submitted?",
            "How do I verify that a document issued to me is genuine?",
            "What documents do I need for my application?",
            "هل يمكنني تقسيط المخالفات المرورية؟",
            "كيف أستعلم عن حالة طلب قدّمته؟",
        ],
    )
    def test_asking_how_a_service_works_is_not_a_lookup(self, question: str) -> None:
        """These name the reader's own records but ask about the procedure."""
        assert not is_personal_case(question)


class TestClassification:
    def test_an_emergency_outranks_a_personal_case(self) -> None:
        """Somebody in trouble needs the number, not an explanation of channels."""
        assert classify("My house is on fire and my case is still open") is Intent.EMERGENCY

    def test_a_personal_case_is_recognised(self) -> None:
        assert classify("What is the status of my fine?") is Intent.PERSONAL_CASE

    def test_an_ordinary_question_is_normal(self) -> None:
        assert classify("How much is a police clearance certificate?") is Intent.NORMAL

    def test_empty_input_is_normal(self) -> None:
        assert classify("") is Intent.NORMAL
