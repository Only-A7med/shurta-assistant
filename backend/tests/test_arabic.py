"""Tests for Arabic normalisation: two spellings of the same word have to end
up as the same token."""

from __future__ import annotations

import pytest

from shurta.arabic import (
    arabic_ratio,
    detect_language,
    normalise,
    strip_diacritics,
    strip_tatweel,
    tokenise,
    unify_letters,
    western_digits,
)


class TestDiacritics:
    def test_vowel_marks_are_removed(self) -> None:
        assert strip_diacritics("شَهَادَة") == "شهادة"

    def test_shadda_and_sukun_are_removed(self) -> None:
        assert strip_diacritics("المُرُورِّ") == "المرور"

    def test_text_without_marks_is_unchanged(self) -> None:
        assert strip_diacritics("شهادة") == "شهادة"


class TestTatweel:
    def test_stretched_word_collapses(self) -> None:
        assert strip_tatweel("شــــهادة") == "شهادة"

    def test_normalisation_matches_the_unstretched_spelling(self) -> None:
        assert normalise("شــــهادة") == normalise("شهادة")


class TestLetterUnification:
    @pytest.mark.parametrize("spelling", ["الإقامة", "الاقامة", "الأقامة", "الآقامة"])
    def test_every_alef_form_unifies(self, spelling: str) -> None:
        assert unify_letters(spelling) == unify_letters("الاقامة")

    def test_ta_marbuta_and_ha_unify(self) -> None:
        assert normalise("شهادة") == normalise("شهاده")

    def test_alef_maqsura_and_ya_unify(self) -> None:
        assert normalise("دعوى") == normalise("دعوي")

    def test_persian_kaf_and_ya_unify_with_arabic(self) -> None:
        assert normalise("شکایی") == normalise("شكايي")


class TestDigits:
    def test_arabic_indic_digits_convert(self) -> None:
        assert western_digits("٥٠٠ درهم") == "500 درهم"

    def test_extended_arabic_indic_digits_convert(self) -> None:
        assert western_digits("۱۲۳") == "123"

    def test_a_fee_matches_across_digit_forms(self) -> None:
        assert normalise("الرسوم ٢٠٠ درهم") == normalise("الرسوم 200 درهم")


class TestNormalise:
    def test_whitespace_is_collapsed(self) -> None:
        assert normalise("  شهادة   حسن    سيرة  ") == "شهاده حسن سيره"

    def test_english_is_lowercased(self) -> None:
        assert normalise("Police  Clearance CERTIFICATE") == "police clearance certificate"

    def test_empty_text_is_handled(self) -> None:
        assert normalise("") == ""

    def test_is_idempotent(self) -> None:
        once = normalise("شَهَادَةُ حُسْنِ السِّيرَةِ ٥٠٠")
        assert normalise(once) == once


class TestTokenise:
    def test_punctuation_separates_tokens(self) -> None:
        assert tokenise("fines, fees; and payment.") == ["fines", "fees", "and", "payment"]

    def test_definite_article_is_stripped(self) -> None:
        assert tokenise("الشهادة") == tokenise("شهادة")

    def test_short_words_keep_their_article(self) -> None:
        # Stripping "ال" from a three-letter word would leave a meaningless stub.
        assert tokenise("الحق") == ["الحق"]

    def test_mixed_script_question_tokenises_both_halves(self) -> None:
        tokens = tokenise("كيف أدفع traffic fine؟")
        assert "traffic" in tokens
        assert "fine" in tokens
        assert "كيف" in tokens

    def test_empty_text_gives_no_tokens(self) -> None:
        assert tokenise("   ") == []


class TestLanguageDetection:
    def test_plain_english(self) -> None:
        assert detect_language("How do I pay a traffic fine?") == "en"

    def test_modern_standard_arabic(self) -> None:
        assert detect_language("كيف أدفع المخالفة المرورية؟") == "ar"

    def test_gulf_dialect(self) -> None:
        assert detect_language("وين أسدد المخالفات؟") == "ar"

    def test_code_switched_question_is_treated_as_arabic(self) -> None:
        assert detect_language("ابغى أسوي police clearance certificate") == "ar"

    def test_digits_alone_do_not_decide(self) -> None:
        assert detect_language("999") == "en"

    def test_ratio_of_pure_scripts(self) -> None:
        assert arabic_ratio("شهادة") == 1.0
        assert arabic_ratio("certificate") == 0.0
