"""Arabic text normalisation, used on both sides of lexical retrieval.

Arabic is written with several interchangeable spellings of the same word. A
question may say ``الاقامة`` where the official page says ``الإقامة``; it may
write ``شهاده`` for ``شهادة``, stretch a word with tatweel as ``شــهادة``, add
vowel marks, or give a fee as ``٥٠٠`` rather than ``500``. None of those
differences carry meaning here, but all of them defeat exact token matching.

Normalising both the indexed text and the query collapses those variants onto
one spelling, which is what makes keyword search usable for Arabic at all. The
transformation is deliberately lossy and is only ever applied to the text used
for *matching* — the text shown to the reader and sent to the model stays
exactly as the source wrote it.
"""

from __future__ import annotations

import re
import unicodedata

# Vowel marks and other combining signs: fatha, damma, kasra, shadda, sukun,
# the tanween forms, and the Quranic superscript alef.
_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

# Tatweel (ـ) stretches a glyph for justification and never changes the word.
_TATWEEL = "ـ"

# Alef with hamza above/below, with madda, and bare alef all collapse to bare alef.
_ALEF_FORMS = str.maketrans(
    {
        "آ": "ا",  # آ  alef with madda
        "أ": "ا",  # أ  alef with hamza above
        "إ": "ا",  # إ  alef with hamza below
        "ٱ": "ا",  # ٱ  alef wasla
    }
)

# Letters that are routinely written as one another in informal text.
_LETTER_VARIANTS = str.maketrans(
    {
        "ة": "ه",  # ة -> ه   ta marbuta written as ha
        "ى": "ي",  # ى -> ي   alef maqsura written as ya
        "ئ": "ي",  # ئ -> ي   ya with hamza
        "ؤ": "و",  # ؤ -> و   waw with hamza
        "ک": "ك",  # ک -> ك   Persian kaf
        "ی": "ي",  # ی -> ي   Persian ya
    }
)

# Arabic-Indic digits (٠-٩) and the Extended Arabic-Indic digits (۰-۹) used in
# Persian and Urdu typography, mapped to the Western forms a fee comparison can
# actually match on.
_DIGITS = str.maketrans(
    {chr(0x0660 + n): str(n) for n in range(10)} | {chr(0x06F0 + n): str(n) for n in range(10)}
)

# Arabic punctuation alongside the Western marks, plus the tatweel-like dashes.
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)

_WHITESPACE = re.compile(r"\s+")

_ARABIC_LETTERS = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")

# The definite article and the single-letter prepositions that attach to a word.
# Stripping the article matters because the same service is written "الشهادة"
# in a page heading and "شهادة" in a question.
_PREFIXES = ("ال",)  # ال


def strip_diacritics(text: str) -> str:
    """Remove vowel marks and other combining signs."""
    return _DIACRITICS.sub("", text)


def strip_tatweel(text: str) -> str:
    """Remove the justification stretch character."""
    return text.replace(_TATWEEL, "")


def unify_letters(text: str) -> str:
    """Collapse the interchangeable alef, ta marbuta and alef maqsura spellings."""
    return text.translate(_ALEF_FORMS).translate(_LETTER_VARIANTS)


def western_digits(text: str) -> str:
    """Convert Arabic-Indic and Extended Arabic-Indic digits to Western digits."""
    return text.translate(_DIGITS)


def normalise(text: str) -> str:
    """Normalise text for matching.

    Applies, in order: Unicode compatibility composition, diacritic removal,
    tatweel removal, letter unification, digit conversion, case folding and
    whitespace collapsing. Safe to call on English text, where it reduces to
    lowercasing and whitespace tidying.
    """
    text = unicodedata.normalize("NFKC", text)
    text = strip_diacritics(text)
    text = strip_tatweel(text)
    text = unify_letters(text)
    text = western_digits(text)
    return _WHITESPACE.sub(" ", text.casefold()).strip()


def tokenise(text: str) -> list[str]:
    """Split normalised text into matching tokens.

    Punctuation becomes a separator, and the Arabic definite article is stripped
    from the front of each token so that ``الشهادة`` and ``شهادة`` match.
    """
    normalised = _PUNCTUATION.sub(" ", normalise(text))
    tokens: list[str] = []
    for token in normalised.split():
        tokens.append(_strip_article(token))
    return [token for token in tokens if token]


def _strip_article(token: str) -> str:
    """Drop a leading Arabic definite article, keeping the token meaningful."""
    for prefix in _PREFIXES:
        # Three characters is the shortest Arabic root worth keeping after the
        # article; below that, stripping would turn a word into noise.
        if token.startswith(prefix) and len(token) > len(prefix) + 2:
            return token[len(prefix) :]
    return token


def arabic_ratio(text: str) -> float:
    """Share of the word characters in ``text`` that are Arabic script."""
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    arabic = sum(1 for character in letters if _ARABIC_LETTERS.match(character))
    return arabic / len(letters)


def detect_language(text: str) -> str:
    """Return ``"ar"`` or ``"en"`` for a piece of user text.

    A mixed question is treated as Arabic once a meaningful share of its letters
    is Arabic script, because a reader who writes any Arabic in their question
    expects to be answered in Arabic. The threshold is low on purpose: a
    code-switched question such as "ابغى أسوي police clearance certificate" is
    mostly Latin letters by count but is an Arabic question.
    """
    return "ar" if arabic_ratio(text) >= 0.2 else "en"
