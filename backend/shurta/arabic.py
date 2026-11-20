"""Arabic text normalisation, used on both sides of lexical retrieval.

The same word is written several ways — ``الاقامة`` for ``الإقامة``, ``شهاده``
for ``شهادة``, ``٥٠٠`` for ``500`` — and none of the differences carry meaning.
Normalising both the indexed text and the query collapses them onto one
spelling.

The transformation is lossy and is applied only to the text used for *matching*.
What the reader sees and what the model is sent is the source's own wording. See
docs/architecture.md for the table of variants.
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

    Compatibility composition, diacritics, tatweel, letter unification, digits,
    case folding, whitespace. On English text this is lowercasing and tidying.
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
        # Three characters is the shortest root worth keeping after the article.
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

    The threshold is low on purpose: "ابغى أسوي police clearance certificate" is
    mostly Latin letters by count but is an Arabic question.
    """
    return "ar" if arabic_ratio(text) >= 0.2 else "en"
