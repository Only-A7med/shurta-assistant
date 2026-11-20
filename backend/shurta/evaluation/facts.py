"""Extracting checkable facts from a service field, and finding them in an answer.

Rule-based rather than model-judged: a fee is a number, a processing time a
number and a unit, a document a phrase. Both sides are Arabic-normalised, so an
answer writing ``٢٠٠`` matches a corpus writing ``200``.

Any pattern matched against normalised text must itself be written in normalised
spelling, or it silently never matches.
"""

from __future__ import annotations

import re

from shurta.arabic import normalise, tokenise
from shurta.models import Language, Service

# Every pattern below is matched against normalised text, so every Arabic
# alternative is written in its normalised spelling. An earlier version wrote
# ``أيام`` where the normaliser produces ``ايام``, and Arabic durations silently
# extracted as nothing.

#: A monetary figure: a number next to a currency word, in either order and in
#: either language. Normalisation has already case-folded the text.
_CURRENCY = r"(?:aed|usd|dirhams?|درهم|دراهم|درهما|\$)"
_AMOUNT = re.compile(
    rf"(?:{_CURRENCY}\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*{_CURRENCY})",
)

#: A duration: a number and a unit. ``عمل`` after an Arabic day word marks it as
#: working days, which is the distinction the corpus draws.
_DURATION = re.compile(
    r"(\d+)\s*(working days?|months?|days?|hours?|"
    r"(?:ايام|يوم|يوما|اشهر|شهر|شهرا|ساعات|ساعه)(?:\s*عمل)?)",
)

#: The word for a service completed immediately, in both languages.
_INSTANT = re.compile(r"\b(instant|immediately|فوري|فورا)\b")

#: How much of a document phrase has to appear in the answer for it to count.
#: A document line can be long, and an answer that paraphrases the condition
#: while naming the document correctly should pass.
DOCUMENT_TOKEN_OVERLAP = 0.6

#: Document phrases shorter than this are skipped: a one-word requirement such
#: as "passport" would match almost any answer.
MINIMUM_DOCUMENT_TOKENS = 2


def amounts_in(text: str) -> list[str]:
    """Every monetary figure in the text, as bare numbers without separators."""
    found: list[str] = []
    for match in _AMOUNT.finditer(normalise(text)):
        number = match.group(1) or match.group(2)
        cleaned = number.replace(",", "").rstrip(".")
        if cleaned and cleaned not in found:
            found.append(cleaned)
    return found


def durations_in(text: str) -> list[str]:
    """Every duration in the text, as a comparable ``count unit`` string."""
    normalised = normalise(text)
    found: list[str] = []
    for match in _DURATION.finditer(normalised):
        value = f"{match.group(1)} {_duration_unit(match.group(2))}"
        if value not in found:
            found.append(value)
    if _INSTANT.search(normalised) and "instant" not in found:
        found.append("instant")
    return found


def _duration_unit(unit: str) -> str:
    """Reduce a unit to one of ``working_day``, ``day``, ``month`` or ``hour``.

    Both languages collapse onto the same token so an English answer can satisfy
    a duration taken from the Arabic side of the corpus and the other way round.
    """
    if "عمل" in unit or "working" in unit:
        return "working_day"
    if unit.startswith(("month", "أشهر", "شهر")):
        return "month"
    if unit.startswith(("hour", "ساعات", "ساعة")):
        return "hour"
    return "day"


def documents_in(items: list[str]) -> list[str]:
    """Document phrases worth checking, dropping ones too short to be evidence."""
    phrases: list[str] = []
    for item in items:
        if len(tokenise(item)) >= MINIMUM_DOCUMENT_TOKENS and item not in phrases:
            phrases.append(item)
    return phrases


def extract(
    service: Service, fields: list[str], language: Language
) -> tuple[list[str], list[str], list[str]]:
    """The amounts, durations and documents a given field of a service states."""
    amounts: list[str] = []
    durations: list[str] = []
    documents: list[str] = []

    for field in fields:
        value = getattr(service, field, None)
        if value is None:
            continue
        content = value.get(language)

        if isinstance(content, list):
            if field == "required_documents":
                documents.extend(
                    phrase for phrase in documents_in(content) if phrase not in documents
                )
            continue

        if field == "fees":
            amounts.extend(amount for amount in amounts_in(content) if amount not in amounts)
        elif field == "processing_time":
            durations.extend(
                duration for duration in durations_in(content) if duration not in durations
            )

    return amounts, durations, documents


def contains_amount(answer: str, amount: str) -> bool:
    """Whether the answer states a figure, allowing for digit and comma forms."""
    return amount in amounts_in(answer)


def contains_duration(answer: str, duration: str) -> bool:
    """Whether the answer states a duration.

    A duration stated in plain days satisfies one recorded in working days: an
    answer saying "5 days" has carried the number across, which is the fact
    being checked, and the corpus wording is quoted alongside it.
    """
    found = durations_in(answer)
    if duration in found:
        return True
    count, _, unit = duration.partition(" ")
    if unit == "working_day":
        return f"{count} day" in found
    return False


def contains_document(answer: str, document: str) -> bool:
    """Whether the answer names a document, by token overlap with the phrase."""
    wanted = set(tokenise(document))
    if not wanted:
        return False
    present = set(tokenise(answer))
    return len(wanted & present) / len(wanted) >= DOCUMENT_TOKEN_OVERLAP
