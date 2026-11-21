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

#: A fee component named after what it funds puts a qualifier between the number
#: and the currency word: "10 Knowledge Dirham", "Innovation Dirham: 10". The
#: plain pattern needs them adjacent, so it cannot see either form, and an answer
#: stating such a fee correctly was scored as having missed it. Restricted to the
#: dirham words and to exactly one intervening word, so an ordinary sentence that
#: happens to put a number some distance from a currency is not swept up.
_DIRHAM = r"(?:dirhams?|درهم|دراهم|درهما)"
_QUALIFIED_AMOUNT = re.compile(
    rf"(?:([\d,]+(?:\.\d+)?)\s+[^\W\d_]+\s+{_DIRHAM}"
    rf"|{_DIRHAM}\s+[^\W\d_]+\s*:?\s*([\d,]+(?:\.\d+)?))",
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

#: The separator the collector uses when it flattens a nested document list into
#: "condition - document", because some document lists hang their entries off a
#: condition label rather than listing them flat.
GROUP_SEPARATOR = " — "

#: The variants of a field a question can ask about, and the category words a
#: field's own labels use for each. A fee field states a figure per applicant
#: category, one per line, and a question asking for one of them should be
#: graded on that one: requiring every figure in the field penalises a correctly
#: scoped answer. A question that asks about the whole field names no variant
#: and keeps every figure.
#:
#: Only category words, in normalised spelling, and never a figure. Matched
#: against normalised text, so any Arabic term here must be written the way the
#: normaliser writes it.
FIELD_VARIANTS: dict[str, tuple[str, ...]] = {
    "citizen": ("citizen", "مواطن"),
    "resident": ("resident", "مقيم"),
    "outside_country": ("outside", "abroad", "خارج"),
    "in_person": ("in person", "شخصيا", "حضوريا"),
    "knowledge_fee": ("knowledge", "معرفه"),
}


def amounts_in(text: str) -> list[str]:
    """Every monetary figure in the text, as bare numbers without separators."""
    normalised = normalise(text)
    found: list[str] = []
    for pattern in (_AMOUNT, _QUALIFIED_AMOUNT):
        for match in pattern.finditer(normalised):
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


def mentions_variant(text: str, variant: str) -> bool:
    """Whether a line of a field is the one stating the named variant."""
    normalised = normalise(text)
    return any(term in normalised for term in FIELD_VARIANTS[variant])


def narrow(content: str, variant: str) -> str:
    """The lines of a field that state the named variant, or all of them.

    A fee field is written one labelled figure per line, so the variant a
    question asks about is a line of it.
    """
    if not variant:
        return content
    return "\n".join(line for line in content.splitlines() if mentions_variant(line, variant))


def narrow_items(items: list[str], variant: str) -> list[str]:
    """The entries of a list field that belong to the named variant.

    A flattened entry is matched on the condition it was grouped under, not on
    the document itself, so a document whose name happens to carry a category
    word is not mistaken for that category's requirement.
    """
    if not variant:
        return items
    return [item for item in items if mentions_variant(document_condition(item), variant)]


def extract(
    service: Service, fields: list[str], language: Language, variant: str = ""
) -> tuple[list[str], list[str], list[str]]:
    """The amounts, durations and documents a given field of a service states.

    ``variant`` narrows the field to one of ``FIELD_VARIANTS`` — the applicant
    category a question asks about — and is empty for a question that asks about
    the field as a whole.
    """
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
                phrases = documents_in(narrow_items(content, variant))
                documents.extend(phrase for phrase in phrases if phrase not in documents)
            continue

        wanted = narrow(content, variant)
        if field == "fees":
            amounts.extend(amount for amount in amounts_in(wanted) if amount not in amounts)
        elif field == "processing_time":
            durations.extend(
                duration for duration in durations_in(wanted) if duration not in durations
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


def document_condition(document: str) -> str:
    """The condition a flattened entry was grouped under, or the whole entry.

    The whole entry when there is no grouping, so an ungrouped list is matched
    on its own text rather than on nothing.
    """
    condition, separator, _name = document.rpartition(GROUP_SEPARATOR)
    return condition if separator else document


def document_name(document: str) -> str:
    """The document itself, without the condition it was grouped under.

    A flattened entry reads "From outside the UAE - Passport copy". The fact
    being checked is that the answer names the *document*; the condition is
    context for it. Matching the whole phrase got this wrong in both
    directions: an answer that listed every document without restating the
    condition scored nothing, while an answer naming only the condition and no
    document at all matched four of the phrase's six tokens and counted as a
    hit.

    The whole phrase is kept when the document half is too short to stand as
    evidence on its own.
    """
    _, separator, name = document.rpartition(GROUP_SEPARATOR)
    if not separator or len(tokenise(name)) < MINIMUM_DOCUMENT_TOKENS:
        return document
    return name


#: The Arabic conjunction, which attaches to the word it joins. "جدول" and
#: "وجدول" are the same word in a list, but they are different tokens, and an
#: answer listing documents with "and" between them was scored as having named
#: the wrong one. Matching allows it on the answer's side rather than stripping
#: it, because the same letter is a root letter in words like "وثيقة".
_CONJUNCTION = "و"


def _names_token(present: set[str], token: str) -> bool:
    return token in present or f"{_CONJUNCTION}{token}" in present


def contains_document(answer: str, document: str) -> bool:
    """Whether the answer names a document, by token overlap with the phrase."""
    wanted = set(tokenise(document_name(document)))
    if not wanted:
        return False
    present = set(tokenise(answer))
    matched = sum(1 for token in wanted if _names_token(present, token))
    return matched / len(wanted) >= DOCUMENT_TOKEN_OVERLAP
