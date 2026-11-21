"""Deriving the key facts an answer must contain, from the local corpus.

A question names a service and a field; this turns that into concrete figures,
durations and document phrases. The result is untracked, being derived from
collected content. Re-run it after every collection.
"""

from __future__ import annotations

from collections.abc import Mapping

from shurta.config import Settings
from shurta.corpus import CorpusMissing, load_configured_corpus
from shurta.evaluation.dataset import (
    KEY_FACTS_FILE,
    Behaviour,
    KeyFacts,
    Question,
    load_questions,
    write_key_facts,
)
from shurta.evaluation.facts import extract
from shurta.models import Corpus, Language, Service


def _gather(
    question: Question, services: Mapping[str, Service], variant: str
) -> tuple[list[str], list[str], list[str]]:
    """The facts the question's fields state, narrowed to ``variant`` if given."""
    amounts: list[str] = []
    durations: list[str] = []
    documents: list[str] = []
    language = Language(question.language)

    for service_id in question.expected_service_ids:
        service = services.get(service_id)
        if service is None:
            continue
        found_amounts, found_durations, found_documents = extract(
            service, question.expected_fields, language, variant
        )
        amounts.extend(amount for amount in found_amounts if amount not in amounts)
        durations.extend(duration for duration in found_durations if duration not in durations)
        documents.extend(document for document in found_documents if document not in documents)

    return amounts, durations, documents


def build_key_facts(corpus: Corpus) -> list[KeyFacts]:
    """Derive the checkable facts for every question that has any.

    A question that names a variant of its fields is given only that variant's
    facts, which is what the recall figure is scored on. Every fact the fields
    state is derived alongside it, so the report can show the unscoped recall
    the same answers used to score.
    """
    services = corpus.by_id()
    facts: list[KeyFacts] = []

    for question in load_questions():
        if question.expected_behaviour is not Behaviour.ANSWER:
            continue

        every = _gather(question, services, "")
        scoped = _gather(question, services, question.expected_variant)
        entry = KeyFacts(
            id=question.id,
            amounts=scoped[0],
            durations=scoped[1],
            documents=scoped[2],
            all_amounts=every[0],
            all_durations=every[1],
            all_documents=every[2],
        )
        if entry:
            facts.append(entry)
        elif question.expected_variant and any(every):
            print(
                f"{question.id}: the variant {question.expected_variant!r} matches no part of "
                f"{', '.join(question.expected_fields)}, so the question has no facts to check."
            )

    return facts


def main(settings: Settings) -> int:
    try:
        corpus = load_configured_corpus(settings)
    except CorpusMissing as error:
        print(f"Cannot derive key facts: {error}")
        return 1

    facts = build_key_facts(corpus)
    write_key_facts(facts)

    scoped = sum(1 for entry in facts if entry.narrowed)
    with_amounts = sum(1 for entry in facts if entry.amounts)
    with_durations = sum(1 for entry in facts if entry.durations)
    with_documents = sum(1 for entry in facts if entry.documents)
    print(
        f"Derived key facts for {len(facts)} questions "
        f"({with_amounts} with figures, {with_durations} with durations, "
        f"{with_documents} with documents) -> {KEY_FACTS_FILE}"
    )
    print(f"{scoped} of them are narrowed to the variant their question asks about.")
    print("This file is derived from collected content and is not committed.")
    return 0
