"""Deriving the key facts an answer must contain, from the local corpus.

The evaluation set names a service and a field; this reads the current corpus
and turns that reference into concrete, checkable facts — the figures, the
durations and the document phrases. The result is written to a file that is not
committed, because it is derived from collected content.

Run it after a collection, and again after any re-collection: a changed fee on
the official page should change what the evaluation expects.
"""

from __future__ import annotations

from shurta.config import Settings
from shurta.corpus import CorpusMissing, load_configured_corpus
from shurta.evaluation.dataset import (
    KEY_FACTS_FILE,
    Behaviour,
    KeyFacts,
    load_questions,
    write_key_facts,
)
from shurta.evaluation.facts import extract
from shurta.models import Corpus, Language


def build_key_facts(corpus: Corpus) -> list[KeyFacts]:
    """Derive the checkable facts for every question that has any."""
    services = corpus.by_id()
    facts: list[KeyFacts] = []

    for question in load_questions():
        if question.expected_behaviour is not Behaviour.ANSWER:
            continue

        amounts: list[str] = []
        durations: list[str] = []
        documents: list[str] = []
        language = Language(question.language)

        for service_id in question.expected_service_ids:
            service = services.get(service_id)
            if service is None:
                continue
            found_amounts, found_durations, found_documents = extract(
                service, question.expected_fields, language
            )
            amounts.extend(amount for amount in found_amounts if amount not in amounts)
            durations.extend(duration for duration in found_durations if duration not in durations)
            documents.extend(document for document in found_documents if document not in documents)

        entry = KeyFacts(id=question.id, amounts=amounts, durations=durations, documents=documents)
        if entry:
            facts.append(entry)

    return facts


def main(settings: Settings) -> int:
    try:
        corpus = load_configured_corpus(settings)
    except CorpusMissing as error:
        print(f"Cannot derive key facts: {error}")
        return 1

    facts = build_key_facts(corpus)
    write_key_facts(facts)

    with_amounts = sum(1 for entry in facts if entry.amounts)
    with_durations = sum(1 for entry in facts if entry.durations)
    with_documents = sum(1 for entry in facts if entry.documents)
    print(
        f"Derived key facts for {len(facts)} questions "
        f"({with_amounts} with figures, {with_durations} with durations, "
        f"{with_documents} with documents) -> {KEY_FACTS_FILE}"
    )
    print("This file is derived from collected content and is not committed.")
    return 0
