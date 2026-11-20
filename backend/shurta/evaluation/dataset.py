"""The evaluation set, and the key facts derived from it.

A question names a service and the *field* an answer has to get right, not the
expected fee itself. So the expectations stay correct when the official page
changes, and no collected content has to live in this repository: the figures are
derived locally by ``python -m shurta eval prepare`` into an untracked file.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from shurta.config import REPOSITORY_ROOT
from shurta.models import Language

QUESTIONS_FILE = REPOSITORY_ROOT / "eval" / "questions.jsonl"
KEY_FACTS_FILE = REPOSITORY_ROOT / "eval" / "key_facts.local.jsonl"


class Behaviour(StrEnum):
    """What a correct response to a question looks like."""

    ANSWER = "answer"
    """Answer from the retrieved context, with citations."""

    REFUSE = "refuse"
    """Decline and name the official channel, for a question about somebody's
    own case, which published information cannot answer."""

    OUT_OF_SCOPE = "out_of_scope"
    """Say the context does not cover it, and point to the official channels."""

    EMERGENCY = "emergency"
    """Give the emergency numbers, before any model is called at all."""


class Question(BaseModel):
    """One evaluation item."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    language: Language
    dialect: str = ""
    """``gulf`` or ``code_switched`` where that is what is being tested."""

    expected_service_ids: list[str] = Field(default_factory=list)
    expected_fields: list[str] = Field(default_factory=list)
    """Service fields an answer must draw its facts from, such as ``fees``."""

    expected_behaviour: Behaviour = Behaviour.ANSWER
    note: str = ""

    @property
    def is_retrieval_item(self) -> bool:
        """Whether this question has a right answer in the corpus to find."""
        return bool(self.expected_service_ids)


class KeyFacts(BaseModel):
    """The facts an answer to one question has to contain, extracted from the
    local corpus rather than written by hand."""

    model_config = ConfigDict(frozen=True)

    id: str
    amounts: list[str] = Field(default_factory=list)
    """Monetary figures, normalised to bare numbers for comparison."""

    durations: list[str] = Field(default_factory=list)
    documents: list[str] = Field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.amounts or self.durations or self.documents)


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    """Read the evaluation set."""
    if not path.exists():
        raise FileNotFoundError(f"no evaluation set at {path}")
    return [Question.model_validate_json(line) for line in _lines(path)]


def load_key_facts(path: Path = KEY_FACTS_FILE) -> dict[str, KeyFacts]:
    """Read the derived key facts, or nothing if they have not been built."""
    if not path.exists():
        return {}
    facts = [KeyFacts.model_validate_json(line) for line in _lines(path)]
    return {entry.id: entry for entry in facts}


def write_key_facts(facts: list[KeyFacts], path: Path = KEY_FACTS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(entry.model_dump_json() + "\n" for entry in facts),
        encoding="utf-8",
    )


def _lines(path: Path) -> Iterator[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            yield stripped


def summarise(questions: list[Question]) -> str:
    """A one-line description of the evaluation set's composition."""
    by_language: dict[str, int] = {}
    by_behaviour: dict[str, int] = {}
    for question in questions:
        key = question.dialect or question.language.value
        by_language[key] = by_language.get(key, 0) + 1
        by_behaviour[question.expected_behaviour.value] = (
            by_behaviour.get(question.expected_behaviour.value, 0) + 1
        )
    languages = ", ".join(f"{name} {count}" for name, count in sorted(by_language.items()))
    behaviours = ", ".join(f"{name} {count}" for name, count in sorted(by_behaviour.items()))
    return f"{len(questions)} questions ({languages}); expected behaviour: {behaviours}"


def json_line(question: Question) -> str:
    """One question as a JSON line, for writing the file back out."""
    return json.dumps(question.model_dump(mode="json", exclude_defaults=True), ensure_ascii=False)
