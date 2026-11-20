"""Splitting services into the passages that retrieval works over.

One chunk is one section of one service in one language. Coarser drags unrelated
sections into the prompt; finer breaks a fee table apart, and "Resident: AED 200"
on its own says nothing about which service it belongs to.

Each chunk carries both names of its service, its section heading, its language
and its source page, so it stands alone as a citation. See docs/architecture.md.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from shurta.models import Corpus, Language, Service

#: The sections a chunk can come from, and their heading in each language, in
#: the order a reader would want them in an answer.
SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("description", "Overview", "نظرة عامة"),
    ("eligibility", "Who can apply and conditions", "من يمكنه التقديم والشروط"),
    ("required_documents", "Required documents", "المستندات المطلوبة"),
    ("fees", "Fees", "الرسوم"),
    ("processing_time", "Processing time and working hours", "مدة الإنجاز وساعات العمل"),
    ("channels", "Where to use the service", "قنوات تقديم الخدمة"),
    ("steps", "Steps", "خطوات التقديم"),
)

SECTION_HEADINGS: dict[str, tuple[str, str]] = {
    section: (english, arabic) for section, english, arabic in SECTIONS
}

_CATEGORY = ("Category", "الباقة")


class Chunk(BaseModel):
    """One retrievable passage."""

    model_config = ConfigDict(frozen=True)

    id: str
    """``<service id>::<section>::<language>``, unique across the corpus."""

    service_id: str
    name_en: str
    name_ar: str
    section: str
    language: Language
    text: str
    """What gets embedded, indexed and placed in the prompt."""

    source_url: str
    collected_at: datetime

    def heading(self) -> str:
        """The section heading in this chunk's language."""
        english, arabic = SECTION_HEADINGS[self.section]
        return english if self.language is Language.EN else arabic


def section_body(service: Service, section: str, language: Language) -> str:
    """The text of one section, or empty when the source did not state it."""
    content = getattr(service, section).get(language)
    if isinstance(content, list):
        return "\n".join(f"- {item}" for item in content if item)
    return str(content)


def chunk_service(service: Service, language: Language) -> list[Chunk]:
    """Every non-empty section of one service, in one language."""
    chunks: list[Chunk] = []
    for section, _english, _arabic in SECTIONS:
        body = section_body(service, section, language)
        if not body.strip():
            continue

        english_heading, arabic_heading = SECTION_HEADINGS[section]
        heading = english_heading if language is Language.EN else arabic_heading
        category_label = _CATEGORY[0 if language is Language.EN else 1]
        category = service.category.get(language)

        # Naming the service in both languages is what lets an Arabic question
        # land on an English passage, and the other way round.
        lines = [
            f"{service.name(language)} ({_other_name(service, language)})",
            f"{category_label}: {category}" if category else "",
            f"## {heading}",
            body.strip(),
        ]

        chunks.append(
            Chunk(
                id=f"{service.id}::{section}::{language.value}",
                service_id=service.id,
                name_en=service.name_en,
                name_ar=service.name_ar,
                section=section,
                language=language,
                text="\n".join(line for line in lines if line),
                source_url=service.source_url(language),
                collected_at=service.collected_at,
            )
        )
    return chunks


def _other_name(service: Service, language: Language) -> str:
    other = Language.AR if language is Language.EN else Language.EN
    return service.name(other)


def chunk_corpus(corpus: Corpus) -> list[Chunk]:
    """Every chunk of the whole corpus, in both languages."""
    chunks: list[Chunk] = []
    for service in corpus.services:
        for language in (Language.EN, Language.AR):
            chunks.extend(chunk_service(service, language))
    return chunks
