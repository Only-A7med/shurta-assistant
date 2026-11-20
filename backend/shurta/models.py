"""The normalised shape of a public service, and the corpus that holds them.

Every text field is bilingual, and either side may be empty. An empty field
means the source did not state this; it is never filled in with a guess.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Language(StrEnum):
    """The two languages the assistant works in."""

    EN = "en"
    AR = "ar"


class Bilingual(BaseModel):
    """One piece of text in both languages."""

    model_config = ConfigDict(frozen=True)

    en: str = ""
    ar: str = ""

    def get(self, language: Language) -> str:
        return self.en if language is Language.EN else self.ar

    def __bool__(self) -> bool:
        return bool(self.en or self.ar)


class BilingualList(BaseModel):
    """A list of short items — documents, steps, channels — in both languages."""

    model_config = ConfigDict(frozen=True)

    en: list[str] = Field(default_factory=list)
    ar: list[str] = Field(default_factory=list)

    def get(self, language: Language) -> list[str]:
        return self.en if language is Language.EN else self.ar

    def __bool__(self) -> bool:
        return bool(self.en or self.ar)


class Service(BaseModel):
    """One public service, normalised from its official page in both languages."""

    model_config = ConfigDict(frozen=True)

    id: str
    """Stable slug, used as the retrieval key and in citations."""

    name_en: str = ""
    name_ar: str = ""
    category: Bilingual = Bilingual()

    description: Bilingual = Bilingual()
    eligibility: Bilingual = Bilingual()
    required_documents: BilingualList = BilingualList()
    fees: Bilingual = Bilingual()
    processing_time: Bilingual = Bilingual()
    channels: BilingualList = BilingualList()
    steps: BilingualList = BilingualList()

    source_urls: dict[Language, str] = Field(default_factory=dict)
    """The official page this was taken from, per language."""

    collected_at: datetime
    """When the source was last read, so an answer can say how fresh it is."""

    def name(self, language: Language) -> str:
        """The service name in one language, falling back to the other if absent."""
        if language is Language.EN:
            return self.name_en or self.name_ar
        return self.name_ar or self.name_en

    def source_url(self, language: Language) -> str:
        """The source page for one language, falling back to the other if absent."""
        other = Language.AR if language is Language.EN else Language.EN
        return self.source_urls.get(language) or self.source_urls.get(other) or ""


class Corpus(BaseModel):
    """The whole collected catalogue, as persisted to ``data/processed``."""

    model_config = ConfigDict(frozen=True)

    collected_at: datetime
    source: str = ""
    """Human-readable note on where the catalogue came from."""

    services: list[Service] = Field(default_factory=list)

    def by_id(self) -> dict[str, Service]:
        return {service.id: service for service in self.services}
