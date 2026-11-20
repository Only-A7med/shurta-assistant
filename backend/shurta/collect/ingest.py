"""Running a collection: fetch, normalise, persist, and report what was found."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shurta.collect import dubai_police, selection
from shurta.collect.normalise import build_service
from shurta.collect.selection import Selected
from shurta.models import Corpus, Language, Service

SOURCE_NOTE = (
    "Collected from the public content interface of the official Dubai Police website "
    "(www.dubaipolice.gov.ae). The content belongs to Dubai Police; this corpus is a "
    "local, unofficial copy built for question answering and is not redistributed."
)

#: The fields the validation report measures, and how to tell whether one is
#: populated for a given language.
FIELDS: tuple[str, ...] = (
    "name",
    "category",
    "description",
    "eligibility",
    "required_documents",
    "fees",
    "processing_time",
    "channels",
    "steps",
    "source_url",
)


@dataclass
class CollectionResult:
    """What a collection run produced, for the command line to report on."""

    corpus: Corpus
    missing_slugs: list[str]
    report: str


def _slug_of(record: dict[str, Any]) -> str:
    url = record.get("url")
    if not isinstance(url, str):
        return ""
    return url.split("?")[0].strip("/")


def _category_names(categories: list[dict[str, Any]]) -> dict[str, str]:
    """Category slug to its published name, in whichever language was fetched."""
    names: dict[str, str] = {}
    for category in categories:
        slug = category.get("category")
        name = category.get("name")
        if isinstance(slug, str) and isinstance(name, str):
            names[slug] = name.strip()
    return names


def _index_by_slug(services: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Site slug to document id, so the two languages can be joined."""
    index: dict[str, str] = {}
    for document_id, record in services.items():
        slug = _slug_of(record)
        if slug:
            index.setdefault(slug, document_id)
    return index


def collect(cache_dir: Path) -> CollectionResult:
    """Collect the selected services in both languages and normalise them."""
    selection.check()
    chosen = selection.selected_by_slug()

    with dubai_police.open_client(cache_dir) as client:
        catalogue = dubai_police.fetch_catalogue(client)

    collected_at = datetime.now(UTC)
    english = catalogue["en"]
    arabic = catalogue["ar"]
    by_slug = _index_by_slug(english["services"])
    category_names = {
        Language.EN: _category_names(english["categories"]),
        Language.AR: _category_names(arabic["categories"]),
    }

    services: list[Service] = []
    missing: list[str] = []

    for slug, entry in chosen.items():
        document_id = by_slug.get(slug)
        if document_id is None:
            missing.append(slug)
            continue
        services.append(_build(entry, document_id, english, arabic, category_names, collected_at))

    services.sort(key=lambda service: service.id)
    corpus = Corpus(collected_at=collected_at, source=SOURCE_NOTE, services=services)
    return CollectionResult(
        corpus=corpus,
        missing_slugs=sorted(missing),
        report=validation_report(corpus, missing_slugs=sorted(missing)),
    )


def _build(
    entry: Selected,
    document_id: str,
    english: dict[str, Any],
    arabic: dict[str, Any],
    category_names: dict[Language, dict[str, str]],
    collected_at: datetime,
) -> Service:
    records: dict[Language, dict[str, Any]] = {}
    english_record = english["services"].get(document_id)
    if english_record is not None:
        records[Language.EN] = english_record
    arabic_record = arabic["services"].get(document_id)
    if arabic_record is not None:
        records[Language.AR] = arabic_record

    return build_service(
        entry=entry,
        records=records,
        category_names={
            language: category_names[language].get(entry.category, "")
            for language in (Language.EN, Language.AR)
        },
        collected_at=collected_at,
        page_url={
            language: dubai_police.service_page_url(entry.slug, language.value)
            for language in (Language.EN, Language.AR)
            if language in records
        },
    )


def _is_populated(service: Service, field: str, language: Language) -> bool:
    if field == "name":
        return bool(service.name_en if language is Language.EN else service.name_ar)
    if field == "source_url":
        return bool(service.source_urls.get(language))
    value = getattr(service, field)
    return bool(value.get(language))


def field_coverage(corpus: Corpus) -> dict[str, dict[Language, int]]:
    """How many services have each field populated, per language."""
    return {
        field: {
            language: sum(
                1 for service in corpus.services if _is_populated(service, field, language)
            )
            for language in (Language.EN, Language.AR)
        }
        for field in FIELDS
    }


def validation_report(corpus: Corpus, *, missing_slugs: list[str]) -> str:
    """A Markdown report on what the collection covers, field by field.

    Written next to the corpus so that a gap in the source is visible rather
    than something to be discovered later through a bad answer.
    """
    total = len(corpus.services)
    coverage = field_coverage(corpus)

    def percentage(count: int) -> str:
        return f"{100 * count / total:.0f}%" if total else "—"

    lines = [
        "# Collection report",
        "",
        f"Collected {total} services on {corpus.collected_at:%Y-%m-%d %H:%M} UTC.",
        "",
        corpus.source,
        "",
        "## Field coverage",
        "",
        "A field is counted as covered when the source stated something for it. An empty",
        "field means the official page did not state it, and is left empty on purpose.",
        "",
        "| Field | English | | Arabic | |",
        "|---|---:|---:|---:|---:|",
    ]
    for field, counts in coverage.items():
        english = counts[Language.EN]
        arabic = counts[Language.AR]
        lines.append(
            f"| `{field}` | {english}/{total} | {percentage(english)} "
            f"| {arabic}/{total} | {percentage(arabic)} |"
        )

    lines += ["", "## Services", "", "| id | English name | Arabic name |", "|---|---|---|"]
    for service in corpus.services:
        lines.append(f"| `{service.id}` | {service.name_en} | {service.name_ar} |")

    if missing_slugs:
        lines += [
            "",
            "## Selected but not found in the catalogue",
            "",
            "These were chosen but the site no longer publishes them under the recorded slug.",
            "",
        ]
        lines += [f"- `{slug}`" for slug in missing_slugs]

    if selection.EXCLUDED:
        lines += [
            "",
            "## Deliberately excluded",
            "",
            "| slug | reason |",
            "|---|---|",
        ]
        lines += [f"| `{slug}` | {reason} |" for slug, reason in sorted(selection.EXCLUDED.items())]

    return "\n".join(lines) + "\n"
