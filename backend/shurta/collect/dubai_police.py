"""Reading the public service catalogue from the official Dubai Police site.

The service pages are a browser-side application, so fetching a page route
returns an empty shell. The same content is served under ``/dpcms/api`` as
structured JSON in both languages, which is lighter on the site than driving a
browser.

Three quirks of the site's edge are worked around here: only the ``/dpcms/api``
prefix is answered, any square bracket in the query string is rejected — which
rules out Strapi's ``pagination[page]`` and ``filters[...]``, so the catalogue
is covered by merging differently ordered ``sort`` slices — and the page size is
fixed at 25.

A service's page address is the same in both languages, since the site holds the
language in browser state. The language is appended as the ``lng`` parameter its
own detector reads, so a citation opens in the answer's language.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shurta.collect.http import CollectionError, PoliteClient

SITE = "https://www.dubaipolice.gov.ae"
API = f"{SITE}/dpcms/api"
REFERER = f"{SITE}/app/home"

LANGUAGES = ("en", "ar")

#: Orderings requested to cover the catalogue, since it cannot be paged through.
#: Each returns the first 25 records under a different sort, and together they
#: reach the great majority of the catalogue. They are ordered so that the
#: cheapest useful coverage comes first and collection can stop early.
SORT_SLICES = (
    "name:asc",
    "name:desc",
    "id:asc",
    "id:desc",
    "mostUsed:desc",
    "dashboardOrder:asc",
    "serviceKey:asc",
)


def service_page_url(slug: str, language: str) -> str:
    """The public page a reader would open for a service, used as the citation."""
    slug = slug.split("?")[0].strip("/")
    if slug.startswith("services/"):
        slug = slug[len("services/") :]
    return f"{SITE}/app/services/{slug}?lng={language}"


def _collection_url(collection: str, language: str, sort: str | None = None) -> str:
    url = f"{API}/{collection}?mode=dp&locale={language}"
    if sort:
        url += f"&sort={sort}"
    return url


def _records(payload: Any, what: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise CollectionError(f"{what} was not a JSON object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise CollectionError(f"{what} had no list of records")
    return [record for record in data if isinstance(record, dict)]


def _key(record: dict[str, Any]) -> str:
    """Identity of a record across languages and across slices.

    ``documentId`` is shared between the two language versions of the same
    record, which is what makes the English and Arabic halves joinable.
    """
    document_id = record.get("documentId")
    if isinstance(document_id, str) and document_id:
        return document_id
    return f"id:{record.get('id')}"


def fetch_services(client: PoliteClient, language: str) -> dict[str, dict[str, Any]]:
    """Collect the service catalogue in one language, keyed by document id."""
    merged: dict[str, dict[str, Any]] = {}
    expected: int | None = None

    for sort in SORT_SLICES:
        response = client.get_json(_collection_url("dp-services", language, sort))
        body = response.payload
        for record in _records(body, f"dp-services sorted by {sort}"):
            merged.setdefault(_key(record), record)

        if isinstance(body, dict):
            pagination = (body.get("meta") or {}).get("pagination") or {}
            total = pagination.get("total")
            expected = int(total) if isinstance(total, int) else expected

        # Every service the project needs is already in hand, so there is no
        # reason to keep asking for further orderings.
        if expected is not None and len(merged) >= expected:
            break

    if not merged:
        raise CollectionError(f"no services were collected for language {language!r}")
    return merged


def fetch_categories(client: PoliteClient, language: str) -> list[dict[str, Any]]:
    """Collect the service categories, which name the group each service sits in."""
    response = client.get_json(_collection_url("service-categories", language))
    return _records(response.payload, "service-categories")


def category_by_service(categories: list[dict[str, Any]]) -> dict[str, str]:
    """Invert the categories into a service document id to category name mapping.

    Most service records carry a null category of their own, but every category
    lists the services inside it, so the association is recoverable from the
    other direction.
    """
    mapping: dict[str, str] = {}
    for category in categories:
        name = category.get("name")
        if not isinstance(name, str) or not name:
            continue
        for service in category.get("dp_services") or []:
            if isinstance(service, dict):
                mapping.setdefault(_key(service), name)
    return mapping


def fetch_catalogue(client: PoliteClient) -> dict[str, Any]:
    """Collect services and categories in both languages.

    Everything passes through the on-disk cache, so a second run issues no
    requests at all and normalisation can be developed without touching the site.
    """
    return {
        language: {
            "services": fetch_services(client, language),
            "categories": fetch_categories(client, language),
        }
        for language in LANGUAGES
    }


def open_client(cache_dir: Path) -> PoliteClient:
    """A client configured for this site."""
    return PoliteClient(cache_dir=cache_dir, referer=REFERER)
