"""Which services the assistant covers, and why.

The official catalogue holds 91 records; the forty below are what remains after
three filters: the record carries real content rather than being a navigation
tile, a member of the public is the applicant, and both language versions are
published.

Each entry fixes a stable ``id`` by hand, because the slugs are not all
well-formed identifiers and a citation should survive the site reorganising its
routes. The category is one of the site's own seven slugs, supplied here because
the content interface leaves most services uncategorised; the category *names*
are read from the site.
"""

from __future__ import annotations

from typing import NamedTuple


class Selected(NamedTuple):
    """One chosen service: a stable identifier, its site slug, its category."""

    id: str
    slug: str
    category: str


#: Category slugs as the site publishes them, kept here so a typo fails loudly
#: when the names are resolved against the collected categories.
CATEGORY_SLUGS = frozenset(
    {
        "inquiries-follow-up",
        "report_complaint",
        "certificate_permit",
        "traffic",
        "community",
        "emergency",
        "business_organization",
    }
)


SELECTED: tuple[Selected, ...] = (
    # --- Permits and certificates -------------------------------------------
    Selected(
        "police-clearance-certificate",
        "services/police-clearance-certificate",
        "certificate_permit",
    ),
    Selected("twimc-certificate", "services/twimc-certificate", "certificate_permit"),
    Selected("twimc-inmate-certificate", "services/TWIMC-inmate-certificate", "certificate_permit"),
    Selected("detainee-visit-permit", "services/detainee-visit", "certificate_permit"),
    Selected("headquarters-entry-permit", "services/hqvisit", "certificate_permit"),
    Selected("police-museum-visit-permit", "services/museum-visit", "certificate_permit"),
    Selected("event-security-permit", "services/event-security", "certificate_permit"),
    Selected("corpse-entry-permit", "services/corpse-entry", "certificate_permit"),
    # --- Traffic -------------------------------------------------------------
    Selected("traffic-fines-inquiry-and-payment", "services/fine-payment", "traffic"),
    Selected("traffic-clearance-certificate", "services/traffic-clearance-certificate", "traffic"),
    Selected("traffic-status-certificate", "services/traffic-status-certificate", "traffic"),
    Selected("vehicle-colour-change-permit", "services/vehicle-color-change", "traffic"),
    Selected("report-reckless-driver", "services/reckless-driver", "traffic"),
    Selected("report-vehicle-obstruction", "services/vehicle-obstruction", "traffic"),
    Selected("traffic-training-courses", "services/traffic-training-courses", "traffic"),
    # --- Reports and complaints ---------------------------------------------
    Selected("report-cybercrime", "services/ecrime", "report_complaint"),
    Selected("file-criminal-complaint", "services/open-criminal-case", "report_complaint"),
    Selected("report-lost-item", "services/lost-and-found/report-lost-item", "report_complaint"),
    Selected("report-bounced-cheque", "services/report-bounce-cheques", "report_complaint"),
    Selected("workers-voice-labour-complaint", "services/labour-complaint", "report_complaint"),
    # --- Inquiries and follow-up --------------------------------------------
    Selected("application-status", "services/application-status", "inquiries-follow-up"),
    Selected("document-verification", "services/document-inquiry", "inquiries-follow-up"),
    Selected("circulars-and-travel-bans", "services/financial-cases", "inquiries-follow-up"),
    Selected("submit-an-inquiry", "services/submit-inquiry", "inquiries-follow-up"),
    Selected("fast-track", "services/fast-track", "inquiries-follow-up"),
    # --- Emergency and response ---------------------------------------------
    Selected("critical-care-response", "services/critical-care-response", "emergency"),
    # --- Community and support ----------------------------------------------
    Selected("smart-home-security", "services/smart-home-security", "community"),
    Selected("tourist-support", "services/tourist-police", "community"),
    Selected("child-protection", "services/child-protection", "community"),
    Selected("women-protection", "services/women-protection", "community"),
    Selected("human-trafficking-victim-care", "services/human-trafficking", "community"),
    Selected("people-of-determination-support", "services/people-of-determination", "community"),
    Selected("police-eye-share-information", "services/share-security-information", "community"),
    Selected(
        "neighbourhood-police",
        "services/volunteer-platform/225177000288/neighborhood-police",
        "community",
    ),
    Selected("volunteer-platform", "services/volunteer-platform", "community"),
    Selected("public-lecture-request", "services/seminar-request", "community"),
    Selected("drug-awareness-lecture", "services/drug-awareness-lecture", "community"),
    Selected("student-awareness", "services/student-awareness", "community"),
    Selected("sports-facilities-booking", "services/request-sports-facility", "community"),
    Selected("dubai-police-activities-request", "services/activity-request", "community"),
)


#: Slugs deliberately left out, with the reason, so the choice is reviewable and
#: a future run does not quietly re-add them.
EXCLUDED: dict[str, str] = {
    "services/lost-and-found": "navigation tile, no content of its own",
    "services/service-attachment": "a step inside other services rather than a service",
    "services/report-bounce-cheques-bank": "addressed to banks; the public version is kept",
    "services/ecocrime": "economic-crime operations centre, addressed to businesses",
    "services/faris": "corporate security partnership programme",
    "services/gold-management": "security audit for gold traders",
    "services/destroy-explosives": "industrial permit, not a public service",
    "services/public-tender": "procurement, addressed to suppliers",
    "services/supplier-application": "procurement, addressed to suppliers",
    "services/lawyer-poa": "addressed to law firms",
    "services/lawyer-search-suspension-letter": "addressed to law firms",
    "services/diplomatic-services": "addressed to diplomatic missions",
    "services/diplomatic-exchange": "addressed to diplomatic missions",
    "services/contact-police-scientists-council": "internal academic channel",
    "services/leaders-at-your-service": "engagement channel, no service content",
    "services/retirees-services": "addressed to Dubai Police retirees",
}


def selected_by_slug() -> dict[str, Selected]:
    """The chosen services keyed by their site slug."""
    return {entry.slug: entry for entry in SELECTED}


def check() -> None:
    """Fail loudly on a malformed selection table.

    Called by the collector before any request goes out, because a duplicate
    identifier would silently drop a service from the corpus.
    """
    identifiers = [entry.id for entry in SELECTED]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("the selection table has duplicate identifiers")

    slugs = [entry.slug for entry in SELECTED]
    if len(set(slugs)) != len(slugs):
        raise ValueError("the selection table has duplicate slugs")

    unknown = {entry.category for entry in SELECTED} - CATEGORY_SLUGS
    if unknown:
        raise ValueError(f"the selection table uses unknown category slugs: {sorted(unknown)}")

    overlap = set(slugs) & set(EXCLUDED)
    if overlap:
        raise ValueError(f"slugs are both selected and excluded: {sorted(overlap)}")
