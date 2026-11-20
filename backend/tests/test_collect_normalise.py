"""Tests for turning content components into the service schema.

The payloads are invented but reproduce the shapes the source uses, including a
duration split into a unit and a count, and documents hanging off sub-keys.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from conftest import component, service_record

from shurta.collect.normalise import build_service
from shurta.collect.selection import Selected
from shurta.models import Language

ENTRY = Selected("invented-certificate", "services/invented-certificate", "certificate_permit")
COLLECTED = datetime(2026, 11, 20, tzinfo=UTC)


def build(*components: dict[str, object], arabic: list[dict[str, object]] | None = None):
    english = service_record(
        name="Invented Certificate", slug=ENTRY.slug, components=list(components)
    )
    records = {Language.EN: english}
    if arabic is not None:
        records[Language.AR] = service_record(
            name="شهادة مُفترضة", slug=ENTRY.slug, components=arabic
        )
    return build_service(
        entry=ENTRY,
        records=records,
        category_names={Language.EN: "Permits and Certificates", Language.AR: "التصاريح والشهادات"},
        collected_at=COLLECTED,
        # A citation is only recorded for a language that was actually collected,
        # which is how the collector behaves.
        page_url={
            language: f"https://example.invalid/services/invented-certificate?lng={language.value}"
            for language in records
        },
    )


class TestEmptyStaysEmpty:
    def test_a_service_with_no_components_has_no_invented_content(self) -> None:
        service = build()
        assert service.description.en == ""
        assert service.fees.en == ""
        assert service.required_documents.en == []
        assert service.processing_time.en == ""
        assert service.channels.en == []

    def test_the_identifier_and_citation_survive_an_empty_service(self) -> None:
        service = build()
        assert service.id == "invented-certificate"
        assert service.source_url(Language.EN).endswith("lng=en")

    def test_steps_are_always_empty_because_the_source_has_none(self) -> None:
        service = build(component("description", text="Anything at all."))
        assert service.steps.en == []
        assert service.steps.ar == []


class TestDescription:
    def test_it_is_carried_across_unchanged(self) -> None:
        text = "A certificate that does not exist, issued by nobody."
        assert build(component("description", text=text)).description.en == text

    def test_several_description_components_are_joined(self) -> None:
        service = build(
            component("description", text="First part."),
            component("description", text="Second part."),
        )
        assert service.description.en == "First part.\nSecond part."

    def test_a_null_text_is_ignored(self) -> None:
        assert build(component("description", text=None)).description.en == ""


class TestDuration:
    """``timing`` holds the unit and ``validity`` the count, so both are needed."""

    def test_instant_needs_no_count(self) -> None:
        service = build(component("period-of-service", timing="instant", validity=None))
        assert service.processing_time.en == "Processing time: Instant"

    def test_a_count_and_unit_become_a_phrase(self) -> None:
        service = build(component("period-of-service", timing="working_days", validity=5))
        assert service.processing_time.en == "Processing time: 5 working days"

    def test_one_is_singular(self) -> None:
        service = build(component("period-of-service", timing="working_days", validity=1))
        assert service.processing_time.en == "Processing time: 1 working day"

    def test_a_unit_with_no_count_still_says_something(self) -> None:
        service = build(component("period-of-service", timing="working_days", validity=None))
        assert service.processing_time.en == "Processing time: Working days"

    def test_an_unrecognised_unit_is_dropped_rather_than_guessed(self) -> None:
        service = build(component("period-of-service", timing="fortnights", validity=2))
        assert service.processing_time.en == ""

    @pytest.mark.parametrize(
        ("count", "expected"),
        [(1, "يوم عمل واحد"), (2, "يوما عمل"), (5, "5 أيام عمل"), (30, "30 يوم عمل")],
    )
    def test_arabic_counts_the_duration_correctly(self, count: int, expected: str) -> None:
        service = build(
            component("description", text="x"),
            arabic=[component("period-of-service", timing="working_days", validity=count)],
        )
        assert service.processing_time.ar == f"مدة الإنجاز: {expected}"

    def test_working_hours_are_listed_alongside(self) -> None:
        service = build(
            component("period-of-service", timing="instant", validity=None),
            component("working-hours", digitalChannels="24/7", policeStations="8:00 - 20:00"),
        )
        assert service.processing_time.en == (
            "Processing time: Instant\nDigital channels: 24/7\nPolice stations: 8:00 - 20:00"
        )


class TestRequiredDocuments:
    def test_plain_entries_become_list_items(self) -> None:
        service = build(
            component(
                "required-documents",
                requiredDocumentOptions=[
                    {"label": "Identity card", "value": None, "subKeys": []},
                    {"label": "Personal photograph", "value": None, "subKeys": []},
                ],
            )
        )
        assert service.required_documents.en == ["Identity card", "Personal photograph"]

    def test_a_grouped_entry_keeps_its_condition_on_every_document(self) -> None:
        """The condition is what makes the document list meaningful."""
        service = build(
            component(
                "required-documents",
                requiredDocumentOptions=[
                    {
                        "label": "Applying from abroad",
                        "value": None,
                        "subKeys": [
                            {"value": "Passport copy"},
                            {"value": "Previous residence permit"},
                        ],
                    }
                ],
            )
        )
        assert service.required_documents.en == [
            "Applying from abroad — Passport copy",
            "Applying from abroad — Previous residence permit",
        ]

    def test_a_label_with_a_value_is_kept_together(self) -> None:
        service = build(
            component(
                "required-documents",
                requiredDocumentOptions=[
                    {"label": "Photograph", "value": "white background", "subKeys": []}
                ],
            )
        )
        assert service.required_documents.en == ["Photograph: white background"]

    def test_the_component_description_leads_the_list(self) -> None:
        service = build(
            component(
                "required-documents",
                description="Originals must be presented.",
                requiredDocumentOptions=[{"label": "Identity card", "subKeys": []}],
            )
        )
        assert service.required_documents.en[0] == "Originals must be presented."


class TestFees:
    def test_fee_lines_pair_the_label_with_the_amount(self) -> None:
        service = build(
            component(
                "service-fees",
                options=[
                    {"label": "Citizen", "value": "75 NRD"},
                    {"label": "Resident", "value": "150 NRD"},
                ],
            )
        )
        assert service.fees.en == "Citizen: 75 NRD\nResident: 150 NRD"

    def test_payment_methods_are_appended_because_a_reader_asks_both_at_once(self) -> None:
        service = build(
            component("service-fees", options=[{"label": "Permit", "value": "100 NRD"}]),
            component("payment-methods", debitCard=True, creditCard=True, cash=None),
        )
        assert service.fees.en == "Permit: 100 NRD\nPayment methods: Debit card, Credit card"

    def test_payment_methods_alone_do_not_imply_a_fee(self) -> None:
        """Otherwise a free service would appear to cost something."""
        service = build(component("payment-methods", debitCard=True))
        assert service.fees.en == ""

    def test_arabic_payment_methods_use_arabic_labels(self) -> None:
        service = build(
            component("description", text="x"),
            arabic=[
                component("service-fees", options=[{"label": "الرسوم", "value": "100 درهم"}]),
                component("payment-methods", cash=True),
            ],
        )
        assert service.fees.ar == "الرسوم: 100 درهم\nطرق الدفع: نقداً"


class TestEligibility:
    def test_beneficiary_switches_become_an_audience_line(self) -> None:
        service = build(component("beneficiaries", individuals=True, businessSector=True))
        assert service.eligibility.en == "Available to: Individuals, Business sector"

    def test_conditions_follow_the_audience(self) -> None:
        service = build(
            component("beneficiaries", individuals=True),
            component(
                "terms-and-conditions",
                options=[{"label": "The applicant must be at least 18 years old."}],
            ),
        )
        assert service.eligibility.en == (
            "Available to: Individuals\nThe applicant must be at least 18 years old."
        )

    def test_arabic_uses_arabic_group_names(self) -> None:
        service = build(
            component("description", text="x"),
            arabic=[component("beneficiaries", individuals=True, tourists=True)],
        )
        assert service.eligibility.ar == "متاح لـ: الأفراد, السياح"


class TestChannels:
    def test_switches_become_named_channels(self) -> None:
        service = build(
            component(
                "service-channels",
                dubaiPoliceApp=True,
                dubaiPoliceWebsite=True,
                callCenter=True,
                policeStations=False,
            )
        )
        assert service.channels.en == [
            "Dubai Police app",
            "Dubai Police website",
            "Call centre 901",
        ]

    def test_free_text_channels_are_kept(self) -> None:
        service = build(
            component("service-channels", dubaiPoliceApp=True, others="Partner service centres")
        )
        assert service.channels.en == ["Dubai Police app", "Partner service centres"]

    def test_delivery_text_is_split_into_one_channel_per_paragraph(self) -> None:
        service = build(
            component(
                "delivery-channels", text="**Email**: sent to you.\n\n**Account**: download it."
            )
        )
        assert service.channels.en == ["**Email**: sent to you.", "**Account**: download it."]

    def test_contact_details_are_appended(self) -> None:
        service = build(
            component("service-channels", dubaiPoliceApp=True),
            component("contact-details", callCenter901="901"),
        )
        assert service.channels.en == ["Dubai Police app", "901"]


class TestBothLanguages:
    def test_names_and_citations_are_recorded_for_each_language(self) -> None:
        service = build(component("description", text="English text."), arabic=[])
        assert service.name_en == "Invented Certificate"
        assert service.name_ar == "شهادة مُفترضة"
        assert service.source_url(Language.AR).endswith("lng=ar")
        assert service.category.ar == "التصاريح والشهادات"

    def test_a_missing_language_falls_back_rather_than_showing_nothing(self) -> None:
        service = build(component("description", text="English only."))
        assert service.name(Language.AR) == "Invented Certificate"
        assert service.source_url(Language.AR).endswith("lng=en")
