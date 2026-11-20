"""Turning the site's content components into this project's service schema.

The components it understands are read; every field with no source is left
empty. Nothing is invented. The only text written here is vocabulary: channels,
payment methods and beneficiary groups arrive as unlabelled booleans, so their
names are spelled out in both languages. The rest is the source's own wording.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from shurta.collect.selection import Selected
from shurta.models import Bilingual, BilingualList, Language, Service

# --- vocabulary for the boolean switches ------------------------------------

CHANNEL_LABELS: dict[str, tuple[str, str]] = {
    "dubaiPoliceApp": ("Dubai Police app", "تطبيق شرطة دبي"),
    "dubaiPoliceWebsite": ("Dubai Police website", "الموقع الإلكتروني لشرطة دبي"),
    "callCenter": ("Call centre 901", "مركز الاتصال 901"),
    "smartPoliceStations": ("Smart Police Station", "مركز شرطة ذكي"),
    "policeStations": ("Police station", "مركز شرطة"),
    "dubaiNow": ("DubaiNow app", "تطبيق دبي الآن"),
    "walkinMachine": ("Walk-in service machine", "جهاز الخدمة الذاتية"),
    "drivethruMachine": ("Drive-through service machine", "جهاز الخدمة من السيارة"),
}

PAYMENT_LABELS: dict[str, tuple[str, str]] = {
    "debitCard": ("Debit card", "بطاقة خصم"),
    "creditCard": ("Credit card", "بطاقة ائتمان"),
    "applePay": ("Apple Pay", "Apple Pay"),
    "cheques": ("Cheque", "شيك"),
    "cash": ("Cash", "نقداً"),
    "tabby": ("Tabby", "تابي"),
}

BENEFICIARY_LABELS: dict[str, tuple[str, str]] = {
    "individuals": ("Individuals", "الأفراد"),
    "generalPublic": ("General public", "الجمهور"),
    "tourists": ("Tourists", "السياح"),
    "schoolStudents": ("School students", "طلبة المدارس"),
    "businessSector": ("Business sector", "قطاع الأعمال"),
    "governmentalEntities": ("Government entities", "الجهات الحكومية"),
    "bankingSector": ("Banking sector", "القطاع المصرفي"),
    "lawFirms": ("Law firms", "مكاتب المحامين"),
    "diplomaticOrganizations": ("Diplomatic missions", "البعثات الدبلوماسية"),
    "policeOfficers": ("Police officers", "ضباط الشرطة"),
    "dubaiPolicePensioners": ("Dubai Police pensioners", "متقاعدو شرطة دبي"),
    "goldShops": ("Gold traders", "متاجر الذهب"),
    "customsPersonnel": ("Customs personnel", "موظفو الجمارك"),
}

WORKING_HOURS_LABELS: dict[str, tuple[str, str]] = {
    "digitalChannels": ("Digital channels", "القنوات الرقمية"),
    "sps": ("Smart Police Stations", "مراكز الشرطة الذكية"),
    "policeStations": ("Police stations", "مراكز الشرطة"),
    "others": ("Other channels", "قنوات أخرى"),
}

#: The period component states a duration as a unit in ``timing`` and a count in
#: ``validity``, so the two have to be read together to mean anything. ``instant``
#: carries no count.
DURATION_UNITS: dict[str, tuple[str, str]] = {
    "instant": ("Instant", "فوري"),
    "working_days": ("working days", "أيام عمل"),
    "months": ("months", "أشهر"),
}

#: Arabic counts a duration differently for one, two, a few and many, so the
#: phrasing is spelled out rather than produced by appending a number.
_ARABIC_DURATION: dict[str, tuple[str, str, str, str]] = {
    # unit: (one, two, three to ten, eleven and above)
    "working_days": ("يوم عمل واحد", "يوما عمل", "{n} أيام عمل", "{n} يوم عمل"),
    "months": ("شهر واحد", "شهران", "{n} أشهر", "{n} شهراً"),
}

_PROCESSING_TIME = ("Processing time", "مدة الإنجاز")

_ELIGIBLE_FOR = ("Available to", "متاح لـ")
_PAYMENT_METHODS = ("Payment methods", "طرق الدفع")


def _label(table: dict[str, tuple[str, str]], key: str, language: Language) -> str:
    english, arabic = table[key]
    return english if language is Language.EN else arabic


# --- reading the components --------------------------------------------------


def _components(record: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """Every component of one kind on a service record."""
    wanted = f"dp-service-options.{kind}"
    return [
        component
        for component in record.get("dPServiceOptions") or []
        if isinstance(component, dict) and component.get("__component") == wanted
    ]


def _first(record: dict[str, Any], kind: str) -> dict[str, Any]:
    components = _components(record, kind)
    return components[0] if components else {}


def _text(value: Any) -> str:
    """A trimmed string, or empty for anything that is not usable text."""
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _options(component: dict[str, Any], field: str = "options") -> list[dict[str, Any]]:
    return [option for option in component.get(field) or [] if isinstance(option, dict)]


def _switched_on(component: dict[str, Any], table: dict[str, tuple[str, str]]) -> list[str]:
    """Keys of ``table`` whose switch is on in ``component``."""
    return [key for key in table if component.get(key) is True]


def _join(parts: list[str], separator: str = "\n") -> str:
    return separator.join(part for part in parts if part)


# --- per-field extraction ----------------------------------------------------


def _duration(unit: Any, count: Any, language: Language) -> str:
    """Render the unit-and-count pair the period component uses as a phrase."""
    if not isinstance(unit, str) or unit not in DURATION_UNITS:
        return ""

    english, arabic = DURATION_UNITS[unit]
    if unit == "instant":
        return english if language is Language.EN else arabic

    try:
        number = int(str(count))
    except (TypeError, ValueError):
        # The unit alone is still worth stating, even without a count.
        return english.capitalize() if language is Language.EN else arabic

    if language is Language.EN:
        singular = english.removesuffix("s") if number == 1 else english
        return f"{number} {singular}"

    one, two, few, many = _ARABIC_DURATION[unit]
    if number == 1:
        return one
    if number == 2:
        return two
    template = few if 3 <= number <= 10 else many
    return template.format(n=number)


def _description(record: dict[str, Any]) -> str:
    return _join([_text(component.get("text")) for component in _components(record, "description")])


def _eligibility(record: dict[str, Any], language: Language) -> str:
    """Who may use the service, and on what conditions.

    Built from the beneficiary switches and the conditions list, which is where
    the site states age limits, ownership requirements and the like.
    """
    lines: list[str] = []

    beneficiaries = _first(record, "beneficiaries")
    groups = [
        _label(BENEFICIARY_LABELS, key, language)
        for key in _switched_on(beneficiaries, BENEFICIARY_LABELS)
    ]
    if groups:
        lines.append(f"{_ELIGIBLE_FOR[0 if language is Language.EN else 1]}: {', '.join(groups)}")
    lines.append(_text(beneficiaries.get("description")))

    conditions = _first(record, "terms-and-conditions")
    lines.append(_text(conditions.get("description")))
    lines.extend(_text(option.get("label")) for option in _options(conditions))

    return _join(lines)


def _required_documents(record: dict[str, Any]) -> list[str]:
    component = _first(record, "required-documents")
    documents: list[str] = []
    description = _text(component.get("description"))
    if description:
        documents.append(description)
    for option in _options(component, "requiredDocumentOptions"):
        label = _text(option.get("label"))
        value = _text(option.get("value"))
        # A document entry may be a group — "From outside the UAE" — whose actual
        # documents hang off it. Each one is emitted on its own line, carrying the
        # group as its condition, so that a question about one document can match
        # it without the condition being lost.
        subkeys = [_text(sub.get("value")) for sub in _options(option, "subKeys")]
        subkeys = [subkey for subkey in subkeys if subkey]
        if subkeys and label:
            documents.extend(f"{label} — {subkey}" for subkey in subkeys)
        elif subkeys:
            documents.extend(subkeys)
        elif label:
            documents.append(f"{label}: {value}" if value else label)
    return documents


def _fees(record: dict[str, Any], language: Language) -> str:
    """The fee table, with the accepted payment methods appended.

    A reader asking what a service costs almost always wants to know how to pay
    it as well, and the site separates the two into different components.
    """
    component = _first(record, "service-fees")
    lines = [_text(component.get("description"))]
    for option in _options(component):
        label = _text(option.get("label"))
        value = _text(option.get("value"))
        if label and value:
            lines.append(f"{label}: {value}")
        else:
            lines.append(label or value)

    if not _join(lines):
        return ""

    payments = _first(record, "payment-methods")
    methods = [
        _label(PAYMENT_LABELS, key, language) for key in _switched_on(payments, PAYMENT_LABELS)
    ]
    if methods:
        heading = _PAYMENT_METHODS[0 if language is Language.EN else 1]
        lines.append(f"{heading}: {', '.join(methods)}")
    lines.append(_text(payments.get("description")))

    return _join(lines)


def _processing_time(record: dict[str, Any], language: Language) -> str:
    """How long the service takes, and when it can be used."""
    lines: list[str] = []

    period = _first(record, "period-of-service")
    lines.append(_text(period.get("description")))
    duration = _duration(period.get("timing"), period.get("validity"), language)
    if duration:
        heading = _PROCESSING_TIME[0 if language is Language.EN else 1]
        lines.append(f"{heading}: {duration}")

    hours = _first(record, "working-hours")
    lines.append(_text(hours.get("description")))
    for key in WORKING_HOURS_LABELS:
        value = _text(hours.get(key))
        if value:
            lines.append(f"{_label(WORKING_HOURS_LABELS, key, language)}: {value}")

    return _join(lines)


def _channels(record: dict[str, Any], language: Language) -> list[str]:
    """Where the service can be used, and how its result is delivered."""
    component = _first(record, "service-channels")
    channels = [
        _label(CHANNEL_LABELS, key, language) for key in _switched_on(component, CHANNEL_LABELS)
    ]
    for extra in (component.get("description"), component.get("others")):
        text = _text(extra)
        if text:
            channels.append(text)

    for delivery in _components(record, "delivery-channels"):
        text = _text(delivery.get("text"))
        # The delivery text is written as one paragraph per channel.
        channels.extend(paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip())

    return channels


def _contact(record: dict[str, Any]) -> list[str]:
    """Contact routes stated on the service, which the channel list omits."""
    component = _first(record, "contact-details")
    details: list[str] = []
    for key in ("callCenter901", "dubaipoliceEmail", "dubaiPoliceLiveCaht", "poBox", "others"):
        text = _text(component.get(key))
        if text:
            details.append(text)
    description = _text(component.get("description"))
    if description:
        details.append(description)
    return details


# --- assembling a service ----------------------------------------------------


def _bilingual(english: str, arabic: str) -> Bilingual:
    return Bilingual(en=english, ar=arabic)


def build_service(
    entry: Selected,
    records: dict[Language, dict[str, Any]],
    category_names: dict[Language, str],
    collected_at: datetime,
    page_url: dict[Language, str],
) -> Service:
    """Assemble one :class:`Service` from its English and Arabic records."""
    english = records.get(Language.EN, {})
    arabic = records.get(Language.AR, {})

    return Service(
        id=entry.id,
        name_en=_text(english.get("name")),
        name_ar=_text(arabic.get("name")),
        category=_bilingual(
            category_names.get(Language.EN, ""), category_names.get(Language.AR, "")
        ),
        description=_bilingual(_description(english), _description(arabic)),
        eligibility=_bilingual(
            _eligibility(english, Language.EN), _eligibility(arabic, Language.AR)
        ),
        required_documents=BilingualList(
            en=_required_documents(english), ar=_required_documents(arabic)
        ),
        fees=_bilingual(_fees(english, Language.EN), _fees(arabic, Language.AR)),
        processing_time=_bilingual(
            _processing_time(english, Language.EN), _processing_time(arabic, Language.AR)
        ),
        channels=BilingualList(
            en=_channels(english, Language.EN) + _contact(english),
            ar=_channels(arabic, Language.AR) + _contact(arabic),
        ),
        # The content interface publishes no step-by-step procedure for any
        # service, so this field stays empty rather than being approximated
        # from the channel or delivery text.
        steps=BilingualList(),
        source_urls={language: url for language, url in page_url.items() if url},
        collected_at=collected_at,
    )
