"""The checks that run before a model is ever asked anything.

An emergency gets a fixed response naming the emergency numbers: a telephone
number in the first line beats a paragraph that arrives seconds later. The
detector is tuned to fire readily, since a false positive costs one unnecessary
line and a miss costs much more.

A personal-case lookup — "what is the status of my fine?" — is declined with the
official channel named, because the answer lives in a record this project has no
access to.

Both detectors run on Arabic-normalised text, and the Arabic side matches tokens
rather than substrings: ``لص`` ("thief") is a substring of ``الصلاحية``
("validity"), which once turned an expired identity card into an emergency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from shurta.arabic import normalise, tokenise

POLICE_NUMBER = "999"
AMBULANCE_NUMBER = "998"
FIRE_NUMBER = "997"
NON_EMERGENCY_NUMBER = "901"


class Intent(StrEnum):
    """What kind of handling a question needs."""

    EMERGENCY = "emergency"
    PERSONAL_CASE = "personal_case"
    NORMAL = "normal"


# --- emergency detection -----------------------------------------------------

_EMERGENCY_ENGLISH = re.compile(
    r"\bhelp\b"
    r"|\bemergency\b"
    r"|\bsos\b"
    r"|\burgent(ly)?\b"
    r"|\bright now\b"
    r"|\bcall (the )?(police|ambulance|fire|civil defence)\b"
    r"|\bbreak(ing)? in(to)?\b|\bbroke in\b|\bburglar"
    r"|\brobbed\b|\brobbery\b|\bbeing robbed\b"
    r"|\battack(ed|ing)\b|\bassault"
    r"|\bstabb(ed|ing)\b|\bshot\b|\bshooting\b|\bgunman\b"
    r"|\bfire\b|\bburning\b|\bon fire\b"
    r"|\binjur(ed|y|ies)\b|\bbleeding\b|\bunconscious\b"
    r"|\bambulance\b"
    r"|\bheart attack\b|\bnot breathing\b|\bcan'?t breathe\b"
    r"|\bdrowning\b"
    r"|\bserious accident\b|\bbad accident\b"
    r"|\bkidnap"
    r"|\bin danger\b|\bthreaten(ing|ed)? to kill\b"
    r"|\bdying\b"
)

#: Single Arabic words signalling an emergency, matched against tokens. Short
#: words are listed both bare and with the definite article, which the tokeniser
#: keeps attached.
_EMERGENCY_WORDS_ARABIC = frozenset(
    {
        "النجده",
        "نجده",
        "ساعدوني",
        "ساعدني",
        "طارئ",
        "طاريء",
        "طارئه",
        "اسعاف",
        "حريق",
        "يحترق",
        "دخان",
        "سرقه",
        "يسرق",
        "سرقوا",
        "لص",
        "اللص",
        "لصوص",
        "اقتحام",
        "يقتحم",
        "هجوم",
        "يهاجمني",
        "اعتداء",
        "طعن",
        "مسدس",
        "مصاب",
        "اصابه",
        "ينزف",
        "اختطاف",
        "خطر",
        "يموت",
        "فورا",
        "حالا",
        "بسرعه",
    }
)

#: Arabic phrases of more than one word, matched on the normalised text.
_EMERGENCY_PHRASES_ARABIC = (
    "حاله طارئه",
    "اتصلوا بالشرطه",
    "اطلبوا الشرطه",
    "يحاول دخول",
    "يحاول الدخول",
    "طلق ناري",
    "فاقد الوعي",
    "لا يتنفس",
    "ازمه قلبيه",
    "حادث خطير",
    "حادث بليغ",
    "يهددني بالقتل",
    "ارجو المساعده",
    "المساعده الفوريه",
)

_EMERGENCY_ARABIC_PHRASE = re.compile(
    "|".join(re.escape(phrase) for phrase in _EMERGENCY_PHRASES_ARABIC)
)

#: Phrasings that ask *about* a service rather than report something happening,
#: so "how do I report a robbery?" reaches the reporting service.
_INFORMATIONAL = re.compile(
    r"\bhow (do|can|would|should) i\b"
    r"|\bhow to\b"
    r"|\bwhat (is|are|documents|fee|paperwork|evidence)\b"
    r"|\bhow much\b|\bhow long\b|\bhow quickly\b"
    r"|\bwhere (do|can) i\b"
    r"|\bfee for\b|\bcost\b|\bdocuments (do|are|needed)\b"
    r"|\bprocedure\b|\brequirements? for\b"
    r"|\bis there a fee\b|\bcan i (apply|pay|submit|report|get|request|use|check)\b"
)

_INFORMATIONAL_ARABIC = re.compile(
    "|".join(
        re.escape(phrase)
        for phrase in (
            "كيف ابلغ",
            "كيف اقدم",
            "كيف احصل",
            "كيف اطلب",
            "كيف ادفع",
            "كيف افتح",
            "كيف استعلم",
            "كيف اتحقق",
            "كيف يقدم",
            "كيف اقدر",
            "هل يمكنني",
            "هل استطيع",
            "هل اقدر",
            "هل توجد رسوم",
            "هل هناك رسوم",
            "ما هي الرسوم",
            "كم الرسوم",
            "كم رسوم",
            "كم تبلغ",
            "كم تستغرق",
            "كم يبيلها",
            "ما المستندات",
            "ما هي المستندات",
            "ما الاوراق",
            "ما الادله",
            "ما هي الاجراءات",
            "ما الفرق",
            "ما الغرض",
            "من اين",
            "وين اسدد",
            "وين اروح",
            "ما هو الدعم",
            "شو الاوراق",
            "شو المستندات",
            "شو اسوي",
            "بكم تطلع",
            "الذي يجب تقديمه",
        )
    )
)

# --- personal-case detection -------------------------------------------------
#
# A personal-case lookup needs two things together: a reference to the reader's
# own record, and a request for a value *from* that record. Either on its own is
# ordinary. "Can I pay my traffic fines in instalments?" refers to the reader's
# own fines but asks how the service works, which published information answers.

_OWN_RECORD = re.compile(
    r"\bmy\b[^.?!]{0,30}\b(fine|fines|case|cases|complaint|certificate|report|application"
    r"|request|permit|status|ban|record|balance|file)\b"
    r"|\b(fine|fines|case|cases|ban|travel ban|record)s?\b[^.?!]{0,20}\b(on|against)\b"
    r"[^.?!]{0,15}\b(me|my name|myself)\b"
    r"|\bmy name\b"
    r"|\bdo i\b|\bam i\b|\bhave i\b"
)

_LOOKUP = re.compile(
    r"\bstatus\b|\bresult\b|\boutcome\b|\bprogress\b"
    r"|\bissued\b|\bready\b|\bapproved\b|\bprocessed\b|\bpending\b|\boutstanding\b"
    r"|\bowe\b|\bowed\b"
    r"|\bhas (my|it) been\b|\bhas my\b|\bis my\b|\bare my\b"
    r"|\bdo i (have|owe)\b|\bam i (banned|wanted|fined|blacklisted)\b"
    r"|\bhow much do i\b"
    r"|\btell me (if|whether)\b|\bis there a\b|\bcheck (my|whether|if)\b"
)

_OWN_RECORD_WORDS_ARABIC = frozenset(
    {
        "مخالفتي",
        "مخالفاتي",
        "قضيتي",
        "قضاياي",
        "شكوايي",
        "شهادتي",
        "بلاغي",
        "طلبي",
        "تصريحي",
        "سجلي",
        "رصيدي",
        "ملفي",
    }
)

_LOOKUP_PHRASES_ARABIC = (
    "حاله مخالفتي",
    "حاله طلبي",
    "هل صدرت",
    "هل جهزت",
    "هل لدي",
    "هل علي",
    "كم علي",
    "المستحق علي",
    "علي من المخالفات",
    "اسمي ممنوع",
    "منع سفر علي",
    "نتيجه طلبي",
    "ما هي حاله",
    "ما حاله",
)

_LOOKUP_ARABIC = re.compile("|".join(re.escape(phrase) for phrase in _LOOKUP_PHRASES_ARABIC))

#: Phrasings that are a personal lookup on their own, naming the reader and the
#: sought value at once — "how much do I owe". No own-record reference needed.
_PERSONAL_DIRECT = re.compile(
    r"\bhow much do i\b[^.?!]{0,20}\bowe\b"
    r"|\bdo i owe\b"
    r"|\bam i (banned|wanted|fined|blacklisted)\b"
    r"|\b(status|result|outcome|progress) of my\b"
    r"|\bhas my\b|\bhave my\b"
    r"|\btravel ban (on|against) (my name|me)\b"
)

_PERSONAL_DIRECT_ARABIC = re.compile(
    "|".join(
        re.escape(phrase)
        for phrase in (
            "المستحق علي",
            "كم علي",
            "علي من المخالفات",
            "هل لدي",
            "هل علي",
            "هل صدرت",
            "هل جهزت",
            "اسمي ممنوع",
            "منع سفر علي",
            "حاله طلبي",
            "حاله مخالفتي",
        )
    )
)

#: Procedure phrasings that rule a personal lookup out. Narrower than the
#: emergency list: "what is the status of my fine" is a lookup all the same.
_PROCEDURE = re.compile(
    r"\bhow (do|can|would|should) i\b|\bhow to\b"
    r"|\bcan i (apply|pay|submit|report|get|request|use|check|settle)\b"
    r"|\bin instal(l)?ments?\b"
    r"|\bwhat (documents|paperwork|evidence)\b"
    r"|\bwhat is the (fee|cost|procedure|difference|purpose)\b"
    r"|\bhow much (is|does|are)\b"
)


@dataclass(frozen=True)
class EmergencyResponse:
    """The fixed response to an emergency intent, in both languages."""

    text: str
    numbers: dict[str, str]


_EMERGENCY_EN_TEXT = f"""**If this is an emergency, call now:**

- **{POLICE_NUMBER}** — Police
- **{AMBULANCE_NUMBER}** — Ambulance
- **{FIRE_NUMBER}** — Civil Defence (fire)

Call rather than read. These numbers work from any telephone, and the operators
answer in Arabic and English.

For something that is not an emergency, call **{NON_EMERGENCY_NUMBER}**.

This is an unofficial demonstration project and cannot contact anybody on your behalf."""

_EMERGENCY_AR_TEXT = f"""**إذا كانت هذه حالة طارئة، اتصل الآن:**

- **{POLICE_NUMBER}** — الشرطة
- **{AMBULANCE_NUMBER}** — الإسعاف
- **{FIRE_NUMBER}** — الدفاع المدني (الحريق)

اتصل ولا تنتظر القراءة. هذه الأرقام تعمل من أي هاتف، ويجيب عليها مشغلون
بالعربية والإنجليزية.

للحالات غير الطارئة، اتصل على **{NON_EMERGENCY_NUMBER}**.

هذا مشروع تجريبي غير رسمي ولا يمكنه الاتصال بأي جهة نيابة عنك."""

EMERGENCY_NUMBERS = {
    "police": POLICE_NUMBER,
    "ambulance": AMBULANCE_NUMBER,
    "fire": FIRE_NUMBER,
    "non_emergency": NON_EMERGENCY_NUMBER,
}


def emergency_response(language: str) -> EmergencyResponse:
    """The fixed emergency response in the reader's language."""
    text = _EMERGENCY_AR_TEXT if language == "ar" else _EMERGENCY_EN_TEXT
    return EmergencyResponse(text=text, numbers=dict(EMERGENCY_NUMBERS))


def _is_informational(normalised: str) -> bool:
    return bool(_INFORMATIONAL.search(normalised) or _INFORMATIONAL_ARABIC.search(normalised))


def is_emergency(question: str) -> bool:
    """Whether the question reports something happening now.

    An informational phrasing overrides the match: asking how to report a
    robbery names one without being one.
    """
    normalised = normalise(question)
    matched = (
        _EMERGENCY_ENGLISH.search(normalised) is not None
        or _EMERGENCY_ARABIC_PHRASE.search(normalised) is not None
        or bool(set(tokenise(question)) & _EMERGENCY_WORDS_ARABIC)
    )
    return matched and not _is_informational(normalised)


def is_personal_case(question: str) -> bool:
    """Whether the question asks for a value from the reader's own record.

    Both an own-record reference and a lookup have to be present: "can I pay my
    traffic fines in instalments?" mentions the reader's fines but asks how the
    service works. Some phrasings carry both at once.
    """
    normalised = normalise(question)
    tokens = set(tokenise(question))

    if _PROCEDURE.search(normalised) or _INFORMATIONAL_ARABIC.search(normalised):
        return False

    if _PERSONAL_DIRECT.search(normalised) or _PERSONAL_DIRECT_ARABIC.search(normalised):
        return True

    own_record = _OWN_RECORD.search(normalised) is not None or bool(
        tokens & _OWN_RECORD_WORDS_ARABIC
    )
    lookup = _LOOKUP.search(normalised) is not None or _LOOKUP_ARABIC.search(normalised) is not None
    return own_record and lookup


def classify(question: str) -> Intent:
    """How a question should be handled. Emergency wins outright."""
    if is_emergency(question):
        return Intent.EMERGENCY
    if is_personal_case(question):
        return Intent.PERSONAL_CASE
    return Intent.NORMAL
