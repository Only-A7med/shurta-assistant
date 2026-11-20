"""Building the prompt, and the fixed responses that need no model.

The prompt has one job: get a small model to restate retrieved text accurately,
in the reader's language, with a citation against each claim.

Three things follow from the model being small: few rules, ordered by
importance; numbered passages, so citing is copying rather than inventing; and
the language instruction stated twice. The system block comes first and is
byte-identical between requests in one language, so the provider can reuse its
cached prompt prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

from shurta.guardrails import NON_EMERGENCY_NUMBER
from shurta.models import Language
from shurta.retrieval.hybrid import Retrieved

OFFICIAL_SITE = "https://www.dubaipolice.gov.ae"


@dataclass(frozen=True)
class Citation:
    """One numbered source behind an answer."""

    number: int
    service_id: str
    service_name: str
    section: str
    language: str
    url: str
    collected_at: str


_SYSTEM_EN = """You answer questions about public Dubai Police services using only the \
numbered passages provided.

Rules, in order of importance:
1. Use only the passages. If they do not contain the answer, say so plainly and point the reader to \
{site}, the Dubai Police app, or {phone}. Never state a fee, a duration, or a document requirement \
that is not in the passages.
2. Answer in English.
3. Put a citation in square brackets after each fact, matching the passage number it came from, \
like [1] or [2].
4. Be brief: a short answer, or a short list. No preamble, no restating the question, no closing \
pleasantries.
5. You cannot look up anybody's own case, fine, application or record. If asked, say so and name \
the official channel instead."""

_SYSTEM_AR = """تجيب على الأسئلة حول خدمات شرطة دبي العامة باستخدام المقاطع المرقَّمة المعطاة فقط.

القواعد، مرتَّبة حسب الأهمية:
١. استخدم المقاطع فقط. إذا لم تتضمن الإجابة، فقل ذلك بوضوح وأرشد القارئ إلى {site} أو تطبيق شرطة دبي \
أو الرقم {phone}. لا تذكر أبداً رسماً أو مدةً أو مستنداً مطلوباً غير موجود في المقاطع.
٢. أجب بالعربية.
٣. ضع رقم المرجع بين قوسين مربعين بعد كل معلومة، مطابقاً لرقم المقطع الذي أخذتها منه، مثل [1] أو [2].
٤. كن موجزاً: إجابة قصيرة أو قائمة قصيرة. بلا مقدمات، وبلا إعادة صياغة السؤال، وبلا خواتيم مجامَلة.
٥. لا يمكنك الاستعلام عن قضية أي شخص أو مخالفته أو طلبه أو سجله. إذا سُئلت عن ذلك فقل ذلك وأرشد إلى \
القناة الرسمية."""

_ANSWER_IN_EN = "Answer in English, using only the passages above, with citations like [1]."
_ANSWER_IN_AR = "أجب بالعربية، باستخدام المقاطع أعلاه فقط، مع مراجع مثل [1]."

_CONTEXT_HEADING_EN = "Passages:"
_CONTEXT_HEADING_AR = "المقاطع:"

_QUESTION_HEADING_EN = "Question:"
_QUESTION_HEADING_AR = "السؤال:"


def grouped_for_prompt(retrieved: tuple[Retrieved, ...]) -> tuple[Retrieved, ...]:
    """The passages reordered so each service's sections sit together.

    The prompt then names a service once instead of once per section, and the
    citation numbers follow the same order. Services keep the order in which
    retrieval first reached them.
    """
    order: list[str] = []
    by_service: dict[str, list[Retrieved]] = {}
    for item in retrieved:
        service = item.chunk.service_id
        if service not in by_service:
            order.append(service)
            by_service[service] = []
        by_service[service].append(item)
    return tuple(item for service in order for item in by_service[service])


def service_name(item: Retrieved, language: Language) -> str:
    """The retrieved passage's service, named in the reader's language."""
    chunk = item.chunk
    if language is Language.AR and chunk.name_ar:
        return chunk.name_ar
    return chunk.name_en or chunk.name_ar


def citations_for(retrieved: tuple[Retrieved, ...], language: Language) -> list[Citation]:
    """Number the retrieved passages, which is what the answer cites."""
    return [
        Citation(
            number=position + 1,
            service_id=item.chunk.service_id,
            service_name=(
                item.chunk.name_ar
                if language is Language.AR and item.chunk.name_ar
                else item.chunk.name_en or item.chunk.name_ar
            ),
            section=item.chunk.section,
            language=item.chunk.language.value,
            url=item.chunk.source_url,
            collected_at=item.chunk.collected_at.date().isoformat(),
        )
        for position, item in enumerate(retrieved)
    ]


def trim_passage(text: str, limit: int | None) -> str:
    """One passage capped to ``limit`` characters, cut at a line or word break.

    Prompt length is most of the time to first token on processor-only
    inference, and the tail of a long fee or document list is rarely what the
    question was about.
    """
    if limit is None or len(text) <= limit:
        return text
    head = text[:limit]
    for separator in ("\n", " "):
        cut = head.rfind(separator)
        if cut > limit // 2:
            head = head[:cut]
            break
    return head.rstrip() + "\n…"


def section_text(item: Retrieved) -> str:
    """A passage's section heading and body, without the service header.

    The chunk text repeats both of its service's names and its category so that
    an Arabic question can retrieve an English passage. In the prompt that is
    redundant — the service is named once per group, in the reader's language —
    and every repeated token is paid for in time to first token.
    """
    body = item.chunk.text
    marker = body.find("## ")
    return body[marker:] if marker != -1 else body


def render_passages(
    retrieved: tuple[Retrieved, ...],
    language: Language,
    *,
    max_passage_chars: int | None = None,
) -> str:
    """The numbered passages, grouped under their service's name."""
    blocks: list[str] = []
    previous = ""
    for position, item in enumerate(retrieved, start=1):
        name = service_name(item, language)
        heading = f"{name}\n" if name != previous else ""
        previous = name
        blocks.append(
            f"{heading}[{position}] {trim_passage(section_text(item), max_passage_chars)}"
        )
    return "\n\n".join(blocks)


def build_prompt(
    question: str,
    retrieved: tuple[Retrieved, ...],
    language: Language,
    *,
    max_passage_chars: int | None = None,
) -> str:
    """Assemble the prompt for one question."""
    arabic = language is Language.AR
    system = (_SYSTEM_AR if arabic else _SYSTEM_EN).format(
        site=OFFICIAL_SITE, phone=NON_EMERGENCY_NUMBER
    )

    passages = render_passages(retrieved, language, max_passage_chars=max_passage_chars)

    return "\n\n".join(
        [
            system,
            _CONTEXT_HEADING_AR if arabic else _CONTEXT_HEADING_EN,
            passages or ("(لا توجد مقاطع)" if arabic else "(no passages)"),
            f"{_QUESTION_HEADING_AR if arabic else _QUESTION_HEADING_EN} {question}",
            _ANSWER_IN_AR if arabic else _ANSWER_IN_EN,
        ]
    )


NO_CONTEXT_EN = (
    f"I could not find this in the service information I have. Please check "
    f"{OFFICIAL_SITE}, the Dubai Police app, or call {NON_EMERGENCY_NUMBER}."
)

NO_CONTEXT_AR = (
    f"لم أجد هذا في معلومات الخدمات المتوفرة لدي. يُرجى التحقق من {OFFICIAL_SITE} "
    f"أو تطبيق شرطة دبي أو الاتصال بالرقم {NON_EMERGENCY_NUMBER}."
)

PERSONAL_CASE_EN = (
    "I cannot look up your own case, fine, application or record — I only hold published "
    "information about how the services work, and I have no access to anybody's records.\n\n"
    "To check your own:\n"
    f"- Use the Dubai Police app or {OFFICIAL_SITE}, where you can sign in and see your "
    "requests and fines.\n"
    "- Use the Application Status service with your reference number.\n"
    f"- Or call {NON_EMERGENCY_NUMBER}.\n\n"
    "I can explain how any of these services works, what it costs, or what documents it needs."
)

PERSONAL_CASE_AR = (
    "لا يمكنني الاستعلام عن قضيتك أو مخالفتك أو طلبك أو سجلك — لدي فقط معلومات منشورة عن طريقة "
    "عمل الخدمات، ولا أملك أي صلاحية للوصول إلى سجلات أحد.\n\n"
    "للاستعلام عن بياناتك:\n"
    f"- استخدم تطبيق شرطة دبي أو {OFFICIAL_SITE}، حيث يمكنك تسجيل الدخول والاطلاع على طلباتك "
    "ومخالفاتك.\n"
    "- استخدم خدمة الاستعلام عن حالة الطلب برقم الطلب المرجعي.\n"
    f"- أو اتصل بالرقم {NON_EMERGENCY_NUMBER}.\n\n"
    "ويمكنني أن أشرح لك طريقة عمل أي من هذه الخدمات أو رسومها أو المستندات المطلوبة لها."
)


def no_context_response(language: Language) -> str:
    return NO_CONTEXT_AR if language is Language.AR else NO_CONTEXT_EN


def personal_case_response(language: Language) -> str:
    return PERSONAL_CASE_AR if language is Language.AR else PERSONAL_CASE_EN
