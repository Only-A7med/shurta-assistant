import type { UiLanguage } from "./types";

export interface Example {
  label: string;
  question: string;
  kind: "service" | "dialect" | "emergency" | "personal";
}

interface Strings {
  title: string;
  tagline: string;
  placeholder: string;
  send: string;
  stop: string;
  clear: string;
  examplesHeading: string;
  citationsHeading: string;
  noCitations: string;
  collectedOn: string;
  thinking: string;
  warming: string;
  degraded: string;
  offline: string;
  failed: string;
  retry: string;
  disclaimer: string;
  emergencyNote: string;
  languageToggle: string;
  catalogue: (services: number, collected: string) => string;
  sourceFor: (name: string) => string;
}

const EN: Strings = {
  title: "Shurta Assistant",
  tagline: "Answers about public police services, from information collected on this machine.",
  placeholder: "Ask about a service, its fee, or the documents it needs",
  send: "Send",
  stop: "Stop",
  clear: "Clear conversation",
  examplesHeading: "Try one of these",
  citationsHeading: "Sources",
  noCitations: "No sources for this answer.",
  collectedOn: "collected",
  thinking: "Looking through the service information…",
  warming: "The model is warming up. The first answer will take longer than the rest.",
  degraded: "Something the assistant needs is missing. Check the health endpoint for details.",
  offline: "The assistant is not reachable. Is the API running?",
  failed: "That answer could not be completed.",
  retry: "Try again",
  disclaimer:
    "Unofficial project, not affiliated with or endorsed by Dubai Police. Information may be out of date. In an emergency call 999.",
  emergencyNote: "In an emergency, call 999.",
  languageToggle: "العربية",
  catalogue: (services, collected) => `${services} services, collected ${collected}`,
  sourceFor: (name) => `Open the official page for ${name}`,
};

const AR: Strings = {
  title: "مساعد شُرطة",
  tagline: "إجابات عن الخدمات الشرطية العامة، من معلومات مجمَّعة على هذا الجهاز.",
  placeholder: "اسأل عن خدمة أو رسومها أو المستندات المطلوبة لها",
  send: "إرسال",
  stop: "إيقاف",
  clear: "مسح المحادثة",
  examplesHeading: "جرّب أحد هذه الأسئلة",
  citationsHeading: "المصادر",
  noCitations: "لا توجد مصادر لهذه الإجابة.",
  collectedOn: "جُمعت في",
  thinking: "يجري البحث في معلومات الخدمات…",
  warming: "يجري تحميل النموذج. ستأخذ الإجابة الأولى وقتاً أطول من غيرها.",
  degraded: "ينقص المساعد شيء يحتاجه. راجع نقطة الفحص لمعرفة التفاصيل.",
  offline: "لا يمكن الوصول إلى المساعد. هل الخدمة قيد التشغيل؟",
  failed: "لم تكتمل هذه الإجابة.",
  retry: "أعد المحاولة",
  disclaimer:
    "مشروع غير رسمي وغير تابع لشرطة دبي ولا معتمد منها. قد تكون المعلومات غير محدَّثة. في الحالات الطارئة اتصل بالرقم ٩٩٩.",
  emergencyNote: "في الحالات الطارئة اتصل بالرقم ٩٩٩.",
  languageToggle: "English",
  catalogue: (services, collected) => `${services} خدمة، جُمعت في ${collected}`,
  sourceFor: (name) => `افتح الصفحة الرسمية لخدمة ${name}`,
};

export function strings(language: UiLanguage): Strings {
  return language === "ar" ? AR : EN;
}

/** Five examples: English, Modern Standard Arabic, Gulf dialect, an emergency
 *  and a question about somebody's own case. The last two show the guardrails. */
export const EXAMPLES: Record<UiLanguage, Example[]> = {
  en: [
    {
      label: "Fee",
      question: "How much does a police clearance certificate cost for a resident?",
      kind: "service",
    },
    {
      label: "بالعربية",
      question: "ما هي المستندات المطلوبة لشهادة بحث الحالة الجنائية؟",
      kind: "service",
    },
    { label: "Gulf dialect", question: "بكم تطلع شهادة لمن يهمه الأمر؟", kind: "dialect" },
    { label: "Emergency", question: "There is a fire in my building, what do I do?", kind: "emergency" },
    { label: "My own case", question: "What is the status of my traffic fine?", kind: "personal" },
  ],
  ar: [
    { label: "الرسوم", question: "كم رسوم شهادة بحث الحالة الجنائية للمقيم؟", kind: "service" },
    {
      label: "In English",
      question: "What documents do I need for a police clearance certificate?",
      kind: "service",
    },
    { label: "بالعامية", question: "بكم تطلع شهادة لمن يهمه الأمر؟", kind: "dialect" },
    { label: "حالة طارئة", question: "فيه حريق في المبنى، شو أسوي؟", kind: "emergency" },
    { label: "قضيتي", question: "ما هي حالة مخالفتي المرورية؟", kind: "personal" },
  ],
};
