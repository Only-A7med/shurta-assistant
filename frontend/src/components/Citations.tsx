import { strings } from "../strings";
import type { Citation, UiLanguage } from "../types";

interface Props {
  citations: Citation[];
  idPrefix: string;
  uiLanguage: UiLanguage;
}

const SECTION: Record<string, Record<UiLanguage, string>> = {
  description: { en: "Overview", ar: "نظرة عامة" },
  eligibility: { en: "Who can apply", ar: "من يمكنه التقديم" },
  required_documents: { en: "Required documents", ar: "المستندات المطلوبة" },
  fees: { en: "Fees", ar: "الرسوم" },
  processing_time: { en: "Processing time", ar: "مدة الإنجاز" },
  channels: { en: "Where to use it", ar: "قنوات الخدمة" },
  steps: { en: "Steps", ar: "الخطوات" },
};

export function Citations({ citations, idPrefix, uiLanguage }: Props) {
  const text = strings(uiLanguage);
  if (citations.length === 0) {
    return <p className="citations-empty">{text.noCitations}</p>;
  }

  return (
    <ol className="citations">
      {citations.map((citation) => (
        <li
          key={citation.number}
          id={`${idPrefix}-citation-${citation.number}`}
          className="citation"
          tabIndex={-1}
        >
          <span className="citation-number" aria-hidden="true">
            {citation.number}
          </span>
          <div className="citation-body" dir="auto">
            <a
              href={citation.url}
              target="_blank"
              rel="noreferrer noopener"
              title={text.sourceFor(citation.service_name)}
            >
              {citation.service_name}
            </a>
            <p className="citation-meta">
              {SECTION[citation.section]?.[uiLanguage] ?? citation.section} · {text.collectedOn}{" "}
              {citation.collected_at}
            </p>
          </div>
        </li>
      ))}
    </ol>
  );
}
