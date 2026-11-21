import { strings } from "../strings";
import type { Message, UiLanguage } from "../types";
import { AnswerText } from "./AnswerText";

interface Props {
  message: Message;
  uiLanguage: UiLanguage;
  /** Element id prefix of the citations panel. */
  anchorPrefix: string;
  selected: boolean;
  onShowSources: () => void;
}

const KIND_LABEL: Record<string, Record<UiLanguage, string>> = {
  emergency: { en: "Emergency", ar: "حالة طارئة" },
  personal_case: { en: "Your own records", ar: "بيانات شخصية" },
  no_context: { en: "Not in the collected information", ar: "غير موجود في المعلومات المجمَّعة" },
};

export function MessageBubble({
  message,
  uiLanguage,
  anchorPrefix,
  selected,
  onShowSources,
}: Props) {
  const text = strings(uiLanguage);
  const label = message.kind ? KIND_LABEL[message.kind] : undefined;

  return (
    <article className={`bubble bubble-${message.role}`}>
      {label && <p className={`bubble-tag tag-${message.kind ?? ""}`}>{label[uiLanguage]}</p>}

      {/* dir="auto" so a message mixing Arabic and English lays itself out. */}
      {/* While streaming, announce the text as it is appended rather than
          repeating the whole answer on every token. */}
      <div
        className="bubble-text"
        dir="auto"
        aria-live={message.streaming ? "polite" : "off"}
        aria-atomic="false"
        aria-busy={message.streaming}
      >
        {message.role === "assistant" ? (
          <AnswerText
            text={message.text}
            citations={message.citations}
            anchorPrefix={anchorPrefix}
            onCite={onShowSources}
          />
        ) : (
          message.text
        )}
        {message.streaming && <span className="caret" aria-hidden="true" />}
      </div>

      {message.error && (
        <p className="bubble-error" role="status">
          {text.failed} <span className="detail">{message.error}</span>
        </p>
      )}

      {message.role === "assistant" && message.citations.length > 0 && (
        <button
          type="button"
          className="button quiet sources-button"
          aria-pressed={selected}
          onClick={onShowSources}
        >
          {text.citationsHeading} · {message.citations.length}
        </button>
      )}
    </article>
  );
}
