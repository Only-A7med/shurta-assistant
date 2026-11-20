import { useRef } from "react";

import { strings } from "../strings";
import type { UiLanguage } from "../types";

interface Props {
  uiLanguage: UiLanguage;
  busy: boolean;
  onSubmit: (question: string) => void;
  onStop: () => void;
}

export function Composer({ uiLanguage, busy, onSubmit, onStop }: Props) {
  const text = strings(uiLanguage);
  const field = useRef<HTMLTextAreaElement>(null);

  function submit() {
    const value = field.current?.value.trim();
    if (!value || busy) return;
    onSubmit(value);
    if (field.current) field.current.value = "";
  }

  return (
    <form
      className="composer"
      onSubmit={(fired) => {
        fired.preventDefault();
        submit();
      }}
    >
      <label className="visually-hidden" htmlFor="question">
        {text.placeholder}
      </label>
      <textarea
        id="question"
        ref={field}
        rows={2}
        dir="auto"
        placeholder={text.placeholder}
        maxLength={2000}
        onKeyDown={(pressed) => {
          // Enter sends; Shift+Enter starts a new line.
          if (pressed.key === "Enter" && !pressed.shiftKey) {
            pressed.preventDefault();
            submit();
          }
        }}
      />
      {busy ? (
        <button type="button" className="button primary" onClick={onStop}>
          {text.stop}
        </button>
      ) : (
        <button type="submit" className="button primary">
          {text.send}
        </button>
      )}
    </form>
  );
}
