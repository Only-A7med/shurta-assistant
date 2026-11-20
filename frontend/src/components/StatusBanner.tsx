import { strings } from "../strings";
import type { Health, UiLanguage } from "../types";

interface Props {
  health: Health | null;
  loaded: boolean;
  uiLanguage: UiLanguage;
}

/** The one place the reader is told the assistant is not ready, and why. */
export function StatusBanner({ health, loaded, uiLanguage }: Props) {
  const text = strings(uiLanguage);
  if (!loaded) return null;

  if (health === null) {
    return (
      <p className="banner banner-error" role="status">
        {text.offline}
      </p>
    );
  }
  if (health.status === "warming") {
    return (
      <p className="banner banner-warming" role="status">
        <span className="spinner" aria-hidden="true" />
        {text.warming}
      </p>
    );
  }
  if (health.status === "degraded") {
    const detail = health.index.detail ?? health.ollama.detail;
    return (
      <p className="banner banner-error" role="status">
        {text.degraded} {detail && <span className="detail">{detail}</span>}
      </p>
    );
  }
  return null;
}
