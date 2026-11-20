import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { askQuestion, fetchHealth } from "./api";
import { Citations } from "./components/Citations";
import { Composer } from "./components/Composer";
import { MessageBubble } from "./components/MessageBubble";
import { StatusBanner } from "./components/StatusBanner";
import { EXAMPLES, strings } from "./strings";
import type { Health, Message, UiLanguage } from "./types";

/** How often the health endpoint is polled while the model is still loading. */
const WARMING_POLL_MS = 3000;

/** Element id prefix of the citations panel, which the answer links into. */
const PANEL = "panel";

function identifier(): string {
  return `m${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

export default function App() {
  const [uiLanguage, setUiLanguage] = useState<UiLanguage>("en");
  const [messages, setMessages] = useState<Message[]>([]);
  const [shownSources, setShownSources] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthLoaded, setHealthLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const inFlight = useRef<AbortController | null>(null);
  const transcript = useRef<HTMLDivElement>(null);

  const text = strings(uiLanguage);
  const direction = uiLanguage === "ar" ? "rtl" : "ltr";

  useEffect(() => {
    document.documentElement.lang = uiLanguage;
    document.documentElement.dir = direction;
  }, [uiLanguage, direction]);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    async function poll() {
      const next = await fetchHealth();
      if (cancelled) return;
      setHealth(next);
      setHealthLoaded(true);
      // Keep polling only while there is something to wait for.
      if (next === null || next.status === "warming") {
        timer = window.setTimeout(() => void poll(), WARMING_POLL_MS);
      }
    }

    void poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    transcript.current?.scrollTo({ top: transcript.current.scrollHeight });
  }, [messages]);

  const update = useCallback((id: string, change: Partial<Message>) => {
    setMessages((current) =>
      current.map((message) => (message.id === id ? { ...message, ...change } : message)),
    );
  }, []);

  const shown = useMemo(() => {
    const answers = messages.filter((message) => message.role === "assistant");
    return answers.find((message) => message.id === shownSources) ?? answers.at(-1) ?? null;
  }, [messages, shownSources]);

  const ask = useCallback(
    async (question: string) => {
      const answerId = identifier();
      setMessages((current) => [
        ...current,
        {
          id: identifier(),
          role: "user",
          text: question,
          language: uiLanguage,
          citations: [],
          streaming: false,
        },
        {
          id: answerId,
          role: "assistant",
          text: "",
          language: uiLanguage,
          citations: [],
          streaming: true,
        },
      ]);
      setShownSources(answerId);

      const controller = new AbortController();
      inFlight.current = controller;
      setBusy(true);

      await askQuestion(
        question,
        uiLanguage,
        {
          onStart: (event) => update(answerId, { kind: event.kind, citations: event.citations }),
          onToken: (piece) =>
            setMessages((current) =>
              current.map((message) =>
                message.id === answerId ? { ...message, text: message.text + piece } : message,
              ),
            ),
          onDone: (event) =>
            update(answerId, {
              text: event.text,
              citations: event.citations,
              kind: event.kind,
              streaming: false,
            }),
          onError: (detail) => update(answerId, { streaming: false, error: detail }),
        },
        controller.signal,
      );

      inFlight.current = null;
      setBusy(false);
      update(answerId, { streaming: false });
    },
    [uiLanguage, update],
  );

  function stop() {
    inFlight.current?.abort();
    inFlight.current = null;
    setBusy(false);
  }

  const waiting = busy && messages.at(-1)?.text === "";

  return (
    <div className="app" dir={direction}>
      <header className="masthead">
        <div className="brand">
          <span className="mark" aria-hidden="true" />
          <div>
            <h1>{text.title}</h1>
            <p className="tagline">{text.tagline}</p>
          </div>
        </div>
        <div className="masthead-actions">
          {health?.corpus.loaded && health.corpus.services && health.corpus.collected_at && (
            <p className="catalogue">
              {text.catalogue(health.corpus.services, health.corpus.collected_at.slice(0, 10))}
            </p>
          )}
          <button
            type="button"
            className="button"
            lang={uiLanguage === "en" ? "ar" : "en"}
            onClick={() => setUiLanguage(uiLanguage === "en" ? "ar" : "en")}
          >
            {text.languageToggle}
          </button>
        </div>
      </header>

      <StatusBanner health={health} loaded={healthLoaded} uiLanguage={uiLanguage} />

      <main className="layout">
        <section className="conversation" aria-label={text.title}>
          <div className="transcript" ref={transcript}>
            {messages.length === 0 ? (
              <div className="examples">
                <h2>{text.examplesHeading}</h2>
                <ul>
                  {EXAMPLES[uiLanguage].map((example) => (
                    <li key={example.question}>
                      <button
                        type="button"
                        className={`chip chip-${example.kind}`}
                        onClick={() => void ask(example.question)}
                        disabled={busy}
                      >
                        <span className="chip-label">{example.label}</span>
                        <span className="chip-question" dir="auto">
                          {example.question}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              messages.map((message) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                  uiLanguage={uiLanguage}
                  anchorPrefix={PANEL}
                  selected={shown?.id === message.id}
                  onShowSources={() => setShownSources(message.id)}
                />
              ))
            )}
            {waiting && (
              <p className="thinking" role="status">
                <span className="spinner" aria-hidden="true" />
                {text.thinking}
              </p>
            )}
          </div>

          <Composer
            uiLanguage={uiLanguage}
            busy={busy}
            onSubmit={(question) => void ask(question)}
            onStop={stop}
          />
          {messages.length > 0 && (
            <button
              type="button"
              className="button quiet"
              onClick={() => {
                setMessages([]);
                setShownSources(null);
              }}
            >
              {text.clear}
            </button>
          )}
        </section>

        <aside className="sources" aria-label={text.citationsHeading}>
          <h2>{text.citationsHeading}</h2>
          <Citations
            citations={shown?.citations ?? []}
            idPrefix={PANEL}
            uiLanguage={uiLanguage}
          />
          <p className="emergency-note">{text.emergencyNote}</p>
        </aside>
      </main>

      <footer className="disclaimer">
        <p>{text.disclaimer}</p>
      </footer>
    </div>
  );
}
