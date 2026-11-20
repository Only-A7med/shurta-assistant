import type { DoneEvent, Health, StartEvent, UiLanguage } from "./types";

/** The chat endpoint is a POST, so EventSource cannot be used: the stream is
 *  read from the response body and the event framing parsed here. */
export interface StreamHandlers {
  onStart: (event: StartEvent) => void;
  onToken: (text: string) => void;
  onDone: (event: DoneEvent) => void;
  onError: (detail: string) => void;
}

interface ServerEvent {
  name: string;
  data: string;
}

/** Split one server-sent event block into its name and payload. */
function parseBlock(block: string): ServerEvent | null {
  let name = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith(":")) continue;
    if (line.startsWith("event:")) name = line.slice(6).trim();
    // Exactly one optional space after the colon is part of the framing; any
    // further whitespace belongs to the payload.
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  return data.length > 0 ? { name, data: data.join("\n") } : null;
}

export async function askQuestion(
  question: string,
  language: UiLanguage | null,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(language ? { question, language } : { question }),
      ...(signal ? { signal } : {}),
    });
  } catch {
    handlers.onError("network");
    return;
  }

  if (!response.ok || !response.body) {
    handlers.onError(`http-${response.status}`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = parseBlock(block);
        if (event) dispatch(event, handlers);
        boundary = buffer.indexOf("\n\n");
      }
    }
  } catch {
    // Stopping the answer aborts the read, which is not a failure to report.
    if (!signal?.aborted) handlers.onError("stream");
  }
}

function dispatch(event: ServerEvent, handlers: StreamHandlers): void {
  switch (event.name) {
    case "start":
      handlers.onStart(JSON.parse(event.data) as StartEvent);
      break;
    case "token":
      handlers.onToken((JSON.parse(event.data) as { text: string }).text);
      break;
    case "done":
      handlers.onDone(JSON.parse(event.data) as DoneEvent);
      break;
    case "error":
      handlers.onError((JSON.parse(event.data) as { detail: string }).detail);
      break;
    default:
      break;
  }
}

export async function fetchHealth(): Promise<Health | null> {
  try {
    const response = await fetch("/api/health");
    return (await response.json()) as Health;
  } catch {
    return null;
  }
}
