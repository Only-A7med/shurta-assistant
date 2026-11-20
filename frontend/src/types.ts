export type UiLanguage = "en" | "ar";

/** How the backend handled a question. */
export type ResponseKind = "answer" | "emergency" | "personal_case" | "no_context";

export interface Citation {
  number: number;
  service_id: string;
  service_name: string;
  section: string;
  language: string;
  url: string;
  collected_at: string;
}

export interface StartEvent {
  kind: ResponseKind;
  language: UiLanguage;
  detected_language: UiLanguage;
  citations: Citation[];
}

export interface DoneEvent {
  kind: ResponseKind;
  language: UiLanguage;
  text: string;
  citations: Citation[];
  emergency_numbers: Record<string, string>;
  best_similarity: number;
  prompt_tokens: number;
  prompt_ms: number;
  model: string | null;
}

export interface Health {
  status: "ok" | "warming" | "degraded";
  warm_state: "off" | "warming" | "ready" | "failed";
  index: { loaded: boolean; chunks?: number; detail?: string };
  corpus: { loaded: boolean; services?: number; collected_at?: string; detail?: string };
  models: Record<string, { name: string; present: boolean }>;
  ollama: { reachable: boolean; detail: string };
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  language: UiLanguage;
  kind?: ResponseKind;
  citations: Citation[];
  streaming: boolean;
  error?: string;
}
