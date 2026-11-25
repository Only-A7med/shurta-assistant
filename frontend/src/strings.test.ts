import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { EXAMPLES } from "./strings";
import type { UiLanguage } from "./types";

/** The question ids the start screen offers, in the order they are shown. */
const SHOWN = ["en-001", "ar-008", "dia-003", "eme-001", "per-001"];

interface Row {
  id: string;
  question: string;
}

/** The evaluation question set, which the examples are copied from. */
function evaluationQuestions(): Map<string, string> {
  // The tests run from frontend/, beside the eval directory.
  const path = resolve(process.cwd(), "../eval/questions.jsonl");
  const rows = readFileSync(path, "utf-8")
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line) as Row);
  return new Map(rows.map((row) => [row.id, row.question]));
}

const LANGUAGES: UiLanguage[] = ["en", "ar"];

describe("EXAMPLES", () => {
  it.each(LANGUAGES)("offers the five evaluated questions in %s", (language) => {
    const asked = EXAMPLES[language].map((example) => example.question);
    const expected = SHOWN.map((id) => evaluationQuestions().get(id));

    expect(asked).toEqual(expected);
  });

  it.each(LANGUAGES)("labels every %s example by what it asks about", (language) => {
    for (const example of EXAMPLES[language]) {
      expect(example.label.trim()).not.toBe("");
    }
  });

  it("labels the English examples by subject", () => {
    expect(EXAMPLES.en.map((example) => example.label)).toEqual([
      "Fee",
      "Fee",
      "Documents",
      "Emergency",
      "My own case",
    ]);
  });

  it("translates the labels for the Arabic interface", () => {
    expect(EXAMPLES.ar.map((example) => example.label)).toEqual([
      "الرسوم",
      "الرسوم",
      "المستندات",
      "حالة طارئة",
      "قضيتي",
    ]);
  });

  it("notes the language only where it is not the interface's", () => {
    expect(EXAMPLES.en.map((example) => example.note)).toEqual([
      undefined,
      "in Arabic",
      "in Gulf dialect",
      undefined,
      undefined,
    ]);
    expect(EXAMPLES.ar.map((example) => example.note)).toEqual([
      "بالإنجليزية",
      undefined,
      "بالعامية الخليجية",
      "بالإنجليزية",
      "بالإنجليزية",
    ]);
  });

  it("keeps the guardrail examples last in both interfaces", () => {
    for (const language of LANGUAGES) {
      expect(EXAMPLES[language].map((example) => example.kind).slice(-2)).toEqual([
        "emergency",
        "personal",
      ]);
    }
  });
});
