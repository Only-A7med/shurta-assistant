import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { Citation } from "../types";
import { AnswerText } from "./AnswerText";

function source(number: number): Citation {
  return {
    number,
    service_id: `service-${number}`,
    service_name: `Service ${number}`,
    section: "fees",
    language: "en",
    url: `https://example.test/${number}`,
    collected_at: "2026-01-01T00:00:00Z",
  };
}

const CITATIONS = [source(1), source(2), source(3)];

function renderAnswer(text: string) {
  return render(<AnswerText text={text} citations={CITATIONS} anchorPrefix="m1" />);
}

/** The citation badges, as the numbers they show. */
function badges(): string[] {
  return screen.queryAllByRole("link").map((link) => link.textContent ?? "");
}

describe("AnswerText", () => {
  afterEach(cleanup);

  it("renders a single citation as one badge", () => {
    const { container } = renderAnswer("The fee is AED 100 [2].");

    expect(badges()).toEqual(["2"]);
    expect(screen.getByRole("link").getAttribute("href")).toBe("#m1-citation-2");
    expect(container.textContent).toBe("The fee is AED 100 2.");
  });

  it("renders a group separated by a comma and a space as two badges", () => {
    renderAnswer("Both passages agree [1, 2].");

    expect(badges()).toEqual(["1", "2"]);
  });

  it("renders a group with no space as two badges", () => {
    renderAnswer("Both passages agree [1,2].");

    expect(badges()).toEqual(["1", "2"]);
  });

  it("renders a group separated by an Arabic comma as two badges", () => {
    renderAnswer("الرسم مذكور في المصدرين [2، 3].");

    expect(badges()).toEqual(["2", "3"]);
  });

  it("leaves an unmatched number in a group as plain text", () => {
    const { container } = renderAnswer("See [2, 9].");

    expect(badges()).toEqual(["2"]);
    expect(container.textContent).toBe("See 2[9].");
  });

  it("leaves bracketed text that is not a citation alone", () => {
    const { container } = renderAnswer("Bring the form [see the notes] and [1a] too.");

    expect(badges()).toEqual([]);
    expect(container.textContent).toBe("Bring the form [see the notes] and [1a] too.");
  });
});
