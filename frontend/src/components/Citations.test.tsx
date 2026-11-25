import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { Citation } from "../types";
import { Citations } from "./Citations";

const SOURCE: Citation = {
  number: 1,
  service_id: "police-clearance-certificate",
  service_name: "Police Clearance Certificate",
  section: "fees",
  language: "en",
  url: "https://example.test/pcc",
  collected_at: "2026-01-01T00:00:00Z",
};

describe("Citations", () => {
  afterEach(cleanup);

  it("invites a question before the first one is asked", () => {
    render(<Citations citations={[]} idPrefix="panel" uiLanguage="en" hasAnswer={false} />);

    expect(screen.getByText("Sources appear here after you ask a question.")).toBeDefined();
  });

  it("invites a question in Arabic before the first one is asked", () => {
    render(<Citations citations={[]} idPrefix="panel" uiLanguage="ar" hasAnswer={false} />);

    expect(screen.getByText("تظهر المصادر هنا بعد أن تسأل سؤالاً.")).toBeDefined();
  });

  it("reports an answer that genuinely has no sources", () => {
    render(<Citations citations={[]} idPrefix="panel" uiLanguage="en" hasAnswer={true} />);

    expect(screen.getByText("No sources for this answer.")).toBeDefined();
  });

  it("reports an answer with no sources in Arabic", () => {
    render(<Citations citations={[]} idPrefix="panel" uiLanguage="ar" hasAnswer={true} />);

    expect(screen.getByText("لا توجد مصادر لهذه الإجابة.")).toBeDefined();
  });

  it("lists the sources an answer does have", () => {
    const { container } = render(
      <Citations citations={[SOURCE]} idPrefix="panel" uiLanguage="en" hasAnswer={true} />,
    );

    expect(screen.getByRole("link").textContent).toBe("Police Clearance Certificate");
    expect(container.querySelector(".citations-empty")).toBeNull();
  });
});
