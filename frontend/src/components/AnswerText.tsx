import type { ReactNode } from "react";

import type { Citation } from "../types";

const CITATION = /\[(\d{1,2})\]/g;

interface Props {
  text: string;
  citations: Citation[];
  /** Element id prefix of the citations panel, so a number can link into it. */
  anchorPrefix: string;
  onCite?: (number: number) => void;
}

/** The answer text, with each citation number linked to its panel entry.
 *  A number that was not supplied is left as plain text rather than linked. */
export function AnswerText({ text, citations, anchorPrefix, onCite }: Props) {
  const known = new Set(citations.map((citation) => citation.number));
  const parts: ReactNode[] = [];
  let cursor = 0;

  for (const match of text.matchAll(CITATION)) {
    const index = match.index ?? 0;
    const number = Number(match[1] ?? 0);
    if (cursor < index) parts.push(text.slice(cursor, index));
    cursor = index + match[0].length;

    if (known.has(number)) {
      parts.push(
        <a
          key={`ref-${index}`}
          className="citation-link"
          href={`#${anchorPrefix}-citation-${number}`}
          onClick={() => onCite?.(number)}
        >
          {number}
        </a>,
      );
    } else {
      parts.push(match[0]);
    }
  }
  if (cursor < text.length) parts.push(text.slice(cursor));

  return <>{parts}</>;
}
