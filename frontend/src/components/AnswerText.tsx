import type { ReactNode } from "react";

import type { Citation } from "../types";

/** One number, or several grouped in a single marker such as `[1, 2]`.
 *  The separator is an ASCII or an Arabic comma, with optional spaces. */
const CITATION = /\[\s*\d{1,2}(?:\s*[,،]\s*\d{1,2})*\s*\]/g;

const SEPARATOR = /[,،]/;

interface Props {
  text: string;
  citations: Citation[];
  /** Element id prefix of the citations panel, so a number can link into it. */
  anchorPrefix: string;
  onCite?: (number: number) => void;
}

/** The numbers carried by one marker, in the order they were written. */
function markerNumbers(marker: string): number[] {
  return marker
    .slice(1, -1)
    .split(SEPARATOR)
    .map((part) => Number(part.trim()));
}

/** The answer text, with each citation number linked to its panel entry.
 *  A grouped marker becomes one link per number. A number that was not
 *  supplied is left as plain text rather than linked. */
export function AnswerText({ text, citations, anchorPrefix, onCite }: Props) {
  const known = new Set(citations.map((citation) => citation.number));
  const parts: ReactNode[] = [];
  let cursor = 0;

  for (const match of text.matchAll(CITATION)) {
    const index = match.index ?? 0;
    if (cursor < index) parts.push(text.slice(cursor, index));
    cursor = index + match[0].length;

    for (const number of markerNumbers(match[0])) {
      if (known.has(number)) {
        parts.push(
          <a
            key={`ref-${index}-${number}`}
            className="citation-link"
            href={`#${anchorPrefix}-citation-${number}`}
            onClick={() => onCite?.(number)}
          >
            {number}
          </a>,
        );
      } else {
        parts.push(`[${number}]`);
      }
    }
  }
  if (cursor < text.length) parts.push(text.slice(cursor));

  return <>{parts}</>;
}
