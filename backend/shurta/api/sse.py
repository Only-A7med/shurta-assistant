"""Server-sent event framing.

Four lines of code, kept here so it can be tested without starting a server. A
payload containing a newline has to be split across several ``data:`` lines or
the stream truncates there, and answer text is full of newlines.
"""

from __future__ import annotations

import json
from typing import Any

#: Keeps an idle connection open while a model is still loading. Every client
#: ignores a comment line.
KEEPALIVE = ": keep-alive\n\n"


def event(name: str, payload: Any) -> str:
    """One server-sent event carrying a JSON payload."""
    body = json.dumps(payload, ensure_ascii=False)
    lines = "".join(f"data: {line}\n" for line in body.split("\n"))
    return f"event: {name}\n{lines}\n"


def comment(text: str) -> str:
    """A comment line, which keeps the connection alive without being an event."""
    return f": {text}\n\n"
