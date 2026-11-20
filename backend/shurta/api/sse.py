"""Server-sent event framing.

Written out rather than taken from a library because the format is four lines of
code and keeping it here means it can be tested without starting a server.

The one subtlety is that a payload containing a newline must be split across
several ``data:`` lines, or the stream silently truncates at the newline. Answer
text is full of newlines — every list is one — so this is the whole reason the
framing is not an f-string at the call site.
"""

from __future__ import annotations

import json
from typing import Any

#: Sent periodically so that an idle connection is not closed by a proxy while
#: a model is still loading. A comment line is ignored by every client.
KEEPALIVE = ": keep-alive\n\n"


def event(name: str, payload: Any) -> str:
    """One server-sent event carrying a JSON payload."""
    body = json.dumps(payload, ensure_ascii=False)
    lines = "".join(f"data: {line}\n" for line in body.split("\n"))
    return f"event: {name}\n{lines}\n"


def comment(text: str) -> str:
    """A comment line, which keeps the connection alive without being an event."""
    return f": {text}\n\n"
