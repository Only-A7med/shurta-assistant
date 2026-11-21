"""A deliberately slow, cached HTTP client for collecting from official sites.

It identifies itself in its user agent, waits at least two seconds between
requests however fast they come back, and caches every response in ``data/raw``
so a run resumes and developing the normalisation step costs no requests.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

USER_AGENT = (
    "ShurtaAssistantBot/0.1 (+https://github.com/Only-A7med/shurta-assistant; "
    "unofficial question-answering project; contact via the repository)"
)

MINIMUM_INTERVAL_SECONDS = 2.0

_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]+")


class CollectionError(RuntimeError):
    """Raised when a source cannot be collected."""


def cache_name(url: str) -> str:
    """A readable, stable file name for a URL.

    The readable part makes the cache directory browsable; the hash suffix keeps
    it unique once query parameters are taken into account.
    """
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    without_scheme = re.sub(r"^https?://", "", url)
    readable = _UNSAFE.sub("-", without_scheme).strip("-")[:90]
    return f"{readable}-{digest}.json"


@dataclass
class CachedResponse:
    """One collected response, as stored on disk."""

    url: str
    fetched_at: datetime
    payload: Any
    from_cache: bool = False


@dataclass
class PoliteClient:
    """Fetch JSON from official sources slowly, once, and with a record of it."""

    cache_dir: Path
    minimum_interval: float = MINIMUM_INTERVAL_SECONDS
    timeout: float = 90.0
    user_agent: str = USER_AGENT
    referer: str = ""
    transport: httpx.BaseTransport | None = None
    """Set only by the tests, so the suite never reaches the network."""

    _last_request_at: float | None = field(default=None, init=False, repr=False)
    _client: httpx.Client | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> PoliteClient:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Accept-Language": "en,ar",
        }
        if self.referer:
            headers["Referer"] = self.referer
        self._client = httpx.Client(
            headers=headers,
            timeout=self.timeout,
            follow_redirects=True,
            transport=self.transport,
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *_exception: object) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def get_json(self, url: str, *, refresh: bool = False) -> CachedResponse:
        """Return the JSON at ``url``, from the cache when it is already there.

        Raises :class:`CollectionError` if the response is not JSON: the site
        answers unsupported requests with an HTML error page under a 200 status.
        """
        path = self.cache_dir / cache_name(url)
        if path.exists() and not refresh:
            stored = json.loads(path.read_text(encoding="utf-8"))
            return CachedResponse(
                url=stored["url"],
                fetched_at=datetime.fromisoformat(stored["fetched_at"]),
                payload=stored["payload"],
                from_cache=True,
            )

        payload = self._request(url)
        fetched_at = datetime.now(UTC)
        path.write_text(
            json.dumps(
                {"url": url, "fetched_at": fetched_at.isoformat(), "payload": payload},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        return CachedResponse(url=url, fetched_at=fetched_at, payload=payload)

    def _request(self, url: str) -> Any:
        if self._client is None:
            raise CollectionError("PoliteClient must be used as a context manager")

        self._wait_turn()
        try:
            response = self._client.get(url)
        except httpx.HTTPError as error:
            raise CollectionError(f"{url} could not be reached: {error}") from error
        finally:
            self._last_request_at = time.monotonic()

        if response.status_code != httpx.codes.OK:
            raise CollectionError(f"{url} returned HTTP {response.status_code}")

        content_type = response.headers.get("content-type", "")
        if "json" not in content_type:
            # The site serves an HTML error page with a 200 status for requests
            # its edge rejects, so the content type is the only honest signal.
            raise CollectionError(
                f"{url} returned {content_type or 'an unknown content type'} rather than JSON, "
                "which means the request was rejected rather than answered"
            )

        try:
            return response.json()
        except ValueError as error:
            raise CollectionError(f"{url} returned malformed JSON: {error}") from error

    def _wait_turn(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.minimum_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
