"""Tests for the collecting client's three obligations: identify, wait, remember."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from shurta.collect.http import USER_AGENT, CollectionError, PoliteClient, cache_name


def _client(tmp_path: Path, handler: httpx.MockTransport, interval: float = 0.0) -> PoliteClient:
    return PoliteClient(cache_dir=tmp_path, minimum_interval=interval, transport=handler)


class TestCacheName:
    def test_is_stable_for_the_same_url(self) -> None:
        url = "https://example.invalid/api/services?locale=en"
        assert cache_name(url) == cache_name(url)

    def test_differs_when_the_query_differs(self) -> None:
        base = "https://example.invalid/api/services?locale="
        assert cache_name(base + "en") != cache_name(base + "ar")

    def test_is_readable_and_a_safe_file_name(self) -> None:
        name = cache_name("https://example.invalid/api/services?locale=ar")
        assert name.startswith("example.invalid-api-services")
        assert name.endswith(".json")
        assert not set(name) & set('<>:"/\\|?*')


class TestIdentifies:
    def test_the_user_agent_names_the_project_and_where_to_find_it(self) -> None:
        assert "ShurtaAssistantBot" in USER_AGENT
        assert "github.com" in USER_AGENT

    def test_every_request_carries_it(self, tmp_path: Path) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers["user-agent"])
            return httpx.Response(200, json={"data": []})

        with _client(tmp_path, httpx.MockTransport(handler)) as client:
            client.get_json("https://example.invalid/api/one")
        assert seen == [USER_AGENT]


class TestRemembers:
    def test_a_second_read_is_served_from_the_cache(self, tmp_path: Path) -> None:
        calls = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={"data": [{"name": "Invented service"}]})

        url = "https://example.invalid/api/services"
        with _client(tmp_path, httpx.MockTransport(handler)) as client:
            first = client.get_json(url)
            second = client.get_json(url)

        assert calls == 1
        assert first.from_cache is False
        assert second.from_cache is True
        assert second.payload == first.payload

    def test_the_cache_survives_a_new_client(self, tmp_path: Path) -> None:
        """This is what makes an interrupted collection resumable."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [1]})

        url = "https://example.invalid/api/services"
        with _client(tmp_path, httpx.MockTransport(handler)) as client:
            client.get_json(url)

        def refuse(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("a cached response must not be requested again")

        with _client(tmp_path, httpx.MockTransport(refuse)) as client:
            assert client.get_json(url).payload == {"data": [1]}

    def test_refresh_reads_the_source_again(self, tmp_path: Path) -> None:
        calls = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={"call": calls})

        url = "https://example.invalid/api/services"
        with _client(tmp_path, httpx.MockTransport(handler)) as client:
            client.get_json(url)
            assert client.get_json(url, refresh=True).payload == {"call": 2}

    def test_what_is_stored_records_the_url_and_the_time(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": []})

        url = "https://example.invalid/api/services"
        with _client(tmp_path, httpx.MockTransport(handler)) as client:
            client.get_json(url)

        stored = json.loads((tmp_path / cache_name(url)).read_text(encoding="utf-8"))
        assert stored["url"] == url
        assert stored["fetched_at"]
        assert stored["payload"] == {"data": []}


class TestWaits:
    def test_it_pauses_between_requests(self, tmp_path: Path) -> None:
        slept: list[float] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": []})

        client = PoliteClient(
            cache_dir=tmp_path,
            minimum_interval=2.0,
            transport=httpx.MockTransport(handler),
        )
        with client as opened:
            import shurta.collect.http as module

            original = module.time.sleep
            module.time.sleep = slept.append  # type: ignore[assignment]
            try:
                opened.get_json("https://example.invalid/api/one")
                opened.get_json("https://example.invalid/api/two")
            finally:
                module.time.sleep = original

        assert len(slept) == 1
        assert 0 < slept[0] <= 2.0

    def test_the_default_interval_is_at_least_two_seconds(self) -> None:
        assert PoliteClient(cache_dir=Path(".")).minimum_interval >= 2.0


class TestRejectsNonContent:
    def test_an_html_error_page_under_a_200_is_not_treated_as_content(self, tmp_path: Path) -> None:
        """The site answers requests its edge refuses with HTML and a 200 status."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>Error Page</html>")

        with (
            _client(tmp_path, httpx.MockTransport(handler)) as client,
            pytest.raises(CollectionError, match="rather than JSON"),
        ):
            client.get_json("https://example.invalid/api/services")

    def test_nothing_is_cached_when_the_request_is_refused(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>Error Page</html>")

        with (
            _client(tmp_path, httpx.MockTransport(handler)) as client,
            pytest.raises(CollectionError),
        ):
            client.get_json("https://example.invalid/api/services")
        assert list(tmp_path.glob("*.json")) == []

    def test_an_error_status_is_reported(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={})

        with (
            _client(tmp_path, httpx.MockTransport(handler)) as client,
            pytest.raises(CollectionError, match="HTTP 503"),
        ):
            client.get_json("https://example.invalid/api/services")

    def test_a_transport_failure_is_reported(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection reset")

        with (
            _client(tmp_path, httpx.MockTransport(handler)) as client,
            pytest.raises(CollectionError, match="could not be reached"),
        ):
            client.get_json("https://example.invalid/api/services")

    def test_using_it_outside_a_context_manager_is_refused(self, tmp_path: Path) -> None:
        client = PoliteClient(cache_dir=tmp_path)
        with pytest.raises(CollectionError, match="context manager"):
            client.get_json("https://example.invalid/api/services")
