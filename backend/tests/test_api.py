"""Tests for the HTTP interface and the answer pipeline behind it.

The model client is a stand-in throughout, so none of this needs Ollama. The
index is built from the invented fixture catalogue using the stand-in's own
embeddings, which makes the whole path — question in, cited answer out —
exercisable in a few milliseconds.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import FakeModelClient
from fastapi.testclient import TestClient

from shurta.answering import SIMILARITY_FLOOR, AnswerPipeline, ResponseKind
from shurta.api.app import create_app
from shurta.api.sse import comment, event
from shurta.chunking import chunk_corpus
from shurta.config import Settings
from shurta.models import Corpus, Language
from shurta.retrieval.hybrid import HybridRetriever
from shurta.retrieval.store import SearchIndex


def _build_index(corpus: Corpus, client: FakeModelClient, directory: Path) -> SearchIndex:
    chunks = chunk_corpus(corpus)
    vectors = np.asarray([client.vector(chunk.text) for chunk in chunks], dtype=np.float32)
    index = SearchIndex.create(chunks, vectors, embedding_model="stand-in")
    index.save(directory)
    return index


def _pipeline(corpus: Corpus, client: FakeModelClient, directory: Path, *, top_k: int = 4):
    index = _build_index(corpus, client, directory)

    async def embed(query: str) -> list[float]:
        return (await client.embed([query]))[0]

    return AnswerPipeline(HybridRetriever(index, embed), client, top_k=top_k)


@pytest.fixture
def prepared(settings: Settings, corpus: Corpus, fake_client: FakeModelClient) -> Settings:
    """Settings with a corpus and an index already in place."""
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.services_file.write_text(corpus.model_dump_json(), encoding="utf-8")
    _build_index(corpus, fake_client, settings.index_path)
    return settings


@pytest.fixture
def client(prepared: Settings, fake_client: FakeModelClient) -> Iterator[TestClient]:
    """An API whose model client is the stand-in."""
    with TestClient(create_app(prepared, fake_client)) as test_client:
        yield test_client


def _events(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse a server-sent event stream back into name and payload pairs."""
    parsed: list[tuple[str, dict[str, Any]]] = []
    for block in body.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        name = ""
        data: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data.append(line.removeprefix("data: "))
        if name:
            parsed.append((name, json.loads("\n".join(data))))
    return parsed


class TestEventFraming:
    def test_an_event_carries_its_name_and_a_json_payload(self) -> None:
        framed = event("token", {"text": "hello"})
        assert framed.startswith("event: token\n")
        assert 'data: {"text": "hello"}' in framed
        assert framed.endswith("\n\n")

    def test_a_newline_in_the_payload_becomes_several_data_lines(self) -> None:
        """A single data line would truncate the answer at the first newline."""
        framed = event("done", {"text": "line one\nline two"})
        payload = json.dumps({"text": "line one\nline two"}, ensure_ascii=False)
        assert all(f"data: {part}\n" in framed for part in payload.split("\n"))

    def test_arabic_is_not_escaped(self) -> None:
        assert "الرسوم" in event("token", {"text": "الرسوم"})

    def test_a_comment_keeps_the_connection_alive_without_being_an_event(self) -> None:
        assert comment("keep-alive") == ": keep-alive\n\n"


class TestPipeline:
    async def test_an_emergency_never_reaches_the_model(
        self, corpus: Corpus, fake_client: FakeModelClient, tmp_path: Path
    ) -> None:
        pipeline = _pipeline(corpus, fake_client, tmp_path)
        answer = await pipeline.answer("Help, there is a fire in my building")
        assert answer.context.kind is ResponseKind.EMERGENCY
        assert fake_client.prompts == []
        assert "999" in answer.text

    async def test_an_arabic_emergency_answers_in_arabic(
        self, corpus: Corpus, fake_client: FakeModelClient, tmp_path: Path
    ) -> None:
        pipeline = _pipeline(corpus, fake_client, tmp_path)
        answer = await pipeline.answer("حريق في المبنى، أرجو المساعدة فوراً")
        assert answer.context.language is Language.AR
        assert "الإسعاف" in answer.text
        assert fake_client.prompts == []

    async def test_a_personal_case_never_reaches_the_model(
        self, corpus: Corpus, fake_client: FakeModelClient, tmp_path: Path
    ) -> None:
        pipeline = _pipeline(corpus, fake_client, tmp_path)
        answer = await pipeline.answer("What is the status of my traffic fine?")
        assert answer.context.kind is ResponseKind.PERSONAL_CASE
        assert fake_client.prompts == []
        assert "cannot look up" in answer.text

    async def test_a_refusal_names_an_official_channel(
        self, corpus: Corpus, fake_client: FakeModelClient, tmp_path: Path
    ) -> None:
        pipeline = _pipeline(corpus, fake_client, tmp_path)
        answer = await pipeline.answer("What is the status of my traffic fine?")
        assert "901" in answer.text

    async def test_an_ordinary_question_builds_a_prompt_with_numbered_passages(
        self, corpus: Corpus, fake_client: FakeModelClient, tmp_path: Path
    ) -> None:
        pipeline = _pipeline(corpus, fake_client, tmp_path)
        answer = await pipeline.answer("What is the fee for a good conduct certificate?")
        assert answer.context.kind is ResponseKind.ANSWER
        assert len(fake_client.prompts) == 1
        assert "[1]" in fake_client.prompts[0]

    async def test_the_prompt_tells_the_model_to_use_only_the_passages(
        self, corpus: Corpus, fake_client: FakeModelClient, tmp_path: Path
    ) -> None:
        pipeline = _pipeline(corpus, fake_client, tmp_path)
        await pipeline.answer("What is the fee for a good conduct certificate?")
        assert "Use only the passages" in fake_client.prompts[0]

    async def test_an_arabic_question_gets_an_arabic_prompt(
        self, corpus: Corpus, fake_client: FakeModelClient, tmp_path: Path
    ) -> None:
        pipeline = _pipeline(corpus, fake_client, tmp_path)
        await pipeline.answer("ما هي رسوم شهادة حسن السيرة والسلوك؟")
        assert "أجب بالعربية" in fake_client.prompts[0]

    async def test_citations_are_numbered_from_one_and_carry_a_source(
        self, corpus: Corpus, fake_client: FakeModelClient, tmp_path: Path
    ) -> None:
        pipeline = _pipeline(corpus, fake_client, tmp_path)
        context = await pipeline.prepare("What is the fee for a good conduct certificate?")
        assert [citation.number for citation in context.citations] == [1, 2, 3, 4]
        for citation in context.citations:
            assert citation.url
            assert citation.collected_at
            assert citation.service_name

    async def test_top_k_bounds_how_much_context_is_used(
        self, corpus: Corpus, fake_client: FakeModelClient, tmp_path: Path
    ) -> None:
        """Context length is the main cost on processor-only inference."""
        pipeline = _pipeline(corpus, fake_client, tmp_path, top_k=2)
        context = await pipeline.prepare("good conduct certificate fee")
        assert len(context.citations) == 2

    async def test_retrieval_timings_are_recorded(
        self, corpus: Corpus, fake_client: FakeModelClient, tmp_path: Path
    ) -> None:
        pipeline = _pipeline(corpus, fake_client, tmp_path)
        context = await pipeline.prepare("good conduct certificate fee")
        assert context.retrieval.timings["total_ms"] >= 0

    async def test_a_model_failure_part_way_is_raised_not_swallowed(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        from shurta.llm.base import ModelError

        failing = FakeModelClient("one two three four", fail_after=2)
        pipeline = _pipeline(corpus, failing, tmp_path, top_k=2)
        with pytest.raises(ModelError):
            await pipeline.answer("good conduct certificate fee")

    def test_the_similarity_floor_is_a_cosine_value(self) -> None:
        assert 0.0 < SIMILARITY_FLOOR < 1.0


class TestChatEndpoint:
    def test_it_streams_a_start_then_tokens_then_done(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={"question": "good conduct certificate fee"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        names = [name for name, _payload in _events(response.text)]
        assert names[0] == "start"
        assert names[-1] == "done"
        assert "token" in names

    def test_the_done_event_carries_the_citations_and_the_full_text(
        self, client: TestClient
    ) -> None:
        response = client.post("/api/chat", json={"question": "good conduct certificate fee"})
        done = next(payload for name, payload in _events(response.text) if name == "done")
        assert done["text"]
        assert done["citations"]
        assert done["citations"][0]["number"] == 1
        assert done["citations"][0]["url"]

    def test_the_streamed_tokens_join_to_the_final_text(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={"question": "good conduct certificate fee"})
        events = _events(response.text)
        tokens = "".join(payload["text"] for name, payload in events if name == "token")
        done = next(payload for name, payload in events if name == "done")
        assert tokens == done["text"]

    def test_the_start_event_reports_the_detected_language(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={"question": "ما هي رسوم الشهادة؟"})
        start = next(payload for name, payload in _events(response.text) if name == "start")
        assert start["detected_language"] == "ar"

    def test_the_language_can_be_overridden_by_the_caller(self, client: TestClient) -> None:
        """The interface's toggle must win over detection."""
        response = client.post("/api/chat", json={"question": "certificate fee", "language": "ar"})
        start = next(payload for name, payload in _events(response.text) if name == "start")
        assert start["language"] == "ar"
        assert start["detected_language"] == "en"

    def test_an_emergency_is_reported_as_such_with_the_numbers(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={"question": "Help, my building is on fire"})
        done = next(payload for name, payload in _events(response.text) if name == "done")
        assert done["kind"] == "emergency"
        assert done["emergency_numbers"]["police"] == "999"
        assert done["citations"] == []

    def test_a_personal_case_is_reported_as_such(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={"question": "What is the status of my fine?"})
        done = next(payload for name, payload in _events(response.text) if name == "done")
        assert done["kind"] == "personal_case"

    def test_a_model_failure_arrives_as_an_error_event_with_what_was_produced(
        self, prepared: Settings
    ) -> None:
        failing = FakeModelClient("one two three four five", fail_after=2)
        with TestClient(create_app(prepared, failing)) as test_client:
            response = test_client.post(
                "/api/chat", json={"question": "good conduct certificate fee"}
            )
        names = [name for name, _payload in _events(response.text)]
        assert names[-1] == "error"
        error = _events(response.text)[-1][1]
        assert error["detail"]
        assert error["partial"] == "one two"

    def test_an_empty_question_is_rejected(self, client: TestClient) -> None:
        assert client.post("/api/chat", json={"question": ""}).status_code == 422

    def test_an_oversized_question_is_rejected(self, client: TestClient) -> None:
        assert client.post("/api/chat", json={"question": "x" * 5000}).status_code == 422


class TestHealthEndpoint:
    def test_it_reports_every_part_when_all_are_present(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["ollama"]["reachable"] is True
        assert body["models"]["generation"]["present"] is True
        assert body["models"]["embedding"]["present"] is True
        assert body["index"]["loaded"] is True
        assert body["index"]["chunks"] > 0
        assert body["corpus"]["services"] == 6

    def test_a_missing_model_makes_it_degraded(self, prepared: Settings) -> None:
        without_generation = FakeModelClient(available=("bge-m3:latest",))
        with TestClient(create_app(prepared, without_generation)) as test_client:
            body = test_client.get("/api/health").json()
        assert body["models"]["generation"]["present"] is False
        assert body["status"] == "degraded"

    def test_an_unreachable_provider_is_reported_rather_than_crashing(
        self, prepared: Settings
    ) -> None:
        stopped = FakeModelClient(reachable=False)
        with TestClient(create_app(prepared, stopped)) as test_client:
            body = test_client.get("/api/health").json()
        assert body["ollama"]["reachable"] is False
        assert body["ollama"]["detail"]
        assert body["status"] == "degraded"

    def test_the_model_tag_is_not_required_to_match_exactly(self, client: TestClient) -> None:
        """``bge-m3`` and ``bge-m3:latest`` are the same model."""
        body = client.get("/api/health").json()
        assert body["models"]["embedding"]["present"] is True


class TestHealthWithoutAnIndex:
    def test_it_says_what_is_missing_and_how_to_build_it(
        self, settings: Settings, fake_client: FakeModelClient
    ) -> None:
        with TestClient(create_app(settings, fake_client)) as test_client:
            body = test_client.get("/api/health").json()
        assert body["status"] == "degraded"
        assert body["index"]["loaded"] is False
        assert "python -m shurta index" in body["index"]["detail"]

    def test_it_says_how_to_build_the_corpus(
        self, settings: Settings, fake_client: FakeModelClient
    ) -> None:
        with TestClient(create_app(settings, fake_client)) as test_client:
            body = test_client.get("/api/health").json()
        assert "python -m shurta ingest" in body["corpus"]["detail"]

    def test_chat_explains_itself_rather_than_failing(
        self, settings: Settings, fake_client: FakeModelClient
    ) -> None:
        with TestClient(create_app(settings, fake_client)) as test_client:
            response = test_client.post("/api/chat", json={"question": "anything"})
        name, payload = _events(response.text)[0]
        assert name == "error"
        assert "python -m shurta index" in payload["detail"]


class TestServicesEndpoint:
    def test_it_lists_the_catalogue_with_both_names_and_sources(self, client: TestClient) -> None:
        body = client.get("/api/services").json()
        assert body["count"] == 6
        first = body["services"][0]
        assert first["name_en"]
        assert first["name_ar"]
        assert first["url_en"].endswith("lng=en")
        assert first["collected_at"]

    def test_it_says_which_services_have_fees_and_documents(self, client: TestClient) -> None:
        body = client.get("/api/services").json()
        sparse = next(item for item in body["services"] if item["id"] == "lost-item-report")
        assert sparse["has_fees"] is False
        assert sparse["has_required_documents"] is False

    def test_it_reports_when_the_corpus_has_not_been_built(
        self, settings: Settings, fake_client: FakeModelClient
    ) -> None:
        with TestClient(create_app(settings, fake_client)) as test_client:
            response = test_client.get("/api/services")
        assert response.status_code == 503
        assert "python -m shurta ingest" in response.json()["detail"]


class TestPrivacy:
    def test_the_question_is_not_echoed_back_in_the_response(self, client: TestClient) -> None:
        """Nothing retains the question; it is not even reflected to the caller."""
        question = "a distinctive phrase about certificates"
        response = client.post("/api/chat", json={"question": question})
        for name, payload in _events(response.text):
            assert question not in json.dumps(payload), f"the {name} event echoed the question"

    def test_logging_questions_is_off_by_default(self) -> None:
        assert Settings(_env_file=None).log_questions is False  # type: ignore[call-arg]


class TestStartUp:
    def test_the_model_is_warmed_when_configured(
        self, prepared: Settings, fake_client: FakeModelClient
    ) -> None:
        warming = prepared.model_copy(update={"warm_model_on_startup": True})
        with TestClient(create_app(warming, fake_client)):
            pass
        assert fake_client.warm_calls == 1

    def test_warming_is_skipped_when_turned_off(
        self, prepared: Settings, fake_client: FakeModelClient
    ) -> None:
        not_warming = prepared.model_copy(update={"warm_model_on_startup": False})
        with TestClient(create_app(not_warming, fake_client)):
            pass
        assert fake_client.warm_calls == 0

    def test_a_failure_to_warm_does_not_stop_the_api_starting(self, prepared: Settings) -> None:
        """A reader whose provider is stopped should still get a health page."""

        class RefusingClient(FakeModelClient):
            async def warm(self, model: str | None = None) -> None:
                raise RuntimeError("connection refused")

        warming = prepared.model_copy(update={"warm_model_on_startup": True})
        with TestClient(create_app(warming, RefusingClient())) as test_client:
            body = test_client.get("/api/health").json()
        assert body["model_warmed"] is False
        assert "connection refused" in body["warm_detail"]
