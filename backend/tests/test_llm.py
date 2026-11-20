"""Tests for the Ollama client, against a mock transport rather than a provider.

The streaming behaviour is the part worth testing carefully. Reasoning leaking
into the answer was a real defect — answers opened with "Okay, the user is
asking…" and ran out of budget before saying anything — and it is invisible
unless something asserts on the separation.
"""

from __future__ import annotations

import json

import httpx
import pytest

from shurta.llm.base import GenerationOptions, ModelClient
from shurta.llm.ollama import OllamaClient, answer_text, model_matches


def _lines(*events: dict[str, object]) -> bytes:
    """Newline-delimited JSON, which is how Ollama streams."""
    return "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events).encode()


def _client(handler: httpx.MockTransport) -> OllamaClient:
    return OllamaClient(
        "http://ollama.invalid:11434",
        generation_model="test-model",
        embedding_model="test-embed",
        transport=handler,
    )


class TestInterface:
    def test_the_client_satisfies_the_provider_interface(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            return httpx.Response(200, json={})

        assert isinstance(_client(httpx.MockTransport(handler)), ModelClient)

    def test_a_trailing_slash_in_the_base_url_is_tolerated(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            return httpx.Response(200, json={})

        client = OllamaClient(
            "http://ollama.invalid:11434/",
            generation_model="m",
            embedding_model="e",
            transport=httpx.MockTransport(handler),
        )
        assert client.base_url == "http://ollama.invalid:11434"


class TestEmbedding:
    async def test_vectors_come_back_in_input_order(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={"embeddings": [[float(len(text))] for text in body["input"]]},
            )

        client = _client(httpx.MockTransport(handler))
        try:
            vectors = await client.embed(["a", "bb", "ccc"])
        finally:
            await client.close()
        assert vectors == [[1.0], [2.0], [3.0]]

    async def test_a_large_list_is_sent_in_batches_and_reassembled(self) -> None:
        batches: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            batches.append(len(body["input"]))
            return httpx.Response(200, json={"embeddings": [[1.0] for _ in body["input"]]})

        client = _client(httpx.MockTransport(handler))
        try:
            vectors = await client.embed([f"text {n}" for n in range(37)])
        finally:
            await client.close()

        assert len(vectors) == 37
        assert len(batches) > 1
        assert sum(batches) == 37

    async def test_an_empty_list_needs_no_request(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be made for an empty list")

        client = _client(httpx.MockTransport(handler))
        try:
            assert await client.embed([]) == []
        finally:
            await client.close()

    async def test_a_short_batch_is_refused_rather_than_silently_misaligned(self) -> None:
        """One vector missing would shift every later chunk onto the wrong text."""
        from shurta.llm.base import ModelError

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [[1.0]]})

        client = _client(httpx.MockTransport(handler))
        try:
            with pytest.raises(ModelError, match="one vector each"):
                await client.embed(["a", "b", "c"])
        finally:
            await client.close()


class TestGeneration:
    async def test_the_answer_is_streamed_piece_by_piece(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_lines(
                    {"response": "The fee ", "done": False},
                    {"response": "is AED 200", "done": False},
                    {"response": "", "done": True},
                ),
            )

        client = _client(httpx.MockTransport(handler))
        try:
            pieces = [piece async for piece in client.generate("prompt")]
        finally:
            await client.close()
        assert pieces == ["The fee ", "is AED 200"]

    async def test_reasoning_is_not_part_of_the_answer(self) -> None:
        """The defect this guards against produced answers made of reasoning."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_lines(
                    {"response": "", "thinking": "Okay, the user is asking about ", "done": False},
                    {"response": "", "thinking": "the fee. Let me check.", "done": False},
                    {"response": "AED 200 [1]", "done": False},
                    {"response": "", "done": True},
                ),
            )

        client = _client(httpx.MockTransport(handler))
        try:
            answer = "".join([piece async for piece in client.generate("prompt")])
        finally:
            await client.close()
        assert answer == "AED 200 [1]"
        assert "Okay" not in answer

    async def test_the_request_does_not_ask_the_provider_to_stop_separating_reasoning(
        self,
    ) -> None:
        """Asking for that is what put the reasoning into the answer field."""
        sent: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, content=_lines({"response": "ok", "done": True}))

        client = _client(httpx.MockTransport(handler))
        try:
            [piece async for piece in client.generate("prompt")]
        finally:
            await client.close()
        assert "think" not in sent[0]

    async def test_the_budget_is_passed_through(self) -> None:
        sent: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, content=_lines({"response": "ok", "done": True}))

        client = _client(httpx.MockTransport(handler))
        try:
            options = GenerationOptions(num_ctx=2048, max_tokens=128, keep_alive="5m")
            [piece async for piece in client.generate("prompt", options=options)]
        finally:
            await client.close()

        body = sent[0]
        assert body["keep_alive"] == "5m"
        assert body["options"] == {"num_ctx": 2048, "num_predict": 128, "temperature": 0.1}

    async def test_a_named_model_overrides_the_default(self) -> None:
        sent: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, content=_lines({"response": "ok", "done": True}))

        client = _client(httpx.MockTransport(handler))
        try:
            [piece async for piece in client.generate("prompt", model="other-model")]
        finally:
            await client.close()
        assert sent[0]["model"] == "other-model"

    async def test_a_malformed_line_is_skipped_rather_than_ending_the_answer(self) -> None:
        """Losing a fragment beats failing an answer that is nearly complete."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b'{"response": "first "}\nnot json at all\n{"response": "second"}\n',
            )

        client = _client(httpx.MockTransport(handler))
        try:
            answer = "".join([piece async for piece in client.generate("prompt")])
        finally:
            await client.close()
        assert answer == "first second"

    async def test_an_error_in_the_stream_is_raised(self) -> None:
        from shurta.llm.base import ModelError

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_lines({"error": "model not found"}))

        client = _client(httpx.MockTransport(handler))
        try:
            with pytest.raises(ModelError, match="model not found"):
                [piece async for piece in client.generate("prompt")]
        finally:
            await client.close()

    async def test_an_error_status_explains_itself(self) -> None:
        from shurta.llm.base import ModelError

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="model not found")

        client = _client(httpx.MockTransport(handler))
        try:
            with pytest.raises(ModelError, match="HTTP 404"):
                [piece async for piece in client.generate("prompt")]
        finally:
            await client.close()

    async def test_an_unreachable_provider_names_the_address(self) -> None:
        from shurta.llm.base import ModelError

        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = _client(httpx.MockTransport(handler))
        try:
            with pytest.raises(ModelError, match=r"ollama.invalid"):
                [piece async for piece in client.generate("prompt")]
        finally:
            await client.close()


class TestAnswerText:
    def test_it_reads_the_answer_field(self) -> None:
        assert answer_text('{"response": "hello"}') == "hello"

    def test_it_ignores_the_reasoning_field(self) -> None:
        assert answer_text('{"response": "", "thinking": "hmm"}') == ""

    def test_a_malformed_line_yields_nothing(self) -> None:
        assert answer_text("not json") == ""

    def test_a_non_object_line_yields_nothing(self) -> None:
        assert answer_text("[1, 2, 3]") == ""


class TestStatus:
    async def test_it_lists_the_models_the_provider_holds(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"models": [{"model": "qwen3:4b"}, {"model": "bge-m3:latest"}]},
            )

        client = _client(httpx.MockTransport(handler))
        try:
            status = await client.status()
        finally:
            await client.close()
        assert status.reachable is True
        assert status.available_models == ("qwen3:4b", "bge-m3:latest")

    async def test_an_unreachable_provider_is_reported_not_raised(self) -> None:
        """The health endpoint has to render something either way."""

        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = _client(httpx.MockTransport(handler))
        try:
            status = await client.status()
        finally:
            await client.close()
        assert status.reachable is False
        assert "connection refused" in status.detail

    async def test_a_malformed_model_list_does_not_crash_the_check(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": "not a list"})

        client = _client(httpx.MockTransport(handler))
        try:
            status = await client.status()
        finally:
            await client.close()
        assert status.reachable is True
        assert status.available_models == ()


class TestWarming:
    async def test_warming_asks_for_no_tokens(self) -> None:
        sent: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={"done": True})

        client = _client(httpx.MockTransport(handler))
        try:
            await client.warm()
        finally:
            await client.close()
        assert sent[0]["prompt"] == ""
        assert sent[0]["stream"] is False
        assert sent[0]["keep_alive"]

    async def test_a_named_model_can_be_warmed(self) -> None:
        sent: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={"done": True})

        client = _client(httpx.MockTransport(handler))
        try:
            await client.warm("gemma3:4b")
        finally:
            await client.close()
        assert sent[0]["model"] == "gemma3:4b"


class TestModelMatching:
    def test_an_exact_name_matches(self) -> None:
        assert model_matches(("qwen3:4b",), "qwen3:4b")

    def test_an_implied_latest_tag_matches(self) -> None:
        """``bge-m3`` and ``bge-m3:latest`` are the same model."""
        assert model_matches(("bge-m3:latest",), "bge-m3")
        assert model_matches(("bge-m3",), "bge-m3:latest")

    def test_a_different_model_does_not_match(self) -> None:
        assert not model_matches(("gemma3:4b",), "qwen3:4b")

    def test_a_different_size_does_not_match(self) -> None:
        assert not model_matches(("qwen3:8b",), "qwen3:4b")

    def test_nothing_available_matches_nothing(self) -> None:
        assert not model_matches((), "qwen3:4b")
