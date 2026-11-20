"""Tests for chunking, BM25, fusion, the stored index and the retriever.

Everything runs against the invented fixture catalogue and a stand-in embedder,
so none of it needs Ollama or any collected content.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shurta.arabic import tokenise
from shurta.chunking import SECTIONS, chunk_corpus, chunk_service
from shurta.models import Corpus, Language
from shurta.retrieval.bm25 import Bm25Index
from shurta.retrieval.fusion import reciprocal_rank_fusion
from shurta.retrieval.hybrid import HybridRetriever
from shurta.retrieval.store import FORMAT_VERSION, IndexError_, SearchIndex, normalise_rows


# A stand-in embedder: a bag-of-words vector over the corpus vocabulary. Texts
# sharing words land near each other, which is enough to test fusion, ranking
# and the language preference without Ollama.
class BagOfWordsEmbedder:
    def __init__(self, documents: list[str]) -> None:
        vocabulary: dict[str, int] = {}
        for document in documents:
            for token in tokenise(document):
                vocabulary.setdefault(token, len(vocabulary))
        self.vocabulary = vocabulary

    def vector(self, text: str) -> list[float]:
        values = [0.0] * max(len(self.vocabulary), 1)
        for token in tokenise(text):
            position = self.vocabulary.get(token)
            if position is not None:
                values[position] += 1.0
        return values

    async def __call__(self, text: str) -> list[float]:
        return self.vector(text)


@pytest.fixture
def index(corpus: Corpus) -> SearchIndex:
    chunks = chunk_corpus(corpus)
    embedder = BagOfWordsEmbedder([chunk.text for chunk in chunks])
    vectors = np.asarray([embedder.vector(chunk.text) for chunk in chunks], dtype=np.float32)
    return SearchIndex.create(chunks, vectors, embedding_model="bag-of-words")


@pytest.fixture
def retriever(index: SearchIndex) -> HybridRetriever:
    embedder = BagOfWordsEmbedder([chunk.text for chunk in index.chunks])
    return HybridRetriever(index, embedder)


class TestChunking:
    def test_every_chunk_is_one_section_of_one_service_in_one_language(
        self, corpus: Corpus
    ) -> None:
        chunks = chunk_corpus(corpus)
        assert len({chunk.id for chunk in chunks}) == len(chunks)
        for chunk in chunks:
            assert chunk.id == f"{chunk.service_id}::{chunk.section}::{chunk.language.value}"

    def test_empty_sections_produce_no_chunk(self, corpus: Corpus) -> None:
        """The sparse fixture service must not generate empty passages."""
        sparse = corpus.by_id()["lost-item-report"]
        sections = {chunk.section for chunk in chunk_service(sparse, Language.EN)}
        assert "fees" not in sections
        assert "required_documents" not in sections
        assert "description" in sections

    def test_no_chunk_is_blank(self, corpus: Corpus) -> None:
        for chunk in chunk_corpus(corpus):
            assert chunk.text.strip()

    def test_both_languages_are_chunked(self, corpus: Corpus) -> None:
        languages = {chunk.language for chunk in chunk_corpus(corpus)}
        assert languages == {Language.EN, Language.AR}

    def test_a_chunk_carries_both_names_so_it_can_be_found_across_languages(
        self, corpus: Corpus
    ) -> None:
        service = corpus.by_id()["good-conduct-certificate"]
        chunk = chunk_service(service, Language.EN)[0]
        assert service.name_en in chunk.text
        assert service.name_ar in chunk.text

    def test_a_chunk_carries_its_citation_and_date(self, corpus: Corpus) -> None:
        for chunk in chunk_corpus(corpus):
            assert chunk.source_url
            assert chunk.collected_at

    def test_list_sections_become_bullet_lines(self, corpus: Corpus) -> None:
        service = corpus.by_id()["good-conduct-certificate"]
        documents = next(
            chunk
            for chunk in chunk_service(service, Language.EN)
            if chunk.section == "required_documents"
        )
        assert documents.text.count("\n- ") == len(service.required_documents.en)

    def test_the_heading_is_in_the_chunk_language(self, corpus: Corpus) -> None:
        service = corpus.by_id()["good-conduct-certificate"]
        arabic = next(
            chunk for chunk in chunk_service(service, Language.AR) if chunk.section == "fees"
        )
        assert arabic.heading() == "الرسوم"
        assert "الرسوم" in arabic.text

    def test_every_section_has_a_heading_in_both_languages(self) -> None:
        for _section, english, arabic in SECTIONS:
            assert english and arabic


class TestBm25:
    def test_a_matching_document_outranks_a_non_matching_one(self) -> None:
        index = Bm25Index.build(
            ["the fee for the certificate is one hundred", "opening hours of the service centre"]
        )
        ranked = index.rank("certificate fee", limit=2)
        assert ranked[0][0] == 0

    def test_documents_matching_nothing_are_left_out(self) -> None:
        index = Bm25Index.build(["fees and charges", "working hours"])
        assert index.rank("certificate", limit=5) == []

    def test_arabic_spelling_variants_match(self) -> None:
        """This is the whole reason BM25 is implemented over the normaliser."""
        index = Bm25Index.build(["رسوم شهادة حسن السيرة", "ساعات العمل"])
        assert index.rank("شهاده", limit=2)[0][0] == 0

    def test_arabic_indic_digits_match_western_ones(self) -> None:
        index = Bm25Index.build(["الرسوم 150 درهماً", "لا توجد رسوم"])
        assert index.rank("١٥٠", limit=2)[0][0] == 0

    def test_a_term_in_every_document_carries_almost_no_weight(self) -> None:
        index = Bm25Index.build(["fee one", "fee two", "fee three"])
        assert index._inverse_document_frequency("fee") < index._inverse_document_frequency("two")

    def test_a_long_document_is_not_favoured_merely_for_its_length(self) -> None:
        short = "certificate fee"
        long = "certificate fee " + "unrelated words " * 40
        index = Bm25Index.build([short, long])
        scores = index.scores("certificate fee")
        assert scores[0] > scores[1]

    def test_an_empty_query_scores_nothing(self) -> None:
        index = Bm25Index.build(["fees", "hours"])
        assert index.scores("") == [0.0, 0.0]

    def test_an_empty_corpus_is_handled(self) -> None:
        index = Bm25Index.build([])
        assert index.size == 0
        assert index.rank("anything", limit=5) == []

    def test_it_survives_a_round_trip_through_json(self) -> None:
        original = Bm25Index.build(["رسوم الشهادة", "certificate fees"])
        restored = Bm25Index.from_dict(original.to_dict())
        assert restored.scores("شهادة") == original.scores("شهادة")


class TestFusion:
    def test_a_document_both_rankings_rate_highly_wins(self) -> None:
        fused = reciprocal_rank_fusion([[5, 1, 2], [5, 3, 4]])
        assert fused[0][0] == 5

    def test_a_document_only_one_ranking_found_still_places(self) -> None:
        fused = reciprocal_rank_fusion([[1], [2]])
        assert {document for document, _score in fused} == {1, 2}

    def test_scores_are_not_needed_only_positions(self) -> None:
        """Which is the point: BM25 and cosine scores are not comparable."""
        first = reciprocal_rank_fusion([[7, 8], [8, 7]])
        second = reciprocal_rank_fusion([[7, 8], [8, 7]])
        assert first == second

    def test_ties_break_on_the_document_index_so_results_are_stable(self) -> None:
        fused = reciprocal_rank_fusion([[2, 1], [1, 2]])
        assert [document for document, _score in fused] == [1, 2]

    def test_weights_change_the_balance(self) -> None:
        equal = reciprocal_rank_fusion([[1], [2]], weights=[1.0, 1.0])
        tilted = reciprocal_rank_fusion([[1], [2]], weights=[0.1, 1.0])
        assert equal[0][0] == 1
        assert tilted[0][0] == 2

    def test_mismatched_weights_are_refused(self) -> None:
        with pytest.raises(ValueError, match="they must match"):
            reciprocal_rank_fusion([[1], [2]], weights=[1.0])

    def test_the_limit_is_respected(self) -> None:
        assert len(reciprocal_rank_fusion([[1, 2, 3, 4]], limit=2)) == 2

    def test_no_rankings_gives_no_results(self) -> None:
        assert reciprocal_rank_fusion([]) == []


class TestStore:
    def test_vectors_are_scaled_to_unit_length(self) -> None:
        unit = normalise_rows(np.asarray([[3.0, 4.0], [1.0, 0.0]]))
        assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)

    def test_a_zero_vector_stays_zero_rather_than_becoming_nan(self) -> None:
        unit = normalise_rows(np.asarray([[0.0, 0.0]]))
        assert not np.isnan(unit).any()
        assert np.allclose(unit, 0.0)

    def test_an_index_survives_a_round_trip_to_disk(
        self, index: SearchIndex, tmp_path: Path
    ) -> None:
        index.save(tmp_path)
        restored = SearchIndex.load(tmp_path)
        assert [chunk.id for chunk in restored.chunks] == [chunk.id for chunk in index.chunks]
        assert np.allclose(restored.vectors, index.vectors)
        assert restored.metadata.embedding_model == index.metadata.embedding_model

    def test_a_missing_index_says_how_to_build_one(self, tmp_path: Path) -> None:
        with pytest.raises(IndexError_, match="python -m shurta index"):
            SearchIndex.load(tmp_path)

    def test_an_index_in_an_older_format_is_refused_rather_than_misread(
        self, index: SearchIndex, tmp_path: Path
    ) -> None:
        index.save(tmp_path)
        metadata = tmp_path / "index.json"
        metadata.write_text(
            metadata.read_text(encoding="utf-8").replace(
                f'"format_version": {FORMAT_VERSION}', '"format_version": 0'
            ),
            encoding="utf-8",
        )
        with pytest.raises(IndexError_, match="Rebuild it"):
            SearchIndex.load(tmp_path)

    def test_chunks_and_vectors_must_line_up(self, index: SearchIndex) -> None:
        with pytest.raises(IndexError_, match="cannot be matched up"):
            SearchIndex(
                chunks=index.chunks[:2],
                vectors=index.vectors,
                lexical=index.lexical,
                metadata=index.metadata,
            )

    def test_the_metadata_records_what_it_was_built_from(self, index: SearchIndex) -> None:
        assert index.metadata.chunk_count == len(index.chunks)
        assert index.metadata.dimensions == index.vectors.shape[1]
        assert index.metadata.corpus_collected_at is None or index.metadata.built_at


class TestRetriever:
    async def test_it_finds_the_right_service(self, retriever: HybridRetriever) -> None:
        result = await retriever.search(
            "What is the fee for a good conduct certificate?", top_k=4, language=Language.EN
        )
        assert result.chunks
        assert result.chunks[0].chunk.service_id == "good-conduct-certificate"

    async def test_it_prefers_a_passage_in_the_reader_s_language(
        self, retriever: HybridRetriever
    ) -> None:
        arabic = await retriever.search(
            "ما هي رسوم شهادة حسن السيرة والسلوك؟", top_k=3, language=Language.AR
        )
        assert arabic.chunks[0].chunk.language is Language.AR

    async def test_it_still_searches_the_other_language(
        self, retriever: HybridRetriever, index: SearchIndex
    ) -> None:
        """Both languages are in one index, so neither is ever excluded."""
        result = await retriever.search("good conduct certificate", top_k=len(index.chunks))
        assert {retrieved.chunk.language for retrieved in result.chunks} == {
            Language.EN,
            Language.AR,
        }

    async def test_it_records_which_method_found_each_passage(
        self, retriever: HybridRetriever
    ) -> None:
        result = await retriever.search("traffic fine instalments", top_k=4)
        assert all(
            retrieved.found_by in {"both", "keyword", "vector"} for retrieved in result.chunks
        )

    async def test_it_reports_its_timings(self, retriever: HybridRetriever) -> None:
        result = await retriever.search("lost item", top_k=2)
        assert set(result.timings) == {"embedding_ms", "search_ms", "total_ms"}
        assert result.total_ms >= 0

    async def test_top_k_is_respected(self, retriever: HybridRetriever) -> None:
        result = await retriever.search("certificate", top_k=2)
        assert len(result.chunks) == 2

    async def test_an_empty_index_returns_nothing_rather_than_failing(
        self, index: SearchIndex
    ) -> None:
        empty = SearchIndex(
            chunks=[],
            vectors=np.zeros((0, 4), dtype=np.float32),
            lexical=Bm25Index.build([]),
            metadata=index.metadata,
        )
        result = await HybridRetriever(empty, BagOfWordsEmbedder([])).search("anything")
        assert result.chunks == ()

    async def test_each_method_can_be_run_alone_for_the_comparison(
        self, retriever: HybridRetriever
    ) -> None:
        lexical = await retriever.search_lexical_only("good conduct certificate", top_k=3)
        dense = await retriever.search_dense_only("good conduct certificate", top_k=3)
        assert lexical and dense
