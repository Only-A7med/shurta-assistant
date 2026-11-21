"""Tests for the service schema, the settings, corpus loading and ``doctor``."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from shurta.config import REPOSITORY_ROOT, Settings
from shurta.corpus import CorpusMissing, load_corpus
from shurta.doctor import collect_checks, run_doctor
from shurta.models import Bilingual, BilingualList, Corpus, Language, Service

COLLECTED = datetime(2026, 11, 20, tzinfo=UTC)


def _service(**overrides: object) -> Service:
    defaults: dict[str, object] = {"id": "invented", "collected_at": COLLECTED}
    defaults.update(overrides)
    return Service(**defaults)  # type: ignore[arg-type]


class TestBilingual:
    def test_it_returns_the_requested_language(self) -> None:
        value = Bilingual(en="Fees", ar="الرسوم")
        assert value.get(Language.EN) == "Fees"
        assert value.get(Language.AR) == "الرسوم"

    def test_an_empty_value_is_falsy_so_a_missing_field_is_easy_to_detect(self) -> None:
        assert not Bilingual()
        assert Bilingual(en="Fees")

    def test_one_language_present_is_still_truthy(self) -> None:
        assert Bilingual(ar="الرسوم")

    def test_a_list_behaves_the_same_way(self) -> None:
        documents = BilingualList(en=["card"], ar=["بطاقة"])
        assert documents.get(Language.AR) == ["بطاقة"]
        assert not BilingualList()


class TestService:
    def test_every_text_field_defaults_to_empty(self) -> None:
        """A field the source did not state must never be invented."""
        service = _service()
        assert service.description.en == ""
        assert service.fees.ar == ""
        assert service.required_documents.en == []
        assert service.source_urls == {}

    def test_a_name_falls_back_to_the_other_language(self) -> None:
        """Better to show the English name than nothing at all."""
        service = _service(name_en="Good Conduct Certificate")
        assert service.name(Language.AR) == "Good Conduct Certificate"

    def test_a_name_in_the_asked_language_is_preferred(self) -> None:
        service = _service(name_en="Certificate", name_ar="شهادة")
        assert service.name(Language.AR) == "شهادة"

    def test_a_citation_falls_back_to_the_other_language(self) -> None:
        service = _service(source_urls={Language.EN: "https://example.invalid/a?lng=en"})
        assert service.source_url(Language.AR).endswith("lng=en")

    def test_no_citation_at_all_gives_an_empty_string_not_an_error(self) -> None:
        assert _service().source_url(Language.EN) == ""

    def test_a_service_is_immutable(self) -> None:
        """The catalogue is read-only once collected."""
        service = _service()
        with pytest.raises(ValueError, match="frozen"):
            service.name_en = "changed"  # type: ignore[misc]


class TestCorpus:
    def test_services_can_be_looked_up_by_identifier(self) -> None:
        corpus = Corpus(
            collected_at=COLLECTED,
            services=[_service(id="one"), _service(id="two")],
        )
        assert set(corpus.by_id()) == {"one", "two"}

    def test_the_fixture_catalogue_validates(self, corpus: Corpus) -> None:
        assert len(corpus.services) >= 5
        assert corpus.source

    def test_every_fixture_service_has_both_names_and_both_citations(self, corpus: Corpus) -> None:
        for service in corpus.services:
            assert service.name_en
            assert service.name_ar
            assert service.source_url(Language.EN)
            assert service.source_url(Language.AR)


class TestCorpusLoading:
    def test_a_missing_catalogue_says_how_to_build_one(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusMissing, match="python -m shurta ingest"):
            load_corpus(tmp_path / "services.json")

    def test_a_catalogue_round_trips(self, corpus: Corpus, tmp_path: Path) -> None:
        path = tmp_path / "services.json"
        path.write_text(corpus.model_dump_json(), encoding="utf-8")
        assert len(load_corpus(path).services) == len(corpus.services)


class TestSettings:
    def test_the_defaults_work_without_any_configuration(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.embedding_model == "bge-m3"
        assert settings.top_k == 4
        assert settings.num_ctx == 4096

    def test_a_trailing_slash_on_the_provider_address_is_removed(self) -> None:
        settings = Settings(ollama_base_url="http://host:11434/", _env_file=None)  # type: ignore[call-arg]
        assert settings.ollama_base_url == "http://host:11434"

    def test_origins_can_be_given_as_a_comma_separated_list(self) -> None:
        """Which is the only way an environment file can carry a list."""
        settings = Settings(cors_origins="http://a:1, http://b:2", _env_file=None)  # type: ignore[arg-type,call-arg]
        assert settings.cors_origins == ["http://a:1", "http://b:2"]

    def test_origins_can_also_be_given_as_a_list(self) -> None:
        settings = Settings(cors_origins=["http://a:1"], _env_file=None)  # type: ignore[call-arg]
        assert settings.cors_origins == ["http://a:1"]

    def test_a_relative_path_resolves_against_the_repository_not_the_shell(self) -> None:
        """So the commands behave the same wherever they are run from."""
        settings = Settings(data_dir=Path("data"), _env_file=None)  # type: ignore[call-arg]
        assert settings.services_file.is_absolute()
        assert settings.services_file.name == "services.json"

    def test_an_absolute_path_is_left_alone(self, tmp_path: Path) -> None:
        settings = Settings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
        assert settings.processed_dir == tmp_path / "processed"

    def test_an_impossible_top_k_is_refused(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            Settings(top_k=0, _env_file=None)  # type: ignore[call-arg]

    def test_the_example_environment_file_only_names_real_settings(self) -> None:
        """Otherwise the documented example silently does nothing."""
        from shurta.config import REPOSITORY_ROOT

        example = (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8")
        named = {
            line.split("=", 1)[0].strip().casefold()
            for line in example.splitlines()
            if "=" in line and not line.strip().startswith("#")
        }
        assert named
        assert named <= set(Settings.model_fields)


class TestDoctor:
    def test_it_reports_what_is_missing_and_the_command_that_fixes_it(
        self, settings: Settings
    ) -> None:
        checks = collect_checks(settings)
        by_name = {check.name: check for check in checks}

        catalogue = next(check for name, check in by_name.items() if "catalogue" in name)
        assert catalogue.passed is False
        assert "python -m shurta ingest" in catalogue.remedy

        index = next(check for name, check in by_name.items() if "index" in name.casefold())
        assert index.passed is False
        assert "python -m shurta index" in index.remedy

    def test_a_failing_required_check_gives_a_non_zero_exit_code(
        self, settings: Settings, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert run_doctor(settings) == 1
        assert "python -m shurta ingest" in capsys.readouterr().out

    def test_the_catalogue_check_passes_when_one_is_present(
        self, settings: Settings, corpus: Corpus
    ) -> None:
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        settings.services_file.write_text(corpus.model_dump_json(), encoding="utf-8")

        catalogue = next(check for check in collect_checks(settings) if "catalogue" in check.name)
        assert catalogue.passed is True
        assert str(len(corpus.services)) in catalogue.detail

    def test_an_index_built_with_a_different_embedding_model_is_refused(
        self, settings: Settings, corpus: Corpus
    ) -> None:
        """Vectors from two models are not comparable, and the result is silent."""
        import numpy as np

        from shurta.chunking import chunk_corpus
        from shurta.retrieval.store import SearchIndex

        chunks = chunk_corpus(corpus)
        vectors = np.zeros((len(chunks), 4), dtype=np.float32)
        SearchIndex.create(chunks, vectors, embedding_model="a-different-model").save(
            settings.index_path
        )

        index = next(check for check in collect_checks(settings) if check.name == "Search index")
        assert index.passed is False
        assert "not comparable" in index.remedy

    def test_an_optional_step_warns_rather_than_failing(self, settings: Settings) -> None:
        checks = collect_checks(settings)
        optional = [check for check in checks if not check.required]
        assert optional
        assert all(check.marker in {"[ ok ]", "[warn]"} for check in optional)


class TestTheShippedDefaults:
    """What a fresh clone gets must match what the documentation promises."""

    def _env_example(self) -> dict[str, str]:
        text = (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8")
        values: dict[str, str] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, _, value = stripped.partition("=")
            values[name.strip()] = value.strip()
        return values

    def test_the_generation_model_default_matches_env_example(self) -> None:
        """The model chosen by the evaluation has to be the one that ships.

        These drifted apart once already: `.env.example` named one model and
        `config.py` another, so a clone with no `.env` ran a different model
        from the documented one.
        """
        assert Settings().generation_model == self._env_example()["GENERATION_MODEL"]

    def test_the_embedding_model_default_matches_env_example(self) -> None:
        assert Settings().embedding_model == self._env_example()["EMBEDDING_MODEL"]

    def test_the_retrieval_budget_matches_env_example(self) -> None:
        example = self._env_example()
        settings = Settings()
        assert settings.top_k == int(example["TOP_K"])
        assert settings.num_ctx == int(example["NUM_CTX"])
        assert settings.max_answer_tokens == int(example["MAX_ANSWER_TOKENS"])

    def test_no_thread_count_is_pinned_by_default(self) -> None:
        """The numbers published are the ones a fresh clone reproduces."""
        assert Settings().num_thread is None
        assert Settings().max_passage_chars is None
