"""``python -m shurta doctor`` — check that everything the assistant needs is there.

The point is that each failure tells the reader the command that fixes it. The
usual first run has nothing built yet, and "no search index" is only useful next
to "run ``python -m shurta index``".
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from shurta.config import Settings
from shurta.corpus import CorpusMissing, load_configured_corpus
from shurta.evaluation.dataset import KEY_FACTS_FILE, QUESTIONS_FILE
from shurta.llm.base import ModelStatus
from shurta.llm.ollama import OllamaClient, model_matches
from shurta.retrieval.store import SearchIndex

TICK = "[ ok ]"
CROSS = "[fail]"
WARN = "[warn]"


def _first_sentence(error: Exception) -> str:
    """The first sentence of an error, keeping any file extension intact.

    Splitting on a bare full stop would cut ``services.json`` in half, so the
    split is on a full stop followed by a space.
    """
    return str(error).split(". ")[0].rstrip(".")


@dataclass
class Check:
    """One thing that was checked, and what to do if it failed."""

    name: str
    passed: bool
    detail: str = ""
    remedy: str = ""
    required: bool = True

    @property
    def marker(self) -> str:
        if self.passed:
            return TICK
        return CROSS if self.required else WARN

    def lines(self) -> list[str]:
        rendered = [f"{self.marker} {self.name}" + (f": {self.detail}" if self.detail else "")]
        if not self.passed and self.remedy:
            rendered.append(f"       -> {self.remedy}")
        return rendered


def _ollama_checks(settings: Settings, status: ModelStatus) -> list[Check]:
    checks = [
        Check(
            name=f"Ollama at {settings.ollama_base_url}",
            passed=status.reachable,
            detail="reachable" if status.reachable else status.detail,
            remedy=(
                "Start Ollama, then run this again. If the API runs in Docker, set "
                "OLLAMA_BASE_URL=http://host.docker.internal:11434"
            ),
        )
    ]

    if not status.reachable:
        # Without a reachable Ollama there is nothing useful to say about models.
        return checks

    for label, wanted in (
        ("Generation model", settings.generation_model),
        ("Embedding model", settings.embedding_model),
    ):
        present = model_matches(status.available_models, wanted)
        checks.append(
            Check(
                name=f"{label} {wanted}",
                passed=present,
                detail="present" if present else "not pulled",
                remedy=f"ollama pull {wanted}",
            )
        )
    return checks


def _corpus_check(settings: Settings) -> Check:
    try:
        corpus = load_configured_corpus(settings)
    except (CorpusMissing, ValueError) as error:
        return Check(
            name="Service catalogue",
            passed=False,
            detail=_first_sentence(error),
            remedy=(
                "python -m shurta ingest  (collects the services from the official "
                "public website into data/processed; this repository ships no "
                "collected content)"
            ),
        )

    return Check(
        name="Service catalogue",
        passed=True,
        detail=(
            f"{len(corpus.services)} services, collected "
            f"{corpus.collected_at:%Y-%m-%d} -> {settings.services_file}"
        ),
    )


def _index_check(settings: Settings) -> Check:
    try:
        index = SearchIndex.load(settings.index_path)
    except RuntimeError as error:
        return Check(
            name="Search index",
            passed=False,
            detail=_first_sentence(error),
            remedy="python -m shurta index",
        )

    mismatch = (
        index.metadata.embedding_model != settings.embedding_model
        and f"{index.metadata.embedding_model}:latest" != settings.embedding_model
        and index.metadata.embedding_model != f"{settings.embedding_model}:latest"
    )
    if mismatch:
        return Check(
            name="Search index",
            passed=False,
            detail=(
                f"built with {index.metadata.embedding_model}, but the configured "
                f"embedding model is {settings.embedding_model}"
            ),
            remedy="python -m shurta index  (vectors from two models are not comparable)",
        )

    return Check(
        name="Search index",
        passed=True,
        detail=(
            f"{index.metadata.chunk_count} chunks, {index.metadata.dimensions} dimensions, "
            f"built {index.metadata.built_at:%Y-%m-%d %H:%M} UTC"
        ),
    )


def _evaluation_checks() -> list[Check]:
    checks = [
        Check(
            name="Evaluation questions",
            passed=QUESTIONS_FILE.exists(),
            detail=f"{QUESTIONS_FILE}" if QUESTIONS_FILE.exists() else "missing",
            remedy="the file is part of the repository; check the clone is complete",
        )
    ]
    present = KEY_FACTS_FILE.exists()
    checks.append(
        Check(
            name="Evaluation key facts",
            passed=present,
            detail=(
                "derived from the local catalogue"
                if present
                else "not derived yet (only needed to score answers)"
            ),
            remedy="python -m shurta eval prepare",
            required=False,
        )
    )
    return checks


def collect_checks(settings: Settings) -> list[Check]:
    """Run every check and return the results."""

    async def ollama_status() -> ModelStatus:
        client = OllamaClient(
            settings.ollama_base_url,
            generation_model=settings.generation_model,
            embedding_model=settings.embedding_model,
        )
        try:
            return await client.status()
        finally:
            await client.close()

    status = asyncio.run(ollama_status())

    return [
        *_ollama_checks(settings, status),
        _corpus_check(settings),
        _index_check(settings),
        *_evaluation_checks(),
    ]


def run_doctor(settings: Settings) -> int:
    """Print the checks and return a shell exit code."""
    checks = collect_checks(settings)

    print("Shurta Assistant — environment check\n")
    for check in checks:
        for line in check.lines():
            print(line)

    failed = [check for check in checks if not check.passed and check.required]
    warned = [check for check in checks if not check.passed and not check.required]

    print()
    if failed:
        print(f"{len(failed)} required check(s) failed. Follow the commands above.")
        return 1
    if warned:
        print(f"Everything required is in place. {len(warned)} optional step(s) not done yet.")
        return 0
    print("Everything is in place.")
    return 0
