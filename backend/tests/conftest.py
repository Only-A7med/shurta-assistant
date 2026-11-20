"""Shared fixtures.

Everything here is either invented for the tests or a small synthetic payload in
the shape of the source's own, so the suite never needs network access or any
collected content.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shurta.config import Settings
from shurta.models import Corpus

FIXTURE_CORPUS = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "services.json"


@pytest.fixture
def corpus() -> Corpus:
    """The invented fixture catalogue."""
    return Corpus.model_validate_json(FIXTURE_CORPUS.read_text(encoding="utf-8"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a scratch directory, so no test touches real data."""
    return Settings(
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "data" / "index",
        warm_model_on_startup=False,
        _env_file=None,  # type: ignore[call-arg]
    )


def service_record(
    *,
    name: str,
    slug: str,
    document_id: str = "doc-1",
    components: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A service record in the shape the source publishes, with invented content."""
    return {
        "id": 1,
        "documentId": document_id,
        "name": name,
        "url": slug,
        "locale": "en",
        "dPServiceOptions": components or [],
    }


def component(kind: str, **fields: Any) -> dict[str, Any]:
    """One content component of the given kind."""
    return {"__component": f"dp-service-options.{kind}", "id": 1, **fields}
