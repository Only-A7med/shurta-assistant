"""Loading the service catalogue from disk."""

from __future__ import annotations

from pathlib import Path

from shurta.config import Settings
from shurta.models import Corpus


class CorpusMissing(RuntimeError):
    """Raised when no catalogue has been built yet."""


def load_corpus(path: Path) -> Corpus:
    """Read a catalogue, explaining how to produce one if it is not there."""
    if not path.exists():
        raise CorpusMissing(
            f"no service catalogue at {path}. Build one with `python -m shurta ingest`, "
            "which collects the services from the official public website."
        )
    return Corpus.model_validate_json(path.read_text(encoding="utf-8"))


def load_configured_corpus(settings: Settings) -> Corpus:
    """The catalogue the settings point at."""
    return load_corpus(settings.services_file)
