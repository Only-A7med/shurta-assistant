"""Application settings, read from the environment and from a local ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from shurta.llm.base import GenerationOptions

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Everything configurable without editing code.

    The defaults work against an Ollama instance on the same machine, with no
    ``.env`` file at all.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    generation_model: str = "qwen3:4b"
    embedding_model: str = "bge-m3"

    # Retrieval and generation budget
    top_k: int = Field(default=4, ge=1, le=20)
    num_ctx: int = Field(default=4096, ge=512)
    max_answer_tokens: int = Field(default=512, ge=32)
    keep_alive: str = "30m"
    warm_model_on_startup: bool = True

    max_passage_chars: int | None = Field(default=None, ge=100)
    """Longest passage placed in a prompt, or ``None`` to send it whole."""

    num_thread: int | None = None
    """Generation threads, or ``None`` to leave the provider's own choice."""

    # Request timeouts, in seconds. Processor-only generation is slow to start,
    # so the generation timeout is generous.
    embedding_timeout: float = 120.0
    generation_timeout: float = 300.0

    # Paths
    data_dir: Path = Path("data")
    index_dir: Path = Path("data/index")

    # Privacy
    log_questions: bool = False

    # API
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8080"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated list, which is how a ``.env`` file carries it."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    def resolve(self, path: Path) -> Path:
        """Turn a possibly relative configured path into an absolute one.

        Relative paths are taken against the repository root, not the working
        directory, so the commands behave the same wherever they run.
        """
        return path if path.is_absolute() else REPOSITORY_ROOT / path

    def generation_options(self) -> GenerationOptions:
        """The configured generation budget."""
        return GenerationOptions(
            num_ctx=self.num_ctx,
            max_tokens=self.max_answer_tokens,
            keep_alive=self.keep_alive,
            num_thread=self.num_thread,
        )

    @property
    def services_file(self) -> Path:
        return self.resolve(self.data_dir) / "processed" / "services.json"

    @property
    def raw_dir(self) -> Path:
        return self.resolve(self.data_dir) / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.resolve(self.data_dir) / "processed"

    @property
    def index_path(self) -> Path:
        return self.resolve(self.index_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, read once."""
    return Settings()
