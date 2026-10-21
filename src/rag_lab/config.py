"""Read .env only when requested, without changing the process environment."""
from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 55432
    postgres_db: str = "rag_lab"
    postgres_user: str = "postgres"
    postgres_password: str = field(default="", repr=False)
    postgres_schema: str | None = None  # Internal isolation for benchmark schemas.
    ollama_host: str = "http://127.0.0.1:11434"
    chat_model: str = "custom_qwen"
    embedding_model: str = "qwen3-embedding:0.6b"
    wikipedia_contact: str = ""
    corpus_dir: Path = PROJECT_ROOT / "astronomy_corpus"

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        # Shell variables take precedence. Literal passwords are not interpolated.
        values = {**dotenv_values(env_file or PROJECT_ROOT / ".env", interpolate=False), **os.environ}
        try:
            port = int(values.get("POSTGRES_PORT") or "55432")
        except ValueError as error:
            raise ValueError("POSTGRES_PORT must be an integer.") from error
        if not 1 <= port <= 65535:
            raise ValueError("POSTGRES_PORT must be between 1 and 65535.")
        corpus_dir = Path(values.get("CORPUS_DIR") or "astronomy_corpus").expanduser()
        if not corpus_dir.is_absolute():
            corpus_dir = PROJECT_ROOT / corpus_dir
        return cls(
            postgres_host=values.get("POSTGRES_HOST") or "127.0.0.1",
            postgres_port=port,
            postgres_db=values.get("POSTGRES_DB") or "rag_lab",
            postgres_user=values.get("POSTGRES_USER") or "postgres",
            postgres_password=values.get("POSTGRES_PASSWORD") or "",
            ollama_host=values.get("OLLAMA_HOST") or "http://127.0.0.1:11434",
            chat_model=values.get("OLLAMA_CHAT_MODEL") or "custom_qwen",
            embedding_model=values.get("OLLAMA_EMBED_MODEL") or "qwen3-embedding:0.6b",
            wikipedia_contact=values.get("WIKIPEDIA_CONTACT") or "",
            corpus_dir=corpus_dir,
        )
