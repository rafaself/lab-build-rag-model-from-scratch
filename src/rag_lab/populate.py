"""Populate PostgreSQL with chunked text and Ollama embeddings."""
import hashlib
import logging
import math
from pathlib import Path

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, ForeignKeyConstraint, Integer, MetaData, Table, Text, delete, select
from sqlalchemy.dialects.postgresql import insert

from rag_lab.config import Settings
from rag_lab.database import create_db_engine
from rag_lab.ollama_client import embed_texts

logger = logging.getLogger(__name__)
metadata = MetaData()

documents = Table(
    "rag_documents", metadata,
    Column("source", Text, primary_key=True),
    Column("model", Text, primary_key=True),
    Column("content_sha256", Text, nullable=False),
    Column("chunk_size", Integer, nullable=False),
    Column("overlap", Integer, nullable=False),
    Column("dimensions", Integer, nullable=False),
    Column("chunk_count", Integer, nullable=False),
)
chunks = Table(
    "rag_chunks", metadata,
    Column("source", Text, primary_key=True),
    Column("model", Text, primary_key=True),
    Column("chunk_index", Integer, primary_key=True),
    Column("content", Text, nullable=False),
    Column("embedding", Vector(), nullable=False),
    ForeignKeyConstraint(
        ["source", "model"], ["rag_documents.source", "rag_documents.model"], ondelete="CASCADE",
    ),
)


def chunk_text(content: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split on whitespace, measuring chunk size and overlap in words."""
    if chunk_size < 1 or not 0 <= overlap < chunk_size:
        raise ValueError("Chunk size must be positive; overlap must be between 0 and chunk size - 1.")
    words = content.split()
    result = []
    for start in range(0, len(words), chunk_size - overlap):
        result.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return result


def embed_chunks(texts: list[str], settings: Settings, model: str, batch_size: int) -> list[list[float]]:
    vectors = []
    dimensions = None
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        embeddings = embed_texts(batch, settings, model=model)
        if len(embeddings) != len(batch):
            raise RuntimeError("Ollama returned a different number of vectors than input chunks.")
        for vector in embeddings:
            if not vector or (dimensions is not None and len(vector) != dimensions):
                raise RuntimeError("Ollama returned empty vectors or inconsistent vector dimensions.")
            if not all(math.isfinite(value) for value in vector) or not any(vector):
                raise RuntimeError("Ollama returned a nonfinite or zero vector.")
            dimensions = len(vector)
        vectors.extend(embeddings)
    return vectors


def populate_database(
    settings: Settings, *, input_dir: Path | None = None, model: str | None = None,
    chunk_size: int = 300, overlap: int = 50, batch_size: int = 16,
    limit: int | None = None, force: bool = False,
) -> dict:
    """Commit each document independently; embedding failures preserve its old rows.

    A source path and model identify a document. Removed files are retained in the
    database. Use force when an existing Ollama model tag has changed its weights.
    """
    chunk_text("", chunk_size, overlap)  # Validate before connecting or writing.
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("Batch size and limit must be positive.")
    directory = (input_dir if input_dir is not None else settings.corpus_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Input directory does not exist: {directory}")
    files = sorted(path for path in directory.rglob("*.txt") if path.is_file())
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError(f"No .txt files found in {directory}. Extract documents to text first.")
    model = model or settings.embedding_model
    summary = {"files": len(files), "populated": 0, "skipped": 0, "empty": 0, "chunks": 0, "model": model}
    engine = create_db_engine(settings)
    try:
        with engine.begin() as connection:
            metadata.create_all(connection)
        for path in files:
            source = str(path.resolve())
            content = path.read_text(encoding="utf-8")
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            key = (documents.c.source == source) & (documents.c.model == model)
            with engine.connect() as connection:
                previous = connection.execute(select(documents).where(key)).mappings().first()
            if not force and previous and (
                previous["content_sha256"], previous["chunk_size"], previous["overlap"]
            ) == (digest, chunk_size, overlap):
                summary["skipped"] += 1
                logger.info("Unchanged: %s", path.name)
                continue
            texts = chunk_text(content, chunk_size, overlap)
            if not texts:
                # Do not replace an earlier nonempty document with an empty file.
                summary["empty"] += 1
                logger.warning("Empty file; retaining any previous records: %s", path.name)
                continue
            logger.info("Embedding %s (%s chunks)", path.name, len(texts))
            vectors = embed_chunks(texts, settings, model, batch_size)
            row = dict(source=source, model=model, content_sha256=digest, chunk_size=chunk_size,
                       overlap=overlap, dimensions=len(vectors[0]), chunk_count=len(texts))
            with engine.begin() as connection:
                statement = insert(documents).values(**row)
                connection.execute(statement.on_conflict_do_update(
                    index_elements=[documents.c.source, documents.c.model],
                    set_={name: statement.excluded[name] for name in row if name not in ("source", "model")},
                ))
                connection.execute(delete(chunks).where((chunks.c.source == source) & (chunks.c.model == model)))
                connection.execute(chunks.insert(), [
                    dict(source=source, model=model, chunk_index=index, content=text, embedding=vector)
                    for index, (text, vector) in enumerate(zip(texts, vectors, strict=True))
                ])
            summary["populated"] += 1
            summary["chunks"] += len(texts)
            logger.info("Saved: %s", path.name)
    finally:
        engine.dispose()
    return summary


if __name__ == "__main__":
    import sys
    from rag_lab.cli import main

    raise SystemExit(main(["populate", *sys.argv[1:]]))
