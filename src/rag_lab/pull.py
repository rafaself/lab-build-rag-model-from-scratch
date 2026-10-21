"""Retrieve stored text chunks by semantic similarity to a query."""
import math

from sqlalchemy import func, select

from rag_lab.config import Settings
from rag_lab.database import create_db_engine
from rag_lab.ollama_client import embed_texts
from rag_lab.populate import chunks, documents


def pull_chunks(query: str, settings: Settings, *, top_k: int = 5, model: str | None = None) -> list[dict]:
    """Read the closest chunks for the same embedding model and dimensions.

    Cosine similarity is a ranking score, not a confidence or probability.
    This function neither writes database rows nor generates an answer.
    """
    if not query.strip():
        raise ValueError("Provide a nonempty search query.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    model = model or settings.embedding_model
    engine = create_db_engine(settings)
    try:
        with engine.connect() as connection:
            dimensions = set(connection.scalars(select(documents.c.dimensions).where(
                documents.c.model == model,
            ).distinct()))
        if not dimensions:
            return []
        if len(dimensions) != 1:
            raise RuntimeError("Stored documents have mixed dimensions for this model. Re-populate with --force.")
        embeddings = embed_texts([query], settings, model=model)
        if len(embeddings) != 1 or not embeddings[0]:
            raise RuntimeError("Ollama must return exactly one nonempty query embedding.")
        vector = embeddings[0]
        if not all(math.isfinite(value) for value in vector) or not any(vector):
            raise RuntimeError("Ollama returned a nonfinite or zero query vector.")
        if len(vector) not in dimensions:
            raise RuntimeError("Query and stored embeddings have different dimensions. Check --model and re-populate with --force if the model changed.")
        distance = chunks.c.embedding.cosine_distance(vector).label("distance")
        statement = select(chunks.c.source, chunks.c.chunk_index, chunks.c.content, distance).where(
            chunks.c.model == model,
            func.vector_dims(chunks.c.embedding) == len(vector),
        ).order_by(distance, chunks.c.source, chunks.c.chunk_index).limit(top_k)
        with engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [
            {"source": row["source"], "chunk_index": row["chunk_index"],
             "similarity": 1.0 - row["distance"], "text": row["content"]}
            for row in rows
        ]
    finally:
        engine.dispose()


if __name__ == "__main__":
    import sys
    from rag_lab.cli import main

    raise SystemExit(main(["pull", *sys.argv[1:]]))
