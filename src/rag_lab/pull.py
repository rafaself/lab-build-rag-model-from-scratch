"""Retrieve stored text chunks by semantic similarity to a query."""
import math

from sqlalchemy import func, select

from rag_lab.config import Settings
from rag_lab.database import create_db_engine
from rag_lab.ollama_client import embed_texts
from rag_lab.populate import chunks, documents
from rag_lab.retrieval import rank_candidates


def pull_chunks(
    query: str, settings: Settings, *, top_k: int = 5, model: str | None = None,
    min_similarity: float | None = None, candidate_k: int | None = None,
    deduplicate: bool = True, rerank: bool = True,
) -> list[dict]:
    """Read the closest chunks for the same embedding model and dimensions.

    Cosine similarity is a ranking score, not a confidence or probability.
    This function neither writes database rows nor generates an answer.
    """
    if not query.strip():
        raise ValueError("Provide a nonempty search query.")
    if not 1 <= top_k <= 1000:
        raise ValueError("top_k must be between 1 and 1000.")
    candidate_k = min(1000, max(40, top_k * 4)) if candidate_k is None else candidate_k
    if not top_k <= candidate_k <= 1000:
        raise ValueError("candidate_k must be between top_k and 1000.")
    if min_similarity is not None and (not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1):
        raise ValueError("min_similarity must be a finite number between -1 and 1.")
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
        statement = select(chunks.c.source, chunks.c.chunk_index, chunks.c.content, distance,
                           chunks.c.start_char, chunks.c.end_char, chunks.c.token_count,
                           chunks.c.heading_path, documents.c.source_metadata).select_from(chunks.join(
            documents, (chunks.c.source == documents.c.source) & (chunks.c.model == documents.c.model),
        )).where(
            chunks.c.model == model,
            func.vector_dims(chunks.c.embedding) == len(vector),
        ).order_by(distance, chunks.c.source, chunks.c.chunk_index).limit(candidate_k)
        with engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        candidates = [
            {"source": row["source"], "chunk_index": row["chunk_index"],
             "similarity": 1.0 - row["distance"], "text": row["content"],
             "start_char": row["start_char"], "end_char": row["end_char"],
             "token_count": row["token_count"], "heading_path": row["heading_path"],
             "metadata": row["source_metadata"]}
            for row in rows
        ]
        return rank_candidates(candidates, query, top_k=top_k, min_similarity=min_similarity,
                               deduplicate=deduplicate, rerank=rerank)
    finally:
        engine.dispose()


if __name__ == "__main__":
    import sys
    from rag_lab.cli import main

    raise SystemExit(main(["pull", *sys.argv[1:]]))
