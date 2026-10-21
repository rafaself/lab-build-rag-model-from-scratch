"""Model-specific HNSW indexes and index-friendly nearest-neighbor queries."""
import hashlib
import time

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Index, MetaData, Table, Text, and_, cast, func, select, text

from rag_lab.config import Settings
from rag_lab.database import create_db_engine
from rag_lab.populate import chunks, documents

DEFAULT_EF_SEARCH = 200


def index_name(model: str, dimensions: int) -> str:
    digest = hashlib.sha256(f"{model}:{dimensions}".encode()).hexdigest()[:16]
    return "rag_chunks_hnsw_" + digest


def schema_name(connection, settings: Settings) -> str:
    return (connection.get_execution_options().get("schema_translate_map", {}).get(None)
            or settings.postgres_schema or connection.scalar(select(func.current_schema())))


def hnsw_definition(model: str, dimensions: int, schema: str) -> Index:
    if not 1 <= dimensions <= 2000:
        raise ValueError("HNSW vector indexes support 1–2000 dimensions; use exact search for this model.")
    # Separate metadata prevents this runtime index from leaking into future
    # create_all calls for other models or isolated benchmark schemas.
    table = Table("rag_chunks", MetaData(), Column("embedding", Vector()), Column("model", Text), schema=schema)
    return Index(
        index_name(model, dimensions), cast(table.c.embedding, Vector(dimensions)).label("embedding"),
        postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"},
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_where=and_(table.c.model == model, func.vector_dims(table.c.embedding) == dimensions),
    )


def has_hnsw_index(connection, model: str, dimensions: int, schema: str) -> bool:
    return bool(connection.scalar(text(
        "SELECT EXISTS (SELECT 1 FROM pg_class i JOIN pg_namespace n ON n.oid=i.relnamespace "
        "JOIN pg_index x ON x.indexrelid=i.oid JOIN pg_am a ON a.oid=i.relam "
        "WHERE n.nspname=:schema AND i.relname=:name AND a.amname='hnsw' AND x.indisvalid)"
    ), {"schema": schema, "name": index_name(model, dimensions)}))


def create_hnsw_index(settings: Settings, *, model: str | None = None) -> dict:
    model = model or settings.embedding_model
    engine = create_db_engine(settings)
    try:
        with engine.begin() as connection:
            dimensions = list(connection.scalars(select(func.vector_dims(chunks.c.embedding)).where(chunks.c.model == model).distinct()))
            if not dimensions:
                raise ValueError("No vectors for this model. Run populate before creating its index.")
            if len(dimensions) != 1:
                raise ValueError("Mixed vector dimensions for this model; re-populate a consistent model first.")
            dimension = dimensions[0]
            schema = schema_name(connection, settings)
            definition = hnsw_definition(model, dimension, schema)
            existed = has_hnsw_index(connection, model, dimension, schema)
            start = time.perf_counter()
            definition.create(connection, checkfirst=True)
            quoted = connection.dialect.identifier_preparer.quote_schema(schema) + ".rag_chunks"
            stats = connection.dialect.identifier_preparer.quote_schema(schema) + ".rag_chunks_model_dimensions_stats"
            connection.execute(text(
                f"CREATE STATISTICS IF NOT EXISTS {stats} (mcv, dependencies) "
                f"ON model, (vector_dims(embedding)) FROM {quoted}"
            ))
            connection.execute(text(f"ANALYZE {quoted}"))
            seconds = time.perf_counter() - start
            size = connection.scalar(text(
                "SELECT pg_relation_size(i.oid) FROM pg_class i JOIN pg_namespace n ON n.oid=i.relnamespace "
                "WHERE n.nspname=:schema AND i.relname=:name"
            ), {"schema": schema, "name": definition.name})
            return {"model": model, "dimensions": dimension, "index": definition.name,
                    "created": not existed, "build_and_analyze_seconds": seconds, "bytes": size,
                    "m": 16, "ef_construction": 64}
    finally:
        engine.dispose()


def candidate_statement(model: str, vector: list[float], limit: int, *, exact: bool = False):
    operator = cast(chunks.c.embedding, Vector(len(vector))).cosine_distance(vector)
    # + 0 deliberately prevents use of approximate indexes in exact mode, while
    # keeping primary-key lookups available for the subsequent metadata join.
    # Reuse the same selected/sorted expression so exact mode does not calculate
    # cosine distance twice per row merely to disable an approximate index.
    distance = (operator + 0 if exact else operator).label("distance")
    nearest = select(chunks.c.source, chunks.c.model, chunks.c.chunk_index, distance).where(
        chunks.c.model == model, func.vector_dims(chunks.c.embedding) == len(vector),
    ).order_by(distance).limit(limit).cte("nearest").prefix_with("MATERIALIZED", dialect="postgresql")
    return select(
        chunks.c.source, chunks.c.chunk_index, chunks.c.content, nearest.c.distance,
        chunks.c.start_char, chunks.c.end_char, chunks.c.token_count, chunks.c.heading_path,
        documents.c.source_metadata,
    ).select_from(nearest.join(chunks, and_(
        nearest.c.source == chunks.c.source, nearest.c.model == chunks.c.model,
        nearest.c.chunk_index == chunks.c.chunk_index,
    )).join(documents, and_(chunks.c.source == documents.c.source, chunks.c.model == documents.c.model))).order_by(
        nearest.c.distance, nearest.c.source, nearest.c.chunk_index,
    )


def configure_search(connection, settings: Settings, model: str, dimensions: int, *, mode: str, ef_search: int, candidate_k: int):
    exists = False
    if mode in ("auto", "hnsw"):
        exists = has_hnsw_index(connection, model, dimensions, schema_name(connection, settings))
        if mode == "hnsw" and not exists:
            raise ValueError("No valid HNSW index for this model/dimension. Run rag-lab index-db first.")
        if exists:
            connection.execute(select(func.set_config("enable_seqscan", "off", True)))
            connection.execute(select(func.set_config("enable_sort", "off", True)))
            # Avoid JIT work triggered solely by a required outer tie sort's
            # disabled-sort planner penalty.
            connection.execute(select(func.set_config("jit", "off", True)))
    if mode != "exact":
        connection.execute(select(func.set_config("hnsw.ef_search", str(max(ef_search, candidate_k)), True)))
        connection.execute(select(func.set_config("hnsw.iterative_scan", "strict_order", True)))
    return exists
