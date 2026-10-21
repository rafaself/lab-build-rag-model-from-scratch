"""Measure exact versus HNSW retrieval on isolated, reproducible scaled data."""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import re
import statistics
import time
import uuid

from ollama import Client

from sqlalchemy import Text, cast, func, select, text
from sqlalchemy.schema import CreateSchema, DropSchema

from rag_lab.config import Settings
from rag_lab.database import create_db_engine
from rag_lab.evaluate import DEFAULT_DATASET, load_dataset
from rag_lab.indexing import DEFAULT_EF_SEARCH, candidate_statement, configure_search, create_hnsw_index
from rag_lab.ollama_client import embed_texts
from rag_lab.populate import chunks, documents, metadata

logger = logging.getLogger(__name__)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def identity(row) -> tuple:
    return row["source"], row["chunk_index"]


def neighbor_recall(exact: list, approximate: list, k: int) -> float:
    expected = {identity(row) for row in exact[:k]}
    return len(expected & {identity(row) for row in approximate[:k]}) / len(expected) if expected else 1.


def explain(connection, statement) -> dict:
    compiled = statement.compile(
        dialect=connection.dialect, compile_kwargs={"literal_binds": True},
        schema_translate_map=connection.get_execution_options().get("schema_translate_map"),
        render_schema_translate=True,
    )
    # literal_binds uses SQLAlchemy type quoting, never string interpolation of
    # caller-provided model/query values into executable SQL.
    result = connection.scalar(text("EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) " + str(compiled)))
    plan = result[0]
    used, schemas = [], set()

    def visit(node):
        if "Index Name" in node:
            used.append(node["Index Name"])
        if "Schema" in node:
            schemas.add(node["Schema"])
        for child in node.get("Plans", []):
            visit(child)

    visit(plan["Plan"])
    def compact(value):
        if isinstance(value, str):
            return re.sub(r"'\[[^\]]+\]'", "'[query vector omitted]'", value)
        if isinstance(value, list):
            return [compact(item) for item in value]
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items()}
        return value

    return {"execution_ms": plan["Execution Time"], "planning_ms": plan["Planning Time"],
            "indexes": used, "uses_hnsw": any(name.startswith("rag_chunks_hnsw_") for name in used),
            "schemas": sorted(schemas), "plan": compact(plan["Plan"])}


def measure(settings: Settings, queries: list[list[float]], *, mode: str, ef_search: int, repeats: int, candidate_k: int = 40) -> tuple[dict, list]:
    engine = create_db_engine(settings)
    durations, results = [], []
    try:
        with engine.begin() as connection:
            approximate = configure_search(connection, settings, settings.embedding_model, len(queries[0]),
                                           mode=mode, ef_search=ef_search, candidate_k=candidate_k)
            statements = [candidate_statement(settings.embedding_model, vector, candidate_k, exact=not approximate) for vector in queries]
            for statement in statements:
                connection.execute(statement).mappings().all()  # Unmeasured warmup.
            for _ in range(repeats):
                iteration = []
                for statement in statements:
                    start = time.perf_counter()
                    rows = connection.execute(statement).mappings().all()
                    durations.append((time.perf_counter() - start) * 1000)
                    iteration.append(rows)
                if not results:
                    results = iteration
            plan = explain(connection, statements[0])
            if settings.postgres_schema and plan["schemas"] != [settings.postgres_schema]:
                raise RuntimeError("Benchmark plan escaped its isolated schema.")
            if approximate and not plan["uses_hnsw"]:
                raise RuntimeError("Requested HNSW benchmark did not use its index; inspect the query plan.")
        return {"p50_ms": statistics.median(durations), "p95_ms": percentile(durations, .95),
                "mean_ms": statistics.mean(durations), "measured_queries": len(durations),
                "mode": mode, "ef_search": max(ef_search, candidate_k), "example_plan": plan}, results
    finally:
        engine.dispose()


def benchmark_indexes(
    settings: Settings, *, dataset: Path = DEFAULT_DATASET,
    sizes=(1000, 5000, 20000), repeats: int = 3, seed: int = 42,
) -> dict:
    if settings.postgres_schema or not sizes or any(not 40 <= n <= 100000 for n in sizes) or not 1 <= repeats <= 20:
        raise ValueError("Use the main lab schema, dataset sizes 40–100000, and repeats 1–20.")
    cases = [row for row in load_dataset(dataset) if not row["unanswerable"]]
    if not cases:
        raise ValueError("Index benchmarking requires answerable query fixtures.")
    admin = create_db_engine(settings)
    try:
        with admin.connect() as connection:
            source_docs = connection.execute(select(documents).where(documents.c.model == settings.embedding_model)).mappings().all()
            source_rows = connection.execute(select(
                chunks.c.source, chunks.c.content, cast(chunks.c.embedding, Text).label("vector"),
                chunks.c.start_char, chunks.c.end_char, chunks.c.token_count, chunks.c.heading_path,
            ).where(chunks.c.model == settings.embedding_model).order_by(chunks.c.source, chunks.c.chunk_index)).mappings().all()
            environment = {"postgres_version": connection.scalar(text("SHOW server_version")),
                           "pgvector_version": connection.scalar(text("SELECT extversion FROM pg_extension WHERE extname='vector'")),
                           "maintenance_work_mem": connection.scalar(text("SHOW maintenance_work_mem"))}
        if not source_rows:
            raise ValueError("Populate the corpus before benchmarking.")
        cohort_hash = hashlib.sha256(json.dumps(
            {"documents": [dict(row) for row in source_docs], "chunks": [dict(row) for row in source_rows]},
            sort_keys=True, ensure_ascii=False,
        ).encode()).hexdigest()
        with Client(host=settings.ollama_host) as client:
            canonical = settings.embedding_model if ":" in settings.embedding_model else settings.embedding_model + ":latest"
            model_digest = next((model.digest for model in client.list().models if model.model == canonical), None)
        bases = [{**row, "vector": json.loads(row["vector"])} for row in source_rows]
        queries = embed_texts([case["question"] for case in cases], settings)
        if len(queries) != len(cases) or any(len(vector) != len(bases[0]["vector"]) for vector in queries):
            raise RuntimeError("Benchmark query embeddings do not match the stored cohort.")
        if any(not any(vector) or not all(math.isfinite(value) for value in vector) for vector in queries):
            raise RuntimeError("Benchmark query embeddings must be finite and nonzero.")
        measured = []
        for count in sizes:
            schema = "rag_index_bench_" + uuid.uuid4().hex
            isolated = replace(settings, postgres_schema=schema)
            local = create_db_engine(isolated)
            logger.info("Index benchmark: %s rows", count)
            try:
                with admin.begin() as connection:
                    connection.execute(CreateSchema(schema))
                generator = random.Random(seed)
                frequencies = Counter(bases[i % len(bases)]["source"] for i in range(count))
                with local.begin() as connection:
                    metadata.create_all(connection)
                    connection.execute(documents.insert(), [
                        {**doc, "chunk_count": frequencies[doc["source"]]} for doc in source_docs
                    ])
                    batch = []
                    for i in range(count):
                        base = bases[i % len(bases)]
                        vector = [value + generator.uniform(-.003, .003) for value in base["vector"]]
                        batch.append({key: value for key, value in base.items() if key != "vector"} | {
                            "model": settings.embedding_model, "chunk_index": i, "embedding": vector,
                        })
                        if len(batch) == 128 or i == count - 1:
                            connection.execute(chunks.insert(), batch)
                            batch = []
                index = create_hnsw_index(isolated)
                logger.info("Built %s-row HNSW index in %.2f seconds", count, index["build_and_analyze_seconds"])
                exact, truth = measure(isolated, queries, mode="exact", ef_search=100, repeats=repeats)
                variants = []
                for mode, ef in (("auto", DEFAULT_EF_SEARCH), ("hnsw", 40), ("hnsw", 100), ("hnsw", 200)):
                    timing, neighbors = measure(isolated, queries, mode=mode, ef_search=ef, repeats=repeats)
                    timing.update({
                        "neighbor_recall@10": statistics.mean(neighbor_recall(a, b, 10) for a, b in zip(truth, neighbors, strict=True)),
                        "neighbor_recall@40": statistics.mean(neighbor_recall(a, b, 40) for a, b in zip(truth, neighbors, strict=True)),
                        "p50_speedup_over_exact": exact["p50_ms"] / timing["p50_ms"],
                    })
                    variants.append(timing)
                measured.append({"rows": count, "index": index, "exact": exact, "variants": variants})
            finally:
                local.dispose()
                with admin.begin() as connection:
                    connection.execute(DropSchema(schema, cascade=True, if_exists=True))
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "model": settings.embedding_model, "dimensions": len(queries[0]),
            "embedding_model_digest": model_digest, "source_cohort_sha256": cohort_hash,
            "original_chunks": len(bases), "queries": len(queries), "repeats": repeats,
            "environment": environment, "measurements": measured,
            "method": "Round-robin copies of stored corpus vectors/text with seeded independent uniform [-0.003,0.003] coordinate noise. Unique perturbed vectors avoid deliberate identical-vector ties. Real benchmark questions supply query vectors. Candidate pool 40, m=16, ef_construction=64; one warmup per query before timing. Latencies include query execution/fetch, exclude embedding and connection creation. Neighbor recall uses the first measured pass and strict source/chunk identity against exact search, not document Recall@K. HNSW mode and auto with a valid index use transaction-local planner hints (sequential scans/sorts discouraged, JIT off). Auto without an index falls back to exact search. Plans preserve the benchmark schema mapping; query vector literals are omitted from stored plans for readability.",
            "limitations": "Warm-cache local CPU database microbenchmark. Synthetic perturbations of the same 99 documents do not model a naturally growing corpus, concurrent workloads, cold IO, or production scale. Timing order is fixed. Strict neighbor identity recall can penalize equal-distance ties. HNSW graph construction is not seeded, so neighbors can vary between builds even with deterministic input data. Results are hardware/data dependent; indexes consume storage and increase write cost.",
        }
    finally:
        admin.dispose()
