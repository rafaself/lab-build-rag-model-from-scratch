"""Compare token chunk sizes in isolated schemas using real Ollama embeddings."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
import uuid

from sqlalchemy import select
from sqlalchemy.schema import CreateSchema, DropSchema

from rag_lab.config import Settings
from rag_lab.database import create_db_engine
from rag_lab.evaluate import DEFAULT_DATASET, evaluate_retrieval, load_dataset
from rag_lab.populate import ensure_population_schema, populate_database
from rag_lab.chunking import TOKENIZER_REPO, QWEN_REVISION, token_chunks


def tune_chunking(settings: Settings, *, dataset: Path = DEFAULT_DATASET, variants=((256, 32), (384, 64), (512, 64))) -> dict:
    """Do not replace the user's indexed corpus while comparing settings."""
    if settings.postgres_schema:
        raise ValueError("Run chunk tuning from the default lab schema.")
    if not variants or not any(not row["unanswerable"] for row in load_dataset(dataset)):
        raise ValueError("Chunk tuning needs variants and at least one answerable benchmark case.")
    for size, overlap in variants:
        token_chunks("", size, overlap)
    from rag_lab.populate import documents

    engine = create_db_engine(settings)
    try:
        ensure_population_schema(engine)
        with engine.connect() as connection:
            baseline_configs = connection.execute(select(documents.c.chunk_size, documents.c.overlap, documents.c.chunking_config).where(
                documents.c.model == settings.embedding_model,
            ).distinct()).mappings().all()
    finally:
        engine.dispose()
    baseline = evaluate_retrieval(settings, dataset=dataset)
    measured = []
    for chunk_size, overlap in variants:
        schema = "rag_chunk_tune_" + uuid.uuid4().hex
        isolated = replace(settings, postgres_schema=schema)
        admin = create_db_engine(settings)
        try:
            with admin.begin() as connection:
                connection.execute(CreateSchema(schema))
            start = time.perf_counter()
            populated = populate_database(isolated, strategy="tokens", chunk_size=chunk_size, overlap=overlap)
            ingest_seconds = time.perf_counter() - start
            evaluation = evaluate_retrieval(isolated, dataset=dataset)
            measured.append({"chunk_size": chunk_size, "overlap": overlap,
                             "population": populated, "ingest_seconds": ingest_seconds, "evaluation": evaluation})
        finally:
            try:
                with admin.begin() as connection:
                    connection.execute(DropSchema(schema, cascade=True, if_exists=True))
            finally:
                admin.dispose()
    if not measured:
        raise ValueError("At least one chunking variant is required.")
    best = max(measured, key=lambda row: (
        row["evaluation"]["improved"]["evidence_hit@3"],
        -(row["evaluation"]["improved"]["unanswerable_returned_match_rate"] or 0),
        row["evaluation"]["improved"]["evidence_hit@1"], -row["population"]["chunks"],
    ))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": {"repo": TOKENIZER_REPO, "revision": QWEN_REVISION},
        "chunker_sha256": hashlib.sha256(Path(__file__).with_name("chunking.py").read_bytes()).hexdigest(),
        "baseline_chunking_configs": [dict(row) for row in baseline_configs], "baseline": baseline,
        "variants": measured, "selected": {key: best[key] for key in ("chunk_size", "overlap")},
        "selection_rule": "Maximize gold evidence hit@3; break ties by fewer unanswerable false matches, then evidence hit@1, then fewer chunks.",
        "limitations": "Exploratory optimization on the same small regression set; no held-out claim. Timings include actual embedding and storage, and can vary with cache/model residency. Public corpus is unchanged by tuning.",
    }
