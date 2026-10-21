"""Reproducible retrieval evaluation against source-checked question fixtures."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

from ollama import Client
from sqlalchemy import select

from rag_lab.config import PROJECT_ROOT, Settings
from rag_lab.pull import pull_chunks
from rag_lab.retrieval import RERANK_WEIGHT, rank_candidates
from rag_lab.database import create_db_engine
from rag_lab.populate import documents
from rag_lab.indexing import DEFAULT_EF_SEARCH

DEFAULT_DATASET = PROJECT_ROOT / "benchmarks/questions.jsonl"


def load_dataset(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in ids:
            raise ValueError("Benchmark cases need unique string IDs.")
        ids.add(row["id"])
        if not isinstance(row.get("question"), str) or not row["question"].strip() or type(row.get("unanswerable")) is not bool:
            raise ValueError("Each benchmark case needs a question and an unanswerable flag.")
        sources = row.get("sources")
        if not isinstance(sources, list) or any(not isinstance(s, str) or Path(s).name != s for s in sources):
            raise ValueError("Benchmark sources must be a list of plain filenames.")
        if not row["unanswerable"] and (not sources or not isinstance(row.get("expected_answer"), str) or not row["expected_answer"].strip() or not isinstance(row.get("gold_quote"), str) or not row["gold_quote"].strip()):
            raise ValueError("Answerable benchmark cases need sources, an expected answer, and a gold quote.")
        if row["unanswerable"] and (sources or row.get("expected_answer") is not None or row.get("gold_quote") is not None):
            raise ValueError("Unanswerable cases must not claim expected sources or answers.")
    if not rows:
        raise ValueError("Benchmark dataset is empty.")
    return rows


def retrieval_metrics(case: dict, matches: list[dict], k: int) -> dict:
    selected = matches[:k]
    expected = set(case["sources"])
    actual = {Path(row["source"]).name for row in selected}
    quote = " ".join((case["gold_quote"] or "").split())
    return {
        "document_recall": len(expected & actual) / len(expected) if expected else None,
        "evidence_hit": any(Path(row["source"]).name in expected and quote in " ".join(row["text"].split()) for row in selected) if expected else None,
        "unique_documents": len(actual), "returned_chunks": len(selected),
    }


def summarize(results: list[dict], variant: str) -> dict:
    answerable = [row for row in results if not row["unanswerable"]]
    unanswerable = [row for row in results if row["unanswerable"]]
    summary = {}
    for k in (1, 3, 5):
        metrics = [row[variant][str(k)] for row in answerable]
        summary[f"document_recall@{k}"] = statistics.mean(m["document_recall"] for m in metrics) if metrics else None
        summary[f"evidence_hit@{k}"] = statistics.mean(m["evidence_hit"] for m in metrics) if metrics else None
    summary["unanswerable_returned_match_rate"] = statistics.mean(bool(row[variant]["1"]["returned_chunks"]) for row in unanswerable) if unanswerable else None
    return summary


def checked_sources(cases: list[dict], corpus_dir: Path) -> dict[str, str]:
    """Check reference quotes against original local files and fingerprint those files."""
    source_hashes = {}
    for case in cases:
        source_texts = []
        for source in case["sources"]:
            path = corpus_dir / source
            if path.is_symlink() or path.resolve().parent != corpus_dir.resolve():
                raise ValueError("Benchmark sources must be regular files inside the corpus directory.")
            content = path.read_text(encoding="utf-8")
            source_texts.append(" ".join(content.split()))
            source_hashes[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_texts and not any(" ".join(case["gold_quote"].split()) in content for content in source_texts):
            raise ValueError(f"Gold quote missing from sources for {case['id']}.")
    return source_hashes


def validate_indexed_sources(settings: Settings, source_hashes: dict[str, str]) -> list:
    """Refuse benchmarks against stale or missing indexed source content."""
    engine = create_db_engine(settings)
    try:
        with engine.connect() as connection:
            indexed = connection.execute(select(documents).where(documents.c.model == settings.embedding_model)).mappings().all()
        by_source = {row["source"]: row for row in indexed}
        for source, digest in source_hashes.items():
            record = by_source.get(str((settings.corpus_dir / source).resolve()))
            if record is None or record["content_sha256"] != digest:
                raise ValueError(f"Indexed source is missing or stale: {source}. Run populate first.")
    finally:
        engine.dispose()
    return indexed


def evaluate_retrieval(settings: Settings, *, dataset: Path = DEFAULT_DATASET, min_similarity: float = 0.3, candidate_k: int = 40,
                       search_mode: str = "exact", ef_search: int = DEFAULT_EF_SEARCH) -> dict:
    if search_mode not in ("auto", "exact", "hnsw") or not 1 <= ef_search <= 1000:
        raise ValueError("Use search mode auto/exact/hnsw and ef-search between 1 and 1000.")
    if not 5 <= candidate_k <= 1000:
        raise ValueError("Evaluation candidate_k must be between 5 and 1000.")
    rank_candidates([], "", top_k=5, min_similarity=min_similarity)
    cases = load_dataset(dataset)
    source_hashes = checked_sources(cases, settings.corpus_dir)
    indexed = validate_indexed_sources(settings, source_hashes)
    with Client(host=settings.ollama_host) as client:
        canonical = settings.embedding_model if ":" in settings.embedding_model else settings.embedding_model + ":latest"
        digest = next((model.digest for model in client.list().models if model.model == canonical), None)
    results, durations = [], []
    sweeps = {str(weight): [] for weight in (0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15)}
    for case in cases:
        candidates = pull_chunks(case["question"], settings, top_k=candidate_k, candidate_k=candidate_k,
                                 deduplicate=False, rerank=False, search_mode=search_mode, ef_search=ef_search)
        start = time.perf_counter()
        improved = rank_candidates(candidates, case["question"], top_k=5, min_similarity=min_similarity)
        durations.append((time.perf_counter() - start) * 1000)
        row = {"id": case["id"], "unanswerable": case["unanswerable"], "baseline": {}, "improved": {}}
        for k in (1, 3, 5):
            row["baseline"][str(k)] = retrieval_metrics(case, candidates, k)
            row["improved"][str(k)] = retrieval_metrics(case, improved, k)
        row["baseline_sources"] = [Path(m["source"]).name for m in candidates[:5]]
        row["improved_sources"] = [Path(m["source"]).name for m in improved]
        results.append(row)
        for weight, measurements in sweeps.items():
            ranked = rank_candidates(candidates, case["question"], top_k=5,
                                     min_similarity=min_similarity, rerank_weight=float(weight))
            measurements.append({"unanswerable": case["unanswerable"], "calibration": {
                str(k): retrieval_metrics(case, ranked, k) for k in (1, 3, 5)
            }})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "source_sha256": source_hashes, "embedding_model": settings.embedding_model,
        "embedding_model_digest": digest, "indexed_documents": len(indexed),
        "indexed_chunks": sum(row["chunk_count"] for row in indexed),
        "candidate_k": candidate_k, "min_similarity": min_similarity,
        "search_mode": search_mode, "ef_search": ef_search,
        "rerank_weight": RERANK_WEIGHT,
        "cases": len(cases), "answerable_cases": sum(not c["unanswerable"] for c in cases),
        "baseline": summarize(results, "baseline"), "improved": summarize(results, "improved"),
        "mean_postprocessing_ms": statistics.mean(durations), "results": results,
        "weight_sweep": {weight: summarize(measurements, "calibration") for weight, measurements in sweeps.items()},
        "limitations": "Small hand-authored regression set, not a held-out generalization study. Source-document recall and gold-quote hit rate are distinct. Timings exclude embedding and database search.",
    }
