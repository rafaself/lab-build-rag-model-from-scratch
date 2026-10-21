# Validation of the five RAG priorities

Audit date: 2026-10-21. The project is a local study lab. Each requested feature is implemented and has direct evidence below; no numerical quality target or production deployment was part of the goal.

| Requested requirement | Implementation | Direct evidence |
| --- | --- | --- |
| Integrate retrieval with Qwen | `ask_question` calls `pull_chunks` and structured Qwen generation | Live 14-case end-to-end `reports/answer-quality.json`; model digests and implementation hashes recorded |
| Generate context-aware answers | Bounded retrieved excerpts; structured claims; source/quote validation and separate grounding check | Published answers, claims, actual cited excerpts, rejection counts, abstentions, and per-case judgments in the answer report; `tests/test_ask.py` |
| Include source references | Application adds source IDs and returns the matching source metadata and exact evidence quotes | All 14 published claims had valid source IDs and normalized verbatim quotes; invalid-source/fabricated-quote tests |
| Configurable similarity threshold | `--min-similarity` in `pull`/`ask`, checked on raw cosine score | Retrieval unit/integration tests and measured 0.3 threshold reports |
| Deduplicate chunks | Exact and near-duplicate word-shingle filtering | Duplicate/near-duplicate tests and `--no-deduplicate` comparison path |
| Rerank for relevance | Configurable retrieval pipeline uses 95% cosine and 5% normalized BM25 | `reports/retrieval-quality.json` weight sweep and distinct document/evidence metrics; lexical-reranking tests |
| Token-based chunks | Pinned Qwen tokenizer, hard retokenized size bounds | `tests/test_chunking.py`; current DB audit verifies the token count of every stored chunk |
| Preserve structure and metadata | Paragraph/sentence boundaries, heading paths, exact offsets, full source metadata; Markdown extraction | Live audit of all 99 documents and 1,548 chunks against original source slices, headings, metadata, and SHA-256; Markdown/Unicode/CRLF tests |
| Optimize chunk size/overlap | Real 256/32, 384/64, 512/64 comparisons; selected 512/64 | `reports/chunking-comparison.json` includes the selection rule, actual metrics, tradeoffs, and applied validation |
| Implement a pgvector index | Model/dimension-specific partial cosine HNSW expression index | Valid public 1024-dimensional HNSW index; PostgreSQL tests check index creation/idempotency and actual query plans |
| Optimize larger searches | Materialized nearest candidates before text/metadata joins; auto/exact/HNSW modes | Index/query-plan tests; transaction-local planner settings restore after use; `reports/retrieval-hnsw.json` agrees with exact corpus metrics on all 14 cases |
| Measure performance/scalability | Isolated 1,000/5,000/20,000-vector datasets, real query embeddings, exact vs HNSW | `reports/index-performance.json`: p50/p95, plans, build/storage, strict neighbor recall; 20k p50 97.19ms exact vs 3.92ms HNSW; Recall@40 98.96–100% across tested sizes |
| Benchmark questions, answers, sources | 14 labeled cases with expected answers, source filenames, and gold quotes; original local source documents | `benchmarks/questions.jsonl`; source hashes and indexed-source freshness checks in measured reports; symlink-escape regression test |
| Measure Recall@K | Document Recall@1/3/5 and separate gold-quote evidence-hit rates | `reports/retrieval-token-chunks.json`: document Recall@3 100%, evidence hit@3 10/12; unanswerable false matches are reported separately |
| Evaluate answer accuracy | Actual `ask` runs, separate structured reference-answer judge, explicit error/abstention denominators | `reports/answer-quality.json` model judgments plus `reports/answer-source-audit.json`: stricter audit 11/12 answerable and 13/14 overall correct |
| Evaluate faithfulness | Separate model judgment against cited full excerpts plus source-based per-claim audit | Model judge 14/14 claims; stricter source audit 13/14, identifying a patent-submission overstatement; `reports/answer-review.md` and calibration preserve disagreement |
| Prompt-injection safeguards | Fixed system vs untrusted JSON, no model tools/private settings/file-reading, ID/quote checks, fallible grounding, bounded input/output | 9 live synthetic post-retrieval attacks + 2 clean controls; no observed successful attacks, useful answers in 5/7 answerable attacks; `reports/security-regression.json`, boundary/error tests, baseline and evaluator-calibration reports |

## Automated and local checks

```bash
RAG_TEST_POSTGRES=1 venv/bin/python -m unittest discover -s tests -q
venv/bin/python -m pip check
venv/bin/python -m compileall -q src/rag_lab tests
venv/bin/python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/rag-lab-wheels .
git diff --check
```

103 tests passed with PostgreSQL integration enabled. Import tests prohibit network access and Docling initialization at import. Package/dependency checks and strict staged secret scanning accompany the commits. The original 101 corpus files remain unchanged; the main database retains 99 documents and 1,548 chunks. Temporary benchmark schemas were removed. Virtual environments, local credentials, corpus files, weights, caches, and build artifacts are ignored and untracked.

## Limits of the evidence

These are development regression cases, not held-out estimates. Model judges are fallible and share the generator's model by default. The patent overstatement remains a known semantic-validation limitation; valid citations do not prove an answer's truth. Two answerable injection cases still abstain, and repeated claims can appear. The synthetic security suite bypasses retrieval to control the injected excerpts; it does not establish safety for arbitrary attacks or tool-using agents. The CLI never executes model output.

Index timings cover warm-cache local database query/fetch, excluding embeddings and generation. Scaled data consists of seeded perturbations of real corpus vectors, not a naturally growing production workload, concurrent traffic, or cold IO. HNSW construction may vary between builds. Hosted CI and production deployment were not performed; neither is required for this local-lab scope.
