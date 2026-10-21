# RAG development priorities

The requested five priorities now have implementations and direct validation evidence, mapped in [VALIDATION.md](VALIDATION.md). Known measurement and model limitations remain explicit.
Each effective change is tested, committed, and pushed before the next change.

| Priority | Requirements | Current evidence and limitations |
| --- | --- | --- |
| 1. Complete the RAG pipeline | Integrate retrieved chunks with Qwen; context-aware generation; source references | `ask` implements bounded context, structured claims, source/quote validation, and a separate grounding check. Automated and live checks accompany its commit. |
| 2. Retrieval quality | Configurable similarity thresholds; chunk deduplication; reranking | Implemented in `pull` and `ask`; `evaluate` compares baseline and improved retrieval. `reports/retrieval-quality.json` records measured Recall@K, gold-quote hit rates, false matches, and a weight sweep on 14 regression cases. Generalization remains unproven. |
| 3. Chunking | Semantic or token-based chunking; document structure, headings, and metadata; optimized size and overlap | Default Qwen token chunking preserves exact slices, offsets, heading hierarchy, and original metadata. Docling exports Markdown. Three real embedding comparisons selected 512/64; `reports/chunking-comparison.json` records both improvements and regressions. Applied validation checks all 99 documents and 1548 chunks. |
| 4. Database performance | HNSW or IVFFlat; optimized larger-dataset searches; measured indexing performance and scalability | `index-db` creates per-model/dimension HNSW indexes; candidate-first queries support auto/exact/HNSW modes. `reports/index-performance.json` measures latency, storage/build cost, strict neighbor recall, and actual isolated plans at 1000/5000/20000 vectors. `reports/retrieval-hnsw.json` checks corpus retrieval metrics against the exact token baseline. |
| 5. Evaluation, testing, and security | Questions with expected answers and source documents; Recall@K; answer accuracy and faithfulness; prompt-injection safeguards | `evaluate`, `evaluate-answers`, and `evaluate-security` provide source-checked retrieval, answer, and live injection measurements. The answer model judge scored 14/14 cases, while a stricter source audit found one overstatement (13/14). Nine synthetic attacks produced no observed success; useful answers survived 5/7 answerable attacks and both clean controls passed. 103 automated tests cover the application boundaries, failure handling, and PostgreSQL paths. Small-set and same-model-judge limitations are documented. |

The validation audit covers the complete five-priority scope. This local lab has measured behavior and documented limits; it does not promise error-free answers or comprehensive security coverage.
Measured improvements and limitations must be reported without invented results.
