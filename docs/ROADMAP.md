# RAG development priorities

The full goal remains active until each requirement has direct validation evidence.
Each effective change is tested, committed, and pushed before the next change.

| Priority | Requirements | Current evidence / work remaining |
| --- | --- | --- |
| 1. Complete the RAG pipeline | Integrate retrieved chunks with Qwen; context-aware generation; source references | `ask` implements bounded context, structured claims, source/quote validation, and a separate grounding check. Automated and live checks accompany its commit. |
| 2. Retrieval quality | Configurable similarity thresholds; chunk deduplication; reranking | Pending implementation and comparison on a benchmark. |
| 3. Chunking | Semantic or token-based chunking; document structure, headings, and metadata; optimized size and overlap | Current population still uses word windows. Replacement and measured settings selection remain. |
| 4. Database performance | HNSW or IVFFlat; optimized larger-dataset searches; measured indexing performance and scalability | Exact search is the current baseline. Index implementation, plan inspection, and scale experiments remain. |
| 5. Evaluation, testing, and security | Questions with expected answers and source documents; Recall@K; answer accuracy and faithfulness; prompt-injection safeguards | Unit/integration tests and initial `ask` safeguards exist. A checked-in dataset, reproducible evaluation, live measurements, and adversarial tests remain. |

Success requires the complete five-priority scope, not just a working `ask` command.
Measured improvements and limitations must be reported without invented results.
