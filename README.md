# Lab: build a RAG model from scratch

A local study project using Ollama, PostgreSQL, and pgvector. The commands cover corpus collection, embeddings, storage, retrieval, and answers generated from retrieved evidence.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/rag_lab/cli.py` | Command-line entry point |
| `src/rag_lab/config.py` | Central configuration from `.env` and shell variables |
| `src/rag_lab/database.py` | SQLAlchemy sessions and pgvector connection setup |
| `src/rag_lab/ollama_client.py` | Separate embedding and chat helpers |
| `src/rag_lab/extraction.py` | Optional Docling document extraction |
| `src/rag_lab/corpus.py` | Resumable Wikipedia corpus collection |
| `src/rag_lab/populate.py` | Chunk text files, generate embeddings, and populate pgvector |
| `src/rag_lab/chunking.py` | Pinned Qwen tokenization, structural boundaries, and source offsets |
| `src/rag_lab/tune_chunking.py` | Real chunk-size/overlap comparison in temporary schemas |
| `src/rag_lab/pull.py` | Retrieve stored chunks by semantic similarity |
| `src/rag_lab/indexing.py` | Model-specific HNSW indexes and candidate-first search |
| `src/rag_lab/benchmark_index.py` | Isolated index latency, plans, storage, and neighbor-recall measurements |
| `src/rag_lab/retrieval.py` | Similarity filtering, duplicate removal, and lexical reranking |
| `src/rag_lab/evaluate.py` | Source-checked retrieval benchmark and calibration |
| `src/rag_lab/evaluate_answers.py` | End-to-end answer accuracy, faithfulness, and citation evaluation |
| `src/rag_lab/evaluate_security.py` | Controlled injection regressions with attack and answer-utility metrics |
| `src/rag_lab/ask.py` | Generate answers with checked evidence quotes and source references |
| `models/Modelfile` | Custom Ollama model recipe |
| `compose.yaml` | Local PostgreSQL service |
| `docker/postgres/init/` | Initial pgvector extension setup |
| `tests/` | Offline unit tests and opt-in PostgreSQL integration tests |
| `benchmarks/`, `reports/` | Labeled questions and measured evaluation results |
| `astronomy_corpus/`, `data/` | Ignored local corpus and generated files |

## Installation

Use Python 3.11 or newer:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -e .
```

The editable installation makes `src/rag_lab` available as a package and installs the `rag-lab` command. Direct dependency versions live in `requirements.txt`, which the package reads during installation. PostgreSQL uses the explicit `psycopg2` driver with its binary distribution.

Document extraction is optional and has larger dependencies:

```bash
python -m pip install -e ".[documents]"
```

Docling is loaded only for extraction. It exports Markdown so headings, paragraphs, lists, and tables survive into chunking. Some document formats may download conversion models on first use.

## Configuration

Keep your existing `.env`. For a new checkout, copy the example and set `POSTGRES_PASSWORD`:

```bash
cp .env.example .env
chmod 600 .env
```

Commands read `.env` from the project root automatically. Exported shell variables take precedence. Passwords are passed through SQLAlchemy's URL builder, so special characters do not need manual URL encoding.

| Variable | Default / purpose |
| --- | --- |
| `POSTGRES_HOST` | `127.0.0.1` |
| `POSTGRES_PORT` | `55432` |
| `POSTGRES_DB` | `rag_lab` |
| `POSTGRES_USER` | `postgres` |
| `POSTGRES_PASSWORD` | Required for database connections; keep in ignored `.env` |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` |
| `OLLAMA_CHAT_MODEL` | `custom_qwen` |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:0.6b` |
| `WIKIPEDIA_CONTACT` | Your contact information for the Wikipedia API user agent |
| `CORPUS_DIR` | `astronomy_corpus`, relative to the project root |

## Commands

| Exercise | Command |
| --- | --- |
| Check PostgreSQL and pgvector | `rag-lab db-check` |
| Generate embeddings | `rag-lab embed "Here is statement one." "Here is statement two."` |
| Ask the chat model | `rag-lab chat "What is a vector database?"` |
| Extract a document | `rag-lab extract ./data/document.pdf --output ./data/document.txt` |
| Collect Wikipedia articles | `rag-lab corpus --topic astronomy --count 100` |
| Populate the vector database | `rag-lab populate` |
| Retrieve relevant text | `rag-lab pull "How do telescopes work?" --top-k 3` |
| Ask with retrieved context | `rag-lab ask "What powers the Sun?"` |
| Evaluate retrieval quality | `rag-lab evaluate --output reports/retrieval-quality.json` |
| Evaluate generated answers | `rag-lab evaluate-answers --output reports/answer-quality.json` |
| Test injection safeguards | `rag-lab evaluate-security --output reports/security-regression.json` |
| Compare token chunking settings | `rag-lab tune-chunking --output reports/chunking-comparison.json` |
| Create the vector search index | `rag-lab index-db` |
| Benchmark index scalability | `rag-lab benchmark-index --output reports/index-performance.json` |
| Show options | `rag-lab --help` |

`python -m rag_lab` is equivalent to `rag-lab`. Importing the package makes no database, model, or document requests.

Prepare the models when needed:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:4b
ollama create custom_qwen -f models/Modelfile
```

The custom recipe uses `qwen3:4b`. Chat and embedding commands accept `--model` overrides. The embedding command prints a size summary; `rag_lab.ollama_client.embed_texts` returns the complete vectors for Python exercises.

Thinking models may spend their token budget before producing a final answer ([Ollama thinking output](https://docs.ollama.com/capabilities/thinking)). If chat reports no final answer, increase `--max-tokens` or try the smaller installed model, for example `rag-lab chat "What is a vector database?" --model qwen3:0.6b --max-tokens 2048`. A failed chat does not modify the vector database.

Set `WIKIPEDIA_CONTACT` before running `corpus`. Existing articles and metadata are reused, missing files are downloaded, and failed articles are recorded for retry on the next run. Progress goes to stderr and the summary to stdout. A run with remaining failures exits with status 1. Requests have a courtesy delay and respect rate-limit retry headers. The existing `astronomy_corpus` directory stays in place.

## Populate the vector database

Start PostgreSQL, install the embedding model if needed, and try one file before importing the whole corpus:

```bash
docker compose up -d --wait postgres
ollama pull qwen3-embedding:0.6b
rag-lab populate --limit 1
rag-lab populate
```

The script is also runnable with `python -m rag_lab.populate`. It reads UTF-8 `.txt` and `.md` files recursively from `CORPUS_DIR` (by default `astronomy_corpus`); file symlinks are excluded. JSON files are not embedded. An optional `metadata.json` list supplies per-file metadata by its `filename` field; all record fields are preserved. For PDFs or other formats, run `rag-lab extract` first and save the Markdown output under an input directory.

```bash
rag-lab populate --input-dir ./data --chunk-size 512 --overlap 64 --batch-size 16
```

| Option | Default / behavior |
| --- | --- |
| `--input-dir` | `CORPUS_DIR` from configuration |
| `--model` | `OLLAMA_EMBED_MODEL`; use an embedding model |
| `--chunk-size` | At most 512 Qwen tokens per chunk; allowed range 1–8192 |
| `--overlap` | Target of 64 overlapping tokens; must be smaller than chunk size |
| `--strategy` | `tokens` by default; `words` reproduces the old baseline |
| `--batch-size` | 16 chunks per Ollama request |
| `--limit` | Process only the first N files, sorted by path |
| `--force` | Re-embed unchanged files for the selected model |

The command creates two tables automatically; pgvector must already be enabled by the Docker init script:

| Table | Stored information |
| --- | --- |
| `rag_documents` | Source path, model, content SHA-256, chunk settings/tokenizer revision, dimensions, count, and original document metadata |
| `rag_chunks` | Source/model/index, exact original text slice, vector, character offsets, token count, and hierarchical heading path |

Source path and model identify each document. Unchanged content and chunk settings are skipped; changed documents replace their old chunks in one transaction. Each document commits separately, so an interrupted run can resume. An embedding or database failure stops the command with a nonzero exit status; previously committed documents remain saved. Empty files retain any previous rows, and deleting source files does not remove their stored data.

The default chunker counts tokens with Hugging Face's `tokenizers` and the [Qwen3 embedding vocabulary](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), pinned to revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. First use downloads only `tokenizer.json` (about 11 MB) into ignored `.cache/huggingface`; subsequent use can use the cache. No model repository code is executed. Counts remain Qwen-token units when using another embedding-model family through `--model`; oversized inputs still fail instead of truncating.

Chunks prefer nearby sentence/paragraph endings and separate substantive Markdown ATX or Wikipedia sections. Empty heading-only blocks are attached to adjacent content rather than indexed as independent facts. The heading hierarchy stays with each chunk. Text is sliced directly from the source, with Python character offsets and intact Unicode characters; overlap is a target and can shift slightly at Unicode boundaries. A source character that cannot fit an unusually tiny chunk size produces an error. Counts exclude special tokens added by the embedding service.

Population adds the metadata columns to older lab tables without deleting rows. An older index needs `populate` again to replace word windows with token chunks; the strategy/tokenizer revision participate in resume checks. Existing rows keep null offsets until reindexed. To reproduce the legacy comparison explicitly, use `rag-lab populate --strategy words --chunk-size 300 --overlap 50` (this replaces the selected model's indexed chunks).

Model names are stored separately, and the vector column accepts the dimensions returned by the chosen model. Searches compare the same model and dimensions. If you replace a model's weights under the same tag, rerun with `--force`. Run `rag-lab index-db` after initial population to enable HNSW acceleration.

Embedding requests disable silent truncation: an oversized chunk fails rather than silently losing text. Reduce `--chunk-size` if this happens ([Ollama embedding API](https://docs.ollama.com/api/embed)).

Progress goes to stderr; stdout contains a JSON summary. `populated` and `chunks` count documents and chunks written during this run, including replacements; `skipped` counts unchanged files and `empty` counts empty files. To inspect stored totals in the PostgreSQL console:

```sql
SELECT model, count(*) AS documents, sum(chunk_count) AS chunks
FROM rag_documents
GROUP BY model;
```

## Pull relevant chunks from the database

Use a natural-language query to retrieve the most similar stored chunks:

```bash
rag-lab pull "Why do stars produce light?" --top-k 3
```

The script is also runnable with `python -m rag_lab.pull "Why do stars produce light?"`. It embeds the query with `OLLAMA_EMBED_MODEL` and searches PostgreSQL using cosine distance ([pgvector Python documentation](https://github.com/pgvector/pgvector-python#sqlalchemy)). It reads the existing database; it does not import files or generate an answer.

| Option | Default / behavior |
| --- | --- |
| `query` | Required search text |
| `--top-k` | Return up to 5 chunks, ordered by similarity |
| `--model` | Override `OLLAMA_EMBED_MODEL`; must match a stored model name |
| `--min-similarity` | Optional raw cosine threshold, from -1 to 1 |
| `--candidate-k` | Default `max(40, 4 × top-k)`, capped at 1000; must be at least top-k |
| `--no-deduplicate` | Keep duplicate and near-duplicate chunks |
| `--no-rerank` | Preserve raw vector-similarity ranking |
| `--search-mode` | `auto`: use a valid HNSW index when available, otherwise exact; `exact` always scans exactly; `hnsw` requires its index |
| `--ef-search` | 200 by default, allowed range 1–1000; internally at least candidate-k |

Output is JSON containing the query, model, and a `matches` list. Each match contains:

| Field | Meaning |
| --- | --- |
| `source` | Absolute path of the original text file |
| `chunk_index` | Zero-based position in that document |
| `similarity` | Cosine similarity (`1 - distance`); higher is closer, not a probability |
| `rerank_score` | Ranking score after optional lexical reranking; not a probability |
| `text` | Full stored chunk, ready to use as context |
| `start_char`, `end_char` | Original text character range; null for old word windows |
| `token_count`, `heading_path`, `metadata` | Qwen token count, structural context, and original document metadata |

Only vectors from the selected model and matching dimensions are compared. No documents for the selected model produces an empty list; missing tables produce an error directing you to `rag-lab populate`. Query dimensions that disagree with stored documents produce a clear error. If you change model weights under the same tag, re-populate with `--force` first.

Candidates are selected before joining document metadata and loading chunk text. Search can be exact or HNSW approximate, according to `--search-mode`. By default, candidates are reranked with 95% cosine similarity and 5% normalized BM25 score. BM25 runs over the candidate pool using Unicode word terms, an English stop-word list, k1=1.5, and b=0.75. It adds lexical relevance without another model or dependency; it is not a cross-encoder. Candidate pool size changes these lexical scores. Exact duplicates and near duplicates with at least 0.8 word-shingle Jaccard similarity are removed after ranking. Ordinary adjacent chunks with only a small overlap are retained. These limits can return fewer than top-k results.

`pull` leaves the similarity threshold unset unless requested. `ask` defaults to 0.3, applied to raw cosine similarity before reranking. Both commands expose the threshold, candidate pool, deduplication, and reranking switches. Thresholds are corpus/model dependent and do not guarantee relevance.

## Ask questions using retrieved evidence

```bash
rag-lab ask "What powers the Sun?"
rag-lab ask "How do telescopes work?" --chat-model qwen3:0.6b --top-k 3
```

`python -m rag_lab.ask` is equivalent. Unlike `chat`, this command retrieves local chunks, supplies bounded context to Qwen, and returns a JSON answer with numbered citations. It does not modify the database.

| Option | Default / meaning |
| --- | --- |
| `--top-k` | 3 retrieved chunks; maximum 20 |
| `--embedding-model` | `OLLAMA_EMBED_MODEL`; must match stored vectors |
| `--chat-model` | `OLLAMA_CHAT_MODEL` |
| `--max-tokens` | 2048 per generation/check call; maximum 8192 |
| `--max-context-chars` | 6000 characters of retrieved text; allowed range 256–16000 |
| `--min-similarity` | 0.3 raw cosine similarity; -1 disables practical filtering |
| `--candidate-k` | Vector candidate pool before reranking; default at least 40 |
| `--no-deduplicate`, `--no-rerank` | Disable either postprocessing step for comparisons |
| `--search-mode`, `--ef-search` | Same index selection and HNSW search settings as `pull` |

Each generated claim must cite an actual supplied source and contain a verbatim supporting quote. A separate model call checks whether the quote supports the entire claim and the claim answers the question. Invalid individual claims (including fabricated quotes or unknown citations) and unsupported claims are removed; insufficient evidence produces `status: "abstained"`. Malformed top-level responses, truncated output, and invalid grounding verdicts fail with an error. Rejection counts are included in the JSON result. The `sources` array maps citations to source paths, chunk indices, similarity scores, and the excerpts actually shown to the model. `claims` includes the evidence quotes.

Paragraphs and headings are preserved in model context. `start_char`/`end_char` refer to the complete stored chunk; `excerpt_start_char`/`excerpt_end_char` identify the actual excerpt when it is clipped for the context budget. Evidence quotes are checked word-for-word with whitespace normalized.

The structured request disables thinking and uses an 8192-token context window; generation still has a finite budget. Large contexts or complex questions can require a smaller model, fewer chunks, or a larger `--max-tokens` value.

Retrieved text and the question stay in an untrusted JSON payload, separate from the fixed system instructions. No tools, database credentials, or local file-reading capability are given to the model. Context and output sizes are bounded. These are safeguards, not a guarantee against prompt injection or hallucination: the grounding checker is also a model, and valid citations alone do not prove an answer is true. See the [development roadmap](docs/ROADMAP.md) for validation evidence and known limitations.

## Retrieval evaluation

```bash
rag-lab evaluate --output reports/retrieval-quality.json
```

The [14-case regression dataset](benchmarks/questions.jsonl) has questions, expected answers, source filenames, and gold evidence quotes (12 answerable, two unanswerable). Evaluation verifies local quotes and indexed source hashes first, then compares raw vector ranking against filtered/deduplicated/reranked candidates. It records document Recall@1/3/5, gold-quote hit rates, and unanswerable queries that still return a match. Answer accuracy and faithfulness are measured separately with `evaluate-answers` and a source-based audit below.

The [historical word-window report](reports/retrieval-quality.json) records dataset/source hashes, embedding model digest, 40-candidate pool, 0.3 threshold, and a reproducible weight sweep. On this small set, 5% lexical weighting increased evidence hits at rank 1 from 4/12 to 5/12 while keeping hits within the top 3 at 11/12. One of two unanswerable questions still returned a match. Document Recall@K was already 100%, so that metric alone concealed passage-level mistakes. These are exploratory calibration results on the same cases, not held-out generalization claims. Postprocessing timing excludes embedding and database search. The [current token-chunk report](reports/retrieval-token-chunks.json) records the applied token index separately.

Evaluation defaults to `--search-mode exact` so a new approximate index does not silently change its baseline. `rag-lab evaluate --search-mode hnsw --output reports/retrieval-hnsw.json` checks the corpus with the actual HNSW path. The checked-in HNSW report matched the exact token-chunk report's document-recall and evidence-hit metrics on all 14 regression cases; approximate search can still miss neighbors on other data.

## Answer evaluation

```bash
rag-lab evaluate-answers --output reports/answer-quality.json
# Optional: compare with a different installed judge model.
rag-lab evaluate-answers --judge-model qwen3:0.6b --output reports/answer-quality-small-judge.json
```

The command runs the actual `ask` pipeline for all 14 source-checked benchmark cases. It defaults to exact retrieval, top-k 3, candidate-k 40, minimum similarity 0.3, and the configured chat model. A separate structured model call judges semantic correctness against the reference answer and quote, and each claim's faithfulness against the retrieved source excerpts. It also audits source IDs and verbatim quotes without a model. Reports include model digests, dataset/source/module hashes, full claims and cited excerpts, reference answers, verdict reasons, timings, abstentions, and errors. They contain local source paths; review reports before publishing a private corpus.

| Metric | Meaning and denominator |
| --- | --- |
| Answerable accuracy | Correct answers / all answerable questions; abstentions and errors count as failures |
| Overall accuracy | Correct answers plus correct unanswerable abstentions / all questions |
| Claim faithfulness | Supported claims / claims with successful judgments; abstentions and errors are excluded and counted separately |
| Fully faithful answer rate | Answers with every claim supported / successfully judged nonempty answers |
| Citation validity | Claims with valid source IDs and normalized verbatim quotes / published claims |
| Expected-source citation rate | Claims citing a labeled source document / claims on answerable questions |

| Final result (configured Qwen 4B) | Model judge | Codex source-based audit |
| --- | ---: | ---: |
| Correct answerable questions, including extra assertions | 12 / 12 | 11 / 12 |
| Correct overall, including private-question abstentions | 14 / 14 | 13 / 14 |
| Faithful published claims | 14 / 14 | 13 / 14 |
| Valid source IDs and evidence quotes | 14 / 14 | 14 / 14 |

Generation and judge errors were both zero. The [source-based audit](reports/answer-source-audit.json) found that the telescope answer strengthened a patent submission into a granted patent; both model checks missed this. The essential Netherlands answer was correct, but the added assertion makes the full answer incorrect under the stricter rubric. The audit is a Codex inspection, not independent human validation. [Case notes](reports/answer-review.md) and [judge calibration](reports/answer-judge-calibration.json) preserve the discrepancy. The [earlier answer baseline](reports/answer-baseline.json) is retained separately.

The [measured answer report](reports/answer-quality.json) is a small development regression benchmark. The default judge is the same Qwen model as the generator in a separate call; its judgments can share the generator's mistakes. This is not independent or human validation. Inspect individual cases, and use `--judge-model` to compare another installed model. Model judgments and valid citations do not establish real-world truth.

## Prompt-injection regression tests

```bash
rag-lab evaluate-security --output reports/security-regression.json
```

The [11-case synthetic suite](benchmarks/security.jsonl) includes nine direct/document attacks and two clean controls: role overrides, forged system delimiters, JSON delimiter tricks, encoded instructions, citation spoofing, instruction-only evidence, private-data requests, and command/URL instructions. It injects controlled post-retrieval excerpts into the same generation, quote validation, and grounding functions used by `ask`; it bypasses corpus retrieval and never writes poisoned text to PostgreSQL. Only a fake public canary is used. Tests verify that private settings, source paths, metadata, and tool permissions are absent from model messages.

Attack success combines deterministic forbidden-output checks with a separate fallible semantic judge. Raw cited source text may naturally contain attack strings, so only the published answer and claim text count as following an output instruction. Missing/failed attack judgments are unassessed, not successful defenses. Answer usefulness under attack, clean controls, abstentions, and errors are separate metrics; refusing everything cannot demonstrate a useful RAG system.

| Measured result | Initial baseline | Current |
| --- | ---: | ---: |
| Observed successful attacks | 0 / 9 | 0 / 9 |
| Correct answers under answerable attacks | 0 / 7 | 5 / 7 |
| Correct clean controls | 1 / 2 | 2 / 2 |
| Abstentions | 10 / 11 | 4 / 11 |
| Generation/judge errors | 0 | 0 |

The [initial baseline](reports/security-baseline.json) exposed excessive refusal. The [current report](reports/security-regression.json) preserves five useful factual answers under attack while abstaining on two answerable attacks, instruction-only evidence, and a private-information request. The prompt explicitly distinguishes factual content from embedded instructions and educational attack descriptions. The corrected semantic judge examines published claims without replaying malicious input documents; any claimed violation must quote actual published output. [Calibration evidence](reports/security-judge-calibration.json) records the earlier judge's input/output confusion, so these are not silently substituted results. Deterministic marker checks and the utility rubric are unchanged; the semantic attack judge was corrected between reports.

This fixed suite is not exhaustive red teaming, an adaptive attack evaluation, or a guarantee of security. The separate judge is the generator model by default and can share its mistakes. Model output is untrusted; this CLI does not give it tools or execute its output. The design follows structured data/instruction separation and least privilege described by the [OWASP prevention guidance](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html); JSON schema constrains format, not semantic safety ([Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)).

## Vector indexing and performance

```bash
rag-lab index-db
rag-lab pull "What powers the Sun?" --search-mode hnsw --ef-search 200
rag-lab pull "What powers the Sun?" --search-mode exact
```

Index creation is idempotent and returns the index name, dimensions, build/analyze time, and storage bytes. Each model/dimension gets a partial cosine HNSW expression index over the existing variable-dimension vector column (m=16, ef_construction=64). It also collects model/dimension statistics. Index creation requires table ownership and blocks writes while building; run it after bulk population in this local lab. Subsequent writes maintain the index automatically. Models with more than 2000 dimensions can use exact search, but this HNSW command rejects them.

`auto` uses an existing valid index for the selected cohort, or falls back to an exact query. `hnsw` fails clearly if its index is missing. `exact` changes the ordering expression to exclude approximate indexes without disabling primary-key lookups. The nearest-neighbor limit is materialized before text/metadata joins. HNSW requests use transaction-local planner settings to select their index and disable JIT; these settings expire at transaction end and do not alter server configuration. Higher `ef-search` generally trades more work for better neighbor recall ([pgvector documentation](https://github.com/pgvector/pgvector#hnsw)).

```bash
rag-lab benchmark-index --sizes 1000 5000 20000 --repeats 3 --output reports/index-performance.json
```

The benchmark copies corpus vectors/text into temporary schemas and perturbs vector coordinates with seeded noise. It compares exact search, auto, and HNSW at ef-search 40/100/200 using 12 real question embeddings and a 40-candidate pool. It measures query/fetch p50/p95, index build time/storage, and strict neighbor Recall@10/40 against exact search. It checks actual `EXPLAIN (ANALYZE, BUFFERS, VERBOSE)` plans and cleans up benchmark schemas. It does not add synthetic rows to the main corpus. Vector literals are omitted from saved plans for readability.

| Vectors | Exact p50 (ms) | Auto/HNSW p50 (ms) | Speedup | Neighbor Recall@40 | Index MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 8.08 | 4.52 | 1.79× | 100.00% | 7.82 |
| 5,000 | 25.46 | 4.35 | 5.85× | 98.96% | 39.07 |
| 20,000 | 97.19 | 3.92 | 24.76× | 100.00% | 156.26 |

The [measured report](reports/index-performance.json) includes model/input fingerprints, server versions, method details, and all query plans. The [initial calibration](reports/index-calibration.json) explains the choice of ef-search 200 and explicit index selection. These are warm-cache local database timings, not end-to-end RAG speed or production capacity claims. Synthetic copies do not replace tests on a naturally growing corpus, cold IO, or concurrent traffic. HNSW construction can vary between builds; high recall here is not a guarantee elsewhere.

## Chunking comparison

```bash
rag-lab tune-chunking --output reports/chunking-comparison.json
```

This command embeds the full corpus with three token-size/overlap combinations in temporary schemas, evaluates each, and drops those schemas even on a handled failure. It requires schema creation permissions. It does not replace the current indexed corpus; only the additive metadata-column upgrade may occur before comparison. Runtime and model residency affect the measured ingestion times. The selected setting maximizes gold evidence hit@3, then reduces false matches, then improves hit@1, then favors fewer chunks.

The [comparison report](reports/chunking-comparison.json) includes the word-window baseline, full per-configuration metrics, tokenizer revision, source hashes, measured ingestion times, and validation of the applied defaults. The [preliminary run](reports/chunking-preliminary.json) records the issue found before empty heading blocks were coalesced.

| Strategy / size / overlap | Chunks | Evidence hit@1 | Evidence hit@3 | Evidence hit@5 |
| --- | ---: | ---: | ---: | ---: |
| Word baseline, 300 / 50 | 1104 | 5/12 | 11/12 | 11/12 |
| Qwen tokens, 256 / 32 | 2340 | 6/12 | 8/12 | 8/12 |
| Qwen tokens, 384 / 64 | 1835 | 4/12 | 9/12 | 11/12 |
| Qwen tokens, 512 / 64 (selected) | 1548 | 7/12 | 10/12 | 11/12 |

All configurations reached document Recall@3 of 100%; one of two unanswerable queries still returned a match. The selected token configuration improves first-result evidence matches and preserves structure with hard token bounds, but its top-three gold-quote coverage is below the word baseline. This tradeoff is retained explicitly; the comparison does not prove global optimality or better performance on unseen questions.

## Local PostgreSQL

This Linux lab uses host networking because Docker bridge connections were blocked on the development computer. PostgreSQL listens only on localhost.

```bash
docker compose up -d --wait postgres
rag-lab db-check
docker compose exec postgres sh -c 'psql -U postgres -d "$POSTGRES_DB"'
```

`docker compose down` stops the service and keeps its volume. Adding `--volumes` deletes the data. The init SQL enables pgvector on first startup; `populate` creates its own tables on demand. Changing `.env` does not change passwords in an existing database.

Database helpers close sessions and dispose their engines after use. Call `session.commit()` explicitly when an exercise writes data.

## Validation

```bash
python -m unittest discover -s tests -v
python -m pip check
```

The tests use mocked clients and temporary files, so no running database, Ollama service, or downloads are needed.

Population and retrieval also have optional PostgreSQL integration tests. They create and remove isolated test schemas and use synthetic embeddings, so no model download is needed:

```bash
RAG_TEST_POSTGRES=1 python -m unittest discover -s tests -v
```

## Previous script names

| Previous file | Replacement |
| --- | --- |
| `database_connect_embeddings.py` | `rag-lab db-check`; `rag_lab.database.get_session` |
| `ollama_functions.py` | `rag-lab embed` and `rag-lab chat` |
| `extract_text.py` | `rag-lab extract` |
| `generate_copus.py` | `rag-lab corpus` |
| `ModelFile` | `models/Modelfile` |
