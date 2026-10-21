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
| `src/rag_lab/pull.py` | Retrieve stored chunks by semantic similarity |
| `src/rag_lab/retrieval.py` | Similarity filtering, duplicate removal, and lexical reranking |
| `src/rag_lab/evaluate.py` | Source-checked retrieval benchmark and calibration |
| `src/rag_lab/ask.py` | Generate answers with checked evidence quotes and source references |
| `models/Modelfile` | Custom Ollama model recipe |
| `compose.yaml` | Local PostgreSQL service |
| `docker/postgres/init/` | Initial pgvector extension setup |
| `tests/` | Offline unit tests |
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

Docling is loaded only for extraction. Some document formats may download conversion models on first use.

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

The script is also runnable with `python -m rag_lab.populate`. It reads UTF-8 `.txt` files recursively from `CORPUS_DIR` (by default `astronomy_corpus`); JSON metadata files are ignored. For PDFs or other formats, run `rag-lab extract` first and save the text under an input directory.

```bash
rag-lab populate --input-dir ./data --chunk-size 300 --overlap 50 --batch-size 16
```

| Option | Default / behavior |
| --- | --- |
| `--input-dir` | `CORPUS_DIR` from configuration |
| `--model` | `OLLAMA_EMBED_MODEL`; use an embedding model |
| `--chunk-size` | 300 words per chunk |
| `--overlap` | 50 words shared between adjacent chunks; must be smaller than chunk size |
| `--batch-size` | 16 chunks per Ollama request |
| `--limit` | Process only the first N files, sorted by path |
| `--force` | Re-embed unchanged files for the selected model |

The command creates two tables automatically; pgvector must already be enabled by the Docker init script:

| Table | Stored information |
| --- | --- |
| `rag_documents` | Absolute source path, model name, content SHA-256, chunk settings, dimensions, and chunk count |
| `rag_chunks` | Source path, model name, zero-based chunk index, text, and embedding vector |

Source path and model identify each document. Unchanged content and chunk settings are skipped; changed documents replace their old chunks in one transaction. Each document commits separately, so an interrupted run can resume. An embedding or database failure stops the command with a nonzero exit status; previously committed documents remain saved. Empty files retain any previous rows, and deleting source files does not remove their stored data.

Model names are stored separately, and the vector column accepts the dimensions returned by the chosen model. Filter by the same model when implementing retrieval. If you replace a model's weights under the same tag, rerun with `--force`. This lab does not create a vector search index yet.

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

Output is JSON containing the query, model, and a `matches` list. Each match contains:

| Field | Meaning |
| --- | --- |
| `source` | Absolute path of the original text file |
| `chunk_index` | Zero-based position in that document |
| `similarity` | Cosine similarity (`1 - distance`); higher is closer, not a probability |
| `rerank_score` | Ranking score after optional lexical reranking; not a probability |
| `text` | Full stored chunk, ready to use as context |

Only vectors from the selected model and matching dimensions are compared. No documents for the selected model produces an empty list; missing tables produce an error directing you to `rag-lab populate`. Query dimensions that disagree with stored documents produce a clear error. If you change model weights under the same tag, re-populate with `--force` first.

The candidate search is exact. By default, candidates are reranked with 95% cosine similarity and 5% normalized BM25 score. BM25 runs over the candidate pool using Unicode word terms, an English stop-word list, k1=1.5, and b=0.75. It adds lexical relevance without another model or dependency; it is not a cross-encoder. Candidate pool size changes these lexical scores. Exact duplicates and near duplicates with at least 0.8 word-shingle Jaccard similarity are removed after ranking. Ordinary adjacent chunks with only a small overlap are retained. These limits can return fewer than top-k results.

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

Each generated claim must cite an actual supplied source and contain a verbatim supporting quote. A separate model call checks whether the quote supports the entire claim. Invalid individual claims (including fabricated quotes or unknown citations) and unsupported claims are removed; insufficient evidence produces `status: "abstained"`. Malformed top-level responses, truncated output, and invalid grounding verdicts fail with an error. Rejection counts are included in the JSON result. The `sources` array maps citations to source paths, chunk indices, similarity scores, and the excerpts actually shown to the model. `claims` includes the evidence quotes.

The structured request disables thinking and uses an 8192-token context window; generation still has a finite budget. Large contexts or complex questions can require a smaller model, fewer chunks, or a larger `--max-tokens` value.

Retrieved text and the question stay in an untrusted JSON payload, separate from the fixed system instructions. No tools, database credentials, or local file-reading capability are given to the model. Context and output sizes are bounded. These are safeguards, not a guarantee against prompt injection or hallucination: the grounding checker is also a model, and valid citations alone do not prove an answer is true. See the [development roadmap](docs/ROADMAP.md) for the remaining evaluation and security work.

## Retrieval evaluation

```bash
rag-lab evaluate --output reports/retrieval-quality.json
```

The [14-case regression dataset](benchmarks/questions.jsonl) has questions, expected answers, source filenames, and gold evidence quotes (12 answerable, two unanswerable). Evaluation verifies local quotes and indexed source hashes first, then compares raw vector ranking against filtered/deduplicated/reranked candidates. It records document Recall@1/3/5, gold-quote hit rates, and unanswerable queries that still return a match. Answer accuracy and faithfulness evaluation are still pending.

The [measured report](reports/retrieval-quality.json) records dataset/source hashes, embedding model digest, 40-candidate pool, 0.3 threshold, and a reproducible weight sweep. On this small set, 5% lexical weighting increased evidence hits at rank 1 from 4/12 to 5/12 while keeping hits within the top 3 at 11/12. One of two unanswerable questions still returned a match. Document Recall@K was already 100%, so that metric alone concealed passage-level mistakes. These are exploratory calibration results on the same cases, not held-out generalization claims. Postprocessing timing excludes embedding and database search.

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
