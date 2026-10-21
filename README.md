# Lab: build a RAG model from scratch

A local study project using Ollama for embeddings and answer generation, with PostgreSQL and pgvector available for storing text and vectors.

## Python environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Local PostgreSQL

One PostgreSQL 17 service with pgvector, a persistent volume, and a readiness check. This local Linux setup uses host networking because Docker bridge connections were blocked on the development computer. PostgreSQL listens only on localhost.

| Setting | Default |
| --- | --- |
| Host | `127.0.0.1` |
| Host port | `55432` |
| Database | `rag_lab` |
| User | `postgres` |
| Password | `POSTGRES_PASSWORD` in `.env` |

Keep your existing `.env`. For a new checkout, copy the example and set `POSTGRES_PASSWORD`:

```bash
cp .env.example .env
chmod 600 .env
```

Start the database:

```bash
docker compose up -d --wait postgres
```

Open a SQL console:

```bash
docker compose exec postgres sh -c 'psql -U postgres -d "$POSTGRES_DB"'
```

Stop the database:

```bash
docker compose down
```

`docker compose down` keeps the database volume. Adding `--volumes` deletes the data. The init SQL enables pgvector on the first startup; document tables and embedding dimensions are left for the lab exercises. Changing `.env` later does not change passwords in an existing database.
