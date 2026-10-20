# Lab: build a RAG model from scratch

A local study project using Ollama for embeddings and answer generation, with PostgreSQL and pgvector available for storing text and vectors.

## Python environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Local PostgreSQL

The Compose service provides PostgreSQL 17 with pgvector 0.8.7, persistent storage, and a readiness health check. It publishes only on the host's loopback interface. A separate application role can create tables in the `public` schema and has no superuser, role-creation, or database-creation privileges.

| Setting | Default |
| --- | --- |
| Host | `127.0.0.1` |
| Host port | `55432` |
| Database | `rag_lab` |
| Application user | `rag_app` |
| Application password | `RAG_DB_PASSWORD` in your ignored `.env` |
| Administrator | `postgres`, with its separate `POSTGRES_PASSWORD` |
| Compose service | `postgres` |
| Persistent storage | Project-scoped `postgres_data` named volume |

### Configure and start

If a local `.env` already exists, keep it. Otherwise:

```bash
cp .env.example .env
chmod 600 .env
```

Set `POSTGRES_PASSWORD` and `RAG_DB_PASSWORD` to two different, strong passwords. For example, run `openssl rand -hex 32` twice and paste each generated value into its corresponding `.env` entry. The example file intentionally has no default passwords, and Compose refuses to start if either is empty. Keep the local `.env` out of Git.

From the repository root:

```bash
docker compose config --quiet
docker compose up -d --wait postgres
docker compose ps
```

The default host port avoids the PostgreSQL instances already using ports 5432 and 5433 on the development computer. Change `POSTGRES_PORT` in `.env` if another port is needed. Processes running in this Compose network should connect to `postgres:5432`; host Python scripts should use `127.0.0.1:55432` by default.

### Linux with a VPN that blocks Docker bridge traffic

On this development computer, PostgreSQL worked inside the container, but the host PostgreSQL handshake timed out through Docker's published port. NordVPN's firewall is enabled and the newly allocated Docker subnet is outside its allowlist.

The optional `compose.host.yaml` uses Linux host networking and explicitly binds PostgreSQL to `127.0.0.1` on `POSTGRES_PORT`. It bypasses the Docker bridge path. To enable it, add this line to your ignored `.env`:

```dotenv
COMPOSE_FILE=compose.yaml:compose.host.yaml
```

Then run the same `docker compose up -d --wait postgres` command. The project name and named data volume remain the same. This mode shares the host's network namespace and has no Docker port mapping or Compose service DNS name; host scripts still connect to `127.0.0.1:55432`, and `docker compose exec` still works. It requires Linux and a Compose version supporting `!reset` (validated with Compose 5.6.0). The local `.env` created during setup already enables this option. A new checkout can use the default bridge setup or enable this option as needed. See Docker's [host networking](https://docs.docker.com/engine/network/drivers/host/) and [Compose merge](https://docs.docker.com/reference/compose-file/merge/) documentation.

### Connect as the application user

With a host PostgreSQL client installed:

```bash
psql -h 127.0.0.1 -p 55432 -U rag_app -d rag_lab -W
```

Enter the value of `RAG_DB_PASSWORD` when prompted. Substitute your configured port, user, and database if you changed them.

An administrative SQL console is available inside the container:

```bash
docker compose exec postgres sh -c 'exec psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

Confirm the vector extension:

```sql
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

### Stop and preserve data

```bash
docker compose down
```

The named volume preserves the database across container recreation. Initialization scripts and the `.env` credentials apply when the volume is first created. Changing passwords in `.env` later does not change existing PostgreSQL role passwords; change them through PostgreSQL and update `.env` together. Do not use `docker compose down --volumes` unless you intend to delete this project's database.

The init script enables `vector` and grants the application role permission to create its own tables. It leaves the document/chunk schema and vector dimensions for the study implementation. The pgvector extension supports vector storage and similarity search; see [its documentation](https://github.com/pgvector/pgvector#readme). Container initialization follows the [PostgreSQL image documentation](https://hub.docker.com/_/postgres).
