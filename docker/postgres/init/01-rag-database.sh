#!/usr/bin/env bash
set -euo pipefail

if [[ "$RAG_DB_USER" == "$POSTGRES_USER" ]]; then
  echo "RAG_DB_USER must differ from the PostgreSQL administrator." >&2
  exit 1
fi

psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  --set=ON_ERROR_STOP=1 \
  --set=app_user="$RAG_DB_USER" \
  --set=app_password="$RAG_DB_PASSWORD" \
  --set=db_name="$POSTGRES_DB" <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE ROLE :"app_user" LOGIN PASSWORD :'app_password';
GRANT CONNECT ON DATABASE :"db_name" TO :"app_user";
GRANT USAGE, CREATE ON SCHEMA public TO :"app_user";
SQL
