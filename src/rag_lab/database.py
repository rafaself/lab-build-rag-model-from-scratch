"""PostgreSQL connections with explicit driver selection and scoped sessions."""
from contextlib import contextmanager
from collections.abc import Iterator

from pgvector.psycopg2 import register_vector
from sqlalchemy import URL, create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from rag_lab.config import Settings


def create_db_engine(settings: Settings) -> Engine:
    if not settings.postgres_password:
        raise ValueError("Set POSTGRES_PASSWORD in .env before connecting to PostgreSQL.")
    url = URL.create(
        "postgresql+psycopg2",
        username=settings.postgres_user,
        password=settings.postgres_password,
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
    )
    engine = create_engine(url, connect_args={"connect_timeout": 5}, pool_pre_ping=True)

    @event.listens_for(engine, "connect")
    def register_types(connection, _connection_record):
        register_vector(connection)

    return engine


@contextmanager
def get_session(settings: Settings) -> Iterator[Session]:
    """Close the session and engine even on failure; callers commit explicitly."""
    engine = create_db_engine(settings)
    try:
        with Session(engine) as session:
            yield session
    finally:
        engine.dispose()


def check_database(settings: Settings) -> dict:
    with get_session(settings) as session:
        row = session.execute(text(
            "SELECT current_database() AS database, current_user AS username, "
            "(SELECT extversion FROM pg_extension WHERE extname = 'vector') AS pgvector"
        )).mappings().one()
        return dict(row)
