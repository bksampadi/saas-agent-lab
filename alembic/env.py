from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import Connection

from app.core.config import get_settings
from app.core.database import create_db_engine
from app.models import Base
from app.models.base import UTCDateTime

config = context.config

if config.config_file_name is not None:
    # Keep loggers configured elsewhere (e.g. by pytest) working.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def render_item(type_: str, obj: Any, autogen_context: Any) -> str | bool:
    # Render our custom column type as a plain DateTime so migration files
    # never import application code, which will change over time.
    if type_ == "type" and isinstance(obj, UTCDateTime):
        return "sa.DateTime(timezone=True)"
    return False


def configure_context(**kwargs: Any) -> None:
    context.configure(
        target_metadata=target_metadata,
        # SQLite cannot ALTER most things in place; batch mode rebuilds the
        # table instead. It is a no-op cost on Postgres.
        render_as_batch=True,
        render_item=render_item,
        compare_type=True,
        **kwargs,
    )


def run_migrations_offline() -> None:
    configure_context(
        url=get_settings().database_url,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_with_connection(connection: Connection) -> None:
    configure_context(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # Tests pass in an existing connection (to an in-memory database), so
    # migrations run against it instead of the configured URL.
    connection = config.attributes.get("connection")
    if connection is not None:
        run_with_connection(connection)
        return

    engine = create_db_engine(get_settings().database_url)
    try:
        with engine.connect() as connection:
            run_with_connection(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
