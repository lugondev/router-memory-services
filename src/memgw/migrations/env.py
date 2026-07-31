"""Alembic environment, driven from code rather than from alembic.ini.

The database URL comes from whatever called us -- ``Catalog`` passes its own
engine straight through -- so there is no second place where a connection string
is written down and no way for the two to disagree.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from memgw.catalog import metadata

target_metadata = metadata


def run_migrations_offline() -> None:
    context.configure(
        url=context.config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # SQLite cannot ALTER most things, so every migration is written through a
        # batch operation: create the new table, copy, swap. Postgres ignores this.
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = context.config.attributes.get("connection", None)
    if connectable is not None:
        do_run_migrations(connectable)
        return

    engine = engine_from_config(
        context.config.get_section(context.config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with engine.connect() as connection:
        do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
