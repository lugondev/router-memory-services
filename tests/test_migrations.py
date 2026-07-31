"""The catalog schema has to be able to change after there is data in it.

Until now ``Catalog.init()`` ran ``CREATE TABLE IF NOT EXISTS`` and nothing else,
which is fine exactly once. The second release that needs a column has no way to
add one, and the failure mode is the worst kind: an old database, a new binary,
and an error at the first query that mentions the new column.

So the schema has a version, migrations move it, and ``memgw doctor`` says when
the database is behind the code -- before the code touches it.
"""

from __future__ import annotations

import pytest

from memgw.catalog import Catalog


class TestAFreshDatabaseIsCurrent:
    async def test_init_leaves_a_new_database_at_head(self, tmp_path):
        catalog = Catalog(f"sqlite+aiosqlite:///{tmp_path}/fresh.db")
        await catalog.init()
        state = await catalog.schema_state()
        assert state.current == state.head
        assert state.up_to_date is True
        await catalog.close()

    async def test_the_head_revision_is_a_real_revision_not_none(self, tmp_path):
        # If head were None the check would pass vacuously forever.
        catalog = Catalog(f"sqlite+aiosqlite:///{tmp_path}/fresh.db")
        await catalog.init()
        state = await catalog.schema_state()
        assert state.head
        await catalog.close()

    async def test_an_in_memory_catalog_still_works(self, tmp_path):
        # Tests and embedded mode use :memory:, which must not need a migration run.
        del tmp_path
        catalog = Catalog("sqlite+aiosqlite:///:memory:")
        await catalog.init()
        assert (await catalog.schema_state()).up_to_date is True
        await catalog.close()


class TestAnOldDatabaseIsCaughtBeforeItIsUsed:
    def test_a_revision_behind_head_is_not_up_to_date(self):
        # Pure, and deliberately so: with one revision in the tree no real database
        # can be behind, but the comparison has to be right *before* the second
        # revision exists, not after somebody discovers it was not.
        from memgw.catalog import SchemaState

        assert SchemaState(current="0001_initial", head="0002_later").up_to_date is False
        assert SchemaState(current=None, head="0001_initial").up_to_date is False
        assert SchemaState(current="0001_initial", head="0001_initial").up_to_date is True

    def test_being_behind_says_what_to_run(self):
        from memgw.catalog import SchemaState

        described = SchemaState(current="0001_initial", head="0002_later").describe()
        assert "memgw migrate" in described
        assert "0002_later" in described

    async def test_tables_that_predate_migrations_are_adopted_rather_than_left_unversioned(
        self, tmp_path
    ):
        """The upgrade path for anyone already running memgw.

        Their catalog was built by ``create_all`` and has no ``alembic_version`` at
        all. Left alone it would look like a database behind every migration, and
        ``memgw migrate`` would try to create tables that are already there.
        """
        from memgw.catalog import metadata

        url = f"sqlite+aiosqlite:///{tmp_path}/legacy.db"
        legacy = Catalog(url)
        async with legacy.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)  # exactly what the old init() did
        assert await legacy._revision() is None
        await legacy.close()

        catalog = Catalog(url)
        await catalog.init()
        assert (await catalog.schema_state()).up_to_date is True
        await catalog.close()

    async def test_upgrade_builds_an_empty_database_from_the_migrations_alone(self, tmp_path):
        catalog = Catalog(f"sqlite+aiosqlite:///{tmp_path}/empty.db")
        await catalog.upgrade()
        assert (await catalog.schema_state()).up_to_date is True
        await catalog.close()

    async def test_migrating_an_already_current_database_changes_nothing(self, tmp_path):
        catalog = Catalog(f"sqlite+aiosqlite:///{tmp_path}/current.db")
        await catalog.init()
        before = await catalog.schema_state()
        await catalog.upgrade()
        assert await catalog.schema_state() == before
        await catalog.close()


class TestDoctorSaysSoBeforeAnythingDependsOnIt:
    async def test_a_behind_schema_is_a_failed_check(self, tmp_path, monkeypatch):
        from memgw.catalog import SchemaState
        from memgw.cli import doctor

        async def behind(self):
            return SchemaState(current="0001_initial", head="0002_later")

        monkeypatch.setattr(Catalog, "schema_state", behind)

        url = f"sqlite+aiosqlite:///{tmp_path}/old.db"
        report = await doctor({**_env(), "MEMGW_CATALOG_URL": url})
        check = next(c for c in report.checks if c.name == "schema")
        assert check.ok is False
        assert "memgw migrate" in check.detail
        assert report.ok is False

    async def test_a_missing_database_driver_is_a_check_and_not_a_traceback(self):
        """Doctor exists so that a misconfiguration reads as a sentence.

        ``create_async_engine`` imports the DBAPI eagerly, so a Postgres URL with no
        ``asyncpg`` installed raised out of the constructor -- before any of doctor's
        error handling, which is wrapped around *using* the catalog rather than
        building it. The user's answer was a stack trace ending in ModuleNotFoundError.
        """
        from memgw.cli import doctor

        report = await doctor(
            {
                **_env(),
                "MEMGW_CATALOG_URL": "postgresql+psycopg_does_not_exist://u:p@localhost/db",
            }
        )
        check = next(c for c in report.checks if c.name == "catalog")
        assert check.ok is False
        assert report.ok is False

    async def test_a_current_schema_passes(self, tmp_path):
        from memgw.cli import doctor

        url = f"sqlite+aiosqlite:///{tmp_path}/new.db"
        report = await doctor({**_env(), "MEMGW_CATALOG_URL": url})
        assert next(c for c in report.checks if c.name == "schema").ok is True


class TestTheMigrationsMatchTheModels:
    async def test_no_column_has_been_added_to_a_table_without_a_migration(self, tmp_path):
        """The drift this guards against is silent: someone adds a column to
        ``memory_index``, tests pass because tests build fresh databases, and the
        first upgraded deployment fails on a column that was never migrated in."""
        from sqlalchemy import inspect

        catalog = Catalog(f"sqlite+aiosqlite:///{tmp_path}/drift.db")
        await catalog.upgrade()  # migrations only -- no create_all

        from memgw.catalog import metadata

        async with catalog.engine.connect() as conn:
            tables = await conn.run_sync(lambda c: inspect(c).get_table_names())
            for table in metadata.sorted_tables:
                assert table.name in tables, f"{table.name} has no migration"
                cols = await conn.run_sync(
                    lambda c, t=table.name: {col["name"] for col in inspect(c).get_columns(t)}
                )
                missing = {c.name for c in table.columns} - cols
                assert not missing, f"{table.name} is missing {missing} in the migrations"
        await catalog.close()


@pytest.fixture(autouse=True)
def _quiet_alembic():
    import logging

    logging.getLogger("alembic").setLevel(logging.WARNING)


def _env() -> dict[str, str]:
    return {
        "OPENAI_API_KEY": "sk-test",
        "MEMGW_PROVIDERS": "pgvector",
        "MEMGW_API_KEYS": "k1:tenant-a",
    }
