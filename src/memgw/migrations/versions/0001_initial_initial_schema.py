"""The schema as it stood when migrations were introduced.

Generated from ``memgw.catalog.metadata`` rather than written by hand, so this
revision and the models cannot disagree about where they started.
``tests/test_migrations.py`` asserts they still agree.

Revision ID: 0001_initial
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "episode_journal",
        sa.Column("episode_id", sa.String(length=48), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("subject", sa.String(length=256), nullable=False),
        sa.Column("agent", sa.String(length=256), nullable=True),
        sa.Column("session", sa.String(length=256), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("ingested_to", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("episode_id"),
    )
    with op.batch_alter_table("episode_journal", schema=None) as batch_op:
        batch_op.create_index("ix_episode_journal_scope", ["tenant_id", "subject"], unique=False)

    op.create_table(
        "memory_index",
        sa.Column("gateway_id", sa.String(length=48), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("native_id", sa.String(length=512), nullable=False),
        sa.Column("subject", sa.String(length=256), nullable=False),
        sa.Column("agent", sa.String(length=256), nullable=True),
        sa.Column("session", sa.String(length=256), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("gateway_id"),
        sa.UniqueConstraint("tenant_id", "provider", "native_id", name="uq_memory_index_native"),
    )
    with op.batch_alter_table("memory_index", schema=None) as batch_op:
        batch_op.create_index(
            "ix_memory_index_scope", ["tenant_id", "subject", "provider"], unique=False
        )

    op.create_table(
        "scope_binding",
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("subject", sa.String(length=256), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("bound_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("migrated_from", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("tenant_id", "subject"),
    )


def downgrade() -> None:
    op.drop_table("scope_binding")
    with op.batch_alter_table("memory_index", schema=None) as batch_op:
        batch_op.drop_index("ix_memory_index_scope")

    op.drop_table("memory_index")
    with op.batch_alter_table("episode_journal", schema=None) as batch_op:
        batch_op.drop_index("ix_episode_journal_scope")

    op.drop_table("episode_journal")
