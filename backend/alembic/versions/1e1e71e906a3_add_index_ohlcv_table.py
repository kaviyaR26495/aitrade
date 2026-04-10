"""add_index_ohlcv_table

Revision ID: 1e1e71e906a3
Revises: a5ebeaf4bb4e
Create Date: 2026-04-10 22:40:14.248762

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1e1e71e906a3'
down_revision: Union[str, Sequence[str], None] = 'a5ebeaf4bb4e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "index_ohlcv",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("interval", sa.String(10), nullable=False, server_default="day"),
        sa.Column("open", sa.Float(), nullable=True),
        sa.Column("high", sa.Float(), nullable=True),
        sa.Column("low", sa.Float(), nullable=True),
        sa.Column("close", sa.Float(), nullable=True),
        sa.Column("volume", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "date", "interval", name="uq_index_ohlcv"),
    )


def downgrade() -> None:
    op.drop_table("index_ohlcv")
