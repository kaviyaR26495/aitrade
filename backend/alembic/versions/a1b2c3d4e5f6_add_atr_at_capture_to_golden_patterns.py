"""add atr_at_capture to golden_patterns

Revision ID: a1b2c3d4e5f6
Revises: 301a773c180c
Create Date: 2026-04-07

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = 'a1b2c3d4e5f6'
down_revision = '301a773c180c'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'golden_patterns',
        sa.Column('atr_at_capture', sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('golden_patterns', 'atr_at_capture')
