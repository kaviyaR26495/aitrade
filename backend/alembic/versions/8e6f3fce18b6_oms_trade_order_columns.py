"""oms_trade_order_columns

Revision ID: 8e6f3fce18b6
Revises: 1e1e71e906a3
Create Date: 2026-04-10 23:23:58.962248

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8e6f3fce18b6'
down_revision: Union[str, Sequence[str], None] = '1e1e71e906a3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""
        ALTER TABLE trade_orders MODIFY COLUMN status ENUM(
            'pending','placed','completed','cancelled','rejected',
            'submitted','partial_fill','filled'
        ) NOT NULL DEFAULT 'pending'
    """)
    op.add_column('trade_orders', sa.Column('filled_quantity', sa.Integer(), nullable=True))
    op.add_column('trade_orders', sa.Column('avg_fill_price', sa.Float(), nullable=True))
    op.add_column('trade_orders', sa.Column('last_reconciled_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('trade_orders', 'last_reconciled_at')
    op.drop_column('trade_orders', 'avg_fill_price')
    op.drop_column('trade_orders', 'filled_quantity')
    op.execute("""
        ALTER TABLE trade_orders MODIFY COLUMN status ENUM(
            'pending','placed','completed','cancelled','rejected'
        ) NOT NULL DEFAULT 'pending'
    """)
