"""merge_heads

Revision ID: 68d371710d95
Revises: a1b2c3d4e5f6, d4c3ab9fa4a2
Create Date: 2026-04-10 21:26:23.904650

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '68d371710d95'
down_revision: Union[str, Sequence[str], None] = ('a1b2c3d4e5f6', 'd4c3ab9fa4a2')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
