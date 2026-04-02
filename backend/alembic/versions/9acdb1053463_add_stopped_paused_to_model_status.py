"""add_stopped_paused_to_model_status

Revision ID: 9acdb1053463
Revises: ceb765788569
Create Date: 2026-04-01 16:39:06.417925

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9acdb1053463'
down_revision: Union[str, Sequence[str], None] = 'ceb765788569'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        "ALTER TABLE rl_models MODIFY COLUMN status "
        "ENUM('pending','training','completed','failed','stopped','paused') NOT NULL"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        "ALTER TABLE rl_models MODIFY COLUMN status "
        "ENUM('pending','training','completed','failed') NOT NULL"
    )
