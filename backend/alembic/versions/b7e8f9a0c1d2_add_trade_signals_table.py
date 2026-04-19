"""add_trade_signals_table

Revision ID: b7e8f9a0c1d2
Revises: f1a2b3c4d5e6
Create Date: 2026-05-01 10:00:00.000000

Adds:
  - trade_signals table (target-price pipeline output)
  - trade_signal_id FK column on trade_orders
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b7e8f9a0c1d2"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── trade_signals ─────────────────────────────────────────────────
    op.create_table(
        "trade_signals",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("stock_id", sa.Integer, sa.ForeignKey("stocks_list.id"), nullable=False),
        sa.Column("signal_date", sa.Date, nullable=False),
        # Price levels
        sa.Column("entry_price", sa.Float, nullable=False),
        sa.Column("target_price", sa.Float, nullable=False),
        sa.Column("stoploss_price", sa.Float, nullable=False),
        sa.Column("current_stoploss", sa.Float, nullable=True),
        # Confidence & quality scores
        sa.Column("pop_score", sa.Float, nullable=True),
        sa.Column("fqs_score", sa.Float, nullable=True),
        sa.Column("confluence_score", sa.Float, nullable=True),
        sa.Column("execution_cost_pct", sa.Float, nullable=True),
        # κ-decay tracking
        sa.Column("initial_rr_ratio", sa.Float, nullable=True),
        sa.Column("current_rr_ratio", sa.Float, nullable=True),
        sa.Column("days_since_signal", sa.Integer, nullable=False, server_default="0"),
        # Model source references
        sa.Column("lstm_mu", sa.Float, nullable=True),
        sa.Column("lstm_sigma", sa.Float, nullable=True),
        sa.Column("knn_median_return", sa.Float, nullable=True),
        sa.Column("knn_win_rate", sa.Float, nullable=True),
        # Trailing stop state
        sa.Column("is_trailing_active", sa.Boolean, nullable=False, server_default="0"),
        sa.Column("trailing_updates_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_gtt_id", sa.String(50), nullable=True),
        # Status & metadata
        sa.Column(
            "status",
            sa.Enum("pending", "active", "target_hit", "sl_hit", "expired", "cancelled",
                    name="signalstatus"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("regime_id", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=True),
        sa.Column("updated_at", sa.DateTime, nullable=True),
        # Constraints
        sa.UniqueConstraint("stock_id", "signal_date", name="uq_signal_stock_date"),
    )
    op.create_index("ix_signal_status", "trade_signals", ["status"])
    op.create_index("ix_signal_date", "trade_signals", ["signal_date"])

    # ── Add trade_signal_id FK to trade_orders ────────────────────────
    op.add_column(
        "trade_orders",
        sa.Column("trade_signal_id", sa.Integer, sa.ForeignKey("trade_signals.id"), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("trade_orders", "trade_signal_id")
    op.drop_index("ix_signal_date", table_name="trade_signals")
    op.drop_index("ix_signal_status", table_name="trade_signals")
    op.drop_table("trade_signals")
    op.execute("DROP TYPE IF EXISTS signalstatus")
