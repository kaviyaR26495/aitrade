"""add_alpha_engine_tables

Revision ID: f1a2b3c4d5e6
Revises: d4c3ab9fa4a2
Create Date: 2026-04-18 00:00:00.000000

Adds:
  - stock_fundamentals_pit        (PIT fundamental snapshots)
  - fundamental_sector_stats      (daily sector PE aggregates)
  - stock_fundamental_zscores     (ML-ready derived z-scores)
  - stock_sentiment               (FinBERT + LLM daily sentiment)
  - lstm_horizon_models           (multi-horizon LSTM registry)
  - lstm_horizon_predictions      (10-day rolling forecasts)
  - regime_ensemble_map           (regime_id → ensemble_config_id)
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "8e6f3fce18b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── stock_fundamentals_pit ────────────────────────────────────────
    op.create_table(
        "stock_fundamentals_pit",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("stock_id", sa.Integer(), sa.ForeignKey("stocks_list.id"), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("pe_ratio", sa.Float(), nullable=True),
        sa.Column("forward_pe", sa.Float(), nullable=True),
        sa.Column("pb_ratio", sa.Float(), nullable=True),
        sa.Column("dividend_yield", sa.Float(), nullable=True),
        sa.Column("roe", sa.Float(), nullable=True),
        sa.Column("debt_equity", sa.Float(), nullable=True),
        sa.Column("source", sa.String(20), nullable=False, server_default="yfinance"),
        sa.Column("ingested_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("stock_id", "date", name="uq_fundamental_pit"),
    )
    op.create_index("ix_fundamental_pit_stock_date", "stock_fundamentals_pit", ["stock_id", "date"])

    # ── fundamental_sector_stats ──────────────────────────────────────
    op.create_table(
        "fundamental_sector_stats",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("sector", sa.String(100), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("sector_pe_avg", sa.Float(), nullable=True),
        sa.Column("sector_pe_std", sa.Float(), nullable=True),
        sa.Column("stock_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("computed_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("sector", "date", name="uq_sector_stats"),
    )
    op.create_index("ix_sector_stats_date", "fundamental_sector_stats", ["date"])

    # ── stock_fundamental_zscores ─────────────────────────────────────
    op.create_table(
        "stock_fundamental_zscores",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("stock_id", sa.Integer(), sa.ForeignKey("stocks_list.id"), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("pe_zscore_3y", sa.Float(), nullable=True),
        sa.Column("pe_zscore_sector", sa.Float(), nullable=True),
        sa.Column("roe_norm", sa.Float(), nullable=True),
        sa.Column("debt_equity_norm", sa.Float(), nullable=True),
        sa.Column("computed_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("stock_id", "date", name="uq_fundamental_zscore"),
    )
    op.create_index("ix_fundamental_zscore_stock_date", "stock_fundamental_zscores", ["stock_id", "date"])

    # ── stock_sentiment ───────────────────────────────────────────────
    op.create_table(
        "stock_sentiment",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("stock_id", sa.Integer(), sa.ForeignKey("stocks_list.id"), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("headline_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("neutral_filtered_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("avg_finbert_score", sa.Float(), nullable=True),
        sa.Column("llm_impact_score", sa.Float(), nullable=True),
        sa.Column("llm_summary", sa.Text(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("stock_id", "date", name="uq_sentiment"),
    )
    op.create_index("ix_sentiment_stock_date", "stock_sentiment", ["stock_id", "date"])

    # ── lstm_horizon_models ───────────────────────────────────────────
    op.create_table(
        "lstm_horizon_models",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("source_rl_model_id", sa.Integer(), sa.ForeignKey("rl_models.id"), nullable=True),
        sa.Column("hidden_size", sa.Integer(), nullable=False, server_default="256"),
        sa.Column("num_layers", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("seq_len", sa.Integer(), nullable=False, server_default="15"),
        sa.Column("horizon", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("model_path", sa.String(500), nullable=True),
        sa.Column("norm_params_path", sa.String(500), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column(
            "status",
            sa.Enum("pending", "training", "completed", "failed", "stopped", "paused", name="modelstatus"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── lstm_horizon_predictions ──────────────────────────────────────
    op.create_table(
        "lstm_horizon_predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_id", sa.Integer(), sa.ForeignKey("lstm_horizon_models.id"), nullable=False),
        sa.Column("stock_id", sa.Integer(), sa.ForeignKey("stocks_list.id"), nullable=False),
        sa.Column("prediction_date", sa.Date(), nullable=False),
        *[
            col
            for i in range(1, 11)
            for col in (
                sa.Column(f"h{i}_action", sa.Integer(), nullable=True),
                sa.Column(f"h{i}_conf", sa.Float(), nullable=True),
            )
        ],
        sa.Column("trend_durability_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_id", "stock_id", "prediction_date", name="uq_horizon_pred"),
    )
    op.create_index("ix_horizon_pred_date", "lstm_horizon_predictions", ["prediction_date"])

    # ── regime_ensemble_map ───────────────────────────────────────────
    op.create_table(
        "regime_ensemble_map",
        sa.Column("regime_id", sa.Integer(), primary_key=True),
        sa.Column(
            "ensemble_config_id",
            sa.Integer(),
            sa.ForeignKey("ensemble_configs.id"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("regime_id"),
    )


def downgrade() -> None:
    op.drop_table("regime_ensemble_map")
    op.drop_index("ix_horizon_pred_date", table_name="lstm_horizon_predictions")
    op.drop_table("lstm_horizon_predictions")
    op.drop_table("lstm_horizon_models")
    op.drop_index("ix_sentiment_stock_date", table_name="stock_sentiment")
    op.drop_table("stock_sentiment")
    op.drop_index("ix_fundamental_zscore_stock_date", table_name="stock_fundamental_zscores")
    op.drop_table("stock_fundamental_zscores")
    op.drop_index("ix_sector_stats_date", table_name="fundamental_sector_stats")
    op.drop_table("fundamental_sector_stats")
    op.drop_index("ix_fundamental_pit_stock_date", table_name="stock_fundamentals_pit")
    op.drop_table("stock_fundamentals_pit")
