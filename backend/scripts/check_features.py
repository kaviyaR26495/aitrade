"""Sanity check: verify 26 stationary ML features load without NaN/inf."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.db.database import get_db


async def main() -> None:
    # Import here so path is set up first
    from app.core.data_service import get_stock_features
    from app.core.normalizer import ML_FEATURES, get_feature_columns

    async for db in get_db():
        df = await get_stock_features(db, stock_id=1, interval="day", normalize=True)
        if df is None or df.empty:
            print("ERROR: no data returned for stock_id=1")
            return

        feature_cols = get_feature_columns(df)
        print(f"Feature count : {len(feature_cols)}  (expected {len(ML_FEATURES)})")
        print(f"Columns       : {feature_cols}")

        missing = [f for f in ML_FEATURES if f not in df.columns]
        if missing:
            print(f"MISSING cols  : {missing}")

        data = df[feature_cols].values.astype(float)
        print(f"Shape         : {data.shape}")
        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())
        print(f"NaN count     : {nan_count}")
        print(f"Inf count     : {inf_count}")

        print()
        print(f"{'Column':<32} {'min':>10} {'max':>10} {'mean':>10} {'NaN':>6}")
        print("-" * 72)
        for col in feature_cols:
            s = df[col].dropna()
            if len(s) == 0:
                print(f"  {col:<30} {'(all NaN)':>42}")
                continue
            nan_n = int(df[col].isna().sum())
            print(
                f"  {col:<30} {s.min():>10.4f} {s.max():>10.4f} {s.mean():>10.4f} {nan_n:>6}"
            )
        break


if __name__ == "__main__":
    asyncio.run(main())
