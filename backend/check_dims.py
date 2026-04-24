import asyncio
from datetime import date
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import async_session_factory
from app.api.routes.backtest import _run_live_inference
from app.config import settings

# monkeypatch to print dimensions
import app.ml.knn_distiller as md

orig_predict_knn = md.predict_knn
def patched_predict(knn_model, X, norm_params=None):
    from sklearn.neighbors import KNeighborsClassifier
    X_flat = X.reshape(len(X), -1)
    
    # Check shape
    print(f"[DEBUG] X shape: {X.shape}, X_flat length: {X_flat.shape[1]}")
    try:
        print(f"[DEBUG] model expected dim (faiss index): {knn_model._index.d if hasattr(knn_model, '_index') else knn_model.n_features_in_}")
    except Exception as e:
        pass
    
    return orig_predict_knn(knn_model, X, norm_params)

md.predict_knn = patched_predict

async def main():
    async with async_session_factory() as session:
        try:
            dates = [date(2026, 4, d) for d in range(1, 23)]
            pred = await _run_live_inference(session, 3, 'day', dates, date(2026,4,1), date(2026,4,22))
            print('Preds:', len(pred))
        except Exception as e:
            pass

asyncio.run(main())
