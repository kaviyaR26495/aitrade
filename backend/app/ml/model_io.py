"""Model Export / Import — portable ZIP bundles for KNN + LSTM ensembles.

Bundle format (ZIP archive):
  aitrade_model_export_YYYYMMDD_HHMM.zip
  ├── manifest.json        ← DB metadata + ensemble config + version tag
  ├── knn/
  │   ├── knn_model.joblib ← FaissKNNClassifier serialised with joblib
  │   ├── norm_params.json ← z-score mean/scale used at inference time
  │   └── metadata.json    ← training metrics + feature column list
  └── lstm/
      ├── lstm_model.pt    ← TradeLSTM state_dict + arch params
      └── metadata.json    ← training metrics + feature column list

Usage
-----
Export from API:
    GET /api/models/export?knn_model_id=3&lstm_model_id=2
    → streams a .zip file download

Import to a new system (or same system for rollback / transfer):
    POST /api/models/import  (multipart form, field name = "file")
    → extracts the bundle, places artifacts, creates DB records,
      returns {knn_model_id, lstm_model_id, ensemble_config_id}

The imported models are immediately usable by:
  • run_daily_predictions(knn_model_id=X, lstm_model_id=Y)
  • run_backtest(model_type="ensemble", knn_model_id=X, lstm_model_id=Y)
"""
from __future__ import annotations

import io
import json
import logging
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

BUNDLE_VERSION = "1.0"
EXPORT_SUBDIR = "exports"          # placed under settings.MODEL_DIR


# ── Export ─────────────────────────────────────────────────────────────

async def export_ensemble(
    db: AsyncSession,
    knn_model_id: int,
    lstm_model_id: int,
    output_dir: Path | None = None,
) -> Path:
    """Build a portable ZIP bundle from trained KNN + LSTM artifacts.

    Parameters
    ----------
    db             : async DB session
    knn_model_id   : primary key of the KNNModel row
    lstm_model_id  : primary key of the LSTMModel row
    output_dir     : directory where the .zip is written
                     (defaults to MODEL_DIR/exports)

    Returns
    -------
    Path to the created .zip file.
    """
    from app.config import settings
    from app.db import crud

    knn_db = await crud.get_knn_model(db, knn_model_id)
    if knn_db is None:
        raise ValueError(f"KNN model id={knn_model_id} not found in DB")
    if not knn_db.model_path or not Path(knn_db.model_path).exists():
        raise FileNotFoundError(
            f"KNN model artifact not found on disk: {knn_db.model_path}"
        )

    lstm_db = await crud.get_lstm_model(db, lstm_model_id)
    if lstm_db is None:
        raise ValueError(f"LSTM model id={lstm_model_id} not found in DB")
    if not lstm_db.model_path or not Path(lstm_db.model_path).exists():
        raise FileNotFoundError(
            f"LSTM model artifact not found on disk: {lstm_db.model_path}"
        )

    # Resolve optional ensemble config that links these two models
    from sqlalchemy import select, desc
    from app.db.models import EnsembleConfig
    q = (
        select(EnsembleConfig)
        .where(
            EnsembleConfig.knn_model_id == knn_model_id,
            EnsembleConfig.lstm_model_id == lstm_model_id,
        )
        .order_by(desc(EnsembleConfig.created_at))
        .limit(1)
    )
    res = await db.execute(q)
    ens_db = res.scalar_one_or_none()

    # ── Build manifest ────────────────────────────────────────────────
    manifest: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "export_date": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "knn": {
            "id": knn_db.id,
            "name": knn_db.name,
            "k_neighbors": knn_db.k_neighbors,
            "seq_len": knn_db.seq_len,
            "interval": knn_db.interval.value if hasattr(knn_db.interval, "value") else str(knn_db.interval),
            "accuracy": knn_db.accuracy,
            "precision_buy": knn_db.precision_buy,
            "precision_sell": knn_db.precision_sell,
            "regime_filter": knn_db.regime_filter,
            "feature_combination": knn_db.feature_combination,
        },
        "lstm": {
            "id": lstm_db.id,
            "name": lstm_db.name,
            "hidden_size": lstm_db.hidden_size,
            "num_layers": lstm_db.num_layers,
            "dropout": lstm_db.dropout,
            "seq_len": lstm_db.seq_len,
            "interval": lstm_db.interval.value if hasattr(lstm_db.interval, "value") else str(lstm_db.interval),
            "accuracy": lstm_db.accuracy,
            "precision_buy": lstm_db.precision_buy,
            "precision_sell": lstm_db.precision_sell,
            "regime_filter": lstm_db.regime_filter,
        },
        "ensemble": {
            "id": ens_db.id if ens_db else None,
            "name": ens_db.name if ens_db else f"Ensemble_{knn_model_id}_{lstm_model_id}",
            "knn_weight": ens_db.knn_weight if ens_db else 0.5,
            "lstm_weight": ens_db.lstm_weight if ens_db else 0.5,
            "agreement_required": ens_db.agreement_required if ens_db else True,
            "interval": ens_db.interval.value if ens_db and hasattr(ens_db.interval, "value") else (
                ens_db.interval if ens_db else (
                    knn_db.interval.value if hasattr(knn_db.interval, "value") else str(knn_db.interval)
                )
            ),
        },
    }

    # ── Collect files to bundle ──────────────────────────────────────
    knn_dir = Path(knn_db.model_path).parent
    lstm_dir = Path(lstm_db.model_path).parent

    def _exists(p: Path) -> Path | None:
        return p if p.exists() else None

    knn_files = {
        "knn/knn_model.joblib": Path(knn_db.model_path),
        "knn/norm_params.json": _exists(knn_dir / "norm_params.json"),
        "knn/metadata.json":    _exists(knn_dir / "metadata.json"),
    }
    lstm_files = {
        "lstm/lstm_model.pt":  Path(lstm_db.model_path),
        "lstm/metadata.json":  _exists(lstm_dir / "metadata.json"),
    }

    # ── Write ZIP ─────────────────────────────────────────────────────
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    zip_name = f"aitrade_model_export_{timestamp}.zip"

    if output_dir is None:
        output_dir = settings.MODEL_DIR / EXPORT_SUBDIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / zip_name

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
        for arc_name, file_path in {**knn_files, **lstm_files}.items():
            if file_path is not None and file_path.exists():
                zf.write(file_path, arcname=arc_name)
            else:
                logger.warning("Skipping missing file in export bundle: %s", arc_name)

    logger.info("Model bundle written to %s (%.1f KB)", zip_path, zip_path.stat().st_size / 1024)
    return zip_path


# ── Import ─────────────────────────────────────────────────────────────

async def import_ensemble(
    db: AsyncSession,
    zip_path: Path,
    target_base_dir: Path | None = None,
) -> dict[str, int]:
    """Import a model bundle ZIP, register the models in DB, and return new IDs.

    Parameters
    ----------
    db              : async DB session
    zip_path        : path to the .zip bundle produced by export_ensemble()
    target_base_dir : root under which the extracted artifacts are placed
                      (defaults to MODEL_DIR / "distill")

    Returns
    -------
    {
      "knn_model_id":      <new DB id>,
      "lstm_model_id":     <new DB id>,
      "ensemble_config_id": <new DB id>,
    }
    """
    from app.config import settings
    from app.db import crud

    if target_base_dir is None:
        target_base_dir = settings.MODEL_DIR / "distill"
    target_base_dir = Path(target_base_dir)

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid ZIP file: {zip_path}")

    # ── Validate & extract to a temp dir first ────────────────────────
    with tempfile.TemporaryDirectory(prefix="aitrade_import_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Guard against path-traversal attacks in the archive
            for member in zf.namelist():
                member_path = (tmp_path / member).resolve()
                if not str(member_path).startswith(str(tmp_path.resolve())):
                    raise ValueError(
                        f"ZIP entry '{member}' would escape the extraction directory."
                    )
            zf.extractall(tmp_path)

        manifest_file = tmp_path / "manifest.json"
        if not manifest_file.exists():
            raise ValueError("Bundle is missing manifest.json")

        with open(manifest_file) as f:
            manifest: dict = json.load(f)

        _require_keys(manifest, ["bundle_version", "knn", "lstm", "ensemble"])
        _require_keys(manifest["knn"], ["k_neighbors", "seq_len", "interval"])
        _require_keys(manifest["lstm"], ["hidden_size", "num_layers", "seq_len", "interval"])

        knn_meta = manifest["knn"]
        lstm_meta = manifest["lstm"]
        ens_meta = manifest["ensemble"]

        knn_joblib = tmp_path / "knn" / "knn_model.joblib"
        lstm_pt    = tmp_path / "lstm" / "lstm_model.pt"

        if not knn_joblib.exists():
            raise FileNotFoundError("Bundle is missing knn/knn_model.joblib")
        if not lstm_pt.exists():
            raise FileNotFoundError("Bundle is missing lstm/lstm_model.pt")

        # Quick sanity checks on model files
        _validate_knn_artifact(knn_joblib)
        _validate_lstm_artifact(lstm_pt)

        # ── Create a stub RL model so FK constraint is satisfied ─────
        # Imported models have no RL parent on this system; we create a
        # placeholder marked as "completed" and flagged with a special name.
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        stub_rl = await crud.create_rl_model(
            db,
            name=f"[Imported stub] {ens_meta.get('name', '')} {timestamp}",
            algorithm="imported",
            interval=knn_meta["interval"],
            status="completed",
            training_config={"source": "import", "bundle_manifest": manifest},
        )

        # ── Determine target directories ─────────────────────────────
        knn_dest_dir  = target_base_dir / f"knn_{stub_rl.id}_imported"
        lstm_dest_dir = target_base_dir / f"lstm_{stub_rl.id}_imported"
        knn_dest_dir.mkdir(parents=True, exist_ok=True)
        lstm_dest_dir.mkdir(parents=True, exist_ok=True)

        # ── Copy artifacts from temp dir to final location ───────────
        knn_model_dest = knn_dest_dir / f"knn_model_{timestamp}.joblib"
        lstm_model_dest = lstm_dest_dir / f"lstm_model_{timestamp}.pt"
        shutil.copy2(knn_joblib, knn_model_dest)
        shutil.copy2(lstm_pt, lstm_model_dest)

        norm_src = tmp_path / "knn" / "norm_params.json"
        norm_dest = knn_dest_dir / "norm_params.json"
        if norm_src.exists():
            shutil.copy2(norm_src, norm_dest)

        knn_meta_src = tmp_path / "knn" / "metadata.json"
        if knn_meta_src.exists():
            shutil.copy2(knn_meta_src, knn_dest_dir / "metadata.json")

        lstm_meta_src = tmp_path / "lstm" / "metadata.json"
        if lstm_meta_src.exists():
            shutil.copy2(lstm_meta_src, lstm_dest_dir / "metadata.json")

    # ── Register KNN model in DB ─────────────────────────────────────
    knn_db = await crud.create_knn_model(
        db,
        name=knn_meta.get("name", f"KNN_imported_{timestamp}"),
        source_rl_model_id=stub_rl.id,
        k_neighbors=knn_meta["k_neighbors"],
        seq_len=knn_meta["seq_len"],
        interval=knn_meta["interval"],
        regime_filter=knn_meta.get("regime_filter"),
        feature_combination=knn_meta.get("feature_combination"),
        status="completed",
        model_path=str(knn_model_dest),
        norm_params_path=str(norm_dest) if norm_src.exists() else None,
        accuracy=knn_meta.get("accuracy"),
        precision_buy=knn_meta.get("precision_buy"),
        precision_sell=knn_meta.get("precision_sell"),
    )

    # ── Register LSTM model in DB ────────────────────────────────────
    lstm_db = await crud.create_lstm_model(
        db,
        name=lstm_meta.get("name", f"LSTM_imported_{timestamp}"),
        source_rl_model_id=stub_rl.id,
        hidden_size=lstm_meta["hidden_size"],
        num_layers=lstm_meta["num_layers"],
        dropout=lstm_meta.get("dropout", 0.3),
        seq_len=lstm_meta["seq_len"],
        interval=lstm_meta["interval"],
        regime_filter=lstm_meta.get("regime_filter"),
        status="completed",
        model_path=str(lstm_model_dest),
        accuracy=lstm_meta.get("accuracy"),
        precision_buy=lstm_meta.get("precision_buy"),
        precision_sell=lstm_meta.get("precision_sell"),
    )

    # ── Register Ensemble config in DB ───────────────────────────────
    ens_db = await crud.create_ensemble_config(
        db,
        name=ens_meta.get("name", f"Ensemble_imported_{timestamp}"),
        knn_model_id=knn_db.id,
        lstm_model_id=lstm_db.id,
        knn_weight=ens_meta.get("knn_weight", 0.5),
        lstm_weight=ens_meta.get("lstm_weight", 0.5),
        agreement_required=ens_meta.get("agreement_required", True),
        interval=ens_meta.get("interval", knn_meta["interval"]),
    )

    logger.info(
        "Import complete — stub_rl=%d  knn=%d  lstm=%d  ensemble=%d",
        stub_rl.id, knn_db.id, lstm_db.id, ens_db.id,
    )

    return {
        "knn_model_id": knn_db.id,
        "lstm_model_id": lstm_db.id,
        "ensemble_config_id": ens_db.id,
        "stub_rl_model_id": stub_rl.id,
    }


# ── Helpers ─────────────────────────────────────────────────────────────

def _require_keys(d: dict, keys: list[str]) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Bundle manifest missing required fields: {missing}")


def _validate_knn_artifact(path: Path) -> None:
    """Load the KNN joblib and verify it has a predict method."""
    import joblib
    try:
        model = joblib.load(path)
    except Exception as exc:
        raise ValueError(f"Cannot load KNN artifact: {exc}") from exc
    if not hasattr(model, "predict"):
        raise ValueError(f"KNN artifact at {path} does not appear to be a valid classifier.")


def _validate_lstm_artifact(path: Path) -> None:
    """Load the LSTM checkpoint and verify it contains the expected keys."""
    import torch
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Cannot load LSTM artifact: {exc}") from exc
    required = {"state_dict", "input_size", "hidden_size", "num_layers", "num_classes"}
    missing = required - set(checkpoint.keys())
    if missing:
        raise ValueError(f"LSTM checkpoint missing keys: {missing}")


# ── In-memory streaming helper (used by the API endpoint) ───────────────

async def export_ensemble_to_bytes(
    db: AsyncSession,
    knn_model_id: int,
    lstm_model_id: int,
) -> tuple[io.BytesIO, str]:
    """Export a model bundle and return it as an in-memory BytesIO buffer.

    The file is written to a temp export directory then read back so we don't
    keep it on disk indefinitely.  Callers should stream the buffer as a
    FileResponse / StreamingResponse.

    Returns
    -------
    (buffer, filename)
    """
    import tempfile
    with tempfile.TemporaryDirectory(prefix="aitrade_export_") as tmp:
        tmp_path = Path(tmp)
        zip_path = await export_ensemble(db, knn_model_id, lstm_model_id, output_dir=tmp_path)
        filename = zip_path.name
        buf = io.BytesIO(zip_path.read_bytes())

    buf.seek(0)
    return buf, filename
