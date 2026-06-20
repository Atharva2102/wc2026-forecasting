"""XGBoost expected-goals adapter.

The saved models consume exactly the feature columns recorded during Stage 8
training. For backtests, callers should pass rows from
data/processed/feature_matrix.parquet: those rows were built in chronological
order and are already point-in-time, so using them avoids leakage-prone
recomputation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import config as cfg
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    import config as cfg


HOME_MODEL_PATH = cfg.DATA_PROCESSED / "xgb_home_goals.json"
AWAY_MODEL_PATH = cfg.DATA_PROCESSED / "xgb_away_goals.json"
FEATURE_METADATA_PATH = cfg.DATA_PROCESSED / "xgb_feature_metadata.json"


@lru_cache(maxsize=1)
def _load_xgb_bundle() -> tuple[Any, Any, list[str]]:
    from xgboost import XGBRegressor

    metadata = json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    feature_columns = list(metadata["feature_columns"])

    home_model = XGBRegressor()
    away_model = XGBRegressor()
    home_model.load_model(HOME_MODEL_PATH)
    away_model.load_model(AWAY_MODEL_PATH)
    return home_model, away_model, feature_columns


def _feature_frame(match_features: pd.Series | dict[str, Any]) -> pd.DataFrame:
    _home_model, _away_model, feature_columns = _load_xgb_bundle()
    row = pd.DataFrame([dict(match_features)])

    if "tournament_tier" in row.columns:
        row = pd.get_dummies(row, columns=["tournament_tier"], dtype=float)

    bool_cols = row.select_dtypes(include=["bool"]).columns
    row.loc[:, bool_cols] = row.loc[:, bool_cols].astype(float)

    return row.reindex(columns=feature_columns, fill_value=0.0)


def xgb_expected_goals(match_features: pd.Series | dict[str, Any]) -> tuple[float, float]:
    """Return model-predicted expected goals as (home_xg, away_xg)."""
    home_model, away_model, _feature_columns = _load_xgb_bundle()
    X = _feature_frame(match_features)
    home_xg = float(home_model.predict(X)[0])
    away_xg = float(away_model.predict(X)[0])
    return (
        float(np.clip(home_xg, 0.05, 5.0)),
        float(np.clip(away_xg, 0.05, 5.0)),
    )
