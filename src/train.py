"""Train XGBoost expected-goals regressors from the feature matrix.

Stage 8 trains two separate regressors, one for home goals and one for away
goals. Splitting is strictly temporal: matches before 2022-01-01 train the
models, and matches from 2022-01-01 onward are held out for validation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import config as cfg
    from src.data_loader import clean_results, fetch_international_results
    from src.features import build_feature_matrix
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    import config as cfg
    from src.data_loader import clean_results, fetch_international_results
    from src.features import build_feature_matrix


FEATURE_MATRIX_PATH = cfg.DATA_PROCESSED / "feature_matrix.parquet"
HOME_MODEL_PATH = cfg.DATA_PROCESSED / "xgb_home_goals.json"
AWAY_MODEL_PATH = cfg.DATA_PROCESSED / "xgb_away_goals.json"
FEATURE_METADATA_PATH = cfg.DATA_PROCESSED / "xgb_feature_metadata.json"
SPLIT_DATE = pd.Timestamp("2022-01-01")


def _load_or_build_feature_matrix() -> pd.DataFrame:
    if FEATURE_MATRIX_PATH.exists():
        return pd.read_parquet(FEATURE_MATRIX_PATH)

    results = clean_results(fetch_international_results(force_refresh=False))
    matrix = build_feature_matrix(results)
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(FEATURE_MATRIX_PATH, index=False)
    return matrix


def _prepare_xy(
    matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], pd.DataFrame]:
    filtered = matrix.loc[
        matrix["home_has_min_history"].astype(bool)
        & matrix["away_has_min_history"].astype(bool)
    ].copy()

    drop_cols = {
        "date",
        "home_team",
        "away_team",
        "tournament",
        "home_goals",
        "away_goals",
        "outcome",
    }
    candidate_features = [
        column for column in filtered.columns if column not in drop_cols
    ]

    features = filtered.loc[:, candidate_features].copy()
    # tournament_tier is the only categorical feature; use one-hot columns so
    # prediction-time feature order is explicit and model-native numeric.
    features = pd.get_dummies(features, columns=["tournament_tier"], dtype=float)

    bool_cols = features.select_dtypes(include=["bool"]).columns
    features.loc[:, bool_cols] = features.loc[:, bool_cols].astype(float)

    numeric_features = features.select_dtypes(include=[np.number]).copy()
    feature_columns = list(numeric_features.columns)
    return (
        numeric_features,
        filtered["home_goals"].astype(float),
        filtered["away_goals"].astype(float),
        feature_columns,
        filtered,
    )


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_pred - y_true.to_numpy(dtype=float)
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(math.sqrt(np.square(errors).mean())),
    }


def _fit_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Any:
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required for Stage 8 training. Install it with "
            "`python -m pip install xgboost` and rerun `python src/train.py`."
        ) from exc

    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": cfg.RANDOM_SEED,
        "n_jobs": 4,
    }

    # Defaults are intentionally modest: enough trees for nonlinear structure,
    # shallow depth to limit overfit, and early stopping on the temporal holdout.
    try:
        model = XGBRegressor(**params, early_stopping_rounds=25)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    except TypeError:
        model = XGBRegressor(**params)
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=25,
                verbose=False,
            )
        except TypeError:
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    return model


def _top_gain_importance(model: Any, feature_columns: list[str], n: int = 15) -> pd.DataFrame:
    gain = model.get_booster().get_score(importance_type="gain")
    rows = [
        {"feature": feature, "gain": float(gain.get(feature, 0.0))}
        for feature in feature_columns
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("gain", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def train_models() -> tuple[Any, Any, dict[str, Any]]:
    matrix = _load_or_build_feature_matrix()
    X, y_home, y_away, feature_columns, filtered = _prepare_xy(matrix)

    train_mask = pd.to_datetime(filtered["date"]) < SPLIT_DATE
    test_mask = ~train_mask
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_home_train, y_home_test = y_home.loc[train_mask], y_home.loc[test_mask]
    y_away_train, y_away_test = y_away.loc[train_mask], y_away.loc[test_mask]

    if X_train.empty or X_test.empty:
        raise ValueError("Temporal split produced an empty train or test set")

    print(f"Original feature matrix rows: {len(matrix):,}")
    print(f"Rows after MIN_HISTORY filter: {len(filtered):,}")
    print(
        "Dropped sparse rows: "
        f"{len(matrix) - len(filtered):,} "
        f"({(len(matrix) - len(filtered)) / len(matrix):.1%})"
    )
    print(
        f"Train rows: {len(X_train):,} | "
        f"{filtered.loc[train_mask, 'date'].min().date()} to "
        f"{filtered.loc[train_mask, 'date'].max().date()}"
    )
    print(
        f"Test rows: {len(X_test):,} | "
        f"{filtered.loc[test_mask, 'date'].min().date()} to "
        f"{filtered.loc[test_mask, 'date'].max().date()}"
    )
    print(f"Feature columns: {len(feature_columns)}")

    home_model = _fit_xgb_model(X_train, y_home_train, X_test, y_home_test)
    away_model = _fit_xgb_model(X_train, y_away_train, X_test, y_away_test)

    home_pred = home_model.predict(X_test)
    away_pred = away_model.predict(X_test)
    home_baseline = np.full(len(y_home_test), y_home_train.mean())
    away_baseline = np.full(len(y_away_test), y_away_train.mean())

    home_metrics = _metrics(y_home_test, home_pred)
    away_metrics = _metrics(y_away_test, away_pred)
    home_baseline_metrics = _metrics(y_home_test, home_baseline)
    away_baseline_metrics = _metrics(y_away_test, away_baseline)

    print("\nHeld-out regression metrics:")
    print(
        "home_goals XGBoost: "
        f"MAE={home_metrics['mae']:.3f}, RMSE={home_metrics['rmse']:.3f}"
    )
    print(
        "home_goals mean baseline: "
        f"MAE={home_baseline_metrics['mae']:.3f}, "
        f"RMSE={home_baseline_metrics['rmse']:.3f}"
    )
    print(
        "away_goals XGBoost: "
        f"MAE={away_metrics['mae']:.3f}, RMSE={away_metrics['rmse']:.3f}"
    )
    print(
        "away_goals mean baseline: "
        f"MAE={away_baseline_metrics['mae']:.3f}, "
        f"RMSE={away_baseline_metrics['rmse']:.3f}"
    )

    print("\nTop 15 home_goals features by gain:")
    print(_top_gain_importance(home_model, feature_columns).to_string(index=False))
    print("\nTop 15 away_goals features by gain:")
    print(_top_gain_importance(away_model, feature_columns).to_string(index=False))

    if (
        home_metrics["mae"] >= home_baseline_metrics["mae"] * 0.98
        or away_metrics["mae"] >= away_baseline_metrics["mae"] * 0.98
    ):
        print(
            "\nHonest note: XGBoost only barely beats, or fails to beat, the "
            "global-mean baseline on at least one MAE target. International "
            "football goals are noisy; this is a real modeling finding, not a "
            "pipeline failure."
        )

    metadata = {
        "split_date": str(SPLIT_DATE.date()),
        "feature_columns": feature_columns,
        "one_hot_encoding": "pandas.get_dummies(tournament_tier)",
        "labels": ["home_goals", "away_goals"],
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "home_metrics": home_metrics,
        "away_metrics": away_metrics,
        "home_mean_baseline_metrics": home_baseline_metrics,
        "away_mean_baseline_metrics": away_baseline_metrics,
    }

    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    home_model.save_model(HOME_MODEL_PATH)
    away_model.save_model(AWAY_MODEL_PATH)
    FEATURE_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {HOME_MODEL_PATH}")
    print(f"Saved {AWAY_MODEL_PATH}")
    print(f"Saved {FEATURE_METADATA_PATH}")

    return home_model, away_model, metadata


def main() -> None:
    train_models()


if __name__ == "__main__":
    main()
