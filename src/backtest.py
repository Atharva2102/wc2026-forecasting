"""No-leakage held-out tournament backtests for the pure Elo pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    import config as cfg
    import src.elo as elo_mod
    from src.data_loader import canonical_team, clean_results, fetch_international_results
    from src.elo import compute_elo_ratings
    from src.model_lambdas import FEATURE_METADATA_PATH, xgb_expected_goals
    from src.optimizer import best_score, ev_optimal_count, most_likely_outcome
    from src.poisson_model import outcome_probs, score_matrix, team_lambdas
    from src.scoring import (
        CORNERS_POINTS,
        REDS_POINTS,
        SCORE_POINTS,
        WINNING_TEAM_GROUP_POINTS,
        YELLOWS_POINTS,
    )
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    import config as cfg
    import src.elo as elo_mod
    from src.data_loader import canonical_team, clean_results, fetch_international_results
    from src.elo import compute_elo_ratings
    from src.model_lambdas import FEATURE_METADATA_PATH, xgb_expected_goals
    from src.optimizer import best_score, ev_optimal_count, most_likely_outcome
    from src.poisson_model import outcome_probs, score_matrix, team_lambdas
    from src.scoring import (
        CORNERS_POINTS,
        REDS_POINTS,
        SCORE_POINTS,
        WINNING_TEAM_GROUP_POINTS,
        YELLOWS_POINTS,
    )


@dataclass(frozen=True)
class HeldOutEvent:
    name: str
    tournament: str
    start: str
    end: str


HELD_OUT_EVENTS: tuple[HeldOutEvent, ...] = (
    HeldOutEvent("2022 FIFA World Cup", "FIFA World Cup", "2022-11-20", "2022-12-18"),
    HeldOutEvent("UEFA Euro 2024", "UEFA Euro", "2024-06-14", "2024-07-14"),
    HeldOutEvent("Copa América 2024", "Copa América", "2024-06-20", "2024-07-14"),
    HeldOutEvent(
        "African Cup of Nations 2023",
        "African Cup of Nations",
        "2024-01-13",
        "2024-02-11",
    ),
)

MODEL_NAME = "Model"
ELO_POISSON_NAME = "Elo-Poisson"
XGBOOST_NAME = "XGBoost"
BASELINE_A_NAME = "Baseline A"
BASELINE_B_NAME = "Baseline B"
OUTCOME_ORDER = ("home", "draw", "away")
FEATURE_MATRIX_PATH = cfg.DATA_PROCESSED / "feature_matrix.parquet"


def _count_rule(points: dict[str, int], within: int | None = None) -> Callable[[int, int], int]:
    def rule(predicted: int, actual: int) -> int:
        diff = abs(predicted - actual)
        if diff == 0:
            return points["exact"]
        if within is not None and diff <= within:
            return points[f"within_{within}"]
        return 0

    return rule


def ev_constant_stats() -> dict[str, int]:
    """Return EV-optimal count predictions for stats with available data."""
    return {
        "corners": ev_optimal_count(10.0, _count_rule(CORNERS_POINTS, within=2)),
        "yellow_cards": ev_optimal_count(3.8, _count_rule(YELLOWS_POINTS, within=1)),
        "red_cards": ev_optimal_count(0.2, _count_rule(REDS_POINTS)),
    }


def load_clean_results() -> pd.DataFrame:
    """Load the cleaned martj42 international results dataset."""
    return clean_results(fetch_international_results(force_refresh=False))


def event_matches(results: pd.DataFrame, event: HeldOutEvent) -> pd.DataFrame:
    """Return matches for one held-out event by exact tournament label and date range."""
    start = pd.Timestamp(event.start)
    end = pd.Timestamp(event.end)
    matches = results.loc[
        (results["tournament"] == event.tournament)
        & (results["date"] >= start)
        & (results["date"] <= end)
    ].copy()
    return matches.sort_values("date").reset_index(drop=True)


def _venue_country(row: pd.Series) -> str | None:
    country = str(row.get("country", "")).strip()
    if country in {"United States", "USA"}:
        return "USA"
    if country in {"Mexico", "Canada"}:
        return country
    return country or None


def _ratings_before_event(
    results: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[dict[str, float], pd.Timestamp]:
    train = results.loc[results["date"] < cutoff].copy()
    max_train_date = train["date"].max()
    assert pd.notna(max_train_date)
    assert max_train_date < cutoff
    ratings, _history = compute_elo_ratings(train)
    return ratings, max_train_date


def _match_matrix(
    home_team: str,
    away_team: str,
    venue_country: str | None,
    ratings: dict[str, float],
) -> np.ndarray:
    lambda_home, lambda_away = team_lambdas(
        ratings.get(home_team, cfg.DEFAULT_RATING),
        ratings.get(away_team, cfg.DEFAULT_RATING),
        team_a=home_team,
        team_b=away_team,
        venue_country=venue_country,
    )
    return score_matrix(lambda_home, lambda_away)


def _actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if away_goals > home_goals:
        return "away"
    return "draw"


def _one_hot(outcome: str) -> tuple[float, float, float]:
    return tuple(1.0 if outcome == item else 0.0 for item in OUTCOME_ORDER)


def _modal_score(matrix: np.ndarray, mode: str = "any") -> tuple[int, int]:
    best: tuple[int, int] | None = None
    best_prob = -1.0
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            if mode == "home" and home_goals <= away_goals:
                continue
            if mode == "away" and home_goals >= away_goals:
                continue
            if mode == "draw" and home_goals != away_goals:
                continue
            prob = float(matrix[home_goals, away_goals])
            if prob > best_prob:
                best_prob = prob
                best = (home_goals, away_goals)
    if best is None:
        raise ValueError(f"No modal candidate for mode={mode!r}")
    return best


def _brier(probs: tuple[float, float, float], actual: str) -> float:
    actual_vec = _one_hot(actual)
    return float(sum((prob - truth) ** 2 for prob, truth in zip(probs, actual_vec)))


def _log_loss(probs: tuple[float, float, float], actual: str) -> float:
    idx = OUTCOME_ORDER.index(actual)
    clipped = max(min(probs[idx], 1.0 - 1e-12), 1e-12)
    return -math.log(clipped)


def _scoreline_points(
    pred_score: tuple[int, int],
    actual_score: tuple[int, int],
) -> int:
    predicted_diff = pred_score[0] - pred_score[1]
    actual_diff = actual_score[0] - actual_score[1]

    if pred_score == actual_score:
        return SCORE_POINTS["exact_score"]
    if predicted_diff == actual_diff:
        return SCORE_POINTS["correct_gd"]
    if sum(pred_score) == sum(actual_score):
        return SCORE_POINTS["correct_total"]
    return 0


def _points(
    pred_score: tuple[int, int],
    pred_outcome: str,
    actual_score: tuple[int, int],
) -> int:
    actual_outcome = _actual_outcome(*actual_score)
    points = _scoreline_points(pred_score, actual_score)
    if pred_outcome == actual_outcome:
        points += WINNING_TEAM_GROUP_POINTS
    return points


def _coverage(matches: pd.DataFrame) -> dict[str, int]:
    coverage_cols = {
        "corners": ["corners", "total_corners"],
        "yellow_cards": ["yellow_cards", "total_yellows"],
        "red_cards": ["red_cards", "total_reds"],
    }
    coverage = {}
    for label, candidates in coverage_cols.items():
        col = next((candidate for candidate in candidates if candidate in matches.columns), None)
        coverage[label] = int(matches[col].notna().sum()) if col else 0
    return coverage


def _record_prediction(
    event_name: str,
    model_name: str,
    row: pd.Series,
    pred_score: tuple[int, int],
    pred_outcome: str,
    probs: tuple[float, float, float],
) -> dict[str, object]:
    actual_score = (int(row["home_score"]), int(row["away_score"]))
    actual_outcome = _actual_outcome(*actual_score)
    return {
        "event": event_name,
        "model": model_name,
        "date": row["date"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "actual_home_goals": actual_score[0],
        "actual_away_goals": actual_score[1],
        "actual_outcome": actual_outcome,
        "predicted_home_goals": pred_score[0],
        "predicted_away_goals": pred_score[1],
        "predicted_outcome": pred_outcome,
        "points": _points(pred_score, pred_outcome, actual_score),
        "wdl_correct": pred_outcome == actual_outcome,
        "exact_score": pred_score == actual_score,
        "brier": _brier(probs, actual_outcome),
        "log_loss": _log_loss(probs, actual_outcome),
    }


def _prediction_from_matrix(
    event_name: str,
    model_name: str,
    row: pd.Series,
    matrix: np.ndarray,
) -> dict[str, object]:
    p_home, p_draw, p_away = outcome_probs(matrix)
    probs = (p_home, p_draw, p_away)
    pred_score = best_score(matrix, mode="any")
    pred_outcome = most_likely_outcome(matrix)
    return _record_prediction(event_name, model_name, row, pred_score, pred_outcome, probs)


def _predict_match_rows(
    event_name: str,
    row: pd.Series,
    ratings: dict[str, float],
) -> list[dict[str, object]]:
    home_team = canonical_team(row["home_team"])
    away_team = canonical_team(row["away_team"])
    matrix = _match_matrix(home_team, away_team, _venue_country(row), ratings)
    p_home, p_draw, p_away = outcome_probs(matrix)
    probs = (p_home, p_draw, p_away)

    model_score = best_score(matrix, mode="any")
    model_outcome = most_likely_outcome(matrix)

    if ratings.get(home_team, cfg.DEFAULT_RATING) > ratings.get(away_team, cfg.DEFAULT_RATING):
        baseline_a_outcome = "home"
    elif ratings.get(away_team, cfg.DEFAULT_RATING) > ratings.get(home_team, cfg.DEFAULT_RATING):
        baseline_a_outcome = "away"
    else:
        baseline_a_outcome = "draw"
    baseline_a_score = _modal_score(matrix, mode=baseline_a_outcome)

    baseline_b_outcome = "draw"
    baseline_b_score = (1, 1)

    return [
        _record_prediction(event_name, MODEL_NAME, row, model_score, model_outcome, probs),
        _record_prediction(
            event_name,
            BASELINE_A_NAME,
            row,
            baseline_a_score,
            baseline_a_outcome,
            _one_hot(baseline_a_outcome),
        ),
        _record_prediction(
            event_name,
            BASELINE_B_NAME,
            row,
            baseline_b_score,
            baseline_b_outcome,
            _one_hot(baseline_b_outcome),
        ),
    ]


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize predictions per event/model plus a pooled block."""
    grouped = (
        predictions.groupby(["event", "model"], as_index=False)
        .agg(
            matches=("points", "size"),
            wdl_accuracy=("wdl_correct", "mean"),
            exact_score_rate=("exact_score", "mean"),
            total_points=("points", "sum"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
        )
    )
    pooled = (
        predictions.groupby("model", as_index=False)
        .agg(
            matches=("points", "size"),
            wdl_accuracy=("wdl_correct", "mean"),
            exact_score_rate=("exact_score", "mean"),
            total_points=("points", "sum"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
        )
    )
    pooled.insert(0, "event", "Pooled")
    summary = pd.concat([grouped, pooled], ignore_index=True)
    order = {
        MODEL_NAME: 0,
        ELO_POISSON_NAME: 0,
        XGBOOST_NAME: 1,
        BASELINE_A_NAME: 2,
        BASELINE_B_NAME: 3,
    }
    summary["model_order"] = summary["model"].map(order)
    return (
        summary.sort_values(["event", "model_order"])
        .drop(columns="model_order")
        .reset_index(drop=True)
    )


def run_backtest(
    results: pd.DataFrame | None = None,
    events: tuple[HeldOutEvent, ...] = HELD_OUT_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all held-out event backtests and return predictions, summary, leakage, coverage."""
    if results is None:
        results = load_clean_results()

    prediction_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for event in events:
        cutoff = pd.Timestamp(event.start)
        matches = event_matches(results, event)
        if matches.empty:
            raise ValueError(f"No matches found for {event.name}")

        ratings, max_train_date = _ratings_before_event(results, cutoff)
        leakage_rows.append(
            {
                "event": event.name,
                "cutoff": cutoff,
                "max_rating_train_date": max_train_date,
                "no_leakage": max_train_date < cutoff,
                "test_matches": len(matches),
            }
        )

        coverage = _coverage(matches)
        coverage_rows.append(
            {
                "event": event.name,
                "matches": len(matches),
                "corners_coverage": coverage["corners"],
                "yellow_cards_coverage": coverage["yellow_cards"],
                "red_cards_coverage": coverage["red_cards"],
            }
        )

        for _idx, row in matches.iterrows():
            prediction_rows.extend(_predict_match_rows(event.name, row, ratings))

    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_predictions(predictions)
    leakage = pd.DataFrame(leakage_rows)
    coverage = pd.DataFrame(coverage_rows)
    assert leakage["no_leakage"].all()
    return predictions, summary, leakage, coverage


def _event_inputs_for_current_k(
    results: pd.DataFrame,
    events: tuple[HeldOutEvent, ...],
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    event_inputs: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for event in events:
        cutoff = pd.Timestamp(event.start)
        matches = event_matches(results, event)
        if matches.empty:
            raise ValueError(f"No matches found for {event.name}")

        ratings, max_train_date = _ratings_before_event(results, cutoff)
        leakage_rows.append(
            {
                "event": event.name,
                "cutoff": cutoff,
                "max_rating_train_date": max_train_date,
                "no_leakage": max_train_date < cutoff,
                "test_matches": len(matches),
            }
        )

        coverage = _coverage(matches)
        coverage_rows.append(
            {
                "event": event.name,
                "matches": len(matches),
                "corners_coverage": coverage["corners"],
                "yellow_cards_coverage": coverage["yellow_cards"],
                "red_cards_coverage": coverage["red_cards"],
            }
        )
        event_inputs.append({"event": event, "matches": matches, "ratings": ratings})

    leakage = pd.DataFrame(leakage_rows)
    assert leakage["no_leakage"].all()
    return event_inputs, leakage, pd.DataFrame(coverage_rows)


def _predictions_from_event_inputs(
    event_inputs: list[dict[str, object]],
) -> pd.DataFrame:
    prediction_rows: list[dict[str, object]] = []
    for event_input in event_inputs:
        event = event_input["event"]
        matches = event_input["matches"]
        ratings = event_input["ratings"]
        for _idx, row in matches.iterrows():
            prediction_rows.extend(_predict_match_rows(event.name, row, ratings))
    return pd.DataFrame(prediction_rows)


def parameter_grid_search(
    spreads: list[float] | tuple[float, ...] = (0.0016, 0.0020, 0.0024, 0.0028, 0.0032),
    k_values: list[float] | tuple[float, ...] = (25, 40, 55, 70),
    results: pd.DataFrame | None = None,
    events: tuple[HeldOutEvent, ...] = HELD_OUT_EVENTS,
) -> pd.DataFrame:
    """Sweep K_BASE and SPREAD, returning pooled Model metrics for each combo."""
    if results is None:
        results = load_clean_results()

    original_spread = cfg.SPREAD
    had_elo_k = "K_BASE" in elo_mod.__dict__
    original_elo_k = getattr(elo_mod, "K_BASE", None)
    rows: list[dict[str, object]] = []

    try:
        for k_base in k_values:
            elo_mod.K_BASE = float(k_base)
            event_inputs, leakage, _coverage = _event_inputs_for_current_k(results, events)
            assert leakage["no_leakage"].all()

            for spread in spreads:
                cfg.SPREAD = float(spread)
                predictions = _predictions_from_event_inputs(event_inputs)
                summary = summarize_predictions(predictions)
                pooled_model = summary.loc[
                    (summary["event"] == "Pooled")
                    & (summary["model"] == MODEL_NAME)
                ].iloc[0]
                rows.append(
                    {
                        "K_BASE": float(k_base),
                        "SPREAD": float(spread),
                        "matches": int(pooled_model["matches"]),
                        "wdl_accuracy": float(pooled_model["wdl_accuracy"]),
                        "exact_score_rate": float(pooled_model["exact_score_rate"]),
                        "total_points": int(pooled_model["total_points"]),
                        "brier": float(pooled_model["brier"]),
                        "log_loss": float(pooled_model["log_loss"]),
                    }
                )
    finally:
        cfg.SPREAD = original_spread
        if had_elo_k:
            elo_mod.K_BASE = original_elo_k
        elif "K_BASE" in elo_mod.__dict__:
            delattr(elo_mod, "K_BASE")

    return pd.DataFrame(rows)


def _load_feature_matrix_for_backtest() -> pd.DataFrame:
    if not FEATURE_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURE_MATRIX_PATH} not found. Build it with `python src/features.py`."
        )
    return pd.read_parquet(FEATURE_MATRIX_PATH)


def _xgb_split_date() -> pd.Timestamp:
    metadata = pd.read_json(FEATURE_METADATA_PATH, typ="series")
    return pd.Timestamp(metadata["split_date"])


def _event_feature_rows(feature_matrix: pd.DataFrame, event: HeldOutEvent) -> pd.DataFrame:
    start = pd.Timestamp(event.start)
    end = pd.Timestamp(event.end)
    rows = feature_matrix.loc[
        (feature_matrix["tournament"] == event.tournament)
        & (pd.to_datetime(feature_matrix["date"]) >= start)
        & (pd.to_datetime(feature_matrix["date"]) <= end)
    ].copy()
    return rows.sort_values("date").reset_index(drop=True)


def _feature_lookup(feature_rows: pd.DataFrame) -> dict[tuple[object, ...], pd.Series]:
    lookup: dict[tuple[object, ...], pd.Series] = {}
    for _idx, row in feature_rows.iterrows():
        key = (
            pd.Timestamp(row["date"]),
            row["home_team"],
            row["away_team"],
            int(row["home_goals"]),
            int(row["away_goals"]),
            row["tournament"],
        )
        lookup[key] = row
    return lookup


def run_xgb_comparison_backtest(
    results: pd.DataFrame | None = None,
    feature_matrix: pd.DataFrame | None = None,
    events: tuple[HeldOutEvent, ...] = HELD_OUT_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare pure Elo-Poisson with XGBoost expected-goals on held-out events."""
    if results is None:
        results = load_clean_results()
    if feature_matrix is None:
        feature_matrix = _load_feature_matrix_for_backtest()

    split_date = _xgb_split_date()
    prediction_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []

    for event in events:
        cutoff = pd.Timestamp(event.start)
        matches = event_matches(results, event)
        features = _event_feature_rows(feature_matrix, event)
        if len(matches) != len(features):
            raise ValueError(
                f"Feature/match count mismatch for {event.name}: "
                f"{len(features)} vs {len(matches)}"
            )
        if not (pd.to_datetime(features["date"]) >= split_date).all():
            raise AssertionError(f"{event.name} includes rows before XGBoost split date")

        ratings, max_train_date = _ratings_before_event(results, cutoff)
        leakage_rows.append(
            {
                "event": event.name,
                "rating_cutoff": cutoff,
                "max_rating_train_date": max_train_date,
                "xgb_train_split": split_date,
                "xgb_all_test_rows_after_split": bool(
                    (pd.to_datetime(features["date"]) >= split_date).all()
                ),
                "no_elo_leakage": max_train_date < cutoff,
                "test_matches": len(matches),
            }
        )

        feature_lookup = _feature_lookup(features)
        for _idx, row in matches.iterrows():
            home_team = canonical_team(row["home_team"])
            away_team = canonical_team(row["away_team"])
            feature_key = (
                pd.Timestamp(row["date"]),
                home_team,
                away_team,
                int(row["home_score"]),
                int(row["away_score"]),
                row["tournament"],
            )
            if feature_key not in feature_lookup:
                raise AssertionError(
                    f"Feature row lookup failed for {event.name}: {feature_key}"
                )
            feature_row = feature_lookup[feature_key]
            if (
                home_team != feature_row["home_team"]
                or away_team != feature_row["away_team"]
                or int(row["home_score"]) != int(feature_row["home_goals"])
                or int(row["away_score"]) != int(feature_row["away_goals"])
            ):
                raise AssertionError(f"Feature row alignment failed for {event.name}")

            elo_matrix = _match_matrix(home_team, away_team, _venue_country(row), ratings)
            prediction_rows.append(
                _prediction_from_matrix(event.name, ELO_POISSON_NAME, row, elo_matrix)
            )

            home_xg, away_xg = xgb_expected_goals(feature_row)
            xgb_matrix = score_matrix(home_xg, away_xg)
            prediction_rows.append(
                _prediction_from_matrix(event.name, XGBOOST_NAME, row, xgb_matrix)
            )

    leakage = pd.DataFrame(leakage_rows)
    assert leakage["no_elo_leakage"].all()
    assert leakage["xgb_all_test_rows_after_split"].all()
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_predictions(predictions)
    return predictions, summary, leakage


if __name__ == "__main__":
    _predictions, _summary, _leakage, _coverage = run_backtest()
    print("Leakage check:")
    print(_leakage.to_string(index=False))
    print("\nCoverage:")
    print(_coverage.to_string(index=False))
    print("\nPooled summary:")
    print(_summary.loc[_summary["event"] == "Pooled"].to_string(index=False))
