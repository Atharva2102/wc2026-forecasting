"""Top-level pipeline orchestrator for writing WC 2026 prediction CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    import config as cfg
    from src.bracket import SlotKey, assign_best_thirds
    from src.data_loader import (
        DATA_RAW,
        NAME_MAP,
        canonical_team,
        clean_results,
        fetch_international_results,
        load_fixtures,
    )
    from src.elo import compute_elo_ratings
    from src.market import blend_ratings, load_market_probs, market_to_rating_scale
    from src.optimizer import best_score, ev_optimal_count, most_likely_outcome
    from src.poisson_model import (
        outcome_probs,
        score_matrix,
        team_lambdas,
        venue_country_from_venue,
    )
    from src.scoring import CORNERS_POINTS, REDS_POINTS, YELLOWS_POINTS
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    import config as cfg
    from src.bracket import SlotKey, assign_best_thirds
    from src.data_loader import (
        DATA_RAW,
        NAME_MAP,
        canonical_team,
        clean_results,
        fetch_international_results,
        load_fixtures,
    )
    from src.elo import compute_elo_ratings
    from src.market import blend_ratings, load_market_probs, market_to_rating_scale
    from src.optimizer import best_score, ev_optimal_count, most_likely_outcome
    from src.poisson_model import (
        outcome_probs,
        score_matrix,
        team_lambdas,
        venue_country_from_venue,
    )
    from src.scoring import CORNERS_POINTS, REDS_POINTS, YELLOWS_POINTS


GROUP_PREDICTION_COLUMNS: list[str] = [
    "predicted_home_goals",
    "predicted_away_goals",
    "corners",
    "yellow_cards",
    "red_cards",
    "winning_team",
]

KNOCKOUT_PREDICTION_COLUMNS: list[str] = [
    "predicted_home_team",
    "predicted_away_team",
    "predicted_home_goals",
    "predicted_away_goals",
    "corners",
    "yellow_cards",
    "red_cards",
    "match_winner",
    "penalties",
]

PENALTY_WIN_PROB_MARGIN: float = 0.08


def _rating(team: str, ratings: dict[str, float]) -> float:
    return ratings.get(team, cfg.DEFAULT_RATING)


def _count_rule(points: dict[str, int], within: int | None = None) -> Callable[[int, int], int]:
    def rule(predicted: int, actual: int) -> int:
        diff = abs(predicted - actual)
        if diff == 0:
            return points["exact"]
        if within is not None and diff <= within:
            return points[f"within_{within}"]
        return 0

    return rule


def _constant_stats() -> tuple[int, int, int]:
    corners = ev_optimal_count(10.0, _count_rule(CORNERS_POINTS, within=2))
    yellows = ev_optimal_count(3.8, _count_rule(YELLOWS_POINTS, within=1))
    reds = ev_optimal_count(0.2, _count_rule(REDS_POINTS, within=None))
    return corners, yellows, reds


def _fixture_style_name(team: str) -> str:
    inverse_name_map = {dataset: fixture for fixture, dataset in NAME_MAP.items()}
    inverse_name_map.update(
        {
            "Cape Verde": "Cabo Verde",
            "Ivory Coast": "Côte d'Ivoire",
            "United States": "USA",
            "Turkey": "Türkiye",
            "Czech Republic": "Czechia",
            "DR Congo": "DR Congo",
        }
    )
    return inverse_name_map.get(team, team)


def _matrix_for_fixture(
    team_a: str,
    team_b: str,
    venue: str,
    ratings: dict[str, float],
) -> np.ndarray:
    venue_country = venue_country_from_venue(venue)
    lambda_a, lambda_b = team_lambdas(
        _rating(team_a, ratings),
        _rating(team_b, ratings),
        team_a=team_a,
        team_b=team_b,
        venue_country=venue_country,
    )
    return score_matrix(lambda_a, lambda_b)


def _expected_goals(matrix: np.ndarray) -> tuple[float, float]:
    goals = np.arange(matrix.shape[0])
    return (
        float((matrix.sum(axis=1) * goals).sum()),
        float((matrix.sum(axis=0) * goals).sum()),
    )


def _deterministic_group_tables(
    group_fixtures: pd.DataFrame,
    ratings: dict[str, float],
) -> tuple[dict[str, str], dict[str, str], list[dict[str, object]]]:
    winners: dict[str, str] = {}
    runners: dict[str, str] = {}
    third_stats: list[dict[str, object]] = []

    for group, fixtures in group_fixtures.groupby("group", sort=True):
        teams = sorted(
            {
                canonical_team(team)
                for team in pd.concat(
                    [fixtures["home_team"], fixtures["away_team"]],
                    ignore_index=True,
                ).dropna()
            }
        )
        stats = {
            team: {
                "team": team,
                "points": 0.0,
                "gd": 0.0,
                "gf": 0.0,
                "rating": _rating(team, ratings),
            }
            for team in teams
        }

        for fixture in fixtures.itertuples(index=False):
            home_team = canonical_team(fixture.home_team)
            away_team = canonical_team(fixture.away_team)
            matrix = _matrix_for_fixture(home_team, away_team, fixture.venue, ratings)
            p_home, p_draw, p_away = outcome_probs(matrix)
            eg_home, eg_away = _expected_goals(matrix)

            stats[home_team]["points"] += 3.0 * p_home + p_draw
            stats[away_team]["points"] += 3.0 * p_away + p_draw
            stats[home_team]["gd"] += eg_home - eg_away
            stats[away_team]["gd"] += eg_away - eg_home
            stats[home_team]["gf"] += eg_home
            stats[away_team]["gf"] += eg_away

        table = sorted(
            stats.values(),
            key=lambda row: (
                -row["points"],
                -row["gd"],
                -row["gf"],
                -row["rating"],
                row["team"],
            ),
        )
        winners[group] = str(table[0]["team"])
        runners[group] = str(table[1]["team"])
        third = dict(table[2])
        third["group"] = group
        third_stats.append(third)

    return winners, runners, third_stats


def _resolve_deterministic_slot(
    slot_text: str,
    match_id: int,
    side: str,
    winners: dict[str, str],
    runners: dict[str, str],
    best3_assignment: dict[SlotKey, str],
    decided_matches: dict[int, dict[str, str]],
) -> str:
    if slot_text.startswith("Winner Group "):
        return winners[slot_text.rsplit(" ", maxsplit=1)[-1]]
    if slot_text.startswith("Runner-up Group "):
        return runners[slot_text.rsplit(" ", maxsplit=1)[-1]]
    if slot_text.startswith("Best 3rd"):
        return best3_assignment[(match_id, side)]
    if slot_text.startswith("Winner Match "):
        return decided_matches[int(slot_text.rsplit(" ", maxsplit=1)[-1])]["winner"]
    if slot_text.startswith("Loser Match "):
        return decided_matches[int(slot_text.rsplit(" ", maxsplit=1)[-1])]["loser"]

    raise ValueError(f"Unrecognized knockout slot: {slot_text!r}")


def _deterministic_knockout_slots(
    group_fixtures: pd.DataFrame,
    knockout_slots: pd.DataFrame,
    ratings: dict[str, float],
) -> dict[SlotKey, str]:
    winners, runners, third_stats = _deterministic_group_tables(group_fixtures, ratings)
    best8 = sorted(
        third_stats,
        key=lambda row: (
            -row["points"],
            -row["gd"],
            -row["gf"],
            -row["rating"],
            row["team"],
        ),
    )[:8]
    best3_assignment = assign_best_thirds(best8, knockout_slots)

    modal_slots: dict[SlotKey, str] = {}
    decided_matches: dict[int, dict[str, str]] = {}

    for fixture in knockout_slots.sort_values("match_id").itertuples(index=False):
        match_id = int(fixture.match_id)
        home_team = _resolve_deterministic_slot(
            str(fixture.slot_home),
            match_id,
            "home",
            winners,
            runners,
            best3_assignment,
            decided_matches,
        )
        away_team = _resolve_deterministic_slot(
            str(fixture.slot_away),
            match_id,
            "away",
            winners,
            runners,
            best3_assignment,
            decided_matches,
        )
        modal_slots[(match_id, "home")] = home_team
        modal_slots[(match_id, "away")] = away_team

        matrix = _matrix_for_fixture(home_team, away_team, fixture.venue, ratings)
        p_home, _p_draw, p_away = outcome_probs(matrix)
        _home_goals, _away_goals, winner_side, _penalties = _knockout_scoreline(
            matrix,
            p_home,
            p_away,
        )
        if winner_side == "home":
            winner, loser = home_team, away_team
        else:
            winner, loser = away_team, home_team
        decided_matches[match_id] = {"winner": winner, "loser": loser}

    r32_teams = [
        team
        for fixture in knockout_slots.loc[knockout_slots["round"] == "Round of 32"]
        .sort_values("match_id")
        .itertuples(index=False)
        for team in (
            modal_slots[(int(fixture.match_id), "home")],
            modal_slots[(int(fixture.match_id), "away")],
        )
    ]
    duplicates = sorted(
        team for team in set(r32_teams) if r32_teams.count(team) > 1
    )
    assert len(r32_teams) == 32 and len(set(r32_teams)) == 32, (
        "Round-of-32 slots must contain 32 distinct teams; "
        f"duplicates: {duplicates}"
    )

    return modal_slots


def predict_group_stage(
    fixtures: pd.DataFrame,
    elo_ratings: dict[str, float],
    constant_stats: tuple[int, int, int] | None = None,
) -> pd.DataFrame:
    """Generate predictions for all group-stage fixtures."""
    corners, yellows, reds = constant_stats or _constant_stats()
    predictions = fixtures.copy()

    rows: list[dict[str, object]] = []
    for fixture in fixtures.itertuples(index=False):
        home_team = canonical_team(fixture.home_team)
        away_team = canonical_team(fixture.away_team)
        matrix = _matrix_for_fixture(home_team, away_team, fixture.venue, elo_ratings)
        predicted_home_goals, predicted_away_goals = best_score(matrix, mode="any")

        rows.append(
            {
                "predicted_home_goals": predicted_home_goals,
                "predicted_away_goals": predicted_away_goals,
                "corners": corners,
                "yellow_cards": yellows,
                "red_cards": reds,
                "winning_team": most_likely_outcome(matrix),
            }
        )

    for column in GROUP_PREDICTION_COLUMNS:
        predictions[column] = [row[column] for row in rows]

    return predictions


def _knockout_scoreline(
    matrix: np.ndarray,
    p_home: float,
    p_away: float,
) -> tuple[int, int, str, bool]:
    margin = abs(p_home - p_away)
    if margin <= PENALTY_WIN_PROB_MARGIN:
        score = best_score(matrix, mode="draw")
        winner = "home" if p_home >= p_away else "away"
        return score[0], score[1], winner, True

    if p_home >= p_away:
        score = best_score(matrix, mode="home")
        return score[0], score[1], "home", False

    score = best_score(matrix, mode="away")
    return score[0], score[1], "away", False


def predict_knockout_stage(
    ko_fixtures: pd.DataFrame,
    elo_ratings: dict[str, float],
    modal_slots: dict[SlotKey, str],
    constant_stats: tuple[int, int, int] | None = None,
) -> pd.DataFrame:
    """Generate predictions for all knockout slots from modal bracket occupants."""
    corners, yellows, reds = constant_stats or _constant_stats()
    predictions = ko_fixtures.copy()

    rows: list[dict[str, object]] = []
    for fixture in ko_fixtures.itertuples(index=False):
        match_id = int(fixture.match_id)
        home_team = modal_slots[(match_id, "home")]
        away_team = modal_slots[(match_id, "away")]
        matrix = _matrix_for_fixture(home_team, away_team, fixture.venue, elo_ratings)
        p_home, _p_draw, p_away = outcome_probs(matrix)
        home_goals, away_goals, match_winner, penalties = _knockout_scoreline(
            matrix,
            p_home,
            p_away,
        )

        rows.append(
            {
                "predicted_home_team": _fixture_style_name(home_team),
                "predicted_away_team": _fixture_style_name(away_team),
                "predicted_home_goals": home_goals,
                "predicted_away_goals": away_goals,
                "corners": corners,
                "yellow_cards": yellows,
                "red_cards": reds,
                "match_winner": match_winner,
                "penalties": bool(penalties),
            }
        )

    for column in KNOCKOUT_PREDICTION_COLUMNS:
        predictions[column] = [row[column] for row in rows]

    return predictions


def _assert_complete(df: pd.DataFrame, expected_columns: list[str], label: str) -> None:
    if list(df.columns) != expected_columns:
        raise AssertionError(f"{label} columns do not match expected schema")
    if df.isna().any().any():
        raise AssertionError(f"{label} contains NaN values")
    if df.astype(object).where(pd.notna(df), None).isnull().any().any():
        raise AssertionError(f"{label} contains None values")


def write_submission(predictions: pd.DataFrame, output_path: str | Path) -> None:
    """Write a prediction DataFrame to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)


def _prediction_ratings(
    elo_ratings: dict[str, float],
    pure_elo: bool = False,
) -> dict[str, float]:
    if pure_elo:
        return elo_ratings

    market_probs = load_market_probs(cfg.MARKET_PROBS_PATH)
    market_ratings = market_to_rating_scale(market_probs, elo_ratings)
    blended = blend_ratings(elo_ratings, market_ratings, cfg.MARKET_WEIGHT)

    print("Top 15 blended ratings:")
    for rank, (team, rating) in enumerate(
        sorted(blended.items(), key=lambda item: item[1], reverse=True)[:15],
        start=1,
    ):
        print(f"{rank}. {_fixture_style_name(team)}: {rating:.1f}")

    return blended


def run_pipeline(pure_elo: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full forecasting pipeline and write group/knockout CSVs."""
    raw_results = fetch_international_results(force_refresh=False)
    results = clean_results(raw_results)
    elo_ratings, _history = compute_elo_ratings(results)
    ratings = _prediction_ratings(elo_ratings, pure_elo=pure_elo)

    group_fixtures, knockout_slots = load_fixtures(
        DATA_RAW / "group_fixtures.csv",
        DATA_RAW / "knockout_slots.csv",
    )

    constant_stats = _constant_stats()
    modal_slots = _deterministic_knockout_slots(
        group_fixtures,
        knockout_slots,
        ratings,
    )

    group_predictions = predict_group_stage(group_fixtures, ratings, constant_stats)
    knockout_predictions = predict_knockout_stage(
        knockout_slots,
        ratings,
        modal_slots,
        constant_stats,
    )

    group_columns = list(group_fixtures.columns) + GROUP_PREDICTION_COLUMNS
    knockout_columns = list(knockout_slots.columns) + KNOCKOUT_PREDICTION_COLUMNS
    group_predictions = group_predictions.loc[:, group_columns]
    knockout_predictions = knockout_predictions.loc[:, knockout_columns]

    _assert_complete(group_predictions, group_columns, "group_predictions")
    _assert_complete(knockout_predictions, knockout_columns, "knockout_predictions")

    write_submission(group_predictions, cfg.OUTPUTS_DIR / "group_predictions.csv")
    write_submission(knockout_predictions, cfg.OUTPUTS_DIR / "knockout_predictions.csv")

    champion = knockout_predictions.loc[
        knockout_predictions["match_id"] == 104,
        "match_winner",
    ].iloc[0]
    final_row = knockout_predictions.loc[knockout_predictions["match_id"] == 104].iloc[0]
    predicted_champion = (
        final_row["predicted_home_team"]
        if champion == "home"
        else final_row["predicted_away_team"]
    )
    semi_finalists = sorted(
        set(
            knockout_predictions.loc[
                knockout_predictions["match_id"].isin([101, 102]),
                ["predicted_home_team", "predicted_away_team"],
            ].to_numpy().ravel()
        )
    )
    r32_team_list = [
        _fixture_style_name(team)
        for fixture in knockout_slots.loc[knockout_slots["round"] == "Round of 32"]
        .sort_values("match_id")
        .itertuples(index=False)
        for team in (
            modal_slots[(int(fixture.match_id), "home")],
            modal_slots[(int(fixture.match_id), "away")],
        )
    ]

    print(f"Predicted champion: {predicted_champion}")
    print(f"Semi-finalists: {', '.join(semi_finalists)}")
    print(f"R32 teams: {', '.join(r32_team_list)}")
    print(f"Filled group rows: {len(group_predictions)} / 72")
    print(f"Filled knockout rows: {len(knockout_predictions)} / 32")

    return group_predictions, knockout_predictions


def run(pure_elo: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alias for run_pipeline()."""
    return run_pipeline(pure_elo=pure_elo)


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
