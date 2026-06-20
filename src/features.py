"""Pre-match feature matrix builder for future goals-model training.

One row is produced per historical match. Features are built in a single
chronological pass: the current tracker state is read first, then the match
result updates the trackers. This keeps every feature strictly pre-match.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import config as cfg
    from src.data_loader import canonical_team, clean_results, fetch_international_results
    from src.elo import classify_tournament, expected_score, is_excluded, mov_multiplier
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    import config as cfg
    from src.data_loader import canonical_team, clean_results, fetch_international_results
    from src.elo import classify_tournament, expected_score, is_excluded, mov_multiplier


MAJOR_TIERS = {"world_cup", "continental_final"}
QUALIFIER_TIERS = {"wc_qual_nations", "continental_qual"}


@dataclass
class TeamState:
    matches: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=10))
    unbeaten_streak: int = 0
    winning_streak: int = 0
    major_games: int = 0
    major_wins: int = 0


@dataclass
class CurrentFeatureState:
    ratings: dict[str, float]
    states: dict[str, TeamState]
    h2h: dict[tuple[str, str], list[dict[str, Any]]]


def _team_country(team: str) -> str | None:
    if team in {"USA", "United States"}:
        return "USA"
    if team in {"Mexico", "Canada"}:
        return team
    return None


def _is_host_team(team: str) -> bool:
    country = _team_country(team)
    return country in cfg.HOSTS if country is not None else False


def _actual_points(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def _outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if away_goals > home_goals:
        return "away"
    return "draw"


def _rolling_features(state: TeamState, window: int) -> dict[str, float]:
    prior = list(state.matches)[-window:]
    if not prior:
        return {
            f"last{window}_ppg": np.nan,
            f"last{window}_gf_avg": np.nan,
            f"last{window}_ga_avg": np.nan,
            f"last{window}_win_rate": np.nan,
        }

    n = len(prior)
    return {
        f"last{window}_ppg": sum(item["points"] for item in prior) / n,
        f"last{window}_gf_avg": sum(item["gf"] for item in prior) / n,
        f"last{window}_ga_avg": sum(item["ga"] for item in prior) / n,
        f"last{window}_win_rate": sum(item["win"] for item in prior) / n,
    }


def _clean_sheet_rate(state: TeamState, window: int = 5) -> float:
    prior = list(state.matches)[-window:]
    if not prior:
        return np.nan
    return sum(item["clean_sheet"] for item in prior) / len(prior)


def _draw_rate(state: TeamState, window: int = 10) -> float:
    prior = list(state.matches)[-window:]
    if not prior:
        return np.nan
    return sum(item["draw"] for item in prior) / len(prior)


def _major_win_rate(state: TeamState) -> float:
    if state.major_games == 0:
        return np.nan
    return state.major_wins / state.major_games


def _state_features(prefix: str, state: TeamState) -> dict[str, float | int | bool]:
    features: dict[str, float | int | bool] = {
        f"{prefix}_prior_matches": len(state.matches),
        f"{prefix}_has_min_history": len(state.matches) >= cfg.MIN_HISTORY,
        f"{prefix}_unbeaten_streak": state.unbeaten_streak,
        f"{prefix}_winning_streak": state.winning_streak,
        f"{prefix}_clean_sheet_rate_l5": _clean_sheet_rate(state, 5),
        f"{prefix}_draw_rate_l10": _draw_rate(state, 10),
        f"{prefix}_major_prior_matches": state.major_games,
        f"{prefix}_major_win_rate": _major_win_rate(state),
    }
    for window in (5, 10):
        for name, value in _rolling_features(state, window).items():
            features[f"{prefix}_{name}"] = value
    return features


def _h2h_features(
    home_team: str,
    away_team: str,
    h2h: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, float | int]:
    key = tuple(sorted((home_team, away_team)))
    meetings = h2h.get(key, [])
    if not meetings:
        return {
            "h2h_count": 0,
            "h2h_home_win_rate": 0.0,
            "h2h_home_avg_gf": 0.0,
            "h2h_away_avg_gf": 0.0,
        }

    home_wins = 0
    home_gf = 0
    away_gf = 0
    for meeting in meetings:
        goals_for_home = meeting["goals"].get(home_team, 0)
        goals_for_away = meeting["goals"].get(away_team, 0)
        home_gf += goals_for_home
        away_gf += goals_for_away
        if goals_for_home > goals_for_away:
            home_wins += 1

    n = len(meetings)
    return {
        "h2h_count": n,
        "h2h_home_win_rate": home_wins / n,
        "h2h_home_avg_gf": home_gf / n,
        "h2h_away_avg_gf": away_gf / n,
    }


def _regress_elo(ratings: dict[str, float]) -> None:
    if not ratings or cfg.REGRESSION_STRENGTH <= 0:
        return
    for team, rating in ratings.items():
        ratings[team] = cfg.DEFAULT_RATING + (
            rating - cfg.DEFAULT_RATING
        ) * (1.0 - cfg.REGRESSION_STRENGTH)


def _update_elo(
    ratings: dict[str, float],
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    neutral: bool,
    tournament: str,
) -> None:
    """Inline Elo update; MUST stay in sync with elo.py.

    This mirrors compute_elo_ratings: tournament-tier K multiplier,
    margin-of-victory multiplier, non-neutral home advantage in expected-score
    calculation only, and annual regression handled by the caller.
    """
    if is_excluded(tournament):
        return

    home_rating = ratings.setdefault(home_team, cfg.DEFAULT_RATING)
    away_rating = ratings.setdefault(away_team, cfg.DEFAULT_RATING)
    home_advantage = 0.0 if neutral else cfg.ELO_HOME_ADVANTAGE
    expected_home = expected_score(home_rating, away_rating, home_advantage)
    if home_goals > away_goals:
        actual_home = 1.0
    elif home_goals == away_goals:
        actual_home = 0.5
    else:
        actual_home = 0.0

    tier = classify_tournament(tournament)
    tier_weight = cfg.TOURNAMENT_WEIGHTS.get(tier, cfg.TOURNAMENT_WEIGHTS["minor"])
    k = cfg.K_BASE * tier_weight * mov_multiplier(home_goals - away_goals)
    delta = k * (actual_home - expected_home)
    ratings[home_team] = home_rating + delta
    ratings[away_team] = away_rating - delta


def _update_team_state(
    state: TeamState,
    gf: int,
    ga: int,
    tier: str,
) -> None:
    points = _actual_points(gf, ga)
    win = gf > ga
    draw = gf == ga
    state.matches.append(
        {
            "points": points,
            "gf": gf,
            "ga": ga,
            "win": float(win),
            "draw": float(draw),
            "clean_sheet": float(ga == 0),
        }
    )
    state.winning_streak = state.winning_streak + 1 if win else 0
    state.unbeaten_streak = state.unbeaten_streak + 1 if points > 0 else 0
    if tier in MAJOR_TIERS:
        state.major_games += 1
        state.major_wins += int(win)


def build_feature_matrix(results: pd.DataFrame) -> pd.DataFrame:
    """Build a strict pre-match feature matrix from cleaned results.

    Rows before both teams have at least cfg.MIN_HISTORY prior games are kept and
    flagged with `home_has_min_history` / `away_has_min_history`; sparse rolling
    values remain NaN and are reported by validation.
    """
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Results missing required columns: {sorted(missing)}")

    work = results.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date", "home_score", "away_score"])
    work = work.sort_values("date").reset_index(drop=True)

    ratings: dict[str, float] = {}
    states: defaultdict[str, TeamState] = defaultdict(TeamState)
    h2h: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    current_year: int | None = None

    for match in work.itertuples(index=False):
        match_year = match.date.year
        if current_year is None:
            current_year = match_year
        elif match_year != current_year:
            _regress_elo(ratings)
            current_year = match_year

        home_team = canonical_team(match.home_team)
        away_team = canonical_team(match.away_team)
        home_goals = int(match.home_score)
        away_goals = int(match.away_score)
        neutral = bool(match.neutral)
        tournament = str(match.tournament)
        tier = classify_tournament(tournament)

        home_state = states[home_team]
        away_state = states[away_team]
        home_elo = ratings.get(home_team, cfg.DEFAULT_RATING)
        away_elo = ratings.get(away_team, cfg.DEFAULT_RATING)

        row: dict[str, Any] = {
            "date": match.date,
            "home_team": home_team,
            "away_team": away_team,
            "tournament": tournament,
            "neutral": neutral,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "outcome": _outcome(home_goals, away_goals),
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "tournament_tier": tier,
            "is_major_tournament": tier in MAJOR_TIERS,
            "is_qualifier": tier in QUALIFIER_TIERS,
            "is_friendly": tier == "friendly",
            "home_is_host": _is_host_team(home_team),
            "away_is_host": _is_host_team(away_team),
        }
        if hasattr(match, "match_id"):
            row["match_id"] = match.match_id

        row.update(_state_features("home", home_state))
        row.update(_state_features("away", away_state))
        row.update(_h2h_features(home_team, away_team, h2h))
        rows.append(row)

        _update_team_state(home_state, home_goals, away_goals, tier)
        _update_team_state(away_state, away_goals, home_goals, tier)
        pair_key = tuple(sorted((home_team, away_team)))
        h2h[pair_key].append(
            {"goals": {home_team: home_goals, away_team: away_goals}}
        )
        _update_elo(
            ratings,
            home_team,
            away_team,
            home_goals,
            away_goals,
            neutral,
            tournament,
        )

    return pd.DataFrame(rows)


def build_current_state(results: pd.DataFrame) -> CurrentFeatureState:
    """Replay all cleaned results and return current trackers for live features."""
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Results missing required columns: {sorted(missing)}")

    work = results.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date", "home_score", "away_score"])
    work = work.sort_values("date").reset_index(drop=True)

    ratings: dict[str, float] = {}
    states: defaultdict[str, TeamState] = defaultdict(TeamState)
    h2h: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    current_year: int | None = None

    for match in work.itertuples(index=False):
        match_year = match.date.year
        if current_year is None:
            current_year = match_year
        elif match_year != current_year:
            _regress_elo(ratings)
            current_year = match_year

        home_team = canonical_team(match.home_team)
        away_team = canonical_team(match.away_team)
        home_goals = int(match.home_score)
        away_goals = int(match.away_score)
        neutral = bool(match.neutral)
        tournament = str(match.tournament)
        tier = classify_tournament(tournament)

        _update_team_state(states[home_team], home_goals, away_goals, tier)
        _update_team_state(states[away_team], away_goals, home_goals, tier)
        pair_key = tuple(sorted((home_team, away_team)))
        h2h[pair_key].append(
            {"goals": {home_team: home_goals, away_team: away_goals}}
        )
        _update_elo(
            ratings,
            home_team,
            away_team,
            home_goals,
            away_goals,
            neutral,
            tournament,
        )

    return CurrentFeatureState(
        ratings=ratings,
        states=dict(states),
        h2h=dict(h2h),
    )


def current_feature_row(
    state: CurrentFeatureState,
    home: str,
    away: str,
    neutral: bool,
    tournament: str = "FIFA World Cup",
) -> dict[str, Any]:
    """Build a live feature row for an arbitrary hypothetical matchup.

    This uses the same formulas and field names as build_feature_matrix(), but
    reads the final current-state trackers instead of updating them.
    """
    home_team = canonical_team(home)
    away_team = canonical_team(away)
    home_state = state.states.get(home_team, TeamState())
    away_state = state.states.get(away_team, TeamState())
    home_elo = state.ratings.get(home_team, cfg.DEFAULT_RATING)
    away_elo = state.ratings.get(away_team, cfg.DEFAULT_RATING)
    tier = classify_tournament(tournament)

    row: dict[str, Any] = {
        "home_team": home_team,
        "away_team": away_team,
        "tournament": tournament,
        "neutral": bool(neutral),
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "tournament_tier": tier,
        "is_major_tournament": tier in MAJOR_TIERS,
        "is_qualifier": tier in QUALIFIER_TIERS,
        "is_friendly": tier == "friendly",
        "home_is_host": _is_host_team(home_team),
        "away_is_host": _is_host_team(away_team),
    }
    row.update(_state_features("home", home_state))
    row.update(_state_features("away", away_state))
    row.update(_h2h_features(home_team, away_team, state.h2h))
    return row


def _feature_columns(matrix: pd.DataFrame) -> list[str]:
    metadata = {
        "date",
        "home_team",
        "away_team",
        "tournament",
        "neutral",
        "match_id",
        "home_goals",
        "away_goals",
        "outcome",
    }
    return [column for column in matrix.columns if column not in metadata]


def _print_validation(matrix: pd.DataFrame, source_results: pd.DataFrame) -> None:
    feature_cols = _feature_columns(matrix)
    print(f"Feature matrix shape: {matrix.shape}")
    print(f"Date range: {matrix['date'].min().date()} to {matrix['date'].max().date()}")
    print(f"Feature columns ({len(feature_cols)}):")
    print(feature_cols)

    mid_idx = len(matrix) // 2
    sample = matrix.iloc[mid_idx]
    home_team = sample["home_team"]
    sample_date = sample["date"]
    prior = source_results.loc[
        (source_results["date"] < sample_date)
        & (
            (source_results["home_team"].map(canonical_team) == home_team)
            | (source_results["away_team"].map(canonical_team) == home_team)
        )
    ].sort_values("date").tail(5)

    print("\nNo-leakage spot check:")
    print(
        f"Sample match: {sample_date.date()} {sample['home_team']} vs {sample['away_team']}"
    )
    spot_cols = [
        "home_last5_ppg",
        "home_last5_gf_avg",
        "home_last5_ga_avg",
        "home_last5_win_rate",
    ]
    print("Computed home last-5 form features:")
    print(sample[spot_cols].to_string())
    print("Raw prior 5 matches for home team:")
    print(
        prior[
            ["date", "home_team", "away_team", "home_score", "away_score", "tournament"]
        ].to_string(index=False)
    )

    null_report = (
        matrix[feature_cols]
        .isna()
        .sum()
        .rename("null_count")
        .to_frame()
        .assign(null_rate=lambda frame: frame["null_count"] / len(matrix))
    )
    print("\nNull / coverage report by feature:")
    print(null_report.to_string())


def main() -> None:
    results = clean_results(fetch_international_results(force_refresh=False))
    matrix = build_feature_matrix(results)
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = cfg.DATA_PROCESSED / "feature_matrix.parquet"
    matrix.to_parquet(output_path, index=False)
    print(f"Wrote {output_path}")
    _print_validation(matrix, results)


if __name__ == "__main__":
    main()
