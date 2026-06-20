"""Elo rating engine with tournament weighting and point-in-time snapshots."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

try:
    import config as cfg
    from src.data_loader import canonical_team
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    import config as cfg
    from src.data_loader import canonical_team


_CONFIG_NAMES = {
    "DEFAULT_RATING",
    "ELO_HOME_ADVANTAGE",
    "EXCLUDED_TOURNAMENT_PATTERNS",
    "K_BASE",
    "REGRESSION_STRENGTH",
    "TOURNAMENT_TIER_PATTERNS",
    "TOURNAMENT_WEIGHTS",
    "USE_MARGIN_OF_VICTORY",
}


def __getattr__(name: str) -> object:
    if name in _CONFIG_NAMES:
        return getattr(cfg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _setting(name: str) -> object:
    return globals().get(name, getattr(cfg, name))


def classify_tournament(name: str) -> str:
    """Return the configured tournament tier for a tournament label."""
    label = str(name)
    label_lower = label.casefold()

    tier_patterns = _setting("TOURNAMENT_TIER_PATTERNS")
    for tier, patterns in tier_patterns.items():
        for pattern in patterns:
            if pattern.casefold() in label_lower:
                return tier

    return "minor"


def is_excluded(name: str) -> bool:
    """Return True for tournament labels excluded from senior FIFA ratings."""
    label_lower = str(name).casefold()
    return any(
        pattern.casefold() in label_lower
        for pattern in _setting("EXCLUDED_TOURNAMENT_PATTERNS")
    )


def expected_score(
    rating_a: float,
    rating_b: float,
    home_advantage: float = 0.0,
) -> float:
    """Return team A's expected Elo score against team B.

    `home_advantage` is added to team A only, so pass ELO_HOME_ADVANTAGE for a
    non-neutral historical home side and 0 otherwise.
    """
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b + home_advantage) / 400.0))


def mov_multiplier(goal_diff: int | float) -> float:
    """Return the margin-of-victory multiplier for a match.

    Formula: max(1, ln(abs(goal_diff) + 1)). This keeps draws and one-goal wins
    at the normal K-factor, while larger wins move ratings a bit more. The
    multiplier is disabled entirely when USE_MARGIN_OF_VICTORY is False.
    """
    if not _setting("USE_MARGIN_OF_VICTORY"):
        return 1.0

    margin = abs(float(goal_diff))
    return max(1.0, math.log(margin + 1.0))


def _actual_home_score(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals == away_goals:
        return 0.5
    return 0.0


def _regress_to_mean(
    ratings: dict[str, float],
    strength: float | None = None,
    mean_rating: float | None = None,
) -> None:
    if strength is None:
        strength = _setting("REGRESSION_STRENGTH")
    if mean_rating is None:
        mean_rating = _setting("DEFAULT_RATING")

    if not ratings or strength <= 0:
        return

    for team, rating in ratings.items():
        ratings[team] = mean_rating + (rating - mean_rating) * (1.0 - strength)


def _neutral_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() == "true"


def compute_elo_ratings(matches: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Compute final team Elo ratings and a rating history table.

    Matches are sorted chronologically, excluded tournaments are removed before
    updates, team names are canonicalized to historical-dataset spellings, and
    ratings are regressed toward DEFAULT_RATING at each calendar-year boundary.
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
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError(f"Matches data is missing required columns: {sorted(missing)}")

    work = matches.loc[:, sorted(required)].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["home_score"] = pd.to_numeric(work["home_score"], errors="coerce")
    work["away_score"] = pd.to_numeric(work["away_score"], errors="coerce")
    work = work.dropna(subset=["date", "home_score", "away_score"])
    work = work.loc[~work["tournament"].map(is_excluded)].copy()

    work["home_team"] = work["home_team"].map(canonical_team)
    work["away_team"] = work["away_team"].map(canonical_team)
    work["tier"] = work["tournament"].map(classify_tournament)
    work = work.sort_values("date").reset_index(drop=True)

    ratings: dict[str, float] = {}
    history_rows: list[dict[str, object]] = []
    current_year: int | None = None

    for row in work.itertuples(index=False):
        match_year = row.date.year
        if current_year is None:
            current_year = match_year
        elif match_year != current_year:
            _regress_to_mean(ratings)
            regression_date = pd.Timestamp(year=match_year, month=1, day=1)
            for team, rating in ratings.items():
                history_rows.append(
                    {"date": regression_date, "team": team, "rating": rating}
                )
            current_year = match_year

        home_team = row.home_team
        away_team = row.away_team
        home_rating = ratings.setdefault(home_team, _setting("DEFAULT_RATING"))
        away_rating = ratings.setdefault(away_team, _setting("DEFAULT_RATING"))

        neutral = _neutral_flag(row.neutral)
        home_advantage = 0.0 if neutral else _setting("ELO_HOME_ADVANTAGE")
        expected_home = expected_score(home_rating, away_rating, home_advantage)
        actual_home = _actual_home_score(int(row.home_score), int(row.away_score))

        tournament_weights = _setting("TOURNAMENT_WEIGHTS")
        tier_weight = tournament_weights.get(row.tier, tournament_weights["minor"])
        goal_diff = int(row.home_score) - int(row.away_score)
        k = _setting("K_BASE") * tier_weight * mov_multiplier(goal_diff)
        delta = k * (actual_home - expected_home)

        ratings[home_team] = home_rating + delta
        ratings[away_team] = away_rating - delta

        history_rows.append(
            {"date": row.date, "team": home_team, "rating": ratings[home_team]}
        )
        history_rows.append(
            {"date": row.date, "team": away_team, "rating": ratings[away_team]}
        )

    history = pd.DataFrame(history_rows, columns=["date", "team", "rating"])
    return ratings, history


def get_ratings_asof(
    history: pd.DataFrame,
    date: str | pd.Timestamp,
) -> dict[str, float]:
    """Return each team's rating strictly before `date`, with no future leakage."""
    if history.empty:
        return {}

    cutoff = pd.Timestamp(date)
    all_teams = set(history["team"])
    past = history.loc[pd.to_datetime(history["date"]) < cutoff].copy()
    if past.empty:
        return {team: _setting("DEFAULT_RATING") for team in all_teams}

    latest = past.sort_values("date").groupby("team", as_index=False).tail(1)
    ratings = {team: _setting("DEFAULT_RATING") for team in all_teams}
    ratings.update(dict(zip(latest["team"], latest["rating"], strict=False)))
    return ratings


def update_ratings(
    rating_a: float,
    rating_b: float,
    result: float,
    k: float,
) -> tuple[float, float]:
    """Apply a single neutral Elo update and return ratings for both teams."""
    expected_a = expected_score(rating_a, rating_b, 0.0)
    delta = k * (result - expected_a)
    return rating_a + delta, rating_b - delta


def apply_home_advantage(
    rating: float,
    is_home: bool,
    is_neutral: bool,
    home_advantage: float,
) -> float:
    """Add the home bonus to a rating only for a non-neutral home side."""
    return rating + home_advantage if is_home and not is_neutral else rating


def get_ratings_snapshot(
    matches: pd.DataFrame,
    as_of_date: str,
    **_: object,
) -> dict[str, float]:
    """Compatibility wrapper returning ratings before `as_of_date`."""
    filtered = matches.loc[pd.to_datetime(matches["date"]) < pd.Timestamp(as_of_date)]
    ratings, _history = compute_elo_ratings(filtered)
    return ratings
