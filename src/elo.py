"""Elo rating engine with recency decay and tournament-importance weighting."""

from __future__ import annotations

import pandas as pd


def compute_elo_ratings(
    matches: pd.DataFrame,
    k_base: float,
    tournament_weights: dict[str, float],
    initial_rating: float = 1500.0,
    recency_decay: float = 1.0,
) -> dict[str, float]:
    """Iterate through sorted matches and return final Elo rating per team.

    Args:
        matches: Cleaned, chronologically sorted results DataFrame.
        k_base: Base K-factor controlling rating volatility.
        tournament_weights: Multiplier per tournament type, e.g. {"FIFA World Cup": 3.0}.
        initial_rating: Starting rating for any team not yet seen.
        recency_decay: Multiplicative annual decay applied to older matches.

    Returns:
        Dict mapping team name -> current Elo rating.
    """
    # TODO
    pass


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return the expected score for team A against team B using the standard Elo formula.

    Args:
        rating_a: Current Elo rating of team A.
        rating_b: Current Elo rating of team B.

    Returns:
        Float in (0, 1) representing team A's expected score.
    """
    # TODO
    pass


def update_ratings(
    rating_a: float,
    rating_b: float,
    result: float,
    k: float,
) -> tuple[float, float]:
    """Apply a single match update and return new ratings for both teams.

    Args:
        rating_a: Pre-match Elo for team A.
        rating_b: Pre-match Elo for team B.
        result: Actual score for A: 1.0 = win, 0.5 = draw, 0.0 = loss.
        k: Effective K-factor (k_base * tournament_weight).

    Returns:
        Tuple (new_rating_a, new_rating_b).
    """
    # TODO
    pass


def apply_home_advantage(
    rating: float,
    is_home: bool,
    is_neutral: bool,
    home_advantage: float,
) -> float:
    """Add or skip home-advantage bonus to a rating before an expected-score calculation.

    Args:
        rating: Team's base Elo rating.
        is_home: True if this team is the designated home side.
        is_neutral: True if the venue is neutral (overrides is_home).
        home_advantage: Elo points to add for home side.

    Returns:
        Adjusted rating.
    """
    # TODO
    pass


def get_ratings_snapshot(
    matches: pd.DataFrame,
    as_of_date: str,
    **kwargs,
) -> dict[str, float]:
    """Return Elo ratings computed from all matches up to (but not including) as_of_date.

    Useful for backtesting without data leakage.

    Args:
        matches: Full cleaned results DataFrame.
        as_of_date: ISO date string 'YYYY-MM-DD'.
        **kwargs: Forwarded to compute_elo_ratings().

    Returns:
        Dict mapping team name -> Elo rating at that snapshot date.
    """
    # TODO
    pass
