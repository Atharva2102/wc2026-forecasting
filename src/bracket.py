"""Group-stage standings, best-third-place qualification logic, and Monte Carlo bracket simulator."""

from __future__ import annotations

import pandas as pd
import numpy as np


def group_standings(results: pd.DataFrame) -> pd.DataFrame:
    """Compute group-stage standings from a DataFrame of simulated match results.

    Applies standard FIFA tiebreakers: points → GD → GF → head-to-head.

    Args:
        results: DataFrame with columns: group, home_team, away_team, home_goals, away_goals.

    Returns:
        DataFrame with one row per team, sorted by group then standing position.
    """
    # TODO
    pass


def assign_best_thirds(standings: pd.DataFrame, n: int = 8) -> list[str]:
    """Select the n best third-placed teams that advance to the Round of 32.

    For WC 2026 with 12 groups, the top 2 from each group (24 teams) plus
    the 8 best third-placed teams = 32 teams advance.

    Args:
        standings: Output of group_standings().
        n: Number of third-placed teams to qualify.

    Returns:
        Sorted list of qualifying team names.
    """
    # TODO
    pass


def simulate_match(
    rating_a: float,
    rating_b: float,
    elo_ratings: dict[str, float],
    rng: np.random.Generator,
    neutral: bool = True,
) -> tuple[int, int]:
    """Draw a single match outcome from the Poisson model.

    Args:
        rating_a: Elo rating of team A.
        rating_b: Elo rating of team B.
        elo_ratings: Full ratings dict (passed through to allow host-advantage lookup).
        rng: NumPy random generator for reproducibility.
        neutral: Whether the venue is neutral.

    Returns:
        Tuple (goals_a, goals_b).
    """
    # TODO
    pass


def simulate_knockout_round(
    bracket: list[tuple[str, str]],
    elo_ratings: dict[str, float],
    rng: np.random.Generator,
) -> list[str]:
    """Simulate one round of knockout matches and return the list of winners.

    Penalty shoot-out is resolved as a 50/50 coin flip after a simulated draw.

    Args:
        bracket: List of (team_a, team_b) matchup tuples for this round.
        elo_ratings: Current Elo ratings.
        rng: NumPy random generator.

    Returns:
        List of winning team names in bracket order.
    """
    # TODO
    pass


def monte_carlo_bracket(
    qualified_teams: list[str],
    group_assignments: dict[str, str],
    elo_ratings: dict[str, float],
    n_sims: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run n_sims full tournament simulations and return win-probability estimates.

    Covers R32 → R16 → QF → SF → third-place play-off → Final.

    Args:
        qualified_teams: 32 teams that advanced from the group stage.
        group_assignments: Maps team name to its group letter (for bracket seeding).
        elo_ratings: Elo ratings dict at the start of the knockout phase.
        n_sims: Number of Monte Carlo iterations.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame indexed by team with columns for each round reach-probability
        (e.g. 'p_r16', 'p_qf', 'p_sf', 'p_final', 'p_winner').
    """
    # TODO
    pass
