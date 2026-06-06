"""Bivariate Poisson goal model: derive lambdas from Elo ratings and compute score distributions."""

from __future__ import annotations

import numpy as np


def team_lambdas(
    rating_a: float,
    rating_b: float,
    base_goals: float,
    spread: float,
    neutral: bool = True,
    host_advantage: float = 0.0,
) -> tuple[float, float]:
    """Convert Elo ratings into Poisson goal-rate parameters for both teams.

    Args:
        rating_a: Elo rating of the home/team-A side.
        rating_b: Elo rating of the away/team-B side.
        base_goals: League-average goals per team per match (tunable).
        spread: Scaling factor mapping Elo difference to lambda offset.
        neutral: If True, no location adjustment is applied.
        host_advantage: Additional lambda bonus if a host nation is playing.

    Returns:
        Tuple (lambda_a, lambda_b) — expected goals for each team.
    """
    # TODO
    pass


def score_matrix(
    lambda_a: float,
    lambda_b: float,
    max_goals: int = 10,
) -> np.ndarray:
    """Build a (max_goals+1) x (max_goals+1) probability matrix P[i, j] = P(A scores i, B scores j).

    Assumes independence between the two Poisson processes (Dixon-Coles correction
    can be added later).

    Args:
        lambda_a: Expected goals for team A.
        lambda_b: Expected goals for team B.
        max_goals: Truncation point; probabilities beyond this are discarded.

    Returns:
        2-D numpy array; rows = goals for A, cols = goals for B.
    """
    # TODO
    pass


def outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    """Collapse the score matrix into win / draw / loss probabilities for team A.

    Args:
        matrix: Score-probability matrix from score_matrix().

    Returns:
        Tuple (p_win, p_draw, p_loss) summing to ~1.0.
    """
    # TODO
    pass


def expected_goals_from_matrix(matrix: np.ndarray) -> tuple[float, float]:
    """Recover expected goals for each team by marginalising the score matrix.

    Args:
        matrix: Score-probability matrix from score_matrix().

    Returns:
        Tuple (exp_goals_a, exp_goals_b).
    """
    # TODO
    pass
