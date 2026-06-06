"""Expected-value optimisation: choose the prediction that maximises expected competition points."""

from __future__ import annotations

import numpy as np


def best_score(
    matrix: np.ndarray,
    mode: str = "ev",
) -> tuple[int, int]:
    """Return the single (goals_a, goals_b) prediction that maximises expected points.

    Args:
        matrix: Score-probability matrix from poisson_model.score_matrix().
        mode: 'ev' uses the full scoring rubric; 'mode' simply picks the most likely scoreline.

    Returns:
        Tuple (predicted_goals_a, predicted_goals_b).
    """
    # TODO
    pass


def ev_optimal_count(
    lam: float,
    rule: str,
    within_n: int | None = None,
) -> int:
    """Find the integer count that maximises expected points under an exact-or-within-N rule.

    Used for corners, yellow cards, red cards.

    Args:
        lam: Poisson rate parameter for the count variable.
        rule: One of 'corners', 'yellows', 'reds'.
        within_n: If provided, overrides the default tolerance for the rule.

    Returns:
        Integer prediction for the count.
    """
    # TODO
    pass


def expected_points_for_scoreline(
    predicted_a: int,
    predicted_b: int,
    matrix: np.ndarray,
    round_multiplier: float,
) -> float:
    """Compute the expected points earned by predicting a specific scoreline.

    Integrates over all possible actual outcomes weighted by their probabilities,
    applying the full scoring rubric for score, GD, total, and winner components.

    Args:
        predicted_a: Predicted goals for team A.
        predicted_b: Predicted goals for team B.
        matrix: Score-probability matrix.
        round_multiplier: Stage multiplier from scoring.ROUND_MULTIPLIERS.

    Returns:
        Float expected points value.
    """
    # TODO
    pass


def build_ev_surface(
    matrix: np.ndarray,
    round_multiplier: float,
) -> np.ndarray:
    """Compute the EV for every possible predicted scoreline, returning a matrix of the same shape.

    Args:
        matrix: Score-probability matrix.
        round_multiplier: Stage multiplier.

    Returns:
        2-D numpy array where [i, j] = EV of predicting A scores i, B scores j.
    """
    # TODO
    pass
