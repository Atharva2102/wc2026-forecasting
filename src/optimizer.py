"""Expected-value optimisation against the competition scoring rubric."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from src.poisson_model import outcome_probs
    from src.scoring import MATCH_WINNER_KO_POINTS, score_match
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.poisson_model import outcome_probs
    from src.scoring import MATCH_WINNER_KO_POINTS, score_match


def _candidate_allowed(home_goals: int, away_goals: int, mode: str) -> bool:
    if mode == "any":
        return True
    if mode == "home":
        return home_goals > away_goals
    if mode == "away":
        return home_goals < away_goals
    if mode == "draw":
        return home_goals == away_goals
    raise ValueError("mode must be one of: any, home, away, draw")


def _scoreline_points(
    predicted_a: int,
    predicted_b: int,
    actual_a: int,
    actual_b: int,
) -> int:
    points = score_match(
        predicted_a,
        predicted_b,
        actual_a,
        actual_b,
        predicted_home="pred_home",
        predicted_away="pred_away",
        actual_home="actual_home",
        actual_away="actual_away",
        stage="R32",
    )
    predicted_result = (predicted_a > predicted_b) - (predicted_a < predicted_b)
    actual_result = (actual_a > actual_b) - (actual_a < actual_b)
    if predicted_result == actual_result:
        points -= MATCH_WINNER_KO_POINTS
    return points


def expected_points_for_scoreline(
    predicted_a: int,
    predicted_b: int,
    matrix: np.ndarray,
) -> float:
    """Expected scoreline-tier points for one predicted scoreline."""
    ev = 0.0
    for actual_a in range(matrix.shape[0]):
        for actual_b in range(matrix.shape[1]):
            ev += matrix[actual_a, actual_b] * _scoreline_points(
                predicted_a,
                predicted_b,
                actual_a,
                actual_b,
            )
    return float(ev)


def best_score(
    matrix: np.ndarray,
    mode: str = "any",
) -> tuple[int, int]:
    """Return the scoreline prediction maximizing expected score-tier points.

    `mode` constrains the predicted result: "any", "home", "away", or "draw".
    """
    best_prediction: tuple[int, int] | None = None
    best_ev = -math.inf

    for predicted_a in range(matrix.shape[0]):
        for predicted_b in range(matrix.shape[1]):
            if not _candidate_allowed(predicted_a, predicted_b, mode):
                continue

            ev = expected_points_for_scoreline(predicted_a, predicted_b, matrix)
            if ev > best_ev:
                best_ev = ev
                best_prediction = (predicted_a, predicted_b)

    if best_prediction is None:
        raise ValueError(f"No candidate scorelines available for mode={mode!r}")

    return best_prediction


def _poisson_probs(lam: float, max_count: int) -> np.ndarray:
    probs = np.empty(max_count + 1, dtype=float)
    probs[0] = math.exp(-lam)
    for count in range(1, max_count + 1):
        probs[count] = probs[count - 1] * lam / count
    return probs / probs.sum()


def ev_optimal_count(
    lam: float,
    rule: Callable[[int, int], int],
) -> int:
    """Return the integer prediction maximizing expected points under `rule`."""
    max_count = max(20, int(math.ceil(lam + 8.0 * math.sqrt(max(lam, 1.0)))))
    probs = _poisson_probs(lam, max_count)

    best_prediction = 0
    best_ev = -math.inf
    for predicted in range(max_count + 1):
        ev = sum(
            prob * rule(predicted, actual)
            for actual, prob in enumerate(probs)
        )
        if ev > best_ev:
            best_ev = ev
            best_prediction = predicted

    return best_prediction


def most_likely_outcome(matrix: np.ndarray) -> str:
    """Return 'home', 'draw', or 'away' from the largest outcome probability."""
    outcomes = ("home", "draw", "away")
    probs = outcome_probs(matrix)
    return outcomes[int(np.argmax(probs))]


def build_ev_surface(matrix: np.ndarray) -> np.ndarray:
    """Return EV for every possible predicted scoreline."""
    surface = np.empty_like(matrix, dtype=float)
    for predicted_a in range(matrix.shape[0]):
        for predicted_b in range(matrix.shape[1]):
            surface[predicted_a, predicted_b] = expected_points_for_scoreline(
                predicted_a,
                predicted_b,
                matrix,
            )
    return surface
