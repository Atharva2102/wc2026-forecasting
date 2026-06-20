"""Poisson goal model: derive lambdas from Elo ratings and score distributions."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import config as cfg
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    import config as cfg


MEXICO_CITIES: set[str] = {"Mexico City", "Guadalajara", "Monterrey"}
CANADA_CITIES: set[str] = {"Toronto", "Vancouver"}


def venue_country_from_venue(venue: str | None) -> str:
    """Map a WC 2026 venue string to its host country from the city name."""
    if venue is None:
        print("Unmapped venue: None; defaulting to USA")
        return "USA"

    parts = str(venue).rsplit(",", maxsplit=1)
    if len(parts) != 2:
        print(f"Unmapped venue: {venue}; defaulting to USA")
        return "USA"

    city = parts[1].strip()
    if city in MEXICO_CITIES:
        return "Mexico"
    if city in CANADA_CITIES:
        return "Canada"
    return "USA"


def _team_country(team: str | None) -> str | None:
    if team is None:
        return None

    cleaned = str(team).strip()
    if cleaned in {"USA", "United States"}:
        return "USA"
    if cleaned == "Mexico":
        return "Mexico"
    if cleaned == "Canada":
        return "Canada"
    return None


def _host_bump(team: str | None, venue_country: str | None) -> float:
    team_country = _team_country(team)
    if team_country is None or venue_country != team_country:
        return cfg.HOST_ADV_AWAY
    if team_country == "Mexico":
        return cfg.HOST_ADV_MEXICO_HOME
    return cfg.HOST_ADV_US_CAN_HOME


def team_lambdas(
    rating_a: float,
    rating_b: float,
    team_a: str | None = None,
    team_b: str | None = None,
    neutral: bool = True,
    venue_country: str | None = None,
) -> tuple[float, float]:
    """Convert Elo ratings into expected goals for both teams.

    Host advantage is applied only when a host plays in its own country. The
    nominal home/away label and neutral flag do not create a host bump.
    """
    del neutral

    rating_gap = rating_a - rating_b
    lambda_a = cfg.BASE_GOALS * math.exp(cfg.SPREAD * rating_gap)
    lambda_b = cfg.BASE_GOALS * math.exp(-cfg.SPREAD * rating_gap)

    lambda_a *= math.exp(_host_bump(team_a, venue_country))
    lambda_b *= math.exp(_host_bump(team_b, venue_country))

    return (
        float(np.clip(lambda_a, 0.15, 5.0)),
        float(np.clip(lambda_b, 0.15, 5.0)),
    )


def _poisson_pmf(lam: float, max_goals: int) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    probs = np.empty(max_goals + 1, dtype=float)
    probs[0] = math.exp(-lam)
    for goal in goals[1:]:
        probs[goal] = probs[goal - 1] * lam / goal
    return probs


def score_matrix(
    lambda_a: float,
    lambda_b: float,
) -> np.ndarray:
    """Build P(A=i, B=j) from independent Poisson goal processes.

    This independence assumption is a simplification; Dixon-Coles correlation is
    a future refinement for low-score dependence.
    """
    probs_a = _poisson_pmf(lambda_a, cfg.MAX_GOALS)
    probs_b = _poisson_pmf(lambda_b, cfg.MAX_GOALS)
    matrix = np.outer(probs_a, probs_b)
    return matrix / matrix.sum()


def outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return (p_home_win, p_draw, p_away_win) from a score matrix."""
    p_home = float(np.tril(matrix, k=-1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, k=1).sum())
    return p_home, p_draw, p_away


def expected_goals_from_matrix(matrix: np.ndarray) -> tuple[float, float]:
    """Recover expected goals for each team by marginalising the score matrix."""
    goals = np.arange(matrix.shape[0])
    exp_goals_a = float((matrix.sum(axis=1) * goals).sum())
    exp_goals_b = float((matrix.sum(axis=0) * goals).sum())
    return exp_goals_a, exp_goals_b
