"""Market-anchored rating adjustment for live WC 2026 predictions.

This is a documented domain adjustment for the final prediction run only. It is
not used in the pure-Elo backtest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import config as cfg
    from src.data_loader import canonical_team, resolve_team
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    import config as cfg
    from src.data_loader import canonical_team, resolve_team


FLOOR_PROB: float = 0.004


def _wc_fixture_teams() -> set[str]:
    group_path = cfg.DATA_RAW / "group_fixtures.csv"
    if not group_path.exists():
        return set()

    fixtures = pd.read_csv(group_path)
    teams = pd.concat(
        [fixtures["home_team"], fixtures["away_team"]],
        ignore_index=True,
    ).dropna()
    return {resolve_team(str(team).strip()) for team in teams}


def load_market_probs(path: str | Path) -> dict[str, float]:
    """Load fixture-style title probabilities and return canonical team probs.

    The provided market probabilities are treated as already margin-free. Unlisted
    WC teams receive FLOOR_PROB, then the full 48-team set is renormalized to sum
    to 1.0. Returned keys use the historical-dataset spelling so they align with
    Elo ratings.
    """
    market = pd.read_csv(path)
    if set(market.columns) != {"team", "win_prob"}:
        raise ValueError("market_probs.csv must have columns: team, win_prob")

    fixture_probs = {
        str(row.team).strip(): float(row.win_prob)
        for row in market.itertuples(index=False)
    }
    for team in _wc_fixture_teams():
        fixture_probs.setdefault(team, FLOOR_PROB)

    canonical_probs: dict[str, float] = {}
    for team, prob in fixture_probs.items():
        canonical = canonical_team(team)
        canonical_probs[canonical] = canonical_probs.get(canonical, 0.0) + prob

    total = sum(canonical_probs.values())
    if total <= 0:
        raise ValueError("Market probabilities must sum to a positive value")

    return {team: prob / total for team, prob in canonical_probs.items()}


def market_to_rating_scale(
    probs: dict[str, float],
    elo_ratings: dict[str, float],
) -> dict[str, float]:
    """Map market title probabilities monotonically onto the Elo rating scale.

    rating = mean(Elo) + SCALE * (log(p) - mean(log(p))). SCALE is chosen so the
    standard deviation of market ratings is approximately the standard deviation
    of the current Elo ratings over the same teams.
    """
    teams = [team for team in probs if team in elo_ratings]
    if not teams:
        raise ValueError("No overlap between market probabilities and Elo ratings")

    elo_values = np.array([elo_ratings[team] for team in teams], dtype=float)
    log_probs = np.log(np.array([probs[team] for team in teams], dtype=float))
    log_centered = log_probs - log_probs.mean()
    log_std = log_centered.std(ddof=0)
    elo_std = elo_values.std(ddof=0)
    scale = 0.0 if log_std == 0 else elo_std / log_std
    mean_elo = elo_values.mean()

    return {
        team: float(mean_elo + scale * (np.log(prob) - log_probs.mean()))
        for team, prob in probs.items()
    }


def blend_ratings(
    elo: dict[str, float],
    market: dict[str, float],
    w: float,
) -> dict[str, float]:
    """Blend Elo and market-implied ratings: (1-w)*Elo + w*market."""
    blended = dict(elo)
    for team, elo_rating in elo.items():
        if team in market:
            blended[team] = (1.0 - w) * elo_rating + w * market[team]
    return blended
