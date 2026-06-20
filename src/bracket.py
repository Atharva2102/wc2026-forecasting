"""Group simulation, best-third assignment, and Monte Carlo bracket logic."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import config as cfg
    from src.data_loader import canonical_team
    from src.poisson_model import (
        outcome_probs,
        score_matrix,
        team_lambdas,
        venue_country_from_venue,
    )
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    import config as cfg
    from src.data_loader import canonical_team
    from src.poisson_model import (
        outcome_probs,
        score_matrix,
        team_lambdas,
        venue_country_from_venue,
    )


Stats = dict[str, dict[str, int]]
SlotKey = tuple[int, str]


def _rating(team: str, ratings: dict[str, float]) -> float:
    return ratings.get(team, getattr(cfg, "DEFAULT_RATING", 1500.0))


def _match_matrix(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    venue_country: str | None,
    cache: dict[tuple[str, str, str | None], np.ndarray] | None = None,
) -> np.ndarray:
    key = (team_a, team_b, venue_country)
    if cache is not None and key in cache:
        return cache[key]

    lambda_a, lambda_b = team_lambdas(
        _rating(team_a, ratings),
        _rating(team_b, ratings),
        team_a=team_a,
        team_b=team_b,
        venue_country=venue_country,
    )
    matrix = score_matrix(lambda_a, lambda_b)
    if cache is not None:
        cache[key] = matrix
    return matrix


def _draw_score(matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flat_index = int(rng.choice(matrix.size, p=matrix.ravel()))
    return np.unravel_index(flat_index, matrix.shape)


def _empty_stats(team: str) -> dict[str, int]:
    return {"team": team, "points": 0, "gd": 0, "gf": 0}


def simulate_group(
    group_fixtures_for_group: pd.DataFrame,
    ratings: dict[str, float],
    rng: np.random.Generator | None = None,
    cache: dict[tuple[str, str, str | None], np.ndarray] | None = None,
) -> Stats:
    """Simulate one group once and return points/GD/GF stats for each team."""
    if rng is None:
        rng = np.random.default_rng()

    stats: Stats = {}
    for fixture in group_fixtures_for_group.itertuples(index=False):
        home_team = canonical_team(fixture.home_team)
        away_team = canonical_team(fixture.away_team)
        venue_country = venue_country_from_venue(fixture.venue)
        stats.setdefault(home_team, _empty_stats(home_team))
        stats.setdefault(away_team, _empty_stats(away_team))

        matrix = _match_matrix(home_team, away_team, ratings, venue_country, cache)
        home_goals, away_goals = _draw_score(matrix, rng)

        stats[home_team]["gf"] += int(home_goals)
        stats[away_team]["gf"] += int(away_goals)
        stats[home_team]["gd"] += int(home_goals - away_goals)
        stats[away_team]["gd"] += int(away_goals - home_goals)

        if home_goals > away_goals:
            stats[home_team]["points"] += 3
        elif away_goals > home_goals:
            stats[away_team]["points"] += 3
        else:
            stats[home_team]["points"] += 1
            stats[away_team]["points"] += 1

    return stats


def group_table(group_points_gd_gf: Stats) -> list[str]:
    """Rank a four-team group by points, goal difference, then goals for."""
    ranked = sorted(
        group_points_gd_gf.values(),
        key=lambda row: (-row["points"], -row["gd"], -row["gf"], row["team"]),
    )
    return [row["team"] for row in ranked]


def rank_third_place(all_third_teams_with_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the eight best third-placed teams by points, GD, then GF."""
    ranked = sorted(
        all_third_teams_with_stats,
        key=lambda row: (-row["points"], -row["gd"], -row["gf"], row["team"]),
    )
    return ranked[:8]


def _eligible_groups(slot_text: str) -> set[str]:
    match = re.search(r"Best 3rd \(Groups ([A-L](?:/[A-L])*)\)", slot_text)
    if not match:
        return set()
    return set(match.group(1).split("/"))


def assign_best_thirds(
    best8_with_groups: list[dict[str, Any]],
    knockout_slots: pd.DataFrame,
) -> dict[SlotKey, str]:
    """Assign best third-placed teams to eligible R32 slots via backtracking."""
    slots: list[tuple[SlotKey, set[str]]] = []
    for row in knockout_slots.itertuples(index=False):
        match_id = int(row.match_id)
        for side, slot_text in (("home", row.slot_home), ("away", row.slot_away)):
            eligible = _eligible_groups(str(slot_text))
            if eligible:
                slots.append(((match_id, side), eligible))

    teams = list(best8_with_groups)
    slots = sorted(
        slots,
        key=lambda item: sum(team["group"] in item[1] for team in teams),
    )

    assignment: dict[SlotKey, str] = {}
    used: set[str] = set()

    def backtrack(index: int) -> bool:
        if index == len(slots):
            return True

        slot_key, eligible = slots[index]
        for third_team in teams:
            team = third_team["team"]
            if team in used or third_team["group"] not in eligible:
                continue

            assignment[slot_key] = team
            used.add(team)
            if backtrack(index + 1):
                return True
            used.remove(team)
            del assignment[slot_key]

        return False

    if not backtrack(0):
        raise ValueError("Could not assign best third-placed teams to R32 slots")

    return assignment


def _resolve_slot(
    slot_text: str,
    match_id: int,
    side: str,
    winners: dict[str, str],
    runners: dict[str, str],
    best3_assignment: dict[SlotKey, str],
    match_results: dict[int, dict[str, Any]],
) -> str:
    if match := re.fullmatch(r"Winner Group ([A-L])", slot_text):
        return winners[match.group(1)]
    if match := re.fullmatch(r"Runner-up Group ([A-L])", slot_text):
        return runners[match.group(1)]
    if slot_text.startswith("Best 3rd"):
        return best3_assignment[(match_id, side)]
    if match := re.fullmatch(r"Winner Match (\d+)", slot_text):
        return match_results[int(match.group(1))]["winner"]
    if match := re.fullmatch(r"Loser Match (\d+)", slot_text):
        return match_results[int(match.group(1))]["loser"]

    raise ValueError(f"Unrecognized knockout slot: {slot_text!r}")


def _play_knockout_match(
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
    rng: np.random.Generator,
    venue_country: str | None,
    cache: dict[tuple[str, str, str | None], np.ndarray] | None = None,
) -> dict[str, Any]:
    matrix = _match_matrix(home_team, away_team, ratings, venue_country, cache)
    home_goals, away_goals = _draw_score(matrix, rng)
    penalties = home_goals == away_goals

    if home_goals > away_goals:
        winner, loser = home_team, away_team
    elif away_goals > home_goals:
        winner, loser = away_team, home_team
    else:
        p_home, _p_draw, p_away = outcome_probs(matrix)
        home_weight = p_home / (p_home + p_away) if (p_home + p_away) > 0 else 0.5
        if rng.random() < home_weight:
            winner, loser = home_team, away_team
        else:
            winner, loser = away_team, home_team

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "winner": winner,
        "loser": loser,
        "penalties": penalties,
    }


def resolve_and_play_knockouts(
    winners: dict[str, str],
    runners: dict[str, str],
    best3_assignment: dict[SlotKey, str],
    knockout_slots: pd.DataFrame,
    ratings: dict[str, float],
    rng: np.random.Generator | None = None,
    cache: dict[tuple[str, str, str | None], np.ndarray] | None = None,
) -> dict[int, dict[str, Any]]:
    """Resolve knockout slot labels in match order and play each match once."""
    if rng is None:
        rng = np.random.default_rng()

    match_results: dict[int, dict[str, Any]] = {}
    ordered_slots = knockout_slots.sort_values("match_id")

    for row in ordered_slots.itertuples(index=False):
        match_id = int(row.match_id)
        home_team = _resolve_slot(
            str(row.slot_home),
            match_id,
            "home",
            winners,
            runners,
            best3_assignment,
            match_results,
        )
        away_team = _resolve_slot(
            str(row.slot_away),
            match_id,
            "away",
            winners,
            runners,
            best3_assignment,
            match_results,
        )
        venue_country = venue_country_from_venue(row.venue)
        result = _play_knockout_match(
            home_team,
            away_team,
            ratings,
            rng,
            venue_country,
            cache,
        )
        result["round"] = row.round
        match_results[match_id] = result

    return match_results


def monte_carlo_bracket(
    group_fixtures: pd.DataFrame,
    knockout_slots: pd.DataFrame,
    ratings: dict[str, float],
    n_sims: int = cfg.N_SIMS,
    seed: int = cfg.RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[SlotKey, str]]:
    """Run full-tournament simulations and return advancement probabilities."""
    rng = np.random.default_rng(seed)
    all_teams = sorted(
        {
            canonical_team(team)
            for team in pd.concat(
                [group_fixtures["home_team"], group_fixtures["away_team"]],
                ignore_index=True,
            ).dropna()
        }
    )
    groups = sorted(group_fixtures["group"].unique())
    grouped_fixtures = {
        group: group_fixtures.loc[group_fixtures["group"] == group]
        for group in groups
    }
    knockout_slots = knockout_slots.sort_values("match_id").reset_index(drop=True)

    slot_counts: dict[SlotKey, Counter[str]] = defaultdict(Counter)
    round_counts: dict[str, Counter[str]] = {
        "R32": Counter(),
        "R16": Counter(),
        "QF": Counter(),
        "SF": Counter(),
        "final": Counter(),
        "win": Counter(),
    }
    matrix_cache: dict[tuple[str, str, str | None], np.ndarray] = {}

    for _ in range(n_sims):
        winners: dict[str, str] = {}
        runners: dict[str, str] = {}
        third_stats: list[dict[str, Any]] = []

        for group in groups:
            stats = simulate_group(grouped_fixtures[group], ratings, rng, matrix_cache)
            table = group_table(stats)
            winners[group] = table[0]
            runners[group] = table[1]
            third_team = table[2]
            third_stats.append(
                {
                    "team": third_team,
                    "group": group,
                    "points": stats[third_team]["points"],
                    "gd": stats[third_team]["gd"],
                    "gf": stats[third_team]["gf"],
                }
            )

        best8 = rank_third_place(third_stats)
        best3_assignment = assign_best_thirds(best8, knockout_slots)
        knockout_results = resolve_and_play_knockouts(
            winners,
            runners,
            best3_assignment,
            knockout_slots,
            ratings,
            rng,
            matrix_cache,
        )

        for match_id, result in knockout_results.items():
            home_team = result["home_team"]
            away_team = result["away_team"]
            slot_counts[(match_id, "home")][home_team] += 1
            slot_counts[(match_id, "away")][away_team] += 1

            round_name = result["round"]
            if round_name == "Round of 32":
                round_counts["R32"][home_team] += 1
                round_counts["R32"][away_team] += 1
            elif round_name == "Round of 16":
                round_counts["R16"][home_team] += 1
                round_counts["R16"][away_team] += 1
            elif round_name == "Quarter-final":
                round_counts["QF"][home_team] += 1
                round_counts["QF"][away_team] += 1
            elif round_name == "Semi-final":
                round_counts["SF"][home_team] += 1
                round_counts["SF"][away_team] += 1
            elif round_name == "Final":
                round_counts["final"][home_team] += 1
                round_counts["final"][away_team] += 1
                round_counts["win"][result["winner"]] += 1

    rows = []
    for team in all_teams:
        rows.append(
            {
                "team": team,
                "p_reach_R32": round_counts["R32"][team] / n_sims,
                "p_reach_R16": round_counts["R16"][team] / n_sims,
                "p_reach_QF": round_counts["QF"][team] / n_sims,
                "p_reach_SF": round_counts["SF"][team] / n_sims,
                "p_reach_final": round_counts["final"][team] / n_sims,
                "p_win_tournament": round_counts["win"][team] / n_sims,
            }
        )

    advancement = (
        pd.DataFrame(rows)
        .sort_values("p_win_tournament", ascending=False)
        .reset_index(drop=True)
    )
    modal_slots = {
        slot_key: counts.most_common(1)[0][0]
        for slot_key, counts in slot_counts.items()
        if counts
    }

    return advancement, modal_slots
