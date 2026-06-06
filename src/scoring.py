"""Competition scoring rubric — single source of truth for all point values and round multipliers."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Base point values (before round multiplier)
# ---------------------------------------------------------------------------

SCORE_POINTS: dict[str, int] = {
    "exact_score": 25,       # predicted scoreline matches exactly
    "correct_gd": 10,        # goal difference is right but total goals differ
    "correct_total": 10,     # total goals are right but goal difference differs
}

CORNERS_POINTS: dict[str, int] = {
    "exact": 10,
    "within_2": 5,
}

YELLOWS_POINTS: dict[str, int] = {
    "exact": 10,
    "within_1": 5,
}

REDS_POINTS: dict[str, int] = {
    "exact": 5,
}

# Group stage only — predicting the winning team
WINNING_TEAM_GROUP_POINTS: int = 40

# Knockout stage — matchup prediction (both teams in fixture correct)
MATCHUP_KO_POINTS: dict[str, int] = {
    "both_correct": 20,
    "one_correct": 10,
}

# Knockout stage — match winner prediction
MATCH_WINNER_KO_POINTS: int = 20

# Knockout stage — penalty shoot-out result
PENALTIES_KO_POINTS: int = 5

# ---------------------------------------------------------------------------
# Round multipliers
# ---------------------------------------------------------------------------

ROUND_MULTIPLIERS: dict[str, float] = {
    "group": 1.0,
    "R32": 1.0,       # Round of 32 (WC 2026 has 48 teams → extra round)
    "R16": 2.0,
    "QF": 4.0,
    "SF": 8.0,
    "third_place": 8.0,
    "final": 16.0,
}

# ---------------------------------------------------------------------------
# Stub functions
# ---------------------------------------------------------------------------


def score_match(
    predicted_a: int,
    predicted_b: int,
    actual_a: int,
    actual_b: int,
    predicted_corners: int | None = None,
    actual_corners: int | None = None,
    predicted_yellows: int | None = None,
    actual_yellows: int | None = None,
    predicted_reds: int | None = None,
    actual_reds: int | None = None,
    predicted_home: str | None = None,
    predicted_away: str | None = None,
    actual_home: str | None = None,
    actual_away: str | None = None,
    predicted_penalties: bool | None = None,
    actual_penalties: bool | None = None,
    stage: str = "group",
) -> int:
    """Return the raw (pre-multiplier) points earned for a single match prediction.

    Args:
        predicted_a: Predicted goals for team A.
        predicted_b: Predicted goals for team B.
        actual_a: Actual goals scored by team A.
        actual_b: Actual goals scored by team B.
        predicted_corners: Predicted total corners (optional).
        actual_corners: Actual total corners (optional).
        predicted_yellows: Predicted total yellow cards (optional).
        actual_yellows: Actual total yellow cards (optional).
        predicted_reds: Predicted total red cards (optional).
        actual_reds: Actual total red cards (optional).
        predicted_home: Predicted home team name for KO matchup (optional).
        predicted_away: Predicted away team name for KO matchup (optional).
        actual_home: Actual home team name in the fixture (optional).
        actual_away: Actual away team name in the fixture (optional).
        predicted_penalties: Whether the predictor expects a penalty shootout — KO only (optional).
        actual_penalties: Whether the match actually went to a penalty shootout — KO only (optional).
        stage: Round identifier; used to select winning-team vs KO scoring rules.

    Returns:
        Integer raw points before the round multiplier is applied.
    """
    points = 0

    predicted_diff = predicted_a - predicted_b
    actual_diff = actual_a - actual_b

    # Score tier — mutually exclusive; only the highest tier fires.
    if predicted_a == actual_a and predicted_b == actual_b:
        points += SCORE_POINTS["exact_score"]
    elif predicted_diff == actual_diff:
        points += SCORE_POINTS["correct_gd"]
    elif predicted_a + predicted_b == actual_a + actual_b:
        points += SCORE_POINTS["correct_total"]

    # Match winner — independent of whether the match went to penalties.
    predicted_result = (predicted_diff > 0) - (predicted_diff < 0)
    actual_result = (actual_diff > 0) - (actual_diff < 0)
    if stage == "group":
        if predicted_result == actual_result:
            points += WINNING_TEAM_GROUP_POINTS
    else:
        if predicted_result == actual_result:
            points += MATCH_WINNER_KO_POINTS

    # KO matchup — unordered team comparison (home/away swap still counts as both correct).
    if stage != "group" and None not in (predicted_home, predicted_away, actual_home, actual_away):
        predicted_pair = {predicted_home, predicted_away}
        actual_pair = {actual_home, actual_away}
        n_correct = len(predicted_pair & actual_pair)
        if n_correct == 2:
            points += MATCHUP_KO_POINTS["both_correct"]
        elif n_correct == 1:
            points += MATCHUP_KO_POINTS["one_correct"]

    # KO penalties — independent True/False prediction; scores for both "yes" and "no" outcomes.
    if stage != "group" and predicted_penalties is not None and actual_penalties is not None:
        if predicted_penalties == actual_penalties:
            points += PENALTIES_KO_POINTS

    # Corners.
    if predicted_corners is not None and actual_corners is not None:
        corners_diff = abs(predicted_corners - actual_corners)
        if corners_diff == 0:
            points += CORNERS_POINTS["exact"]
        elif corners_diff <= 2:
            points += CORNERS_POINTS["within_2"]

    # Yellow cards.
    if predicted_yellows is not None and actual_yellows is not None:
        yellows_diff = abs(predicted_yellows - actual_yellows)
        if yellows_diff == 0:
            points += YELLOWS_POINTS["exact"]
        elif yellows_diff <= 1:
            points += YELLOWS_POINTS["within_1"]

    # Red cards.
    if predicted_reds is not None and actual_reds is not None:
        if predicted_reds == actual_reds:
            points += REDS_POINTS["exact"]

    return points


def total_points(
    raw_points: int,
    stage: str,
) -> float:
    """Apply the round multiplier to raw points.

    Args:
        raw_points: Output of score_match().
        stage: Round identifier matching a key in ROUND_MULTIPLIERS.

    Returns:
        Float total points for this prediction.
    """
    return raw_points * ROUND_MULTIPLIERS[stage]
