"""Top-level pipeline orchestrator: runs end-to-end and writes submission CSVs to outputs/."""

from __future__ import annotations

import pandas as pd


def run_pipeline(config_path: str = "config.py") -> None:
    """Execute the full forecasting pipeline and write all submission files.

    Steps:
        1. Load & clean historical results via data_loader.
        2. Compute Elo ratings via elo.compute_elo_ratings().
        3. Build score matrices for each fixture via poisson_model.
        4. Select EV-optimal predictions via optimizer.
        5. Simulate knockout bracket via bracket.monte_carlo_bracket().
        6. Write submission CSVs to outputs/.

    Args:
        config_path: Path to config.py (resolved at runtime).
    """
    # TODO
    pass


def predict_group_stage(
    fixtures: pd.DataFrame,
    elo_ratings: dict[str, float],
) -> pd.DataFrame:
    """Generate predictions for all group-stage matches.

    Args:
        fixtures: Group-stage fixture list from data_loader.load_fixtures().
        elo_ratings: Ratings dict from elo.compute_elo_ratings().

    Returns:
        DataFrame with columns: match_id, home_team, away_team, pred_home_goals,
        pred_away_goals, pred_corners, pred_yellows, pred_reds, winning_team.
    """
    # TODO
    pass


def predict_knockout_stage(
    ko_fixtures: pd.DataFrame,
    elo_ratings: dict[str, float],
    mc_results: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions for knockout-stage matches.

    Uses Monte Carlo bracket probabilities to fill predicted matchups and then
    applies the Poisson model within each likely matchup.

    Args:
        ko_fixtures: Knockout fixture skeleton (matchup slots, not yet filled).
        elo_ratings: Ratings dict.
        mc_results: Output of bracket.monte_carlo_bracket().

    Returns:
        DataFrame with the same schema as predict_group_stage() plus matchup columns.
    """
    # TODO
    pass


def write_submission(predictions: pd.DataFrame, output_path: str) -> None:
    """Serialise the predictions DataFrame to a CSV file.

    Args:
        predictions: Combined group + KO predictions.
        output_path: Destination file path under outputs/.
    """
    # TODO
    pass
