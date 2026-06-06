"""Project-wide configuration. All values are tunable placeholders — calibrate against backtest metrics."""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

# URL for the ~49K international results dataset.
RESULTS_URL: str = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Local filename after download
RESULTS_FILENAME: str = "results.csv"
FIXTURES_FILENAME: str = "wc2026_fixtures.csv"

# Earliest year to include in Elo training data.
ELO_START_YEAR: int = 2000
MIN_YEAR: int = ELO_START_YEAR

# ---------------------------------------------------------------------------
# Elo parameters
# ---------------------------------------------------------------------------

# Default starting rating for any team with no prior history.
ELO_INITIAL_RATING: float = 1500.0  # TODO: tune

# Base K-factor — controls how fast ratings move after each result.
# Higher K = more reactive; lower K = more stable.
K_BASE: float = 20.0  # TODO: tune

# Elo points added to the home team's effective rating at non-neutral venues.
HOME_ADVANTAGE: float = 100.0  # TODO: tune

# Per-tournament importance multiplier applied to K_BASE.
# Friendlies matter less; World Cup matches matter more.
TOURNAMENT_WEIGHTS: dict[str, float] = {
    "FIFA World Cup": 4.0,
    "UEFA Euro": 3.0,
    "Copa America": 3.0,
    "Africa Cup of Nations": 2.5,
    "AFC Asian Cup": 2.5,
    "CONCACAF Gold Cup": 2.0,
    "Nations League": 1.5,
    "Friendly": 0.5,
}  # TODO: tune multipliers

# Annual decay factor applied to match K so older games carry less weight.
# 1.0 = no decay; 0.95 = 5% reduction per year.
RECENCY_DECAY: float = 0.98  # TODO: tune

# ---------------------------------------------------------------------------
# Poisson model parameters
# ---------------------------------------------------------------------------

# Average goals per team per match used as the baseline lambda.
BASE_GOALS: float = 1.3  # TODO: tune (typically 1.1–1.5 for international football)

# Scaling factor that maps Elo rating difference to lambda offset.
# Higher spread = larger favourites get a bigger lambda boost.
SPREAD: float = 400.0  # TODO: tune

# ---------------------------------------------------------------------------
# Host-nation advantage
# ---------------------------------------------------------------------------

# Teams hosting WC 2026 — they receive a lambda boost in Poisson model.
HOSTS: set[str] = {"USA", "Mexico", "Canada"}

# Additional expected goals awarded to a host nation per match.
HOST_ADV: float = 0.2  # TODO: tune

# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

# Number of full-tournament simulations. Higher = tighter confidence intervals.
# 10 000 is fast; 100 000 is preferred for final submission.
N_SIMS: int = 10_000  # TODO: increase to 100_000 for final run

# Random seed for reproducibility.
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Submission output
# ---------------------------------------------------------------------------

SUBMISSION_FILENAME: str = "submission.csv"
