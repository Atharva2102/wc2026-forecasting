"""Project-wide configuration. Tune these values against backtest metrics."""

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

RESULTS_URL: str = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

RESULTS_FILENAME: str = "results.csv"
FIXTURES_FILENAME: str = "wc2026_fixtures.csv"
MARKET_PROBS_PATH = DATA_RAW / "market_probs.csv"

ELO_START_YEAR: int = 2000
MIN_YEAR: int = ELO_START_YEAR

# Keep early rows but flag teams with fewer prior matches than this threshold.
MIN_HISTORY: int = 5

# ---------------------------------------------------------------------------
# Elo parameters
# ---------------------------------------------------------------------------

DEFAULT_RATING: float = 1500.0
ELO_INITIAL_RATING: float = DEFAULT_RATING

K_BASE: float = 40.0

# Historical Elo home advantage, used only in Elo expected-score calculations.
ELO_HOME_ADVANTAGE: float = 65.0
HOME_ADVANTAGE: float = ELO_HOME_ADVANTAGE

# Light annual pull toward 1500; 0.25 over-compressed the rating spread.
REGRESSION_STRENGTH: float = 0.03
USE_MARGIN_OF_VICTORY: bool = True

TOURNAMENT_WEIGHTS: dict[str, float] = {
    "world_cup": 1.0,
    "continental_final": 0.85,
    "wc_qual_nations": 0.7,
    "continental_qual": 0.55,
    "friendly": 0.3,
    "minor": 0.15,
}

EXCLUDED_TOURNAMENT_PATTERNS: list[str] = [
    "CONIFA",
    "ConIFA",
    "Viva World Cup",
    "Island Games",
    "Olympic",
    "U-",
    "U23",
    "U21",
    "U20",
    "U17",
    "U19",
    "Youth",
    "Universiade",
]

TOURNAMENT_TIER_PATTERNS: dict[str, list[str]] = {
    "wc_qual_nations": [
        "FIFA World Cup qualification",
        "UEFA Nations League",
        "CONCACAF Nations League",
        "African Nations Championship",
    ],
    "continental_qual": [
        "UEFA Euro qualification",
        "Copa América qualification",
        "Copa America qualification",
        "African Cup of Nations qualification",
        "Africa Cup of Nations qualification",
        "AFC Asian Cup qualification",
        "CONCACAF Gold Cup qualification",
        "OFC Nations Cup qualification",
    ],
    "world_cup": ["FIFA World Cup"],
    "continental_final": [
        "UEFA Euro",
        "Copa América",
        "Copa America",
        "African Cup of Nations",
        "Africa Cup of Nations",
        "AFC Asian Cup",
        "CONCACAF Gold Cup",
        "Gold Cup",
        "OFC Nations Cup",
    ],
    "friendly": ["Friendly"],
}

# Annual decay factor kept for compatibility with earlier placeholders.
RECENCY_DECAY: float = 0.98

# ---------------------------------------------------------------------------
# Poisson model parameters
# ---------------------------------------------------------------------------

BASE_GOALS: float = 1.35
SPREAD: float = 0.0024
MAX_GOALS: int = 8

# ---------------------------------------------------------------------------
# Host-nation advantage
# ---------------------------------------------------------------------------

HOSTS: set[str] = {"USA", "Mexico", "Canada"}
HOST_ADV_MEXICO_HOME: float = 0.22
HOST_ADV_US_CAN_HOME: float = 0.10
HOST_ADV_AWAY: float = 0.0

# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

N_SIMS: int = 10_000
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Submission output
# ---------------------------------------------------------------------------

MARKET_WEIGHT: float = 0.30
SUBMISSION_FILENAME: str = "submission.csv"
