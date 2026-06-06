"""Data ingestion and cleaning for international results and WC 2026 fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from config import DATA_RAW, ELO_START_YEAR, RESULTS_URL
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    from config import DATA_RAW, ELO_START_YEAR, RESULTS_URL


RESULT_COLUMNS: list[str] = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]

GROUP_FIXTURE_COLUMNS: list[str] = [
    "match_id",
    "group",
    "home_team",
    "away_team",
    "date_utc",
    "venue",
]

KNOCKOUT_SLOT_COLUMNS: list[str] = [
    "match_id",
    "round",
    "multiplier",
    "date_utc",
    "venue",
    "slot_home",
    "slot_away",
]

PLAYOFF_MAP: dict[str, str] = {
    "UEFA Playoff A": "Bosnia and Herzegovina",
    "UEFA Playoff B": "Sweden",
    "UEFA Playoff C": "Türkiye",
    "UEFA Playoff D": "Czechia",
    "FIFA Playoff 1": "DR Congo",
    "FIFA Playoff 2": "Iraq",
    "UEFA A": "Bosnia and Herzegovina",
    "UEFA B": "Sweden",
    "UEFA C": "Türkiye",
    "UEFA D": "Czechia",
    "FIFA 1": "DR Congo",
    "FIFA 2": "Iraq",
}

NAME_MAP: dict[str, str] = {
    "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Ivory Coast",
    "CÃ´te d'Ivoire": "Ivory Coast",
    "USA": "United States",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Congo DR": "DR Congo",
}


def resolve_team(name: str) -> str:
    """Resolve WC 2026 playoff placeholders to their assumed winning teams."""
    cleaned = str(name).strip()
    if cleaned in PLAYOFF_MAP:
        return PLAYOFF_MAP[cleaned]

    normalized = cleaned.casefold()
    for fixture_name, resolved_name in PLAYOFF_MAP.items():
        if fixture_name.casefold() == normalized:
            return resolved_name

    return cleaned


def canonical_team(name: str) -> str:
    """Resolve playoff placeholders, then map to the dataset's exact spelling."""
    resolved = resolve_team(str(name).strip())
    return NAME_MAP.get(resolved, resolved)


def fetch_international_results(force_refresh: bool = False) -> pd.DataFrame:
    """Load the raw international results dataset, downloading and caching if needed.

    Args:
        force_refresh: Download from RESULTS_URL even when the cache exists.

    Returns:
        Raw DataFrame with the source columns needed by the pipeline.
    """
    cache_path = DATA_RAW / "results.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path)
    else:
        df = pd.read_csv(RESULTS_URL)
        df.to_csv(cache_path, index=False)

    missing = [col for col in RESULT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Results data is missing required columns: {missing}")

    return df.loc[:, RESULT_COLUMNS].copy()


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Clean historical results for downstream Elo work without tournament filtering.

    Parses dates, drops unplayed fixtures, keeps matches from ELO_START_YEAR onward,
    and preserves the neutral flag for later home-advantage handling.
    """
    missing = [col for col in RESULT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Results data is missing required columns: {missing}")

    clean = df.loc[:, RESULT_COLUMNS].copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["home_score"] = pd.to_numeric(clean["home_score"], errors="coerce")
    clean["away_score"] = pd.to_numeric(clean["away_score"], errors="coerce")

    clean = clean.dropna(subset=["date", "home_score", "away_score"])
    clean = clean.loc[clean["date"].dt.year >= ELO_START_YEAR].copy()
    clean["home_score"] = clean["home_score"].astype(int)
    clean["away_score"] = clean["away_score"].astype(int)

    if clean["neutral"].dtype != bool:
        clean["neutral"] = clean["neutral"].map(
            lambda value: str(value).strip().lower() == "true"
        )

    return clean.sort_values("date").reset_index(drop=True)


def load_fixtures(
    group_path: str | Path,
    knockout_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load group fixtures and knockout slot definitions from CSV files."""
    group_df = pd.read_csv(group_path)
    knockout_df = pd.read_csv(knockout_path)

    missing_group = [col for col in GROUP_FIXTURE_COLUMNS if col not in group_df.columns]
    if missing_group:
        raise ValueError(f"Group fixtures are missing required columns: {missing_group}")

    missing_knockout = [
        col for col in KNOCKOUT_SLOT_COLUMNS if col not in knockout_df.columns
    ]
    if missing_knockout:
        raise ValueError(
            f"Knockout slots are missing required columns: {missing_knockout}"
        )

    group_df = group_df.loc[:, GROUP_FIXTURE_COLUMNS].copy()
    knockout_df = knockout_df.loc[:, KNOCKOUT_SLOT_COLUMNS].copy()

    group_df["date_utc"] = pd.to_datetime(group_df["date_utc"], errors="coerce")
    knockout_df["date_utc"] = pd.to_datetime(knockout_df["date_utc"], errors="coerce")

    return group_df, knockout_df


def filter_by_tournament(df: pd.DataFrame, tournaments: list[str]) -> pd.DataFrame:
    """Return only rows whose tournament column matches the given list."""
    return df.loc[df["tournament"].isin(tournaments)].copy()


def verify_fixture_team_names(
    group_path: str | Path = DATA_RAW / "group_fixtures.csv",
    knockout_path: str | Path = DATA_RAW / "knockout_slots.csv",
) -> list[str]:
    """Assert every canonicalized WC 2026 group fixture team exists in results."""
    results = clean_results(fetch_international_results(force_refresh=False))
    group_fixtures, _ = load_fixtures(group_path, knockout_path)

    fixture_teams = pd.concat(
        [group_fixtures["home_team"], group_fixtures["away_team"]],
        ignore_index=True,
    ).dropna()
    canonical_fixture_teams = sorted({canonical_team(team) for team in fixture_teams})
    result_teams = set(results["home_team"]) | set(results["away_team"])
    missing = [team for team in canonical_fixture_teams if team not in result_teams]

    print(f"Canonical WC fixture teams: {len(canonical_fixture_teams)}")
    print(f"Missing from historical results: {len(missing)}")
    if missing:
        print("Missing teams:")
        for team in missing:
            print(f"- {team}")

    assert not missing, "Some canonical fixture teams are missing from results.csv"
    return missing


if __name__ == "__main__":
    verify_fixture_team_names()
