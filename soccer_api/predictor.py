"""Cached prediction service for XGBoost and Elo-Poisson match forecasts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

import config as cfg
from src.data_loader import (
    NAME_MAP,
    canonical_team,
    clean_results,
    fetch_international_results,
    load_fixtures,
    resolve_team,
)
from src.elo import compute_elo_ratings
from src.features import CurrentFeatureState, build_current_state, current_feature_row
from src.optimizer import best_score
from src.poisson_model import outcome_probs, score_matrix, team_lambdas


HOME_MODEL_PATH = cfg.DATA_PROCESSED / "xgb_home_goals.json"
AWAY_MODEL_PATH = cfg.DATA_PROCESSED / "xgb_away_goals.json"
FEATURE_METADATA_PATH = cfg.DATA_PROCESSED / "xgb_feature_metadata.json"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ANALYSIS_SYSTEM_PROMPT = """You translate soccer prediction numbers into plain-language analysis.
Use ONLY the numbers provided in the user message.
Write 2-4 concise sentences for a non-technical fan.
Do not mention players, injuries, recent matches, tactical narratives, news, or any fact not present in the numbers.
Cover who is favored and how strongly, likely scoreline, whether the matchup is tight, and whether the two models notably disagree."""

DISPLAY_NAME_OVERRIDES = {
    "Cape Verde": "Cabo Verde",
    "Ivory Coast": "Côte d'Ivoire",
    "United States": "USA",
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
    "Congo DR": "DR Congo",
    "DR Congo": "DR Congo",
    "TÃ¼rkiye": "Türkiye",
    "CÃ´te d'Ivoire": "Côte d'Ivoire",
    "CÃƒÂ´te d'Ivoire": "Côte d'Ivoire",
}


@dataclass
class PredictionState:
    elo_ratings: dict[str, float]
    elo_history: pd.DataFrame
    feature_state: CurrentFeatureState
    home_model: XGBRegressor
    away_model: XGBRegressor
    feature_columns: list[str]
    teams: list[str]


STATE: PredictionState | None = None


def _display_name(team: str) -> str:
    canonical = canonical_team(team)
    inverse_map = {value: key for key, value in NAME_MAP.items()}
    display = inverse_map.get(canonical, canonical)
    return DISPLAY_NAME_OVERRIDES.get(display, DISPLAY_NAME_OVERRIDES.get(canonical, display))


def _load_teams() -> list[str]:
    group_fixtures, _knockout_slots = load_fixtures(
        cfg.DATA_RAW / "group_fixtures.csv",
        cfg.DATA_RAW / "knockout_slots.csv",
    )
    fixture_teams = pd.concat(
        [group_fixtures["home_team"], group_fixtures["away_team"]],
        ignore_index=True,
    ).dropna()
    return sorted({_display_name(resolve_team(team)) for team in fixture_teams})


def _load_models() -> tuple[XGBRegressor, XGBRegressor, list[str]]:
    metadata = json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    feature_columns = list(metadata["feature_columns"])

    home_model = XGBRegressor()
    away_model = XGBRegressor()
    home_model.load_model(HOME_MODEL_PATH)
    away_model.load_model(AWAY_MODEL_PATH)
    return home_model, away_model, feature_columns


def initialize() -> PredictionState:
    """Build and cache all expensive state once at API startup."""
    global STATE
    if STATE is not None:
        return STATE

    if load_dotenv is not None:
        load_dotenv()

    results = clean_results(fetch_international_results(force_refresh=False))
    elo_ratings, elo_history = compute_elo_ratings(results)
    feature_state = build_current_state(results)
    home_model, away_model, feature_columns = _load_models()
    teams = _load_teams()

    STATE = PredictionState(
        elo_ratings=elo_ratings,
        elo_history=elo_history,
        feature_state=feature_state,
        home_model=home_model,
        away_model=away_model,
        feature_columns=feature_columns,
        teams=teams,
    )
    return STATE


def teams() -> list[str]:
    return initialize().teams


def _feature_frame(row: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame([row])
    if "tournament_tier" in frame.columns:
        frame = pd.get_dummies(frame, columns=["tournament_tier"], dtype=float)

    bool_cols = frame.select_dtypes(include=["bool"]).columns
    frame.loc[:, bool_cols] = frame.loc[:, bool_cols].astype(float)
    frame = frame.reindex(columns=feature_columns, fill_value=0.0)

    assert list(frame.columns) == feature_columns, (
        "Feature column order mismatch between live row and training metadata"
    )
    return frame


def _top_scorelines(matrix: np.ndarray, n: int = 5) -> list[dict[str, float | str]]:
    flat = matrix.ravel()
    order = np.argsort(flat)[::-1][:n]
    rows: list[dict[str, float | str]] = []
    width = matrix.shape[1]
    for idx in order:
        home_goals = int(idx // width)
        away_goals = int(idx % width)
        rows.append(
            {
                "score": f"{home_goals}-{away_goals}",
                "prob": float(flat[idx]),
            }
        )
    return rows


def _poisson_summary(matrix: np.ndarray, exp_home: float, exp_away: float) -> dict[str, Any]:
    p_home, p_draw, p_away = outcome_probs(matrix)
    ev_home, ev_away = best_score(matrix, mode="any")
    top_scorelines = _top_scorelines(matrix)
    return {
        "p_home": float(p_home),
        "p_draw": float(p_draw),
        "p_away": float(p_away),
        "exp_home_goals": float(exp_home),
        "exp_away_goals": float(exp_away),
        "top_scorelines": top_scorelines,
        "most_likely_score": top_scorelines[0]["score"],
        "ev_score": f"{ev_home}-{ev_away}",
    }


def predict(home: str, away: str, neutral: bool) -> dict[str, Any]:
    """Return XGBoost and Elo-Poisson predictions for one hypothetical match."""
    state = initialize()
    home_team = canonical_team(home)
    away_team = canonical_team(away)

    feature_row = current_feature_row(
        state.feature_state,
        home_team,
        away_team,
        bool(neutral),
        tournament="FIFA World Cup",
    )
    X = _feature_frame(feature_row, state.feature_columns)

    home_xg = float(np.clip(state.home_model.predict(X)[0], 0.05, 5.0))
    away_xg = float(np.clip(state.away_model.predict(X)[0], 0.05, 5.0))
    xgb_matrix = score_matrix(home_xg, away_xg)

    home_elo = float(state.elo_ratings.get(home_team, cfg.DEFAULT_RATING))
    away_elo = float(state.elo_ratings.get(away_team, cfg.DEFAULT_RATING))
    elo_home_xg, elo_away_xg = team_lambdas(
        home_elo,
        away_elo,
        team_a=home_team,
        team_b=away_team,
        neutral=bool(neutral),
        venue_country=None,
    )
    elo_matrix = score_matrix(elo_home_xg, elo_away_xg)

    response = {
        "home": _display_name(home_team),
        "away": _display_name(away_team),
        "neutral": bool(neutral),
        "xgboost": _poisson_summary(xgb_matrix, home_xg, away_xg),
        "elo_poisson": {
            key: value
            for key, value in _poisson_summary(
                elo_matrix,
                elo_home_xg,
                elo_away_xg,
            ).items()
            if key
            in {
                "p_home",
                "p_draw",
                "p_away",
                "exp_home_goals",
                "exp_away_goals",
            }
        },
        "context": {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "home_last5_ppg": _json_float(feature_row.get("home_last5_ppg")),
            "away_last5_ppg": _json_float(feature_row.get("away_last5_ppg")),
            "h2h_count": int(feature_row.get("h2h_count", 0)),
        },
    }
    response["analysis"] = _analysis(response)
    return response


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def _analysis(prediction: dict[str, Any]) -> str:
    payload = {
        "home": prediction["home"],
        "away": prediction["away"],
        "neutral": prediction["neutral"],
        "xgboost": prediction["xgboost"],
        "elo_poisson": prediction["elo_poisson"],
        "context": prediction["context"],
    }
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=180,
            temperature=0.2,
            system=ANALYSIS_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, indent=2),
                }
            ],
        )
        text_blocks = [
            block.text.strip()
            for block in message.content
            if getattr(block, "type", None) == "text" and block.text.strip()
        ]
        if not text_blocks:
            raise RuntimeError("Anthropic returned no text content")
        return " ".join(text_blocks)
    except Exception:
        return _fallback_analysis(payload)


def _fallback_analysis(payload: dict[str, Any]) -> str:
    home = payload["home"]
    away = payload["away"]
    xgb = payload["xgboost"]
    elo = payload["elo_poisson"]
    labels = [home, "draw", away]
    xgb_probs = [xgb["p_home"], xgb["p_draw"], xgb["p_away"]]
    elo_probs = [elo["p_home"], elo["p_draw"], elo["p_away"]]
    xgb_pick = labels[int(np.argmax(xgb_probs))]
    elo_pick = labels[int(np.argmax(elo_probs))]
    confidence = max(xgb_probs)
    top_score = xgb.get("most_likely_score", "a low-scoring result")

    if xgb_pick == "draw":
        lead = f"The XGBoost model sees this as tight, with the draw the single most likely outcome at {confidence:.0%}."
    else:
        lead = f"The XGBoost model leans toward {xgb_pick}, giving that outcome a {confidence:.0%} chance."

    score = (
        f"Its most likely scoreline is {top_score}, with expected goals of "
        f"{xgb['exp_home_goals']:.2f} for {home} and {xgb['exp_away_goals']:.2f} for {away}."
    )
    if xgb_pick != elo_pick:
        comparison = f"The rating-only baseline points to {elo_pick}, so the two models disagree."
    else:
        comparison = f"The rating-only baseline also points to {elo_pick}, so the models broadly agree."
    return f"{lead} {score} {comparison}"
