# World Cup 2026 Forecasting Engine — Project Guide

A data-driven forecasting pipeline that predicts international football matches, applied to
the 2026 FIFA World Cup. Originally built for a 3,400-entrant prediction competition (entry
submitted); now being extended as a **portfolio piece** demonstrating end-to-end ML:
feature engineering, model training, leakage-free temporal validation, and an interactive
demo. This document is the single source of truth for the project's vision, design
decisions, and the responsibility of each file. Build against it, one stage at a time.

---

## 0. Build Status

### Phase 1 — Elo + Poisson pipeline (COMPLETE; competition entry submitted)

| Stage | Scope | Status |
|---|---|---|
| 1 | `data_loader.py` + `01_eda.ipynb` — load, clean, team-name reconciliation, EDA | DONE (48 teams, 0 name mismatches) |
| — | `scoring.py` — competition rubric, single source of truth | DONE (3 bugs fixed) |
| 2 | `elo.py` + `02_elo_ratings.ipynb` — data-driven Elo ratings | DONE (std ~145, sane top order) |
| 3 | `poisson_model.py` + `optimizer.py` — ratings to goals to EV-optimal picks | DONE (SPREAD=0.0024) |
| 4 | `bracket.py` — group tables, best-thirds, Monte Carlo + venue-based host advantage | DONE |
| 5 | `predict.py` — orchestrate, write submission CSVs | DONE (bracket-consistency fix) |
| 6 | `03_backtest.ipynb` + `backtest.py` — leakage-free metrics on 4 held-out tournaments | DONE |
| 6b | `market.py` — prediction-market rating blend (live 2026 only) | DONE (w=0.30) |

### Phase 2 — Path 2 upgrade (IN PROGRESS; portfolio depth)

| Stage | Scope | Status |
|---|---|---|
| 7 | `features.py` — chronological, no-leakage feature matrix (Elo + form + H2H + momentum + tournament + venue) | NEXT |
| 8 | `train.py` — train XGBoost goals regressors (Option B); feature importance | TODO |
| 9 | Rewire prediction to use the trained model; re-run backtest vs Elo-Poisson baseline | TODO |
| 10 | `README.md` — recruiter-facing, written with final numbers | TODO |
| 11 | Interactive frontend demo (optional, high portfolio value) | TODO |

---

## 1. Vision

A clean, reproducible, *defensible* forecasting pipeline where every number traces back to
data rather than gut feel, and where each stage can be explained on its own in an interview.
Now that the competition is over, the audience is purely a recruiter/interviewer reading the
GitHub repo. Success is measured by **demonstrated rigor** — honest, leakage-free metrics and
clear reasoning — not by any single accuracy number.

The headline idea the whole repo serves: **form an honest probability distribution over
outcomes from data, then act optimally given that distribution.** Phase 1 did this with Elo
to Poisson. Phase 2 deepens the *belief* layer with a richer trained model while keeping the
same honest-validation discipline.

---

## 2. The Problem & Scoring (competition origin)

The competition required predicting, for 104 fixtures: exact scoreline, total corners, total
yellow/red cards, group-stage winning team, and knockout matchup/winner/penalties. The
scoring rubric (encoded in `src/scoring.py`) and round multipliers drove Phase 1's design.

Scoring rubric: exact score 25 / correct goal-difference 10 / correct total 10; corners
exact 10, within-2 5; yellows exact 10, within-1 5; reds exact 5; group winning-team 40;
knockout matchup 20 (both) / 10 (one); match winner 20; penalties 5. Round multipliers:
group x1, R32 x1, R16 x2, QF x4, SF x8, third-place x8, final x16.

For the **portfolio extension**, the prediction target narrows to what tells the cleanest ML
story and can be validated honestly: **match outcome (W/D/L) and expected goals per team.**
Corners/cards remain a competition-specific footnote (predicted as tournament means because
the historical dataset lacks per-match corner/card data) and are not part of the Phase 2
headline.

---

## 3. The Approach

### Phase 1 (built)
```
results -> Elo ratings -> Poisson goal model -> score distribution
                                                      |
                       +------------------------------+
                       v                              v
            EV-optimizer (vs rubric)        Monte-Carlo bracket sim
                       |                              |
                       +---------> predict.py <-------+
```

### Phase 2 (Path 2 — trained goals model)
```
results -> features.py (Elo + form + H2H + momentum + tournament + venue)
                 |  (chronological, no-leakage feature matrix; labels = actual goals)
                 v
         train.py — XGBoost regressors -> expected home/away goals
                 |
                 v
   expected goals feed the SAME Poisson -> score_matrix -> optimizer -> bracket
                 |
                 v
         backtest vs the Elo-Poisson baseline (moment of truth)
```

**Key architectural decision (Path 2, Option B):** XGBoost predicts each team's *expected
goals* (regression), which become the lambda inputs to the existing Poisson machinery. This
reuses `poisson_model`, `optimizer`, `bracket`, and `scoring` unchanged — only the *source*
of the lambdas changes (trained model instead of the Elo-gap formula). Elo is **retained as
one feature** among many, not discarded.

---

## 4. Key Design Decisions

- **Data-driven, not hand-tuned.** Strength is earned from ~25K international results
  (2000-present). Phase 2 adds more earned signal (form, H2H, momentum, tournament context).
- **No leakage, always.** Features and ratings are computed point-in-time (strictly before
  each match), then state updates *after*. This is the discipline behind the trustworthy
  backtest and it is non-negotiable in `features.py`.
- **Tournament-importance weighting** (Elo K-factor tiers; non-FIFA/youth/Olympic excluded —
  see section 6 `elo.py`).
- **Regression toward the mean** keeps favorites from looking invincible (upset honesty).
- **Host advantage is venue-based, not label-based.** Real home edge applies only when a host
  (USA/Mexico/Canada) actually plays in its own country; it dissolves on the road. Full bump
  for Mexico-at-home (0.22), muted for USA/Canada-at-home (0.10), zero away. This corrected a
  bug where hosts were over-credited throughout the knockouts.
- **Market as a sanity anchor (live only).** Prediction-market (Polymarket) implied
  probabilities are blended into the 2026 ratings at w=0.30 — a documented domain adjustment,
  explicitly NOT a backtest-validated gain (no historical odds, so it cannot be).
- **Decision-theoretic optimization.** Predictions maximize expected points vs the real
  rubric in `scoring.py`. Belief (probabilities) and action (EV-optimal pick) are decoupled.
- **Monte Carlo carries uncertainty.** Simulating the tournament thousands of times lets
  upsets ripple forward and yields advancement *probabilities* rather than a single chalk path.

---

## 5. Directory Structure

```
wc2026-forecasting/
|-- data/
|   |-- raw/            # international results + competition fixture CSVs + market_probs.csv
|   +-- processed/      # built ratings, feature_matrix, intermediate data
|-- src/
|   |-- __init__.py
|   |-- data_loader.py      DONE
|   |-- elo.py              DONE
|   |-- poisson_model.py    DONE
|   |-- optimizer.py        DONE
|   |-- scoring.py          DONE
|   |-- bracket.py          DONE
|   |-- market.py           DONE
|   |-- backtest.py         DONE
|   |-- features.py         TODO  Phase 2 — feature matrix
|   +-- train.py            TODO  Phase 2 — XGBoost goals model
|-- notebooks/
|   |-- 01_eda.ipynb            DONE
|   |-- 02_elo_ratings.ipynb    DONE
|   |-- 03_backtest.ipynb       DONE
|   +-- 04_final_predictions.ipynb DONE
|-- outputs/            # submission CSVs, figures
|-- config.py
|-- requirements.txt
|-- .gitignore
+-- README.md           TODO  written last, with final numbers
```

---

## 6. File-by-File Reference

### Done (Phase 1)

**`data_loader.py`** — `fetch_international_results()` (cache to data/raw), `clean_results()`
(parse dates, drop unplayed NaN-score future fixtures, filter >= `ELO_START_YEAR`=2000,
preserve `neutral`), `load_fixtures()`. **Name reconciliation** is the key deliverable:
`PLAYOFF_MAP`/`resolve_team` (placeholder to real), `NAME_MAP` (fixture to dataset spelling:
Cabo Verde to Cape Verde, Cote d'Ivoire to Ivory Coast, USA to United States), and
`canonical_team()` = resolve then map. **Every stage uses `canonical_team` for lookups.**
Verified 48 teams, 0 missing.

**`scoring.py`** — rubric constants + `score_match()`/`total_points()`. Three fixed bugs:
knockout matchup points (unordered team match), penalties as an independent True/False item,
score tiers as mutually-exclusive if/elif/elif.

**`elo.py`** — `compute_elo_ratings(matches)` (chronological, K scaled by tournament tier,
margin-of-victory, annual regression `REGRESSION_STRENGTH=0.03`, home advantage only when
`neutral==False`); `get_ratings_asof(history, date)` (strict `<` cutoff, defaults unknown
teams to `DEFAULT_RATING` — the no-leakage accessor). Tournament tiers: exclude
CONIFA/Viva/Island Games/youth/Olympic entirely; WC finals highest K, continental finals
high, WC-qual/Nations League medium-high, continental qual medium, friendlies low, minor
invitationals very low, default to minor.

**`poisson_model.py`** — `team_lambdas(rating_a, rating_b, venue_country, ...)` (lambda =
BASE_GOALS*exp(SPREAD*rating_diff), SPREAD=0.0024, venue-based host bump, clamped exponent);
`score_matrix`, `outcome_probs`. Independent-Poisson simplification noted.

**`optimizer.py`** — `best_score(matrix, mode)` (EV-max vs real `scoring.py`),
`ev_optimal_count(lam, rule)`, `most_likely_outcome(matrix)`.

**`bracket.py`** — `simulate_group`, `group_table`, `rank_third_place`,
`assign_best_thirds` (backtracking, group-eligibility), `monte_carlo_bracket(n_sims=10000)`
to advancement probabilities + modal slot teams.

**`market.py`** — `load_market_probs` (renormalize to sum 1), `market_to_rating_scale`
(map onto Elo scale), `blend_ratings(elo, market, w=0.30)`. Live 2026 only.

**`backtest.py` / `03_backtest.ipynb`** — leakage-free evaluation on 2022 WC, Euro 2024,
Copa America 2024, AFCON 2023. Pooled result: W/D/L ~50.3%, model beats naive baseline on
points and dominates on calibration (Brier 0.62 vs 0.99; log-loss 1.04 vs 13.7). Honest
finding: weaker on AFCON (~44%) — confederation imbalance. Includes a SPREAD/K sensitivity
sweep (flat surface -> defaults retained, not overfit to the test set).

**`predict.py`** — orchestrates Phase 1 end-to-end; resolves placeholders, builds a single
internally-consistent deterministic bracket (no double-booked teams — asserted), writes
submission CSVs in the grader's spelling (fixture spelling for output, dataset spelling
internally). Uses market-blended ratings for the live 2026 run.

### To build (Phase 2)

**`features.py`** TODO *(Stage 7, NEXT)*
- Purpose: build a per-match feature matrix for training a goals model. One row per
  historical match; PRE-MATCH features only (strict no-leakage); labels = actual home/away
  goals (+ outcome).
- Method: **single chronological forward pass** — read tracker state to build the row, THEN
  update with the match result. Do NOT call `get_ratings_asof` per match (O(n^2)); maintain
  running Elo state inline reusing `elo.py`'s update logic. Use `canonical_team`.
- Families (all computable from existing results data — no new datasets): Elo (home/away/
  diff), rolling form (last-5/last-10 PPG, goals for/against, win rate), head-to-head
  (count, home win rate, avg goals), momentum (streak length, clean-sheet rate, draw rate),
  tournament context (major/qual/friendly tier, each team's major-tournament win rate),
  venue (neutral flag, host indicator).
- API: `build_feature_matrix(results) -> DataFrame`; save to
  `data/processed/feature_matrix.parquet`. `MIN_HISTORY` constant for sparse early matches.
- Validation: shape/columns, a no-leakage spot check on one match, per-column coverage report.

**`train.py`** TODO *(Stage 8)*
- Build the feature matrix, **temporal** train/test split (train pre-2022; validate on the
  four held-out tournaments — same as the backtest, no leakage). Train XGBoost regressors
  for home and away expected goals (Option B). Save the model. Report **feature importance**
  (key portfolio artifact — shows which signals matter).

**Stage 9 — rewire + re-backtest.** Feed trained-model expected goals into `team_lambdas`'s
role (the lambda source) so the existing Poisson/optimizer/bracket stack is reused unchanged.
Re-run the backtest and compare XGBoost vs the Elo-Poisson baseline on the same held-out
tournaments. **Moment of truth — report the result honestly even if the trained model only
ties the simpler baseline; that is itself a strong, judgment-demonstrating finding.**

---

## 7. Success Criteria & Honest Metrics

- **W/D/L accuracy ~50-55%** on held-out internationals. >58-60% implies suspect leakage.
- **Calibration** (Brier/log-loss) clearly better than baselines — the headline result.
- **Points beat a naive baseline** modestly; beat a zero-skill baseline clearly.
- Phase 2: report whether XGBoost beats Elo-Poisson out-of-sample — **either outcome is a
  legitimate, reportable result.** Do not force a win; do not overfit the test set.
- Never claim a leaderboard placement or "predicted the champion" (sample of one).

---

## 8. Limitations & Next Steps

- Team-level only — no squad/injury/individual-form signal (Phase 2 partially addresses via
  form/momentum but still no player data). Underrates tournament-overperformers (e.g.
  Croatia); overrates soft-draw/AFCON teams pre-market-anchor.
- Corners/cards predicted as constants (no per-match data in the dataset).
- Home/away goals modeled independently (Dixon-Coles correlation a future refinement).
- Market blend is a fixed judgment weight, not backtested (no historical odds).
- Future: interactive demo frontend; richer feature families (goalscorer/player data) if a
  player-level dataset is added.
