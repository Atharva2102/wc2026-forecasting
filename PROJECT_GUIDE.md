# World Cup 2026 Forecasting Engine — Project Guide

A data-driven forecasting pipeline that predicts every match of the 2026 FIFA World Cup
(all 104 fixtures) and submits to a 3,400+ entrant prediction competition. This document
is the single reference for the project's vision, design decisions, and what each file in
the repository is responsible for. Build against it stage by stage.

---

## 0. Build Status

| Stage | Scope | Status |
|---|---|---|
| 1 | `data_loader.py` + `01_eda.ipynb` — data load, cleaning, name reconciliation, EDA | ✅ **Done** |
| — | `scoring.py` — rubric, single source of truth | ✅ **Done** (3 bugs fixed) |
| 2 | `elo.py` + `02_elo_ratings.ipynb` — data-driven ratings | ⬜ **Next** |
| 3 | `poisson_model.py` + `optimizer.py` — single-match predictions | ⬜ |
| 4 | `bracket.py` — standings, best-thirds, Monte Carlo | ⬜ |
| 5 | `predict.py` — orchestrate, write submission | ⬜ |
| 6 | `03_backtest.ipynb` — real metrics | ⬜ |
| 7 | `README.md` — written with numbers in hand | ⬜ |

---

## 1. Vision

This is a portfolio project first and a competition entry second. Both goals point the
same direction: a clean, reproducible, *defensible* forecasting pipeline where every
number traces back to data rather than gut feel, and where each stage can be explained on
its own in an interview.

The headline idea, which the whole repo serves: **we hold beliefs (a probability
distribution over what might happen) and we know the reward function (the competition's
scoring rubric), so the best prediction is the one that maximizes expected reward under
those beliefs — which is usually *not* the single most likely outcome.** Everything below
is machinery for (a) forming honest beliefs from data and (b) optimizing predictions
against the real rubric.

---

## 2. The Problem

For every one of the 104 matches we predict an exact scoreline, total corners, total
yellow cards, and total red cards. Group games also need a winning team (home / away /
draw). Knockout games also need the two teams we expect in that slot, the match winner,
and whether it goes to penalties.

### Scoring rubric (encoded in `src/scoring.py`)

| Category | Condition | Points |
|---|---|---|
| Score | Exact scoreline | 25 |
| Score | Correct goal difference, wrong score | 10 |
| Score | Correct total goals, wrong score | 10 |
| Corners | Exact | 10 |
| Corners | Within 2 | 5 |
| Yellow cards | Exact | 10 |
| Yellow cards | Within 1 | 5 |
| Red cards | Exact | 5 |
| Winning team *(group only)* | Correct | 40 |
| Matchup *(knockout only)* | Both teams correct | 20 |
| Matchup *(knockout only)* | One team correct | 10 |
| Match winner *(knockout only)* | Correct | 20 |
| Penalties *(knockout only)* | Correct (did it go to a shootout?) | 5 |

Round multipliers: group ×1, Round of 32 ×1, Round of 16 ×2, Quarter-final ×4,
Semi-final ×8, Third-place ×8, Final ×16.

### Strategic insights that shape the model

- **Winning team (40 pts) is the single biggest category** and is scored *separately*
  from the scoreline. Predict the single most likely outcome; never default to "draw"
  (a draw is rarely the most likely *single* outcome).
- **The score tiers (25 / 10 / 10) are mutually exclusive** and reward being close, so
  the optimal scoreline is an expected-value pick, not the modal one. Even matchups
  collapse to 1–1 (a strong hedge); clear favorites push to 2–1, 3–0.
- **Corners and cards are match totals with wide bands.** Without per-match data the
  EV-optimal play is the tournament mean for *every* match: ~10 corners, ~3–4 yellows,
  0 reds. Red cards = 0 is nearly free points (base rate ~0.2/match).
- **Points are back-loaded by multipliers.** The Final alone is ×16. The QF→Final stretch
  decides the leaderboard.
- **Knockout points are gated.** To score the matchup (20) and winner (20), you must
  predict the right team in the slot — so simulating the bracket accurately is the
  highest-leverage work in the project.
- **Upsets are priced, not predicted.** See §4.

---

## 3. The Approach (pipeline)

```
historical results ─▶ Elo ratings ─▶ Poisson goal model ─▶ score distribution
                                                                    │
                          ┌─────────────────────────────────────────┤
                          ▼                                          ▼
                EV-optimizer (vs rubric)                  Monte-Carlo bracket sim
                  scores, stats, winner                   advancement probabilities
                          │                                          │
                          └──────────────▶ predict.py ◀──────────────┘
                                               │
                                    submission CSVs + backtest metrics
```

Data flows one direction: results become ratings, ratings become expected goals, expected
goals become a probability distribution, and that distribution is optimized against the
rubric for single matches and simulated forward for the bracket.

---

## 4. Key Design Decisions

- **Data-driven ratings, not hand-typed.** Team strength is *earned* from ~25K
  international results (2000-present), not assigned by opinion. This is the core of the
  data-science story and what makes every number defensible.
- **Recency + tournament-importance weighting.** A 2026 Copa América match should move a
  rating far more than a 2015 friendly. Implemented via Elo K-factor scaling and/or time
  decay, not by hunting separate tournament files — all the tournaments live in the one
  dataset. See §6 `elo.py` for the tier table.
- **Regression toward the mean.** International football is high-variance, so a
  results-only model tends to be *overconfident* in favorites. Damp the rating spread so
  no team is treated as near-invincible. This is one of the levers that keeps upset
  probabilities honest.
- **Host advantage is by identity, not by label.** At a World Cup nearly every game is at
  a neutral venue, so the `home_team` column is an administrative label, *not* a crowd
  advantage. Real home advantage attaches only to the three hosts (USA, Mexico, Canada).
  The historical dataset's `neutral` flag is used to learn home advantage correctly.
- **Markets as a sanity anchor, not the backbone.** Current betting odds (and FIFA
  rankings) are folded in lightly to keep the top tier of teams honest, since markets are
  very good at the elite end. The historical data does the heavy lifting.
- **Decision-theoretic optimization.** Predictions maximize expected points against the
  *actual* rubric in `scoring.py` — change the rubric and the picks change.
- **Monte Carlo carries the uncertainty.** A deterministic bracket silently assumes the
  favorite wins every game (a bracket that has never happened). Simulating the tournament
  thousands of times lets upsets ripple forward and yields advancement *probabilities*.

---

## 5. Directory Structure

```
wc2026-forecasting/
├── data/
│   ├── raw/            # pulled international results + the two competition fixture CSVs
│   └── processed/      # built ratings, cleaned/intermediate data
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── elo.py
│   ├── poisson_model.py
│   ├── optimizer.py
│   ├── scoring.py
│   ├── bracket.py
│   └── predict.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_elo_ratings.ipynb
│   ├── 03_backtest.ipynb
│   └── 04_final_predictions.ipynb
├── outputs/            # final submission CSVs, figures
├── config.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 6. File-by-File Build Guide

### Config & meta

**`config.py`** — All tunable constants in one place so nothing is hard-coded mid-pipeline.
Holds: the results dataset URL; `ELO_START_YEAR = 2000`; file paths for raw/processed/
outputs; Poisson params (`BASE_GOALS`, `SPREAD`); `HOST_ADV` and
`HOSTS = {"USA", "Mexico", "Canada"}`; Elo params (`K_BASE`, default/starting rating,
regression strength); the tournament-importance tier map (see `elo.py`); `N_SIMS` for the
Monte Carlo. Values are placeholders to be tuned during the backtest.

**`requirements.txt`** — pandas, numpy, scipy, matplotlib, seaborn, jupyter, ipykernel.

**`.gitignore`** — Standard Python ignores plus `data/raw/*`, `data/processed/*`,
`outputs/*` (keep `.gitkeep`s). The results CSV is re-pullable from the URL, so it stays
out of git; document the fetch in the README. *Optional:* commit the two small competition
fixture CSVs if you want the repo self-contained.

**`README.md`** — The recruiter-facing front door. Written *last*, once real numbers exist.
Sections: Overview, The Problem & Scoring, Approach, Results/Backtest (metrics table),
Project Structure, How to Run, Limitations & Next Steps.

### `src/` modules — build in this order

**`data_loader.py`** ✅ *(done)*
- Purpose: get data in and clean. `fetch_international_results()` pulls the ~49K-row
  results CSV from the URL and caches it to `data/raw/`; `clean_results()` parses dates,
  drops unplayed (`NaN`-score) future fixtures, filters to `ELO_START_YEAR`+ (→ ~25,256
  matches), and preserves the `neutral` flag; `load_fixtures()` loads the two competition
  CSVs (`group_fixtures.csv`, `knockout_slots.csv`).
- **Name reconciliation (the Stage 1 deliverable):** the historical dataset spells some
  teams differently than the fixtures, and the fixtures contain 6 playoff placeholders.
  Two maps + one function reconcile everything to the **dataset's** spelling:
  - `PLAYOFF_MAP` + `resolve_team()` — placeholder → real team:
    UEFA A→Bosnia and Herzegovina, B→Sweden, C→Türkiye, D→Czechia,
    FIFA 1→DR Congo, FIFA 2→Iraq.
  - `NAME_MAP` — fixture/standard spelling → dataset spelling. Confirmed entries:
    Cabo Verde→Cape Verde, Côte d'Ivoire→Ivory Coast, USA→United States.
    (All 6 resolved playoff names were verified to already exist in the dataset under
    those spellings.)
  - `canonical_team(name)` — applies `resolve_team` **then** `NAME_MAP`, in that order.
    **Every downstream stage must use `canonical_team` for team lookups.**
  - `verify_fixture_team_names()` — asserts all 48 fixture teams resolve to names present
    in the results data. Verified result: **48 teams, 0 missing.**
- Data columns: `date, home_team, away_team, home_score, away_score, tournament, city,
  country, neutral`.
- Source: `https://raw.githubusercontent.com/martj42/international_results/master/results.csv`

**`scoring.py`** ✅ *(done — rubric, single source of truth)*
- `score_match()` and `total_points()`. The three original bugs are fixed: knockout
  matchup points (20/10) now scored from predicted vs. actual teams (unordered);
  penalties scored as an independent True/False item covering both the "yes" and "no"
  cases; score tiers are a mutually-exclusive `if/elif/elif` chain.
- Everything downstream optimizes against this file — treat it as the contract.

**`elo.py`** ⬜ *(NEXT — the interesting one, most judgment calls live here)*
- Purpose: turn match history into one strength number per team.
  `compute_elo_ratings(matches, ...)`.
- Mechanics: process matches chronologically; after each game move points between teams
  based on result vs. expectation. Recent games dominate naturally because the rating
  keeps updating. Use `canonical_team` from `data_loader` for all team names.
- Decisions to implement:
  1. **K-factor scaled by tournament importance** — use a tiered map (defined in
     `config.py`, classified by pattern, not 126 hand entries). Tiers settled from the
     Stage-1 audit:
     - **Exclude entirely (filter out, not downweight):** non-FIFA / non-senior comps —
       any tournament matching CONIFA / ConIFA / "Viva World Cup" / "Island Games" style
       non-FIFA events, and any youth/Olympic markers (`Olympic`, `U-`, `U23`, `U21`,
       `U20`, `Youth`, `Universiade`). These are different entities; they must never enter
       the ratings dict.
     - **Highest K:** FIFA World Cup (finals).
     - **High K:** continental finals — UEFA Euro, Copa América, African Cup of Nations,
       AFC Asian Cup, CONCACAF Gold Cup.
     - **Medium-high K:** FIFA World Cup qualification, UEFA Nations League.
     - **Medium K:** continental qualification (Euro/AFCON/Asian/etc. qualifiers).
     - **Low K:** Friendlies.
     - **Very low K:** minor invitationals that ARE real national teams — King's Cup,
       Nehru Cup, Merdeka, Lunar New Year Cup, CFU Caribbean Cup (+ qual), AFC Challenge
       Cup (+ qual), Indian Ocean Island Games, etc.
     - **Default:** anything unmatched falls to the friendly/minor tier.
  2. optional **margin-of-victory** scaling (bigger wins move ratings a bit more).
  3. **regression toward the mean** between cycles so ratings don't drift to extremes
     (keeps favorites from looking invincible — an upset-honesty lever).
  4. **home advantage only when `neutral == False`.**
- Critical for the backtest: expose ratings **as of a given date** (point-in-time), so
  the backtest never uses future information. No leakage.
- Output: a `team → rating` mapping (and ideally the rating time series for plots).

**`poisson_model.py`** ⬜
- Purpose: ratings → expected goals → probability distribution over scorelines.
  `team_lambdas(rating_a, rating_b, neutral)`, `score_matrix(lh, la)`,
  `outcome_probs(lh, la)`.
- Mechanics: convert the Elo rating gap to each team's expected goals (λ) via
  `BASE_GOALS · exp(SPREAD · rating_diff)`, bounded to a sane range; build the score grid
  as the outer product of two Poisson distributions; derive win/draw/loss probabilities.
- Gotchas: apply `HOST_ADV` by **team identity** (is the team in `HOSTS`?), only for
  non-neutral games; calibrate `SPREAD` so average goal output matches reality (~1.35/team).
- Simplification to note in the README: home/away goals treated as independent
  (Dixon–Coles correlation is a future refinement).

**`optimizer.py`** ⬜
- Purpose: pick the EV-maximizing prediction given the distribution and the rubric.
  `best_score(matrix, mode)`, `ev_optimal_count(lam, rule)`.
- `best_score` sweeps candidate scorelines and maximizes expected points **using
  `scoring.py`** (import the real function — do not re-hardcode the rubric); `mode`
  constrains the domain for knockout games (`home` / `away` / `draw` to stay consistent
  with the predicted winner and penalties flag). `ev_optimal_count` finds the best total
  for corners/yellows under their bands. Winning-team / match-winner picked by argmax of
  outcome probabilities.

**`bracket.py`** ⬜
- Purpose: resolve the knockout structure and carry uncertainty.
  `group_standings()`, `assign_best_thirds()`, `monte_carlo_bracket(n_sims)`.
- `group_standings` ranks each group; `assign_best_thirds` matches the 8 best third-placed
  teams to bracket slots respecting each slot's group-eligibility list (backtracking).
- `monte_carlo_bracket` simulates the whole tournament `N_SIMS` times, rolling every match
  by its probabilities, and returns **advancement probabilities per slot** plus the
  single most-likely team for the submission. This is where upsets are represented.
- Gotchas: knockout slots are conditions ("Winner Group A", "Best 3rd (Groups C/E/F/H/I)",
  "Winner Match 73") — resolve in `match_id` order so winners feed forward.

**`predict.py`** ⬜ *(orchestrator, build last before backtest)*
- Purpose: run the full pipeline and write the submission. Resolves placeholders via
  `canonical_team`, applies the model to all 72 group + 32 knockout fixtures, fills the
  exact submission columns, writes to `outputs/`, and asserts no `None`/`NaN` remain.
- Submission columns — **group**: `predicted_home_goals, predicted_away_goals, corners,
  yellow_cards, red_cards, winning_team` (`"home"/"away"/"draw"`). **Knockout**:
  `predicted_home_team, predicted_away_team, predicted_home_goals, predicted_away_goals,
  corners, yellow_cards, red_cards, match_winner` (`"home"/"away"`), `penalties` (bool).
- Gotcha: the knockout `predicted_*_team` strings the grader expects may differ from the
  dataset spelling — for the SUBMISSION, write the names the competition/grader uses (the
  fixture spelling), even though internal ratings use the dataset spelling. Verify against
  a sample submission/team list if one exists.

### `notebooks/` — the story and the evidence

- **`01_eda.ipynb`** ✅ — explores the results data: shape/coverage/date-range, goal
  distributions, home vs. neutral effects, the team-name consistency check (now passing,
  0 mismatches), and the competition-type audit (tier decisions above). Justifies the
  modeling choices.
- **`02_elo_ratings.ipynb`** ⬜ — build and visualize ratings; show the top teams and
  rating trajectories (e.g. a rising side beating giants in recent friendlies); tune K and
  the importance weights; sanity-check against FIFA ranking and market odds.
- **`03_backtest.ipynb`** ⬜ — **the metrics notebook.** Run the engine *as of before* the
  2022 World Cup and Euro 2024 (point-in-time ratings, no leakage), predict, score with
  `scoring.py`, and report: W/D/L accuracy, EV-points lift vs. a naive modal/FIFA-rank
  baseline, and calibration (Brier / log-loss). These numbers become the README and résumé.
- **`04_final_predictions.ipynb`** ⬜ — run the full pipeline for 2026, generate the
  submission, eyeball sanity (champion, plausible bracket), export to `outputs/`.

### Data & output folders

- **`data/raw/`** — the pulled results CSV and the two competition fixture CSVs.
- **`data/processed/`** — built ratings and cleaned intermediates.
- **`outputs/`** — final submission CSVs and figures used in the README.

---

## 7. Build Roadmap

1. ✅ `data_loader.py` + `01_eda.ipynb` — data local, cleaned, names reconciled (0 missing).
2. ✅ `scoring.py` — rubric correct (3 bugs fixed).
3. ⬜ `elo.py` + `02_elo_ratings.ipynb` — build and validate ratings. **← next**
4. ⬜ `poisson_model.py` + `optimizer.py` — single-match predictions.
5. ⬜ `bracket.py` — standings, best-thirds, Monte Carlo.
6. ⬜ `predict.py` — wire it together, produce a submission.
7. ⬜ `03_backtest.ipynb` — generate the real metrics.
8. ⬜ `README.md` — write it with the numbers in hand.

---

## 8. Success Criteria & Honest Metrics

Targets that are realistic and defensible in an interview (measure them; don't assume them):

- **W/D/L accuracy ~50–55%** on held-out internationals. Above ~58–60% almost certainly
  means data leakage (training on the test period) — treat that as a red flag, not a win.
- **EV-points lift ~10–20%** over a naive modal-scoreline baseline, measured by scoring
  both prediction sets against real results with `scoring.py`. This is the most
  *distinctive* result — it shows expected-value reasoning, not just model-fitting.
- **Calibration** (Brier / log-loss) better than a FIFA-rank baseline.

Do **not** claim: a leaderboard placement before results exist, or "predicted the
champion" (a sample of one — luck, not skill).

---

## 9. Limitations & Next Steps

- No squad/injury/form-of-individuals signal — team-level only.
- Corners and cards are constant (tournament means); per-match models would add edge.
- Home/away goals modeled independently (Dixon–Coles correlation is a refinement).
- Market blend is light; a fuller probabilistic blend is possible.
- For the competition specifically: one deliberate, defensible contrarian pick on a single
  high-multiplier knockout slot is the rational way to separate from a chalk-picking field
  — a calculated variance bet on standings, not a belief that upsets can be called.