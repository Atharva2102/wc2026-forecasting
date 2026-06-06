# WC 2026 Forecasting Engine

## Overview

<!-- One-line description: A data-driven forecasting system for the 2026 FIFA World Cup that combines Elo ratings, Poisson goal modelling, expected-value optimisation, and Monte Carlo bracket simulation to generate competition predictions. -->

## The Problem & Scoring

<!-- Describe the competition scoring rubric: exact score (25 pts), correct GD (10 pts), correct total (10 pts), corners/yellows/reds partial credit, winning team (group, 40 pts), KO matchup + winner + penalties; round multipliers up to 16× for the Final. -->

## Approach

### 1 — Elo Ratings

<!-- Build team strength estimates from ~49K historical results, weighted by tournament importance and recency decay. -->

### 2 — Poisson Goal Model

<!-- Convert Elo rating differences into per-team expected goal rates (λ) and generate full score-probability matrices. -->

### 3 — EV-Optimal Predictions

<!-- For each match, pick the scoreline (and count predictions) that maximise expected competition points under the scoring rubric. -->

### 4 — Monte Carlo Bracket Simulation

<!-- Simulate the full 48-team → 32-team knockout bracket 10 000+ times to estimate round-advancement probabilities and fill projected KO fixtures. -->

## Results / Backtest

<!-- Table placeholder — fill after running notebooks/03_backtest.ipynb -->

| Tournament | Metric | Score |
|---|---|---|
| 2022 FIFA World Cup | Mean points per match | TBD |
| UEFA Euro 2024 | Mean points per match | TBD |
| 2022 FIFA World Cup | Log-loss (scoreline) | TBD |

## Project Structure

```
wc2026-forecasting/
├── data/
│   ├── raw/            # raw CSVs (git-ignored)
│   └── processed/      # intermediate outputs (git-ignored)
├── src/
│   ├── data_loader.py  # ingestion & cleaning
│   ├── elo.py          # Elo rating engine
│   ├── poisson_model.py
│   ├── optimizer.py    # EV-optimal prediction selection
│   ├── scoring.py      # competition rubric constants
│   ├── bracket.py      # group standings + Monte Carlo
│   └── predict.py      # end-to-end pipeline
├── notebooks/          # EDA, rating analysis, backtest, final predictions
├── outputs/            # submission CSVs (git-ignored)
├── config.py           # all tunable constants
└── requirements.txt
```

## How to Run

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place raw data CSVs in data/raw/

# 4. Run the full pipeline
python -m src.predict

# 5. Or explore interactively via notebooks
jupyter lab
```

## Limitations & Next Steps

<!-- Known limitations and planned improvements go here, e.g.: Dixon-Coles correlation correction, Bayesian hyperparameter optimisation, corner/card count sub-models, ensemble with market odds. -->
