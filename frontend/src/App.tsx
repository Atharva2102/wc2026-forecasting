import { useEffect, useMemo, useState } from 'react';
import { API_URL, fetchTeams, predictMatch } from './api';
import { flagForTeam, TEAM_FLAG } from './flags';
import type { ModelSummary, PredictionResponse } from './types';

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function num(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'n/a';
  return value.toFixed(digits);
}

function TeamCombobox({
  label,
  value,
  teams,
  blockedTeam,
  onChange,
}: {
  label: string;
  value: string;
  teams: string[];
  blockedTeam?: string;
  onChange: (team: string) => void;
}) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);

  const visibleTeams = useMemo(() => {
    const q = query.trim().toLowerCase();
    return teams
      .filter((team) => team !== blockedTeam)
      .filter((team) => team.toLowerCase().includes(q))
      .slice(0, 12);
  }, [blockedTeam, query, teams]);

  return (
    <div className="relative">
      <label className="mb-2 block text-xs font-bold uppercase tracking-[0.16em] text-muted">
        {label}
      </label>
      <button
        type="button"
        className="flex w-full items-center justify-between rounded-md border border-line bg-white px-5 py-5 text-left text-lg shadow-sm transition hover:border-green focus:outline-none focus:ring-2 focus:ring-green/25 active:scale-[0.99]"
        onClick={() => setOpen((next) => !next)}
      >
        <span className="flex items-center gap-3">
          <span className="text-3xl leading-none">{value ? flagForTeam(value) : '-'}</span>
          <span className={value ? 'font-semibold text-ink' : 'text-muted'}>
            {value || 'Choose a team'}
          </span>
        </span>
        <span className="text-muted">v</span>
      </button>
      {open && (
        <div className="absolute z-30 mt-2 w-full overflow-hidden rounded-md border border-line bg-white shadow-soft">
          <input
            autoFocus
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Type to filter"
            className="w-full border-b border-line px-5 py-4 text-base outline-none"
          />
          <div className="max-h-72 overflow-auto p-1">
            {visibleTeams.length === 0 ? (
              <p className="px-4 py-4 text-sm text-muted">No teams found.</p>
            ) : (
              visibleTeams.map((team) => (
                <button
                  type="button"
                  key={team}
                  onClick={() => {
                    onChange(team);
                    setQuery('');
                    setOpen(false);
                  }}
                  className="flex w-full items-center gap-3 rounded px-4 py-3 text-left transition hover:bg-grass active:scale-[0.99]"
                >
                  <span className="text-2xl">{flagForTeam(team)}</span>
                  <span className="text-base font-medium">{team}</span>
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function ProbabilityBar({
  model,
  home,
  away,
  large = false,
}: {
  model: ModelSummary;
  home: string;
  away: string;
  large?: boolean;
}) {
  const segments = [
    { label: home, value: model.p_home, className: 'bg-green text-white' },
    { label: 'Draw', value: model.p_draw, className: 'bg-straw text-ink' },
    { label: away, value: model.p_away, className: 'bg-clay text-white' },
  ];

  return (
    <div>
      <div
        className={`mb-3 flex justify-between gap-3 font-semibold text-muted ${
          large ? 'text-base' : 'text-sm'
        }`}
      >
        {segments.map((segment) => (
          <span key={segment.label}>
            {segment.label} <b className="text-ink">{pct(segment.value)}</b>
          </span>
        ))}
      </div>
      <div
        className={`flex overflow-hidden rounded-md border border-line bg-white ${
          large ? 'h-16' : 'h-10'
        }`}
      >
        {segments.map((segment) => (
          <div
            key={segment.label}
            className={`bar-segment flex items-center justify-center font-bold ${
              large ? 'text-sm md:text-base' : 'text-xs'
            } ${segment.className}`}
            style={{ width: `${Math.max(segment.value * 100, 5)}%` }}
            title={`${segment.label}: ${pct(segment.value)}`}
          >
            {pct(segment.value)}
          </div>
        ))}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-line bg-paper/60 px-3 py-3">
      <p className="text-xs font-bold uppercase tracking-[0.14em] text-muted">{label}</p>
      <p className="mt-1 text-2xl font-bold text-ink">{value}</p>
    </div>
  );
}

function XGBoostHero({ prediction }: { prediction: PredictionResponse }) {
  return (
    <section className="rounded-md border border-line bg-white p-6 shadow-sm md:p-8">
      <div className="mb-7 flex flex-col justify-between gap-4 md:flex-row md:items-start">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.18em] text-green">
            Primary forecast
          </p>
          <h3 className="mt-2 font-display text-4xl font-bold text-ink">
            XGBoost goals model
          </h3>
          <p className="mt-2 max-w-2xl text-base text-muted">
            Feature-trained expected goals model using the same Poisson scoreline layer.
          </p>
        </div>
        <div className="rounded border border-line bg-paper px-4 py-2 text-sm font-semibold text-muted">
          Most likely: {prediction.xgboost.most_likely_score}
        </div>
      </div>

      <ProbabilityBar
        model={prediction.xgboost}
        home={prediction.home}
        away={prediction.away}
        large
      />

      <div className="mt-6 grid grid-cols-2 gap-4">
        <Metric label={`${prediction.home} xG`} value={num(prediction.xgboost.exp_home_goals, 2)} />
        <Metric label={`${prediction.away} xG`} value={num(prediction.xgboost.exp_away_goals, 2)} />
      </div>

      <div className="mt-6">
        <p className="mb-3 text-xs font-bold uppercase tracking-[0.16em] text-muted">
          Top scorelines
        </p>
        <div className="grid gap-2 md:grid-cols-5">
          {prediction.xgboost.top_scorelines.map((scoreline, index) => (
            <div
              key={scoreline.score}
              className="rounded border border-line bg-paper/60 px-4 py-3"
            >
              <p className="text-xs font-bold uppercase tracking-[0.14em] text-muted">
                #{index + 1}
              </p>
              <p className="mt-1 text-2xl font-bold text-ink">{scoreline.score}</p>
              <p className="text-sm font-semibold text-muted">{pct(scoreline.prob)}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function BaselineStrip({ prediction }: { prediction: PredictionResponse }) {
  return (
    <section className="rounded-md border border-line bg-white p-4 shadow-sm">
      <div className="mb-3 flex flex-col justify-between gap-1 md:flex-row md:items-center">
        <p className="text-sm font-bold text-ink">Elo-Poisson rating-only baseline</p>
        <p className="text-sm text-muted">
          Kept for comparison; it can legitimately disagree with XGBoost.
        </p>
      </div>
      <ProbabilityBar
        model={prediction.elo_poisson}
        home={prediction.home}
        away={prediction.away}
      />
    </section>
  );
}

function ResultPanel({ prediction }: { prediction: PredictionResponse }) {
  return (
    <div className="result-enter space-y-6">
      <div className="rounded-md border border-line bg-white p-5 shadow-sm">
        <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.16em] text-muted">
              Featured matchup
            </p>
            <h2 className="mt-1 font-display text-4xl font-bold text-ink">
              {flagForTeam(prediction.home)} {prediction.home}
              <span className="mx-3 text-muted">vs</span>
              {flagForTeam(prediction.away)} {prediction.away}
            </h2>
          </div>
          <div className="rounded border border-line bg-paper px-4 py-2 text-sm font-semibold text-muted">
            {prediction.neutral ? 'Neutral venue' : 'Home venue'}
          </div>
        </div>
      </div>

      <XGBoostHero prediction={prediction} />
      <BaselineStrip prediction={prediction} />

      <section className="grid gap-3 rounded-md border border-line bg-white p-5 shadow-sm md:grid-cols-5">
        <Metric label={`${prediction.home} Elo`} value={num(prediction.context.home_elo, 0)} />
        <Metric label={`${prediction.away} Elo`} value={num(prediction.context.away_elo, 0)} />
        <Metric label={`${prediction.home} last 5 PPG`} value={num(prediction.context.home_last5_ppg, 2)} />
        <Metric label={`${prediction.away} last 5 PPG`} value={num(prediction.context.away_last5_ppg, 2)} />
        <Metric label="Head to head" value={String(prediction.context.h2h_count)} />
      </section>

      <section className="rounded-md border border-line bg-grass/70 p-6 shadow-sm">
        <p className="mb-2 text-xs font-bold uppercase tracking-[0.18em] text-green">
          Model read
        </p>
        <p className="font-display text-xl leading-8 text-ink">{prediction.analysis}</p>
      </section>
    </div>
  );
}

export default function App() {
  const [teams, setTeams] = useState<string[]>([]);
  const [home, setHome] = useState('');
  const [away, setAway] = useState('');
  const [neutral, setNeutral] = useState(true);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loadingTeams, setLoadingTeams] = useState(true);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchTeams()
      .then((loadedTeams) => {
        const missingFlags = loadedTeams.filter((team) => !TEAM_FLAG[team]);
        if (missingFlags.length) {
          console.warn('Teams missing flag mappings:', missingFlags);
        }
        setTeams(loadedTeams);
        setHome(loadedTeams.includes('Spain') ? 'Spain' : loadedTeams[0] ?? '');
        setAway(loadedTeams.includes('Brazil') ? 'Brazil' : loadedTeams[1] ?? '');
      })
      .catch(() => {
        setError(
          `Could not reach the backend at ${API_URL}. Start FastAPI first, then refresh this page.`,
        );
      })
      .finally(() => setLoadingTeams(false));
  }, []);

  async function submitPrediction() {
    if (!home || !away || home === away) return;
    setError('');
    setLoadingPrediction(true);
    try {
      const result = await predictMatch(home, away, neutral);
      setPrediction(result);
    } catch {
      setError(
        `Prediction failed. Check that the FastAPI backend is running at ${API_URL}.`,
      );
    } finally {
      setLoadingPrediction(false);
    }
  }

  return (
    <main className="min-h-screen bg-paper text-ink">
      <div className="mx-auto flex w-full max-w-7xl flex-col px-5 py-8 md:px-8">
        <header className="mb-10 border-b border-line pb-8">
          <p className="mb-3 text-xs font-bold uppercase tracking-[0.2em] text-green">
            WC 2026 Match Lab
          </p>
          <div className="space-y-8">
            <div>
              <h1 className="font-display text-5xl font-bold leading-tight text-ink md:text-6xl">
                A live football forecast, built from the tournament model.
              </h1>
              <p className="mt-5 max-w-2xl text-lg leading-7 text-muted">
                Pick two World Cup teams to compare the feature-trained XGBoost
                forecast against a clean Elo-Poisson baseline.
              </p>
            </div>
            <section className="rounded-md border border-line bg-white p-6 shadow-soft md:p-8 lg:p-10">
              <div className="mb-6 flex flex-col justify-between gap-2 md:flex-row md:items-end">
                <div>
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-green">
                    Build a matchup
                  </p>
                  <h2 className="mt-2 font-display text-3xl font-bold text-ink">
                    Pick teams, then ask the model.
                  </h2>
                </div>
                <p className="max-w-md text-sm text-muted">
                  The selectors call the backend team list, so the demo stays tied to the live model.
                </p>
              </div>
              <div className="grid gap-5 md:grid-cols-2">
                <TeamCombobox
                  label="Home team"
                  value={home}
                  teams={teams}
                  blockedTeam={away}
                  onChange={setHome}
                />
                <TeamCombobox
                  label="Away team"
                  value={away}
                  teams={teams}
                  blockedTeam={home}
                  onChange={setAway}
                />
              </div>
              <div className="mt-7 flex flex-col justify-between gap-5 border-t border-line pt-7 sm:flex-row sm:items-center">
                <label className="flex cursor-pointer items-center gap-3 text-base font-semibold text-ink">
                  <span
                    className={`flex h-6 w-11 items-center rounded-full p-1 transition ${
                      neutral ? 'bg-green' : 'bg-muted'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={neutral}
                      onChange={(event) => setNeutral(event.target.checked)}
                      className="sr-only"
                    />
                    <span
                      className={`h-4 w-4 rounded-full bg-white transition-transform ${
                        neutral ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </span>
                  Neutral venue
                </label>
                <button
                  type="button"
                  disabled={loadingTeams || loadingPrediction || !home || !away || home === away}
                  onClick={submitPrediction}
                  className="rounded-md bg-green px-9 py-5 text-base font-bold uppercase tracking-[0.14em] text-white transition hover:bg-green/90 active:scale-[0.98] disabled:cursor-not-allowed disabled:bg-muted"
                >
                  {loadingPrediction ? 'Predicting...' : 'Predict'}
                </button>
              </div>
            </section>
          </div>
        </header>

        {error && (
          <div className="mb-6 rounded-md border border-clay/30 bg-white px-5 py-4 text-clay shadow-sm">
            {error}
          </div>
        )}

        {loadingTeams ? (
          <section className="rounded-md border border-line bg-white p-8 text-muted shadow-sm">
            Loading teams from the prediction API...
          </section>
        ) : prediction ? (
          <ResultPanel prediction={prediction} />
        ) : (
          <section className="rounded-md border border-line bg-white p-8 text-center shadow-sm">
            <p className="font-display text-3xl font-bold text-ink">
              Choose a matchup to open the model card.
            </p>
            <p className="mx-auto mt-3 max-w-xl text-muted">
              The response uses live FastAPI predictions only; no mocked data is rendered.
            </p>
          </section>
        )}

        <footer className="mt-12 border-t border-line pt-6 text-sm text-muted">
          Predictions come from a model trained on roughly 25k international
          matches and validated with leakage-free temporal backtests.
        </footer>
      </div>
    </main>
  );
}
