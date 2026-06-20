export type TeamName = string;

export type Scoreline = {
  score: string;
  prob: number;
};

export type ModelSummary = {
  p_home: number;
  p_draw: number;
  p_away: number;
  exp_home_goals: number;
  exp_away_goals: number;
  top_scorelines?: Scoreline[];
  most_likely_score?: string;
  ev_score?: string;
};

export type PredictionResponse = {
  home: string;
  away: string;
  neutral: boolean;
  analysis: string;
  xgboost: ModelSummary & {
    top_scorelines: Scoreline[];
    most_likely_score: string;
  };
  elo_poisson: ModelSummary;
  context: {
    home_elo: number;
    away_elo: number;
    home_last5_ppg: number | null;
    away_last5_ppg: number | null;
    h2h_count: number;
  };
};
