import type { PredictionResponse } from './types';

const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000';

export async function fetchTeams(): Promise<string[]> {
  const response = await fetch(`${API_URL}/teams`);
  if (!response.ok) {
    throw new Error(`Could not load teams from ${API_URL}`);
  }
  return response.json();
}

export async function predictMatch(
  home: string,
  away: string,
  neutral: boolean,
): Promise<PredictionResponse> {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ home, away, neutral }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Prediction request failed at ${API_URL}`);
  }
  return response.json();
}

export { API_URL };
