"""FastAPI entrypoint for live soccer predictions."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from soccer_api import predictor


class PredictionRequest(BaseModel):
    home: str
    away: str
    neutral: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.initialize()
    yield


app = FastAPI(title="WC 2026 Forecast API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/teams")
def teams() -> list[str]:
    return predictor.teams()


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    return predictor.predict(request.home, request.away, request.neutral)
