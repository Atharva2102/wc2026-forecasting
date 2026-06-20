"""FastAPI entrypoint for live soccer predictions."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from soccer_api import predictor


PREDICT_RATE_LIMIT_PER_MINUTE = os.getenv("PREDICT_RATE_LIMIT_PER_MINUTE", "10/minute")
PREDICT_RATE_LIMIT_PER_DAY = os.getenv("PREDICT_RATE_LIMIT_PER_DAY", "100/day")


class PredictionRequest(BaseModel):
    home: str
    away: str
    neutral: bool = True


def client_ip_key(request: Request) -> str:
    """Resolve the original visitor IP behind DigitalOcean/Cloudflare proxies."""
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",", maxsplit=1)[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


# In-memory limits are fine for one container. If the app is scaled to multiple
# containers, use a shared store such as Redis so limits apply globally.
limiter = Limiter(
    key_func=client_ip_key,
    storage_uri=os.getenv("RATE_LIMIT_STORAGE_URI", "memory://"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.initialize()
    yield


app = FastAPI(title="WC 2026 Forecast API", lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded, please wait a moment."},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/teams")
def teams() -> list[str]:
    return predictor.teams()


@app.post("/predict")
@limiter.limit(PREDICT_RATE_LIMIT_PER_DAY)
@limiter.limit(PREDICT_RATE_LIMIT_PER_MINUTE)
def predict(request: Request, payload: PredictionRequest) -> dict:
    return predictor.predict(payload.home, payload.away, payload.neutral)
