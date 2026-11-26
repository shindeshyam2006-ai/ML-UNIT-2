"""FastAPI server for CIE-2 Activity 2: Simple House Price Prediction Studio."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import FRONTEND_DIR, get_settings
from backend.external_apis import (
    ask_gemini_analyst,
    fetch_market_comps,
    fetch_real_estate_news,
    fetch_weather,
)
from backend.ml_engine import HousingMLPipeline

_pipeline: HousingMLPipeline | None = None


def get_pipeline() -> HousingMLPipeline:
    """Lazy singleton accessor for the trained Activity 2 ML pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = HousingMLPipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_pipeline()
    yield


app = FastAPI(
    title="ML Unit 2 — Activity 2: Simple House Price Prediction",
    description="Interactive Regression Studio, Notebook Walkthrough & Live Market Intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    gr_liv_area: float = Field(..., ge=100, le=15000, description="Above-grade living area in sq. ft.")
    overall_qual: int | None = Field(default=None, ge=1, le=10)
    total_bsmt_sf: float | None = Field(default=None, ge=0, le=10000)
    garage_cars: int | None = Field(default=None, ge=0, le=6)
    full_bath: int | None = Field(default=None, ge=0, le=6)
    year_built: int | None = Field(default=None, ge=1850, le=2026)


class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)


class AIAnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=3000)
    prediction_context: dict[str, Any] | None = None
    market_context: dict[str, Any] | None = None
    history: list[ChatTurn] = Field(default_factory=list)


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    pipeline = get_pipeline()
    settings = get_settings()
    summary = pipeline.get_summary()
    return {
        "status": "ok",
        "model_ready": True,
        "dataset_rows": summary["dataset"]["rows"],
        "dataset_columns": summary["dataset"]["columns"],
        "r2_test": summary["metrics"]["r2"],
        "apis": settings.api_status(),
    }


@app.get("/api/activity-summary")
def activity_summary() -> dict[str, Any]:
    return get_pipeline().get_summary()


@app.post("/api/predict")
def predict_house_price(payload: PredictRequest) -> dict[str, Any]:
    return get_pipeline().predict(
        gr_liv_area=payload.gr_liv_area,
        overall_qual=payload.overall_qual,
        total_bsmt_sf=payload.total_bsmt_sf,
        garage_cars=payload.garage_cars,
        full_bath=payload.full_bath,
        year_built=payload.year_built,
    )


@app.get("/api/market/weather")
def market_weather(city: str = Query(default="Ames,US", min_length=2, max_length=80)) -> dict[str, Any]:
    return fetch_weather(city, get_settings())


@app.get("/api/market/news")
def market_news(
    query: str = Query(default="housing market home prices", min_length=2, max_length=120),
    page_size: int = Query(default=5, ge=1, le=10),
) -> dict[str, Any]:
    return fetch_real_estate_news(query, get_settings(), page_size=page_size)


@app.get("/api/market/comps")
def market_comps(
    query: str = Query(default="Ames Iowa median home price real estate market", min_length=2, max_length=160),
    num: int = Query(default=5, ge=1, le=8),
) -> dict[str, Any]:
    return fetch_market_comps(query, get_settings(), num=num)


@app.get("/api/market/bundle")
def market_bundle(
    city: str = Query(default="Ames,US", min_length=2, max_length=80),
    news_query: str = Query(default="US housing market home prices", min_length=2, max_length=120),
    comps_query: str = Query(default="Ames Iowa median home price real estate market", min_length=2, max_length=160),
) -> dict[str, Any]:
    settings = get_settings()
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_weather = pool.submit(fetch_weather, city, settings)
        f_news = pool.submit(fetch_real_estate_news, news_query, settings, 5)
        f_comps = pool.submit(fetch_market_comps, comps_query, settings, 5)
        return {
            "weather": f_weather.result(),
            "news": f_news.result(),
            "comps": f_comps.result(),
        }


@app.post("/api/ai/analyze")
def ai_analyze(payload: AIAnalyzeRequest) -> dict[str, Any]:
    summary = get_pipeline().get_summary()
    settings = get_settings()

    grounded_context = {
        "dataset": {
            "name": summary["dataset"]["name"],
            "location": summary["dataset"]["location"],
            "rows": summary["dataset"]["rows"],
            "columns": summary["dataset"]["columns"],
            "primary_feature": summary["dataset"]["primary_feature"],
            "target": summary["dataset"]["target"],
            "area_stats": summary["dataset"]["area_stats"],
            "price_stats": summary["dataset"]["price_stats"],
        },
        "split": summary["split"],
        "model": summary["model"],
        "metrics": summary["metrics"],
        "active_prediction": payload.prediction_context,
        "live_market_context": payload.market_context,
    }
    history = [{"role": t.role, "content": t.content} for t in payload.history]
    result = ask_gemini_analyst(payload.prompt, grounded_context, settings, history=history)
    if not result["available"]:
        raise HTTPException(status_code=503, detail=result.get("error", "Gemini AI unavailable"))
    return result


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_frontend() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)
