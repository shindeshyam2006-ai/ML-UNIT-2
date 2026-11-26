"""External API integrations (Gemini, OpenWeatherMap, NewsAPI, SerpAPI) with TTL caching."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import requests

from backend.config import Settings


@dataclass
class TTLCache:
    """Simple in-memory TTL cache to avoid redundant external API calls."""

    ttl_seconds: int = 300
    _store: dict[str, tuple[float, Any]] = field(default_factory=dict)

    def get(self, key: str) -> Any | None:
        item = self._store.get(key)
        if not item:
            return None
        expires_at, value = item
        if time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time() + self.ttl_seconds, value)


_MARKET_CACHE = TTLCache(ttl_seconds=300)


def fetch_weather(city: str, settings: Settings) -> dict[str, Any]:
    """Fetch live weather and climate snapshot for a city via OpenWeatherMap."""
    city_clean = (city or "Ames,US").strip()
    if not settings.openweather_api_key:
        return {
            "available": False,
            "provider": "OpenWeatherMap",
            "error": "OPENWEATHER_API_KEY is not configured in .env.",
        }

    cache_key = f"weather:{city_clean.lower()}"
    cached = _MARKET_CACHE.get(cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city_clean,
                "appid": settings.openweather_api_key,
                "units": "metric",
            },
            timeout=10,
        )
        data = resp.json()
        if resp.status_code != 200:
            return {
                "available": False,
                "provider": "OpenWeatherMap",
                "error": data.get("message", f"HTTP {resp.status_code}"),
            }

        result = {
            "available": True,
            "provider": "OpenWeatherMap",
            "cached": False,
            "city": data.get("name", city_clean),
            "country": data.get("sys", {}).get("country", ""),
            "temp_c": round(float(data["main"]["temp"]), 1),
            "feels_like_c": round(float(data["main"]["feels_like"]), 1),
            "humidity_pct": int(data["main"]["humidity"]),
            "wind_mps": round(float(data.get("wind", {}).get("speed", 0)), 1),
            "condition": data["weather"][0]["main"] if data.get("weather") else "Clear",
            "description": data["weather"][0]["description"].title() if data.get("weather") else "",
            "coord": data.get("coord", {}),
        }
        _MARKET_CACHE.set(cache_key, result)
        return result
    except requests.RequestException as exc:
        return {"available": False, "provider": "OpenWeatherMap", "error": str(exc)}


def fetch_real_estate_news(query: str, settings: Settings, page_size: int = 5) -> dict[str, Any]:
    """Fetch recent real-estate and housing market headlines via NewsAPI."""
    q = (query or "housing market home prices").strip()
    if not settings.news_api_key:
        return {
            "available": False,
            "provider": "NewsAPI",
            "error": "NEWS_API_KEY is not configured in .env.",
            "articles": [],
        }

    cache_key = f"news:{q.lower()}:{page_size}"
    cached = _MARKET_CACHE.get(cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": q,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(max(int(page_size), 1), 10),
                "apiKey": settings.news_api_key,
            },
            timeout=10,
        )
        data = resp.json()
        if resp.status_code != 200 or data.get("status") != "ok":
            return {
                "available": False,
                "provider": "NewsAPI",
                "error": data.get("message", f"HTTP {resp.status_code}"),
                "articles": [],
            }

        articles = [
            {
                "title": art.get("title", ""),
                "source": (art.get("source") or {}).get("name", "Unknown"),
                "published_at": (art.get("publishedAt") or "")[:10],
                "description": art.get("description") or "",
                "url": art.get("url") or "",
            }
            for art in data.get("articles", [])
            if art.get("title") and art.get("title") != "[Removed]"
        ]
        result = {
            "available": True,
            "provider": "NewsAPI",
            "cached": False,
            "query": q,
            "articles": articles,
        }
        _MARKET_CACHE.set(cache_key, result)
        return result
    except requests.RequestException as exc:
        return {"available": False, "provider": "NewsAPI", "error": str(exc), "articles": []}


def fetch_market_comps(query: str, settings: Settings, num: int = 5) -> dict[str, Any]:
    """Search live housing market benchmarks and comparable insights via SerpAPI."""
    q = (query or "Ames Iowa median home price real estate market").strip()
    if not settings.serp_api_key:
        return {
            "available": False,
            "provider": "SerpAPI",
            "error": "SERP_API_KEY is not configured in .env.",
            "results": [],
        }

    cache_key = f"serp:{q.lower()}:{num}"
    cached = _MARKET_CACHE.get(cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": q,
                "num": min(max(int(num), 1), 8),
                "api_key": settings.serp_api_key,
            },
            timeout=12,
        )
        data = resp.json()
        if resp.status_code != 200 or "error" in data:
            return {
                "available": False,
                "provider": "SerpAPI",
                "error": data.get("error", f"HTTP {resp.status_code}"),
                "results": [],
            }

        results = [
            {
                "position": item.get("position"),
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "displayed_link": item.get("displayed_link", ""),
                "snippet": item.get("snippet", ""),
            }
            for item in data.get("organic_results", [])[:num]
        ]

        answer_box = data.get("answer_box") or {}
        highlight = answer_box.get("snippet") or answer_box.get("answer") or ""

        result = {
            "available": True,
            "provider": "SerpAPI",
            "cached": False,
            "query": q,
            "highlight": highlight,
            "results": results,
        }
        _MARKET_CACHE.set(cache_key, result)
        return result
    except requests.RequestException as exc:
        return {"available": False, "provider": "SerpAPI", "error": str(exc), "results": []}


def ask_gemini_analyst(
    prompt: str,
    model_context: dict[str, Any],
    settings: Settings,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Generate a grounded property & ML analysis response via Gemini API."""
    if not settings.gemini_api_key:
        return {
            "available": False,
            "provider": "Google Gemini",
            "error": "GEMINI_API_KEY is not configured in .env.",
            "reply": "",
        }

    system_instruction = (
        "You are an expert Machine Learning & Real Estate Valuation Analyst assisting a T.Y. B.Tech AI & DS student "
        "presenting CIE-2 Activity 2 (Simple House Price Prediction, Course ADPC509 at Zeal College of Engineering & Research). "
        "Answer clearly, concisely, and accurately using ONLY the verified ground-truth numbers in the provided JSON context. "
        "Do not invent metrics or dataset figures. Format your response cleanly with Markdown headings, bullet points, and clear USD/sq.ft. units.\n\n"
        f"VERIFIED MODEL & CONTEXT JSON:\n{model_context}"
    )

    contents: list[dict[str, Any]] = []
    for turn in (history or [])[-6:]:
        role = "model" if turn.get("role") == "assistant" else "user"
        text = str(turn.get("content", "")).strip()
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})

    contents.append({"role": "user", "parts": [{"text": prompt.strip()}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent"
    try:
        resp = requests.post(
            url,
            params={"key": settings.gemini_api_key},
            json={
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.35,
                    "maxOutputTokens": 1200,
                },
            },
            timeout=25,
        )
        data = resp.json()
        if resp.status_code != 200:
            err_msg = data.get("error", {}).get("message", f"HTTP {resp.status_code}")
            return {
                "available": False,
                "provider": f"Google Gemini ({settings.gemini_model})",
                "error": err_msg,
                "reply": "",
            }

        candidates = data.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        reply_text = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
        if not reply_text:
            return {
                "available": False,
                "provider": f"Google Gemini ({settings.gemini_model})",
                "error": "Empty response received from Gemini API.",
                "reply": "",
            }

        return {
            "available": True,
            "provider": f"Google Gemini ({settings.gemini_model})",
            "model": settings.gemini_model,
            "reply": reply_text,
        }
    except requests.RequestException as exc:
        return {
            "available": False,
            "provider": f"Google Gemini ({settings.gemini_model})",
            "error": str(exc),
            "reply": "",
        }
