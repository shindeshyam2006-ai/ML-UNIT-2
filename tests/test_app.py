"""Automated test suite for Activity 2: Simple House Price Prediction."""

from __future__ import annotations

from pathlib import Path
import nbformat
import pandas as pd
from fastapi.testclient import TestClient

from backend.config import CLEAN_DATA_PATH, PROJECT_ROOT, RAW_DATA_PATH, Settings
from backend.main import app, get_pipeline


client = TestClient(app)


def test_dataset_and_cleaned_csv_integrity():
    assert RAW_DATA_PATH.exists()
    assert CLEAN_DATA_PATH.exists()
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_clean = pd.read_csv(CLEAN_DATA_PATH)
    assert df_raw.shape == (1460, 81)
    assert df_clean.shape == (1460, 81)
    assert int(df_clean.isna().sum().sum()) == 0
    assert int(df_clean.duplicated().sum()) == 0


def test_notebook_executed_cleanly_without_errors():
    nb_path = PROJECT_ROOT / "Activity_2_Simple_House_Price_Prediction.ipynb"
    assert nb_path.exists()
    nb = nbformat.read(nb_path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 10
    image_outputs = 0
    for cell in code_cells:
        assert cell.execution_count is not None
        for out in cell.outputs:
            assert out.output_type != "error", f"Notebook error in cell: {out}"
            if out.output_type == "display_data" and "image/png" in out.get("data", {}):
                image_outputs += 1
    assert image_outputs >= 3


def test_ml_pipeline_split_and_metrics():
    pipeline = get_pipeline()
    summary = pipeline.get_summary()
    assert summary["split"]["train_count"] == 1168
    assert summary["split"]["test_count"] == 292
    assert len(set(pipeline.X_train.index).intersection(set(pipeline.X_test.index))) == 0
    assert abs(summary["model"]["slope"] - 102.4889) < 0.05
    assert abs(summary["model"]["intercept"] - 24899.75) < 0.5
    assert 0.54 < summary["metrics"]["r2"] < 0.57
    assert summary["metrics"]["mae"] < summary["metrics"]["baseline_mae"]


def test_health_and_frontend_routes():
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "HousePrice" in r_root.text

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    data = r_health.json()
    assert data["status"] == "ok"
    assert data["dataset_rows"] == 1460
    assert set(data["apis"].keys()) == {"gemini", "openweather", "newsapi", "serpapi"}
    for val in data["apis"].values():
        assert isinstance(val, bool)


def test_predict_endpoint_valid_and_invalid():
    r_ok = client.post("/api/predict", json={"gr_liv_area": 1800})
    assert r_ok.status_code == 200
    body = r_ok.json()
    slr = body["simple_linear_regression"]
    assert abs(slr["predicted_price_usd"] - 209379.87) < 1.0
    assert slr["units"] == "USD"
    assert slr["in_training_range"] is True
    assert len(body["comparables"]) == 5

    r_bad = client.post("/api/predict", json={"gr_liv_area": -100})
    assert r_bad.status_code == 422


def test_market_and_ai_endpoints_mocked(monkeypatch):
    monkeypatch.setattr(
        "backend.main.fetch_weather",
        lambda city, s: {"available": True, "provider": "OpenWeatherMap", "city": city, "temp_c": 12.0},
    )
    monkeypatch.setattr(
        "backend.main.fetch_real_estate_news",
        lambda q, s, page_size=5: {"available": True, "provider": "NewsAPI", "articles": [{"title": "Market Update"}]},
    )
    monkeypatch.setattr(
        "backend.main.fetch_market_comps",
        lambda q, s, num=5: {"available": True, "provider": "SerpAPI", "results": [{"title": "Ames Median"}]},
    )
    monkeypatch.setattr(
        "backend.main.ask_gemini_analyst",
        lambda prompt, ctx, s, history=None: {
            "available": True,
            "provider": "Google Gemini (gemini-2.5-flash)",
            "reply": "Grounded analysis for 1,800 sq. ft.",
        },
    )

    r_bundle = client.get("/api/market/bundle?city=Ames,US")
    assert r_bundle.status_code == 200
    b = r_bundle.json()
    assert b["weather"]["temp_c"] == 12.0
    assert len(b["news"]["articles"]) == 1
    assert len(b["comps"]["results"]) == 1

    r_ai = client.post("/api/ai/analyze", json={"prompt": "Explain R2"})
    assert r_ai.status_code == 200
    assert "1,800" in r_ai.json()["reply"]


def test_no_secret_keys_in_tracked_files():
    settings = Settings.from_env()
    secrets = [
        k
        for k in [
            settings.gemini_api_key,
            settings.serp_api_key,
            settings.openweather_api_key,
            settings.news_api_key,
        ]
        if k
    ]
    gitignore_text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore_text

    if not secrets:
        return

    for folder in ["backend", "frontend", "tests"]:
        for path in (PROJECT_ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".html", ".css", ".js", ".md"}:
                content = path.read_text(encoding="utf-8", errors="ignore")
                for secret in secrets:
                    assert secret not in content, f"Secret leaked in {path}"


def test_frontend_dom_ids_match_html():
    import re
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    js = (
        (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        + (PROJECT_ROOT / "frontend" / "js" / "charts.js").read_text(encoding="utf-8")
    )
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    js_ids = set(re.findall(r'getElementById\("([^"]+)"\)', js))
    missing = js_ids - html_ids
    assert not missing, f"JS references missing DOM IDs: {missing}"

