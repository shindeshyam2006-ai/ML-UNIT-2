# CIE-2 Activity 2 — Simple House Price Prediction Studio

> **Zeal Education Society's Zeal College of Engineering and Research, Narhe, Pune – 41**  
> **Department:** Artificial Intelligence & Data Science (AI & DS) · **Class:** T.Y. B.Tech (Semester V) · **Academic Year:** 2026–27  
> **Course:** Machine Learning (`ADPC509`) · **Unit II:** Supervised Learning: Regression · **Marks:** 5

An end-to-end, verified implementation of **CIE-2 Activity 2: Simple House Price Prediction**, pairing a fully executed, submission-ready **Jupyter Notebook** (`Steps 1–10`) with an editorial, Claude-inspired **FastAPI + Interactive SVG Web Studio** powered by **Google Gemini (`gemini-2.5-flash`)**, **OpenWeatherMap**, **NewsAPI**, and **SerpAPI**.

---

## Key Verified Results (Activity 2)

| Parameter / Metric | Observed Value |
| :--- | :--- |
| **Dataset & Source** | [Ames Housing Dataset (De Cock, 2011) — OpenML ID 42165](https://www.openml.org/d/42165) |
| **Raw / Cleaned Shape** | `1,460` rows × `81` columns (`0` duplicates, `0` remaining nulls after cleaning) |
| **Primary Feature ($X$) & Target ($y$)** | `GrLivArea` (sq. ft.) → `SalePrice` (USD), Pearson $r = 0.7086$ (`Id` excluded) |
| **Train / Test Split** | `80:20` (`random_state=42`) → `1,168` train / `292` unseen test (`0` overlap) |
| **Learned Linear Equation** | $\widehat{\text{SalePrice}} = 24{,}899.75 + 102.49 \times \text{GrLivArea}$ |
| **Test MAE** | **`$38,341.20`** (`21.44%` of test mean price; `38.73%` lower error than mean baseline) |
| **Test MSE** | **`3,418,946,311.18 USD²`** |
| **Test RMSE** | **`$58,471.76`** |
| **Test $R^2$ Score** | **`0.5543`** (`55.43%` variance explained vs. `-0.0009` for mean-only baseline) |
| **New Example (`1,800` sq. ft.)** | **`$209,379.87 USD`** (comfortably inside training range `334–5,642` sq. ft.) |
| **Educational 6-Feature Reference** | Test $R^2 = 0.7944$, Test MAE $= \$25,319.86$ (shows multi-factor limitations of 1-feature OLS) |

---

## Repository Structure

```text
ML-UNIT-2/
├── Activity_2_Simple_House_Price_Prediction.ipynb  # Complete executed notebook (Steps 1–10 + embedded plots)
├── data/
│   ├── ames_housing.csv                            # Raw open-source Ames Housing dataset (1,460 × 81)
│   └── ames_housing_cleaned.csv                    # Cleaned CSV with zero nulls and preserved records
├── backend/
│   ├── __init__.py
│   ├── config.py                                   # Secure .env loader & non-secret API status flags
│   ├── ml_engine.py                                # Cleaning pipeline, OLS model, metrics & comparables
│   ├── external_apis.py                            # Gemini, OpenWeatherMap, NewsAPI, SerpAPI + TTL cache
│   └── main.py                                     # FastAPI application serving REST APIs & frontend
├── frontend/
│   ├── index.html                                  # Editorial workspace UI (light/dark themes, responsive)
│   ├── css/styles.css                              # Warm alabaster & charcoal styling
│   └── js/
│       ├── charts.js                               # Interactive vector SVG charts with hover tooltips
│       └── app.js                                  # Live valuation, diagnostics, walkthrough, chat & market
├── tests/
│   └── test_app.py                                 # Automated verification of notebook, ML, routes & security
├── .env.example                                    # Safe environment template (copy to .env)
├── .gitignore                                      # Keeps .env and local caches out of Git history
├── pyproject.toml                                  # Pytest configuration
├── requirements.txt                                # Python dependencies
└── run_app.py                                      # One-command web studio launcher
```

---

## Activity 2 Submission Checklist (Items 47–91)

- **Step 1 — Dataset Selection:** Ames Housing (`https://www.openml.org/d/42165`), continuous target `SalePrice` (USD), primary input `GrLivArea` (sq. ft.).
- **Step 2 — Load & Clean:** Inspected head/tail, types, summary stats, `0` duplicates, and `19` columns with missing values; preserved all `1,460` properties via domain-aware cleaning (`'Absent'` for structural feature absence, neighborhood median for `LotFrontage`).
- **Step 3 — Feature Selection:** Justified `GrLivArea` relevance (`r = 0.7086`) and explicitly excluded `Id` (`r = -0.0219`).
- **Step 4 — Visualise Relationship:** Scatter plot highlighting positive trend, heteroscedasticity, and 2 outliers, noting *correlation does not prove causation*.
- **Step 5 — Split Data:** Reproducible `80:20` split (`1,168` train / `292` test) with verified zero overlap.
- **Step 6 — Train Linear Regression:** Recorded intercept (`$24,899.75`), slope (`$102.49 / sq. ft.`), equation, and physical interpretation.
- **Step 7 — Generate Test Predictions:** Side-by-side unseen test comparison with closest matches (e.g., `Id = 980` within `0.57%`) and largest luxury misses (`Id = 899`, `Id = 692`).
- **Step 8 — Evaluate Model:** Computed and contextualized MAE, MSE, RMSE, and $R^2$ against the mean baseline (`R² = -0.0009`).
- **Step 9 — Visualise Actual vs. Predicted:** Side-by-side `Actual vs. Predicted` plot with $y = x$ reference line and `Residuals` plot.
- **Step 10 — New Example & Conclusion:** Predicted `1,800` sq. ft. (`$209,379.87 USD`) and explained multi-factor limitations.

---

## Interactive Web Studio Features

1. **Valuation Studio:** Real-time `GrLivArea` slider & numeric input, exact formula trace, MAE typical interval, 6-feature comparison drawer, interactive SVG scatter/regression chart (hover any of the 1,460 Ames homes), and 5 nearest historical comparables.
2. **Model Diagnostics:** KPI metric cards, interactive **Actual vs. Predicted ($y = x$)** and **Residuals** SVG plots, correlation bar chart, and filterable test predictions explorer.
3. **Steps 1–10 Report:** Clean academic breakdown synced directly with the ML pipeline.
4. **AI Analyst (`Gemini 2.5 Flash`):** Conversational Q&A grounded in verified model parameters, active property valuation, and optional live market signals.
5. **Live Market & Location Intel:**
   - **OpenWeatherMap:** Current neighborhood weather and climate metrics (`Ames,US` or custom city).
   - **SerpAPI:** Real-time Google market benchmarks and median price search results.
   - **NewsAPI:** Latest housing market headlines, plus one-click **AI Market Briefing** synthesis.

---

## Setup & Running Locally

### 1. Clone and Install Dependencies
```bash
git clone https://github.com/shindeshyam2006-ai/ML-UNIT-2.git
cd ML-UNIT-2
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Keys (Optional for Live AI & Market Features)
Copy `.env.example` to `.env` and add your keys (`.env` is gitignored so credentials stay private):
```bash
cp .env.example .env
```
Variables in `.env`:
- `GEMINI_API_KEY` — Powers the AI Analyst and Executive Market Briefing
- `OPENWEATHER_API_KEY` — Powers live city climate & weather lookup
- `NEWS_API_KEY` — Powers recent housing market headlines
- `SERP_API_KEY` — Powers live housing market benchmark search

> The core ML Valuation Studio, Model Diagnostics, and Steps 1–10 Walkthrough run completely offline without external API keys.

### 3. Launch the Web Application
```bash
python run_app.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

### 4. Run the Notebook & Automated Tests
```bash
# Re-execute notebook if desired:
jupyter nbconvert --to notebook --execute --inplace Activity_2_Simple_House_Price_Prediction.ipynb

# Run test suite:
pytest -v
```

---

## REST API Endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Pipeline readiness & non-secret boolean API availability |
| `GET` | `/api/activity-summary` | Full dataset audit, split stats, regression equation, metrics & plot data |
| `POST` | `/api/predict` | Simple Linear Regression valuation + comparables + 6-feature benchmark |
| `GET` | `/api/market/weather?city=Ames,US` | OpenWeatherMap live climate snapshot |
| `GET` | `/api/market/news?query=...` | NewsAPI housing market headlines |
| `GET` | `/api/market/comps?query=...` | SerpAPI Google Search market benchmarks |
| `GET` | `/api/market/bundle` | Parallel fetch of weather + news + comps |
| `POST` | `/api/ai/analyze` | Grounded Gemini 2.5 Flash conversational analysis |

---

## Author

**Shyam Shinde** — T.Y. B.Tech (AI & DS), Zeal College of Engineering and Research, Pune  
Repository: [github.com/shindeshyam2006-ai/ML-UNIT-2](https://github.com/shindeshyam2006-ai/ML-UNIT-2)
