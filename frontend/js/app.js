/* Frontend controller for CIE-2 Activity 2: Simple House Price Prediction Studio */

(() => {
  const state = {
    summary: null,
    prediction: null,
    market: null,
    chatHistory: [],
    currentView: "studio",
    debounceTimer: null,
  };

  const VIEW_TITLES = {
    studio: "Property Valuation Studio",
    diagnostics: "Model Evaluation & Diagnostics",
    walkthrough: "Activity 2 Academic Walkthrough (Steps 1–10)",
    analyst: "Gemini AI Property & ML Analyst",
    market: "Live Market & Location Intelligence",
  };

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderSimpleMarkdown(raw) {
    const escaped = escapeHtml(raw);
    const lines = escaped.split(/\r?\n/);
    const out = [];
    let inList = false;

    const formatInline = (text) =>
      text
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/\*([^*]+)\*/g, "<em>$1</em>");

    for (const line of lines) {
      const trimmed = line.trim();
      if (/^#{1,4}\s+/.test(trimmed)) {
        if (inList) { out.push("</ul>"); inList = false; }
        const level = Math.min(4, trimmed.match(/^#+/)[0].length);
        out.push(`<h${level}>${formatInline(trimmed.replace(/^#{1,4}\s+/, ""))}</h${level}>`);
      } else if (/^[-*]\s+/.test(trimmed)) {
        if (!inList) { out.push("<ul>"); inList = true; }
        out.push(`<li>${formatInline(trimmed.replace(/^[-*]\s+/, ""))}</li>`);
      } else if (trimmed === "") {
        if (inList) { out.push("</ul>"); inList = false; }
      } else {
        if (inList) { out.push("</ul>"); inList = false; }
        out.push(`<p>${formatInline(trimmed)}</p>`);
      }
    }
    if (inList) out.push("</ul>");
    return out.join("");
  }

  function fmtUsd(val) {
    return "$" + Number(val).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function fmtUsd0(val) {
    return "$" + Number(val).toLocaleString("en-US", { maximumFractionDigits: 0 });
  }

  async function apiFetch(url, options = {}) {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || data.error || `HTTP ${res.status}`);
    }
    return data;
  }

  function showGlobalError(msg) {
    const banner = document.getElementById("globalErrorBanner");
    if (!banner) return;
    if (!msg) {
      banner.classList.add("hidden");
      banner.textContent = "";
    } else {
      banner.textContent = msg;
      banner.classList.remove("hidden");
    }
  }

  function switchView(viewName) {
    state.currentView = viewName;
    document.querySelectorAll(".nav-item").forEach((btn) => {
      const active = btn.dataset.view === viewName;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", String(active));
    });
    document.querySelectorAll(".view-pane").forEach((pane) => {
      pane.classList.toggle("active", pane.id === `view-${viewName}`);
    });
    document.getElementById("viewTitle").textContent = VIEW_TITLES[viewName] || "Studio";
    closeMobileSidebar();

    if (viewName === "market" && !state.market) {
      loadMarketBundle();
    }
  }

  function closeMobileSidebar() {
    document.getElementById("sidebar")?.classList.remove("open");
    document.getElementById("sidebarBackdrop")?.classList.remove("visible");
  }

  function initTheme() {
    const saved = localStorage.getItem("ml_unit2_theme") || "light";
    applyTheme(saved);
    document.getElementById("themeToggleBtn")?.addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      applyTheme(next);
      localStorage.setItem("ml_unit2_theme", next);
    });
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const icon = document.getElementById("themeIcon");
    const label = document.getElementById("themeLabel");
    if (icon) icon.textContent = theme === "dark" ? "☀" : "☾";
    if (label) label.textContent = theme === "dark" ? "Light appearance" : "Dark appearance";
  }

  function updateApiStatus(health) {
    const grid = document.getElementById("apiStatusGrid");
    if (!grid) return;
    grid.querySelectorAll(".status-chip").forEach((chip) => {
      const apiName = chip.dataset.api;
      if (!apiName) {
        chip.classList.add("online");
      } else if (health.apis && health.apis[apiName]) {
        chip.classList.add("online");
      }
    });
  }

  function renderStudioScatter() {
    if (!state.summary) return;
    const box = document.getElementById("scatterChartBox");
    const live = state.prediction
      ? {
          area: state.prediction.input.gr_liv_area,
          price: state.prediction.simple_linear_regression.predicted_price_usd,
        }
      : null;
    ChartKit.renderScatterRegression(box, state.summary.scatter_points, state.summary.model, live);
  }

  function renderPredictionUI(pred) {
    state.prediction = pred;
    const slr = pred.simple_linear_regression;
    document.getElementById("predictedPriceDisplay").textContent = fmtUsd(slr.predicted_price_usd);
    document.getElementById("formulaTraceDisplay").textContent = slr.formula_evaluated;
    document.getElementById("maeBandDisplay").textContent = `${fmtUsd0(slr.mae_band[0])} – ${fmtUsd0(slr.mae_band[1])}`;
    document.getElementById("percentileDisplay").textContent = `${slr.area_percentile}th / ${slr.price_percentile}th pct`;
    document.getElementById("multiRefDisplay").textContent = fmtUsd(pred.multi_feature_reference.predicted_price_usd);

    const rangeBadge = document.getElementById("areaRangeBadge");
    if (slr.in_training_range) {
      rangeBadge.textContent = "In training range (334–5,642)";
      rangeBadge.classList.remove("warn");
    } else {
      rangeBadge.textContent = "Extrapolation outside training range";
      rangeBadge.classList.add("warn");
    }

    const tbody = document.querySelector("#comparablesTable tbody");
    tbody.innerHTML = pred.comparables
      .map(
        (c) => `
        <tr>
          <td class="mono">#${c.id}</td>
          <td>${escapeHtml(c.neighborhood)}</td>
          <td class="mono">${Number(c.gr_liv_area).toLocaleString()} sq.ft.</td>
          <td class="mono">${c.overall_qual}/10</td>
          <td class="mono">${c.year_built}</td>
          <td class="mono"><strong>${fmtUsd0(c.actual_price)}</strong></td>
        </tr>`
      )
      .join("");

    document.getElementById("chatContextPill").textContent =
      `Context: ${Number(pred.input.gr_liv_area).toLocaleString()} sq.ft. → ${fmtUsd(slr.predicted_price_usd)}`;

    renderStudioScatter();
  }

  async function runPrediction() {
    const areaVal = parseFloat(document.getElementById("areaInput").value);
    if (!Number.isFinite(areaVal) || areaVal < 100 || areaVal > 15000) return;

    const payload = {
      gr_liv_area: areaVal,
      overall_qual: parseInt(document.getElementById("qualInput").value || "6", 10),
      total_bsmt_sf: parseFloat(document.getElementById("bsmtInput").value || "991"),
      garage_cars: parseInt(document.getElementById("garageInput").value || "2", 10),
      full_bath: parseInt(document.getElementById("bathInput").value || "2", 10),
      year_built: parseInt(document.getElementById("yearInput").value || "1973", 10),
    };

    try {
      const pred = await apiFetch("/api/predict", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      renderPredictionUI(pred);
    } catch (err) {
      showGlobalError(`Prediction failed: ${err.message}`);
    }
  }

  function schedulePrediction() {
    clearTimeout(state.debounceTimer);
    state.debounceTimer = setTimeout(runPrediction, 120);
  }

  function renderDiagnostics(summary) {
    const m = summary.metrics;
    const kpiRow = document.getElementById("diagnosticsKpiRow");
    const cards = [
      { label: "Test R² Score", value: m.r2.toFixed(4), note: "55.43% variance explained vs -0.0009 baseline" },
      { label: "Mean Absolute Error", value: fmtUsd(m.mae), note: `${m.mae_pct_of_mean}% of mean test price (${m.mae_improvement_pct}% better)` },
      { label: "Root Mean Squared Error", value: fmtUsd(m.rmse), note: "Same USD units; highlights luxury outliers" },
      { label: "Train / Test Split", value: `${summary.split.train_count} / ${summary.split.test_count}`, note: `80:20 reproducible split (seed=${summary.split.random_state})` },
    ];
    kpiRow.innerHTML = cards
      .map(
        (c) => `
      <div class="kpi-card">
        <div class="kpi-label">${escapeHtml(c.label)}</div>
        <div class="kpi-value">${escapeHtml(c.value)}</div>
        <div class="kpi-note">${escapeHtml(c.note)}</div>
      </div>`
      )
      .join("");

    ChartKit.renderActualVsPredicted(document.getElementById("actualPredChartBox"), summary.test_points);
    ChartKit.renderResiduals(document.getElementById("residualsChartBox"), summary.test_points);
    ChartKit.renderCorrelations(document.getElementById("correlationChartBox"), summary.dataset.top_correlations);

    const bmBody = document.querySelector("#benchmarkTable tbody");
    bmBody.innerHTML = `
      <tr>
        <td>Mean-Only Baseline</td>
        <td class="mono">${fmtUsd(m.baseline_mae)}</td>
        <td class="mono">${fmtUsd(m.baseline_rmse)}</td>
        <td class="mono">-0.0009</td>
      </tr>
      <tr>
        <td><strong>Simple Linear Regression (GrLivArea)</strong></td>
        <td class="mono"><strong>${fmtUsd(m.mae)}</strong></td>
        <td class="mono"><strong>${fmtUsd(m.rmse)}</strong></td>
        <td class="mono"><strong>${m.r2.toFixed(4)}</strong></td>
      </tr>
      <tr>
        <td>6-Feature Reference (Educational)</td>
        <td class="mono">${fmtUsd(m.multi_feature_benchmark.mae)}</td>
        <td class="mono">${fmtUsd(m.multi_feature_benchmark.rmse)}</td>
        <td class="mono">${m.multi_feature_benchmark.r2.toFixed(4)}</td>
      </tr>`;

    renderTestTable("sample");
  }

  function renderTestTable(filterMode) {
    if (!state.summary) return;
    const rows =
      filterMode === "closest"
        ? state.summary.closest_predictions
        : filterMode === "largest"
        ? state.summary.largest_errors
        : state.summary.test_predictions_sample;

    const tbody = document.querySelector("#testPredictionsTable tbody");
    tbody.innerHTML = rows
      .map(
        (r) => `
      <tr>
        <td class="mono">#${r.id}</td>
        <td>${escapeHtml(r.neighborhood)}</td>
        <td class="mono">${r.overall_qual}</td>
        <td class="mono">${Number(r.gr_liv_area).toLocaleString()}</td>
        <td class="mono">${fmtUsd0(r.actual_price)}</td>
        <td class="mono">${fmtUsd0(r.predicted_price)}</td>
        <td class="mono">${r.residual >= 0 ? "+" : ""}${fmtUsd0(r.residual)}</td>
        <td class="mono">${r.pct_error.toFixed(2)}%</td>
      </tr>`
      )
      .join("");
  }

  function renderWalkthrough(summary) {
    const d = summary.dataset;
    const s = summary.split;
    const mod = summary.model;
    const met = summary.metrics;

    const steps = [
      {
        num: "Step 01",
        title: "Select and Understand the Dataset",
        bullets: [
          `<strong>Dataset:</strong> ${escapeHtml(d.name)} — Source: <a href="${escapeHtml(d.source_url)}" target="_blank" rel="noopener">${escapeHtml(d.source_url)}</a>`,
          `<strong>Target (y):</strong> <code>${escapeHtml(d.target)}</code> — continuous numerical sale price in USD ($${d.price_stats.min.toLocaleString()} to $${d.price_stats.max.toLocaleString()}).`,
          `<strong>Feature (X):</strong> <code>${escapeHtml(d.primary_feature)}</code> — above-grade living area in sq. ft. (${d.area_stats.min} to ${d.area_stats.max} sq. ft.).`,
        ],
      },
      {
        num: "Step 02",
        title: "Load and Inspect Data Quality",
        bullets: [
          `Loaded <strong>${d.rows.toLocaleString()} rows × ${d.columns} columns</strong>; inspected head/tail samples and data types.`,
          `Detected <strong>${d.duplicates} duplicate rows</strong> and <strong>${d.total_missing_before.toLocaleString()}</strong> missing values across <strong>${d.missing_columns_count}</strong> columns (structural absences like <code>PoolQC</code>, <code>Alley</code>, <code>Garage*</code>, <code>Bsmt*</code> plus <code>LotFrontage</code>).`,
          `Applied domain-aware cleaning (<code>'Absent'</code> for structural absences, neighborhood median for <code>LotFrontage</code>) → <strong>${d.total_missing_after}</strong> remaining nulls.`,
        ],
      },
      {
        num: "Step 03",
        title: "Select Input Feature & Exclude Record ID",
        bullets: [
          `Selected <code>GrLivArea</code> (Pearson correlation <code>r = 0.7086</code>) as the strongest physical-size predictor of <code>SalePrice</code>.`,
          `Explicitly excluded <code>Id</code> (correlation <code>${d.id_correlation}</code>) because sequential record IDs carry zero causal property value.`,
        ],
      },
      {
        num: "Step 04",
        title: "Visualise the Relationship",
        bullets: [
          `Scatter plot shows a strong positive linear trend between <code>GrLivArea</code> and <code>SalePrice</code>, with widening variance above 2,200 sq. ft. and 2 high-area/low-price outliers.`,
          `<strong>Key Principle:</strong> Correlation / visible trend does not by itself prove causation.`,
        ],
      },
      {
        num: "Step 05",
        title: "Split into Training & Testing Sets",
        bullets: [
          `Divided data <strong>${s.ratio}</strong> (<code>random_state=${s.random_state}</code>): <strong>${s.train_count.toLocaleString()}</strong> training homes and <strong>${s.test_count.toLocaleString()}</strong> unseen test homes.`,
          `Verified zero index overlap so test records remain completely isolated from parameter fitting.`,
        ],
      },
      {
        num: "Step 06",
        title: "Train Simple Linear Regression",
        bullets: [
          `Fitted OLS <code>LinearRegression</code> on training set: Intercept <code>${fmtUsd(mod.intercept)}</code>, Slope <code>${fmtUsd(mod.slope)} / sq. ft.</code>`,
          `<strong>Equation:</strong> <code>${escapeHtml(mod.equation)}</code> — each additional sq. ft. adds ~$102.49 to predicted price.`,
        ],
      },
      {
        num: "Step 07",
        title: "Generate Unseen Test Predictions",
        bullets: [
          `Predicted all <strong>${s.test_count}</strong> test properties; closest matches (e.g., ID #980 at 1,121 sq.ft.) lie within 0.57% of actual price.`,
          `Largest misses occur on custom luxury homes (e.g., ID #899, OverallQual 9) where quality premiums go beyond floor area.`,
        ],
      },
      {
        num: "Step 08",
        title: "Evaluate Regression Metrics",
        bullets: [
          `<strong>MAE:</strong> ${fmtUsd(met.mae)} (${met.mae_pct_of_mean}% of mean test price, ${met.mae_improvement_pct}% lower than mean baseline).`,
          `<strong>MSE:</strong> ${Number(met.mse).toLocaleString()} USD² | <strong>RMSE:</strong> ${fmtUsd(met.rmse)} (in USD units).`,
          `<strong>R²:</strong> ${met.r2} (explains 55.43% of unseen price variance vs. <code>-0.0009</code> for predicting training mean).`,
        ],
      },
      {
        num: "Step 09",
        title: "Visualise Actual vs. Predicted & Residuals",
        bullets: [
          `Mid-market homes ($100k–$240k) cluster symmetrically around the 45° <code>y = x</code> reference line.`,
          `Residual fanning above $350k demonstrates systematic under-prediction of luxury finishes.`,
        ],
      },
      {
        num: "Step 10",
        title: "New Example Prediction & Conclusion",
        bullets: [
          `New in-range input <code>GrLivArea = 1,800 sq. ft.</code> predicts <strong>$209,379.87 USD</strong>.`,
          `Conclusion: Floor area captures over half of price variance, while multi-feature context (quality, basement, garage, neighborhood) boosts test R² to <code>${met.multi_feature_benchmark.r2}</code>.`,
        ],
      },
    ];

    document.getElementById("stepsTimeline").innerHTML = steps
      .map(
        (st) => `
      <article class="step-card">
        <div class="step-header">
          <span class="step-num">${escapeHtml(st.num)}</span>
          <h3 class="step-title">${escapeHtml(st.title)}</h3>
        </div>
        <div class="step-body">
          <ul>${st.bullets.map((b) => `<li>${b}</li>`).join("")}</ul>
        </div>
      </article>`
      )
      .join("");
  }

  function appendChatMessage(role, content) {
    const container = document.getElementById("chatMessages");
    const welcome = container.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const row = document.createElement("div");
    row.className = `msg-row ${role}`;
    const roleLabel = document.createElement("div");
    roleLabel.className = "msg-role";
    roleLabel.textContent = role === "user" ? "You" : "Gemini Analyst";

    const bubble = document.createElement("div");
    bubble.className = "msg-bubble markdown-body";
    if (role === "user") {
      bubble.textContent = content;
    } else {
      bubble.innerHTML = renderSimpleMarkdown(content);
    }

    row.appendChild(roleLabel);
    row.appendChild(bubble);
    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
    return row;
  }

  async function sendChatPrompt(text) {
    const prompt = text.trim();
    if (!prompt) return;

    const sendBtn = document.getElementById("sendChatBtn");
    const input = document.getElementById("chatInput");
    input.value = "";
    sendBtn.disabled = true;
    sendBtn.textContent = "Thinking...";

    appendChatMessage("user", prompt);
    const loadingRow = appendChatMessage("assistant", "_Analyzing verified model metrics and context..._");

    const includeMarket = document.getElementById("includeMarketToggle")?.checked;
    try {
      const data = await apiFetch("/api/ai/analyze", {
        method: "POST",
        body: JSON.stringify({
          prompt,
          prediction_context: state.prediction,
          market_context: includeMarket ? state.market : null,
          history: state.chatHistory,
        }),
      });
      loadingRow.remove();
      appendChatMessage("assistant", data.reply);
      state.chatHistory.push({ role: "user", content: prompt });
      state.chatHistory.push({ role: "assistant", content: data.reply });
    } catch (err) {
      loadingRow.remove();
      appendChatMessage("assistant", `**Error:** ${err.message}`);
    } finally {
      sendBtn.disabled = false;
      sendBtn.textContent = "Send";
    }
  }

  function renderMarketBundle(bundle) {
    state.market = bundle;

    // Weather
    const wBox = document.getElementById("weatherCardBody");
    const w = bundle.weather;
    if (!w || !w.available) {
      wBox.innerHTML = `<p class="muted">${escapeHtml(w?.error || "Weather unavailable.")}</p>`;
    } else {
      wBox.innerHTML = `
        <div class="weather-hero">
          <div>
            <strong>${escapeHtml(w.city)}${w.country ? ", " + escapeHtml(w.country) : ""}</strong>
            <div class="muted">${escapeHtml(w.description || w.condition)}</div>
          </div>
          <div class="weather-temp">${escapeHtml(w.temp_c)}°C</div>
        </div>
        <div class="weather-metrics">
          <div class="weather-stat"><span>Feels Like</span><strong>${escapeHtml(w.feels_like_c)}°C</strong></div>
          <div class="weather-stat"><span>Humidity</span><strong>${escapeHtml(w.humidity_pct)}%</strong></div>
          <div class="weather-stat"><span>Wind</span><strong>${escapeHtml(w.wind_mps)} m/s</strong></div>
        </div>`;
    }

    // SerpAPI Comps
    const cBox = document.getElementById("compsCardBody");
    const c = bundle.comps;
    if (!c || !c.available) {
      cBox.innerHTML = `<p class="muted">${escapeHtml(c?.error || "Market benchmarks unavailable.")}</p>`;
    } else {
      const highlight = c.highlight ? `<div class="comp-highlight">${escapeHtml(c.highlight)}</div>` : "";
      const items = (c.results || [])
        .map(
          (r) => `
        <div class="comp-item">
          <a href="${escapeHtml(r.link)}" target="_blank" rel="noopener">${escapeHtml(r.title)}</a>
          <div class="comp-snippet">${escapeHtml(r.snippet)}</div>
        </div>`
        )
        .join("");
      cBox.innerHTML = highlight + (items || `<p class="muted">No organic results found.</p>`);
    }

    // NewsAPI
    const nBox = document.getElementById("newsCardBody");
    const n = bundle.news;
    if (!n || !n.available) {
      nBox.innerHTML = `<p class="muted">${escapeHtml(n?.error || "News feed unavailable.")}</p>`;
    } else {
      nBox.innerHTML = (n.articles || [])
        .map(
          (a) => `
        <article class="news-card">
          <div class="news-source">${escapeHtml(a.source)} · ${escapeHtml(a.published_at)}</div>
          <a class="news-title" href="${escapeHtml(a.url)}" target="_blank" rel="noopener">${escapeHtml(a.title)}</a>
          <div class="news-desc">${escapeHtml(a.description)}</div>
        </article>`
        )
        .join("");
    }
  }

  async function loadMarketBundle() {
    const btn = document.getElementById("refreshMarketBtn");
    const city = document.getElementById("marketCityInput").value.trim() || "Ames,US";
    const q = document.getElementById("marketQueryInput").value.trim() || "Ames Iowa housing market median home price";
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Fetching...";
    }
    try {
      const bundle = await apiFetch(
        `/api/market/bundle?city=${encodeURIComponent(city)}&news_query=${encodeURIComponent("housing market home prices")}&comps_query=${encodeURIComponent(q)}`
      );
      renderMarketBundle(bundle);
    } catch (err) {
      showGlobalError(`Live market signals error: ${err.message}`);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = "Fetch Live Signals";
      }
    }
  }

  async function generateMarketBriefing() {
    const btn = document.getElementById("briefingBtn");
    const card = document.getElementById("marketBriefingCard");
    const body = document.getElementById("marketBriefingBody");
    btn.disabled = true;
    btn.textContent = "Synthesizing...";
    try {
      if (!state.market) await loadMarketBundle();
      const data = await apiFetch("/api/ai/analyze", {
        method: "POST",
        body: JSON.stringify({
          prompt:
            "Write a concise executive market briefing connecting our Ames Simple Linear Regression valuation " +
            "with the live weather, SerpAPI real-estate benchmarks, and recent NewsAPI headlines in our context.",
          prediction_context: state.prediction,
          market_context: state.market,
        }),
      });
      body.innerHTML = renderSimpleMarkdown(data.reply);
      card.classList.remove("hidden");
    } catch (err) {
      showGlobalError(`Briefing failed: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "AI Market Briefing";
    }
  }

  function bindEvents() {
    document.querySelectorAll(".nav-item").forEach((btn) => {
      btn.addEventListener("click", () => switchView(btn.dataset.view));
    });

    document.getElementById("openSidebarBtn")?.addEventListener("click", () => {
      document.getElementById("sidebar").classList.add("open");
      document.getElementById("sidebarBackdrop").classList.add("visible");
    });
    document.getElementById("closeSidebarBtn")?.addEventListener("click", closeMobileSidebar);
    document.getElementById("sidebarBackdrop")?.addEventListener("click", closeMobileSidebar);

    const areaInput = document.getElementById("areaInput");
    const areaSlider = document.getElementById("areaSlider");

    areaInput.addEventListener("input", () => {
      const v = Math.min(5000, Math.max(400, Number(areaInput.value) || 1800));
      areaSlider.value = String(v);
      document.querySelectorAll(".chip-btn").forEach((b) => b.classList.remove("active"));
      schedulePrediction();
    });

    areaSlider.addEventListener("input", () => {
      areaInput.value = areaSlider.value;
      document.querySelectorAll(".chip-btn").forEach((b) => b.classList.remove("active"));
      schedulePrediction();
    });

    ["qualInput", "bsmtInput", "garageInput", "bathInput", "yearInput"].forEach((id) => {
      document.getElementById(id)?.addEventListener("input", schedulePrediction);
    });

    document.querySelectorAll(".chip-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".chip-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        areaInput.value = btn.dataset.area;
        areaSlider.value = Math.min(5000, Math.max(400, Number(btn.dataset.area)));
        runPrediction();
      });
    });

    document.getElementById("resetExampleBtn")?.addEventListener("click", () => {
      areaInput.value = "1800";
      areaSlider.value = "1800";
      document.getElementById("qualInput").value = "6";
      document.getElementById("bsmtInput").value = "991";
      document.getElementById("garageInput").value = "2";
      document.getElementById("bathInput").value = "2";
      document.getElementById("yearInput").value = "1973";
      document.querySelectorAll(".chip-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.area === "1800");
      });
      runPrediction();
    });

    document.querySelectorAll("#testFilterTabs .seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("#testFilterTabs .seg-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        renderTestTable(btn.dataset.filter);
      });
    });

    document.getElementById("quickAnalyzeBtn")?.addEventListener("click", () => {
      switchView("analyst");
      const area = state.prediction?.input?.gr_liv_area || 1800;
      const price = state.prediction?.simple_linear_regression?.predicted_price_usd || 209379.87;
      sendChatPrompt(`Analyze my property at ${area.toLocaleString()} sq. ft. (predicted ${fmtUsd(price)}): explain the formula, MAE band, and key limitations.`);
    });

    document.getElementById("chatMessages")?.addEventListener("click", (e) => {
      const chip = e.target.closest(".prompt-chip");
      if (chip) sendChatPrompt(chip.textContent);
    });

    document.getElementById("chatForm")?.addEventListener("submit", (e) => {
      e.preventDefault();
      sendChatPrompt(document.getElementById("chatInput").value);
    });

    document.getElementById("chatInput")?.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatPrompt(e.target.value);
      }
    });

    document.getElementById("clearChatBtn")?.addEventListener("click", () => {
      state.chatHistory = [];
      const box = document.getElementById("chatMessages");
      box.innerHTML = `
        <div class="chat-welcome">
          <div class="welcome-spark" aria-hidden="true">✦</div>
          <h3>New conversation started</h3>
          <p>Ask any question about Activity 2, regression metrics, or your current prediction.</p>
          <div class="prompt-chips">
            <button class="prompt-chip" type="button">Explain my current house price prediction, confidence interval, and comparable sales.</button>
            <button class="prompt-chip" type="button">Why did Simple Linear Regression miss luxury homes like Id 899 and Id 692 in the test set?</button>
            <button class="prompt-chip" type="button">Prepare crisp 2-minute viva answers for Activity 2 (slope, intercept, R², MAE vs RMSE).</button>
            <button class="prompt-chip" type="button">Compare the Ames 2010 model valuation against current real estate market trends.</button>
          </div>
        </div>`;
    });

    document.getElementById("marketSearchForm")?.addEventListener("submit", (e) => {
      e.preventDefault();
      loadMarketBundle();
    });

    document.getElementById("briefingBtn")?.addEventListener("click", generateMarketBriefing);
    document.getElementById("closeBriefingBtn")?.addEventListener("click", () => {
      document.getElementById("marketBriefingCard").classList.add("hidden");
    });
  }

  async function bootstrap() {
    initTheme();
    bindEvents();

    try {
      const [health, summary] = await Promise.all([
        apiFetch("/api/health"),
        apiFetch("/api/activity-summary"),
      ]);
      state.summary = summary;
      updateApiStatus(health);

      document.getElementById("sidebarFormula").textContent = summary.model.equation;
      document.getElementById("sidebarDatasetMeta").textContent = `${summary.dataset.rows.toLocaleString()} × ${summary.dataset.columns}`;
      document.getElementById("sidebarR2Meta").textContent = summary.metrics.r2.toFixed(4);

      renderDiagnostics(summary);
      renderWalkthrough(summary);
      await runPrediction();
    } catch (err) {
      showGlobalError(`Failed to initialize studio: ${err.message}`);
    }
  }

  document.addEventListener("DOMContentLoaded", bootstrap);
})();
