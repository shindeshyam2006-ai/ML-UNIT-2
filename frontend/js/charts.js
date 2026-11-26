/* Interactive vector SVG charts for Activity 2: Simple House Price Prediction */

const ChartKit = (() => {
  const SVG_NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs = {}, parent = null) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      node.setAttribute(k, String(v));
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function formatUsdK(val) {
    return `$${Math.round(val / 1000)}k`;
  }

  function formatUsd(val) {
    return `$${Number(val).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
  }

  function showTooltip(evt, html) {
    const tip = document.getElementById("chartTooltip");
    if (!tip) return;
    tip.innerHTML = html;
    tip.classList.remove("hidden");
    const pad = 14;
    const x = Math.min(window.innerWidth - 250, evt.clientX + pad);
    const y = Math.min(window.innerHeight - 130, evt.clientY + pad);
    tip.style.left = `${Math.max(8, x)}px`;
    tip.style.top = `${Math.max(8, y)}px`;
  }

  function hideTooltip() {
    const tip = document.getElementById("chartTooltip");
    if (tip) tip.classList.add("hidden");
  }

  function niceTicks(minVal, maxVal, count = 5) {
    const span = maxVal - minVal || 1;
    const rawStep = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const norm = rawStep / mag;
    let step = mag;
    if (norm > 5) step = 10 * mag;
    else if (norm > 2) step = 5 * mag;
    else if (norm > 1) step = 2 * mag;
    const start = Math.floor(minVal / step) * step;
    const ticks = [];
    for (let v = start; v <= maxVal + step * 0.5; v += step) {
      if (v >= minVal - step * 0.01) ticks.push(v);
    }
    return ticks;
  }

  function createSvgFrame(container, w, h) {
    container.innerHTML = "";
    return el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" }, container);
  }

  function renderScatterRegression(container, points, model, livePoint = null) {
    const W = 740, H = 370;
    const m = { top: 20, right: 24, bottom: 48, left: 66 };
    const pw = W - m.left - m.right;
    const ph = H - m.top - m.bottom;

    const maxArea = Math.max(5800, livePoint ? livePoint.area * 1.05 : 0);
    const maxPrice = Math.max(780000, livePoint ? livePoint.price * 1.1 : 0);
    const xMin = 200, xMax = maxArea;
    const yMin = 0, yMax = maxPrice;

    const sx = (v) => m.left + ((v - xMin) / (xMax - xMin)) * pw;
    const sy = (v) => m.top + ph - ((v - yMin) / (yMax - yMin)) * ph;

    const svg = createSvgFrame(container, W, H);

    // Gridlines & ticks
    for (const xt of niceTicks(xMin, xMax, 6)) {
      const gx = sx(xt);
      el("line", { x1: gx, y1: m.top, x2: gx, y2: m.top + ph, stroke: "var(--border-hairline)", "stroke-dasharray": "3 3" }, svg);
      const t = el("text", { x: gx, y: m.top + ph + 18, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 11 }, svg);
      t.textContent = xt.toLocaleString("en-US");
    }
    for (const yt of niceTicks(yMin, yMax, 5)) {
      const gy = sy(yt);
      el("line", { x1: m.left, y1: gy, x2: m.left + pw, y2: gy, stroke: "var(--border-hairline)", "stroke-dasharray": "3 3" }, svg);
      const t = el("text", { x: m.left - 8, y: gy + 4, "text-anchor": "end", fill: "var(--text-muted)", "font-size": 11 }, svg);
      t.textContent = formatUsdK(yt);
    }

    // Axes
    el("line", { x1: m.left, y1: m.top + ph, x2: m.left + pw, y2: m.top + ph, stroke: "var(--border-strong)" }, svg);
    el("line", { x1: m.left, y1: m.top, x2: m.left, y2: m.top + ph, stroke: "var(--border-strong)" }, svg);

    const xl = el("text", { x: m.left + pw / 2, y: H - 8, "text-anchor": "middle", fill: "var(--text-secondary)", "font-size": 12, "font-weight": 600 }, svg);
    xl.textContent = "Above-Grade Living Area — GrLivArea (sq. ft.)";
    const yl = el("text", { x: 16, y: m.top + ph / 2, "text-anchor": "middle", fill: "var(--text-secondary)", "font-size": 12, "font-weight": 600, transform: `rotate(-90 16 ${m.top + ph / 2})` }, svg);
    yl.textContent = "Sale Price (USD)";

    // Points
    const gPts = el("g", {}, svg);
    for (const pt of points) {
      const cx = sx(pt.area);
      const cy = sy(pt.price);
      const fill = pt.outlier ? "#235789" : pt.split === "test" ? "#C96442" : "#7D8471";
      const r = pt.outlier ? 5.2 : 3.2;
      const dot = el("circle", {
        cx, cy, r,
        fill,
        "fill-opacity": pt.outlier ? 0.95 : 0.56,
        stroke: pt.outlier ? "#FFFFFF" : "none",
        "stroke-width": pt.outlier ? 1.2 : 0,
      }, gPts);
      dot.addEventListener("mouseenter", (e) => {
        dot.setAttribute("r", r + 2.5);
        showTooltip(
          e,
          `<strong>Property #${pt.id} (${pt.neighborhood})</strong><br/>` +
          `Split: ${pt.split.toUpperCase()} · Qual: ${pt.qual}/10<br/>` +
          `Living Area: ${pt.area.toLocaleString("en-US")} sq. ft.<br/>` +
          `Actual Price: <strong>${formatUsd(pt.price)}</strong>`
        );
      });
      dot.addEventListener("mouseleave", () => {
        dot.setAttribute("r", r);
        hideTooltip();
      });
    }

    // Regression Line
    const lx1 = 334;
    const lx2 = Math.min(xMax, 5650);
    const ly1 = model.intercept + model.slope * lx1;
    const ly2 = model.intercept + model.slope * lx2;
    el("line", {
      x1: sx(lx1), y1: sy(ly1),
      x2: sx(lx2), y2: sy(ly2),
      stroke: "var(--text-primary)",
      "stroke-width": 2.4,
    }, svg);

    // Live prediction marker
    if (livePoint) {
      const px = sx(livePoint.area);
      const py = sy(Math.min(yMax, Math.max(yMin, livePoint.price)));
      el("line", { x1: px, y1: m.top + ph, x2: px, y2: py, stroke: "#16A34A", "stroke-dasharray": "4 3", "stroke-width": 1.4 }, svg);
      el("line", { x1: m.left, y1: py, x2: px, y2: py, stroke: "#16A34A", "stroke-dasharray": "4 3", "stroke-width": 1.4 }, svg);
      el("circle", { cx: px, cy: py, r: 9, fill: "#16A34A", "fill-opacity": 0.2 }, svg);
      const liveDot = el("circle", { cx: px, cy: py, r: 5.5, fill: "#16A34A", stroke: "#FFFFFF", "stroke-width": 2 }, svg);
      liveDot.addEventListener("mouseenter", (e) => {
        showTooltip(
          e,
          `<strong>Your Active Input</strong><br/>Living Area: ${livePoint.area.toLocaleString("en-US")} sq. ft.<br/>Predicted: <strong>${formatUsd(livePoint.price)}</strong>`
        );
      });
      liveDot.addEventListener("mouseleave", hideTooltip);
    }
  }

  function renderActualVsPredicted(container, testPoints) {
    const W = 560, H = 330;
    const m = { top: 20, right: 20, bottom: 46, left: 62 };
    const pw = W - m.left - m.right;
    const ph = H - m.top - m.bottom;

    const maxV = 780000;
    const sx = (v) => m.left + (v / maxV) * pw;
    const sy = (v) => m.top + ph - (v / maxV) * ph;

    const svg = createSvgFrame(container, W, H);
    for (const tick of niceTicks(0, maxV, 5)) {
      const gx = sx(tick), gy = sy(tick);
      el("line", { x1: gx, y1: m.top, x2: gx, y2: m.top + ph, stroke: "var(--border-hairline)", "stroke-dasharray": "3 3" }, svg);
      el("line", { x1: m.left, y1: gy, x2: m.left + pw, y2: gy, stroke: "var(--border-hairline)", "stroke-dasharray": "3 3" }, svg);
      const tx = el("text", { x: gx, y: m.top + ph + 17, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 10.5 }, svg);
      tx.textContent = formatUsdK(tick);
      const ty = el("text", { x: m.left - 8, y: gy + 4, "text-anchor": "end", fill: "var(--text-muted)", "font-size": 10.5 }, svg);
      ty.textContent = formatUsdK(tick);
    }

    el("line", { x1: m.left, y1: m.top + ph, x2: m.left + pw, y2: m.top + ph, stroke: "var(--border-strong)" }, svg);
    el("line", { x1: m.left, y1: m.top, x2: m.left, y2: m.top + ph, stroke: "var(--border-strong)" }, svg);

    const xl = el("text", { x: m.left + pw / 2, y: H - 8, "text-anchor": "middle", fill: "var(--text-secondary)", "font-size": 11.5, "font-weight": 600 }, svg);
    xl.textContent = "Actual Sale Price (USD)";
    const yl = el("text", { x: 15, y: m.top + ph / 2, "text-anchor": "middle", fill: "var(--text-secondary)", "font-size": 11.5, "font-weight": 600, transform: `rotate(-90 15 ${m.top + ph / 2})` }, svg);
    yl.textContent = "Predicted Sale Price (USD)";

    // Perfect prediction reference line y = x
    el("line", {
      x1: sx(30000), y1: sy(30000),
      x2: sx(760000), y2: sy(760000),
      stroke: "var(--text-primary)", "stroke-width": 2, "stroke-dasharray": "6 4",
    }, svg);
    const refLabel = el("text", {
      x: sx(615000), y: sy(645000),
      fill: "var(--text-secondary)", "font-size": 10.5, "font-weight": 600,
    }, svg);
    refLabel.textContent = "y = x (Perfect)";

    for (const pt of testPoints) {
      const dot = el("circle", {
        cx: sx(pt.actual_price),
        cy: sy(pt.predicted_price),
        r: 3.6,
        fill: "#C96442",
        "fill-opacity": 0.65,
      }, svg);
      dot.addEventListener("mouseenter", (e) => {
        dot.setAttribute("r", 6);
        showTooltip(
          e,
          `<strong>Test ID #${pt.id} (${pt.neighborhood})</strong><br/>` +
          `Area: ${pt.gr_liv_area.toLocaleString()} sq.ft. · Qual: ${pt.overall_qual}<br/>` +
          `Actual: ${formatUsd(pt.actual_price)}<br/>` +
          `Predicted: ${formatUsd(pt.predicted_price)}<br/>` +
          `Residual: <strong>${formatUsd(pt.residual)}</strong>`
        );
      });
      dot.addEventListener("mouseleave", () => { dot.setAttribute("r", 3.6); hideTooltip(); });
    }
  }

  function renderResiduals(container, testPoints) {
    const W = 560, H = 330;
    const m = { top: 20, right: 20, bottom: 46, left: 66 };
    const pw = W - m.left - m.right;
    const ph = H - m.top - m.bottom;

    const xMin = 40000, xMax = 500000;
    const yMin = -360000, yMax = 380000;
    const sx = (v) => m.left + ((v - xMin) / (xMax - xMin)) * pw;
    const sy = (v) => m.top + ph - ((v - yMin) / (yMax - yMin)) * ph;

    const svg = createSvgFrame(container, W, H);
    for (const xt of niceTicks(xMin, xMax, 5)) {
      const gx = sx(xt);
      el("line", { x1: gx, y1: m.top, x2: gx, y2: m.top + ph, stroke: "var(--border-hairline)", "stroke-dasharray": "3 3" }, svg);
      const tx = el("text", { x: gx, y: m.top + ph + 17, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 10.5 }, svg);
      tx.textContent = formatUsdK(xt);
    }
    for (const yt of niceTicks(yMin, yMax, 5)) {
      const gy = sy(yt);
      el("line", { x1: m.left, y1: gy, x2: m.left + pw, y2: gy, stroke: "var(--border-hairline)", "stroke-dasharray": "3 3" }, svg);
      const ty = el("text", { x: m.left - 8, y: gy + 4, "text-anchor": "end", fill: "var(--text-muted)", "font-size": 10.5 }, svg);
      ty.textContent = formatUsdK(yt);
    }

    el("line", { x1: m.left, y1: m.top + ph, x2: m.left + pw, y2: m.top + ph, stroke: "var(--border-strong)" }, svg);
    el("line", { x1: m.left, y1: m.top, x2: m.left, y2: m.top + ph, stroke: "var(--border-strong)" }, svg);
    el("line", { x1: m.left, y1: sy(0), x2: m.left + pw, y2: sy(0), stroke: "var(--text-primary)", "stroke-width": 2, "stroke-dasharray": "6 4" }, svg);

    const xl = el("text", { x: m.left + pw / 2, y: H - 8, "text-anchor": "middle", fill: "var(--text-secondary)", "font-size": 11.5, "font-weight": 600 }, svg);
    xl.textContent = "Predicted Sale Price (USD)";
    const yl = el("text", { x: 15, y: m.top + ph / 2, "text-anchor": "middle", fill: "var(--text-secondary)", "font-size": 11.5, "font-weight": 600, transform: `rotate(-90 15 ${m.top + ph / 2})` }, svg);
    yl.textContent = "Residual = Actual − Predicted (USD)";

    for (const pt of testPoints) {
      const clampedX = Math.max(xMin, Math.min(xMax, pt.predicted_price));
      const clampedY = Math.max(yMin, Math.min(yMax, pt.residual));
      const dot = el("circle", {
        cx: sx(clampedX),
        cy: sy(clampedY),
        r: 3.6,
        fill: pt.residual >= 0 ? "#2D6A4F" : "#C96442",
        "fill-opacity": 0.65,
      }, svg);
      dot.addEventListener("mouseenter", (e) => {
        dot.setAttribute("r", 6);
        showTooltip(
          e,
          `<strong>Test ID #${pt.id}</strong><br/>Predicted: ${formatUsd(pt.predicted_price)}<br/>Residual: <strong>${formatUsd(pt.residual)}</strong>`
        );
      });
      dot.addEventListener("mouseleave", () => { dot.setAttribute("r", 3.6); hideTooltip(); });
    }
  }

  function renderCorrelations(container, correlations) {
    const top = correlations.slice(0, 7);
    const W = 560, H = 250;
    const m = { top: 12, right: 56, bottom: 20, left: 118 };
    const pw = W - m.left - m.right;
    const rowH = (H - m.top - m.bottom) / top.length;

    const svg = createSvgFrame(container, W, H);
    top.forEach((item, idx) => {
      const y = m.top + idx * rowH + 6;
      const barH = Math.max(14, rowH - 12);
      const isPrimary = item.feature === "GrLivArea";
      const barW = Math.max(4, item.correlation * pw);

      const label = el("text", {
        x: m.left - 8, y: y + barH / 2 + 4,
        "text-anchor": "end", fill: isPrimary ? "var(--accent)" : "var(--text-secondary)",
        "font-size": 12, "font-weight": isPrimary ? 700 : 500,
      }, svg);
      label.textContent = item.feature;

      el("rect", {
        x: m.left, y, width: barW, height: barH, rx: 5,
        fill: isPrimary ? "var(--accent)" : "#7D8471",
        "fill-opacity": isPrimary ? 0.95 : 0.72,
      }, svg);

      const valTxt = el("text", {
        x: m.left + barW + 7, y: y + barH / 2 + 4,
        fill: "var(--text-primary)", "font-size": 11.5, "font-weight": 600,
      }, svg);
      valTxt.textContent = item.correlation.toFixed(4);
    });
  }

  return {
    renderScatterRegression,
    renderActualVsPredicted,
    renderResiduals,
    renderCorrelations,
  };
})();
