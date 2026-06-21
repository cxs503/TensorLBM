/**
 * app_convergence.js – TensorLBM Convergence Monitor & Report Panel
 *
 * Real-time convergence monitoring (force coefficients, residuals) and
 * engineering report generation for completed jobs.
 */
"use strict";

/* =========================================================================
   Convergence monitor
   ========================================================================= */

let _cvg_poll_timer = null;
let _cvg_job_id = null;
let _cvg_chart_data = {};   // { seriesName: [values…] }

/** Called when the Reports tab is activated or a job is selected. */
async function convergenceLoad(jobId) {
  if (!jobId) return;
  _cvg_job_id = jobId;
  _stopPoll();
  await _fetchConvergence(jobId);

  // Poll every 2 s if the job is running
  const statusEl = document.getElementById("cvg-job-status");
  if (statusEl && (statusEl.textContent || "").toLowerCase().includes("running")) {
    _cvg_poll_timer = setInterval(() => _fetchConvergence(jobId), 2000);
  }
}

function _stopPoll() {
  if (_cvg_poll_timer) { clearInterval(_cvg_poll_timer); _cvg_poll_timer = null; }
}

async function _fetchConvergence(jobId) {
  try {
    const resp = await fetch(`/api/postprocess/convergence/${jobId}`);
    if (!resp.ok) return;
    const data = await resp.json();
    _cvg_chart_data = data.series || {};
    _renderConvergenceCharts(data);

    // Stop polling if job is no longer running
    if (data.job_status !== "running") _stopPoll();
  } catch { /* silent – job may not be found during polling */ }
}

function _renderConvergenceCharts(data) {
  const el = document.getElementById("cvg-charts");
  if (!el) return;

  const steps = data.steps || [];
  const series = data.series || {};
  const seriesNames = Object.keys(series);

  if (!seriesNames.length) {
    el.innerHTML = `<div class="text-muted small p-2">${t("convergence.no_data")}</div>`;
    return;
  }

  const charts = seriesNames.slice(0, 6).map(name => {
    const vals = series[name] || [];
    return _miniLineChart(steps, vals, name);
  }).join("");

  const statusEl = document.getElementById("cvg-job-status");
  if (statusEl) statusEl.textContent = data.job_status || "";

  el.innerHTML = charts;
}

/** Build a minimal inline SVG sparkline chart. */
function _miniLineChart(xData, yData, label) {
  const W = 480, H = 110, pad = 30;
  if (!yData.length) return "";
  const mn = Math.min(...yData), mx = Math.max(...yData);
  const rng = mx - mn || 1;
  const n = yData.length;

  const pts = yData.map((v, i) => {
    const x = pad + (i / Math.max(n - 1, 1)) * (W - 2 * pad);
    const y = H - pad - ((v - mn) / rng) * (H - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");

  // Last value
  const lastVal = yData[yData.length - 1];
  const color = label.toLowerCase().includes("drag") || label.toLowerCase().includes("cd")
    ? "#0d6efd"
    : label.toLowerCase().includes("lift") || label.toLowerCase().includes("cl")
    ? "#198754"
    : "#6f42c1";

  return `
<div class="mb-3">
  <div class="small fw-semibold mb-1">${_cvgEsc(label)}
    <span class="text-muted ms-2 fw-normal">(last: ${lastVal != null ? lastVal.toFixed(5) : "—"})</span>
  </div>
  <svg width="${W}" height="${H}" style="display:block;background:#f8f9fa;border-radius:.35rem;border:1px solid #dee2e6">
    <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2"/>
    <text x="${pad}" y="${H - 8}" font-size="10" fill="#adb5bd">step ${xData[0] != null ? xData[0] : 0}</text>
    <text x="${W - pad}" y="${H - 8}" font-size="10" fill="#adb5bd" text-anchor="end">step ${xData[xData.length - 1] != null ? xData[xData.length - 1] : n}</text>
    <text x="${pad}" y="14" font-size="10" fill="#adb5bd">max ${mx.toFixed(4)}</text>
    <text x="${pad}" y="${H - pad + 14}" font-size="10" fill="#adb5bd">min ${mn.toFixed(4)}</text>
  </svg>
</div>`;
}

/* =========================================================================
   Report panel
   ========================================================================= */

async function reportsLoad(jobId) {
  if (!jobId) return;
  const summaryEl = document.getElementById("report-summary");
  if (summaryEl) summaryEl.innerHTML = `<div class="text-muted small">Loading…</div>`;

  try {
    const resp = await fetch(`/api/reports/${jobId}/summary`);
    if (!resp.ok) throw new Error(await resp.text());
    const s = await resp.json();
    _renderReportSummary(s);
  } catch (e) {
    if (summaryEl) summaryEl.innerHTML = `<div class="alert alert-danger small">${e.message}</div>`;
  }
}

function _renderReportSummary(s) {
  const el = document.getElementById("report-summary");
  if (!el) return;
  const statusColor = {
    completed: "success", failed: "danger", running: "warning", queued: "secondary"
  }[s.status] || "secondary";

  el.innerHTML = `
<div class="row g-2 mb-3">
  <div class="col"><div class="card text-center p-2"><div class="fw-bold">${_cvgEsc(s.status)}</div><div class="small text-muted">Status</div></div></div>
  <div class="col"><div class="card text-center p-2"><div class="fw-bold">${s.diagnostic_steps}</div><div class="small text-muted">Diag Steps</div></div></div>
  <div class="col"><div class="card text-center p-2"><div class="fw-bold">${s.force_rows}</div><div class="small text-muted">Force Rows</div></div></div>
  <div class="col"><div class="card text-center p-2"><div class="fw-bold">${s.image_count}</div><div class="small text-muted">Images</div></div></div>
</div>
<div class="d-flex gap-2 mb-2">
  <a class="btn btn-primary btn-sm" href="/api/reports/${_cvgEsc(s.job_id)}" target="_blank">
    <i class="bi bi-file-earmark-text me-1"></i>${t("reports.open_html")}
  </a>
</div>
<div class="small text-muted">
  Created: ${s.created_at || '—'} | Completed: ${s.completed_at || '—'}
</div>`;
}

/* =========================================================================
   Combined Reports+Convergence tab initialisation
   ========================================================================= */

function reportsTabInit(jobId) {
  if (!jobId) {
    const el = document.getElementById("cvg-charts");
    if (el) el.innerHTML = `<div class="text-muted small p-2">${t("reports.no_job")}</div>`;
    const sel = document.getElementById("report-summary");
    if (sel) sel.innerHTML = `<div class="text-muted small p-2">${t("reports.no_job")}</div>`;
    return;
  }
  convergenceLoad(jobId);
  reportsLoad(jobId);
}

/* =========================================================================
   Utility
   ========================================================================= */

function _cvgEsc(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}
