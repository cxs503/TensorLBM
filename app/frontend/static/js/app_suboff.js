// SUBOFF surrogate model frontend — calls library-function-backed API
const SUBOFF_API = "/api/ai/suboff";
let _suboffPollTimers = {};  // { jobId: intervalId }

async function _suboffFetch(url, opts = {}) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ── Status ──
async function suboffRefreshStatus() {
  try {
    const s = await _suboffFetch(`${SUBOFF_API}/status`);
    const jobs = document.querySelectorAll("#suboff-results-table tbody tr.job-row");
    document.getElementById("suboff-status-panel").innerHTML =
      `<span class="text-success">● Checkpoints: ${s.checkpoints}</span>
       &nbsp; <span class="text-muted">Latest: ${s.latest || "none"}</span>
       &nbsp; <span class="text-muted">Jobs: ${jobs.length}</span>`;
  } catch (e) {
    document.getElementById("suboff-status-panel").innerHTML =
      `<span class="text-danger">${e.message}</span>`;
  }
}

// ── Data ──
async function suboffScanData() {
  const dir = document.getElementById("suboff-data-dir").value;
  const info = document.getElementById("suboff-data-info");
  info.textContent = "Scanning…";
  try {
    const d = await _suboffFetch(`${SUBOFF_API}/data?data_dir=${encodeURIComponent(dir)}`);
    if (d.exists === false) { info.innerHTML = `<span class="text-danger">Not found: ${dir}</span>`; return; }
    if (d.multi_re) {
      let html = `<span class="text-success">${d.total_snapshots} snapshots (${d.re_groups} Re groups)</span><br>`;
      for (const [re, n] of Object.entries(d.per_re)) {
        html += `<span class="text-muted small">${re}: ${n}</span> `;
      }
      info.innerHTML = html;
    } else {
      info.innerHTML = `<span class="text-success">${d.total_snapshots} snapshots</span>
        p:${d.channels?.p||0} ux:${d.channels?.ux||0} uy:${d.channels?.uy||0} uz:${d.channels?.uz||0}`;
    }
  } catch (e) { info.innerHTML = `<span class="text-danger">${e.message}</span>`; }
}

// ── Training ──
async function suboffStartTrain() {
  const status = document.getElementById("suboff-train-status");
  status.textContent = "Starting…";
  try {
    const j = await _suboffFetch(`${SUBOFF_API}/train`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        iters: parseInt(document.getElementById("suboff-epochs").value),
        lr: parseFloat(document.getElementById("suboff-lr").value),
        n_points: parseInt(document.getElementById("suboff-npoints").value),
        data_dir: document.getElementById("suboff-data-dir").value,
        device: document.getElementById("suboff-device").value,
      }),
    });
    status.innerHTML = `<span class="text-primary">Started ${j.job_id}</span>`;
    _suboffPollJob(j.job_id);
  } catch (e) { status.innerHTML = `<span class="text-danger">${e.message}</span>`; }
}

async function suboffStartFinetune() {
  const status = document.getElementById("suboff-train-status");
  status.textContent = "Starting finetune…";
  try {
    const j = await _suboffFetch(`${SUBOFF_API}/finetune`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        iters: parseInt(document.getElementById("suboff-epochs").value),
        lr: parseFloat(document.getElementById("suboff-lr").value) * 0.1,
        n_points: parseInt(document.getElementById("suboff-npoints").value),
        data_dir: document.getElementById("suboff-data-dir").value,
        device: document.getElementById("suboff-device").value,
      }),
    });
    status.innerHTML = `<span class="text-primary">Started ${j.job_id}</span>`;
    _suboffPollJob(j.job_id);
  } catch (e) { status.innerHTML = `<span class="text-danger">${e.message}</span>`; }
}

function _suboffPollJob(jobId) {
  // Clear old timer for this job if any
  if (_suboffPollTimers[jobId]) clearInterval(_suboffPollTimers[jobId]);

  const jobShort = jobId.slice(-8);
  const container = document.getElementById("suboff-progress-container");
  const tbody = document.querySelector("#suboff-results-table tbody");
  let row = null;

  // Create a per-job progress card
  const cardId = "suboff-card-" + jobShort;
  let card = null;
  let progressBar = null;
  let progressInfo = null;
  if (container) {
    card = document.getElementById(cardId);
    if (!card) {
      container.style.display = "block";
      card = document.createElement("div");
      card.id = cardId;
      card.className = "card mb-2";
      card.innerHTML = `
        <div class="card-header fw-semibold py-1" style="font-size:.85rem">
          <i class="bi bi-activity"></i> Job <code>${jobShort}</code>
        </div>
        <div class="card-body py-2">
          <div class="progress" style="height:22px">
            <div class="progress-bar progress-bar-striped progress-bar-animated bg-primary" role="progressbar" style="width:0%">0%</div>
          </div>
          <div class="mt-1 small text-muted" style="font-size:.78rem">starting</div>
        </div>`;
      container.prepend(card);
    }
    progressBar = card.querySelector(".progress-bar");
    progressInfo = card.querySelector(".text-muted");
  }

  const statusEl = document.getElementById("suboff-train-status");
  _suboffPollTimers[jobId] = setInterval(async () => {
    try {
      const j = await _suboffFetch(`${SUBOFF_API}/train/${jobId}`);
      const pct = j.total > 0 ? Math.min(100, (j.epoch / j.total) * 100) : 0;
      if (progressBar) {
        progressBar.style.width = pct.toFixed(1) + "%";
        progressBar.textContent = `${j.epoch}/${j.total} (${pct.toFixed(1)}%)`;
      }

      const phase = j.phase || "preparing";
      if (progressBar) {
        if (phase === "testing") {
          progressBar.className = "progress-bar progress-bar-striped bg-warning";
          progressBar.textContent = `Testing @ iter ${j.epoch} (${pct.toFixed(1)}%)`;
        } else if (phase === "completed") {
          progressBar.className = "progress-bar bg-success";
        } else if (phase === "failed") {
          progressBar.className = "progress-bar bg-danger";
        } else {
          progressBar.className = "progress-bar progress-bar-striped progress-bar-animated bg-primary";
        }
      }

      if (progressInfo) {
        let info = phase === "testing"
          ? `<span class="badge bg-warning text-dark">TESTING</span>`
          : `<b>${phase.toUpperCase()}</b>`;
        if (j.loss != null) info += ` | loss(1e-4): ${j.loss.toFixed(3)}`;
        else if (phase === "testing") info += ` | loss: computing…`;
        if (j.mse != null) info += ` | mse(1e-4): ${j.mse.toFixed(3)}`;
        if (j.lr != null) info += ` | lr: ${j.lr.toExponential(3)}`;
        if (j.best_loss) info += ` | best: ${j.best_loss.toFixed(3)}`;
        progressInfo.innerHTML = info;
      }

      if (card) {
        const header = card.querySelector(".card-header");
        if (header) {
          const phaseClass = phase === "completed" ? "bg-success"
            : phase === "failed" ? "bg-danger"
            : phase === "testing" ? "bg-warning text-dark"
            : "bg-secondary";
          header.innerHTML = `<i class="bi bi-activity"></i> Job <code>${jobShort}</code> <span class="badge ${phaseClass}">${phase.toUpperCase()}</span>`;
        }
      }

      if (statusEl) statusEl.innerHTML = `<span class="text-primary">Job ${jobShort} running</span>`;

      // Update results table row
      if (tbody) {
        if (!row) {
          row = tbody.insertRow(0);
          row.className = "job-row";
          row.innerHTML = `<td>${jobShort}</td><td></td><td></td><td></td><td></td><td></td><td></td>`;
        }
        row.cells[1].textContent = phase;
        row.cells[2].textContent = `${j.epoch}/${j.total}`;
        row.cells[3].textContent = j.loss != null ? j.loss.toFixed(3) : "-";
        row.cells[4].textContent = j.mse != null ? j.mse.toFixed(3) : "-";
        row.cells[5].textContent = j.lr != null ? j.lr.toExponential(3) : "-";
        row.cells[6].textContent = j.best_loss != null ? j.best_loss.toFixed(3) : "-";
        if (phase === "failed") row.style.color = "red";
        if (phase === "completed") row.style.color = "green";
      }

      if (phase === "completed") {
        if (progressBar) progressBar.classList.remove("progress-bar-animated");
        clearInterval(_suboffPollTimers[jobId]);
        delete _suboffPollTimers[jobId];
        suboffRefreshStatus();
      } else if (phase === "failed") {
        if (progressBar) progressBar.classList.remove("progress-bar-animated");
        if (progressInfo) progressInfo.innerHTML = `<span class="text-danger">Error: ${j.error || "unknown"}</span>`;
        clearInterval(_suboffPollTimers[jobId]);
        delete _suboffPollTimers[jobId];
      }
    } catch (e) {
      clearInterval(_suboffPollTimers[jobId]);
      delete _suboffPollTimers[jobId];
    }
  }, 2000);
}

// ── Inference (predict + auto-render cloud map) ──
async function suboffPredict() {
  const result = document.getElementById("suboff-predict-result");
  const vizResult = document.getElementById("suboff-viz-result");
  result.textContent = "Predicting…";
  vizResult.textContent = "Waiting for predict to finish…";
  try {
    const p = await _suboffFetch(`${SUBOFF_API}/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n_points: parseInt(document.getElementById("suboff-infer-points").value), device: document.getElementById("suboff-device").value }),
    });
    result.innerHTML = `<span class="text-success"><i class="bi bi-check-circle"></i> Done</span> <span class="text-muted small">(see 推理 Results in right panel)</span>`;
    // Also render formatted table in the right panel
    const tableEl = document.getElementById("suboff-predict-table");
    if (tableEl) {
      let tbl = `<table class="table table-sm table-borderless mb-0" style="font-size:.8rem">`;
      tbl += `<tr><td class="text-muted" style="width:120px">Device</td><td><span class="badge bg-info">${p.device}</span></td></tr>`;
      tbl += `<tr><td class="text-muted">Checkpoint</td><td class="text-truncate" style="max-width:300px">${p.checkpoint.split("/").pop()}</td></tr>`;
      tbl += `<tr><td class="text-muted">MAPE</td><td><span class="badge bg-success">${p.mape}%</span></td></tr>`;
      tbl += `<tr><td class="text-muted">rel_L2(1e-4)</td><td>${p.rel_l2_avg_1e4}</td></tr>`;
      tbl += `<tr><td class="text-muted">MSE(1e-4)</td><td>${p.mse_avg_1e4}</td></tr>`;
      tbl += `<tr><td colspan="2" class="fw-semibold pt-2 border-top">Velocity Stats</td></tr>`;
      tbl += `<tr><td class="text-muted">vx range</td><td>[${p.stats.vx.min.toFixed(3)}, ${p.stats.vx.max.toFixed(3)}] mean=${p.stats.vx.mean.toFixed(4)}</td></tr>`;
      tbl += `<tr><td class="text-muted">vy range</td><td>[${p.stats.vy.min.toFixed(3)}, ${p.stats.vy.max.toFixed(3)}] mean=${p.stats.vy.mean.toFixed(4)}</td></tr>`;
      tbl += `<tr><td colspan="2" class="fw-semibold pt-2 border-top">Reconstruction Error</td></tr>`;
      tbl += `<tr><td class="text-muted">vx rel_L2</td><td>${p.recon_error.vx_rel_l2.toFixed(4)}</td></tr>`;
      tbl += `<tr><td class="text-muted">vy rel_L2</td><td>${p.recon_error.vy_rel_l2.toFixed(4)}</td></tr>`;
      tbl += `</table>`;
      tableEl.innerHTML = tbl;
    }
    // Auto-render cloud map after predict
    suboffViz();
  } catch (e) { result.innerHTML = `<span class="text-danger">${e.message}</span>`; }
}

// ── Error Analysis ──
async function suboffRunError() {
  const leftEl = document.getElementById("suboff-error-result");
  const rightEl = document.getElementById("suboff-error-right");
  leftEl.textContent = "Analyzing…";
  if (rightEl) rightEl.innerHTML = "Analyzing…";
  try {
    const e = await _suboffFetch(`${SUBOFF_API}/error`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data_dir: document.getElementById("suboff-data-dir").value,
        n_points: parseInt(document.getElementById("suboff-infer-points").value),
      }),
    });
    let html = `<span class="text-success">${e.n_snapshots} snapshots, ${e.time_ms}ms, ckpt: ${e.checkpoint}</span><br>`;
    html += "<table class='table table-sm table-borderless mb-0'><tr><th>Channel</th><th>rel_L2 mean</th><th>rel_L2 max</th></tr>";
    for (const [ch, s] of Object.entries(e.summary)) {
      html += `<tr><td>${ch}</td><td>${s.rel_l2_mean}</td><td>${s.rel_l2_max}</td></tr>`;
    }
    html += "</table>";
    leftEl.innerHTML = `<span class="text-success"><i class="bi bi-check-circle"></i> Done</span> <span class="text-muted small">(see 误差分析结果 in right panel)</span>`;
    if (rightEl) rightEl.innerHTML = html;
  } catch (e) {
    leftEl.innerHTML = `<span class="text-danger">${e.message}</span>`;
    if (rightEl) rightEl.innerHTML = `<span class="text-danger">${e.message}</span>`;
  }
}

// ── Multi-dataset slice comparison ──────────────────────────────────────────

async function suboffSliceCompare() {
  const infoEl = document.getElementById("suboff-compare-info");
  const gridEl = document.getElementById("suboff-compare-grid");
  infoEl.textContent = "加载中…";
  gridEl.innerHTML = "";

  // Collect checked datasets
  const checkboxes = document.querySelectorAll("#suboff-compare-datasets input[type=checkbox]:checked");
  const dsList = Array.from(checkboxes).map(cb => cb.value).join(",");
  if (!dsList) {
    infoEl.innerHTML = '<span class="text-danger">请至少选择一个数据集</span>';
    return;
  }

  const sliceAxis = document.getElementById("suboff-compare-axis").value;
  const sliceIdxRaw = document.getElementById("suboff-compare-slice").value;
  const snapIdx = parseInt(document.getElementById("suboff-compare-snap").value) || 1499;
  const sliceIdx = sliceIdxRaw === "" ? null : parseInt(sliceIdxRaw);

  const params = new URLSearchParams({
    datasets: dsList,
    snap_idx: snapIdx,
    slice_axis: sliceAxis,
    velocity_only: "true",
  });
  if (sliceIdx !== null) params.set("slice_idx", sliceIdx);

  try {
    const d = await _suboffFetch(`${SUBOFF_API}/slice-compare?${params}`);
    const nDs = d.datasets.length;
    // Layout: each dataset gets a column, each velocity component gets a row
    // Plus a speed row
    const cols = nDs;
    gridEl.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

    const chLabels = {ux: "轴向速度 ux", uy: "横向速度 uy", uz: "垂向速度 uz", speed: "速度幅值 |u|"};
    const chCmaps  = {ux: "RdBu_r", uy: "PiYG", uz: "PiYG", speed: "Viridis"};

    // Find global color range for each channel across all datasets (for consistent comparison)
    const chRanges = {};
    for (const ch of ["ux", "uy", "uz", "speed"]) {
      let gMin = Infinity, gMax = -Infinity;
      for (const ds of d.datasets) {
        if (ds.error) continue;
        const s = ds.slices[ch];
        const flat = s.data.flat();
        const mn = Math.min(...flat), mx = Math.max(...flat);
        gMin = Math.min(gMin, mn); gMax = Math.max(gMax, mx);
      }
      chRanges[ch] = {min: gMin, max: gMax};
    }

    // For each channel, create one row of plots (one per dataset)
    const channels = ["speed", "ux", "uy", "uz"];
    for (const ch of channels) {
      for (const ds of d.datasets) {
        if (ds.error) {
          const div = document.createElement("div");
          div.innerHTML = `<span class="text-danger small">${ds.error}</span>`;
          gridEl.appendChild(div);
          continue;
        }
        const s = ds.slices[ch];
        const ny = s.shape[0], nx = s.shape[1];
        const xArr = Array.from({length: nx}, (_, i) => i);
        const yArr = Array.from({length: ny}, (_, i) => i);
        const div = document.createElement("div");
        div.id = `sc-plot-${ds.name}-${ch}`;
        gridEl.appendChild(div);

        const range = chRanges[ch];
        const isSymmetric = (ch === "ux" || ch === "uy" || ch === "uz");
        const plotConfig = {
          z: s.data, x: xArr, y: yArr, type: "contour",
          contours: {coloring: "heatmap"}, colorscale: chCmaps[ch],
        };
        if (isSymmetric) {
          const absMax = Math.max(Math.abs(range.min), Math.abs(range.max));
          plotConfig.zmin = -absMax; plotConfig.zmax = absMax;
        } else {
          plotConfig.zmin = range.min; plotConfig.zmax = range.max;
        }

        const sliceLabel = `${sliceAxis}=${ds.slice_idx} (${ds.plane_name})`;
        Plotly.newPlot(div, [plotConfig], {
          title: `${ds.name} — ${chLabels[ch]}<br><span class="small text-muted">${sliceLabel}, snap=${ds.snap_idx}</span>`,
          margin: {t: 45, b: 20, l: 40, r: 10},
          xaxis: {title: "X"}, yaxis: {title: "Y", scaleanchor: "x"},
          width: 320, height: 240,
        }, {responsive: true});
      }
    }

    // Info line
    let infoHtml = `<span class="text-success">${nDs} datasets loaded</span> — `;
    for (const ds of d.datasets) {
      if (ds.error) { infoHtml += `<span class="text-danger">${ds.name}: ${ds.error}</span> `; continue; }
      infoHtml += `<span class="text-muted">${ds.name}: ${ds.shape.join("×")}, ${ds.slice_axis}=${ds.slice_idx} (${ds.plane_name}), speed=[${ds.speed_stats.min.toFixed(4)}, ${ds.speed_stats.max.toFixed(4)}]</span> `;
    }
    infoEl.innerHTML = infoHtml;

  } catch (e) {
    infoEl.innerHTML = `<span class="text-danger">${e.message}</span>`;
  }
}

// ── Init ──
document.addEventListener("DOMContentLoaded", () => {
  const origShowTab = window.showTab;
  window.showTab = function (name, el) {
    origShowTab(name, el);
    if (name === "suboff") {
      suboffRefreshStatus();
      suboffScanData();
    }
  };
});

// ── Animation (time-series playback) ──────────────────────────────────────

let _animFrames = null;   // loaded animation data
let _animTimer = null;    // setInterval timer for playback
let _animIdx = 0;         // current frame index
let _animPlots = {};      // Plotly div references per channel

const ANIM_CH_LABELS = {p: "Pressure (p)", ux: "Axial velocity (ux)", uy: "Lateral velocity (uy)", uz: "Vertical velocity (uz)"};
const ANIM_CH_CMAPS  = {p: "Viridis", ux: "RdBu_r", uy: "PiYG", uz: "PiYG"};

async function suboffLoadAnimation() {
  const leftInfo = document.getElementById("suboff-anim-info");
  const rightInfo = document.getElementById("suboff-anim-info-right");
  const rightGrid = document.getElementById("suboff-anim-grid-right");
  leftInfo.textContent = "Loading frames…";

  // Stop any running animation
  suboffPauseAnimation();

  const params = new URLSearchParams({
    data_dir: document.getElementById("suboff-data-dir").value,
    snap_start: document.getElementById("suboff-anim-start").value,
    snap_end: document.getElementById("suboff-anim-end").value,
    slice_axis: document.getElementById("suboff-anim-axis").value,
    slice_idx: document.getElementById("suboff-anim-slice").value,
    channels: document.getElementById("suboff-anim-channels").value,
    view: document.getElementById("suboff-anim-view").value,
  });

  try {
    const d = await _suboffFetch(`${SUBOFF_API}/animate?${params}`);
    _animFrames = d;
    _animIdx = 0;

    const viewLabel = d.view === "full"
      ? `Full tail field (${d.full_shape.join("×")})`
      : `Training crop (${d.cropped_shape.join("×")})`;
    leftInfo.innerHTML = `<span class="text-success"><i class="bi bi-check-circle"></i> Done</span> <span class="text-muted small">(see 动画结果 in right panel)</span>`;
    if (rightInfo) rightInfo.innerHTML = `<span class="text-success">${d.n_frames} frames loaded</span> — ${viewLabel}, ${d.slice_axis}=${d.slice_idx} (${d.plane_name}), snaps ${d.snap_start}..${d.snap_end-1}, channels: ${d.channels.join(", ")}`;

    // Create Plotly divs for each channel in the right panel grid
    _animPlots = {};
    const firstFrame = d.frames[0];
    if (rightGrid) rightGrid.innerHTML = "";
    for (const ch of d.channels) {
      if (!firstFrame.channels[ch]) continue;
      const chObj = firstFrame.channels[ch];
      const zData = chObj.data;
      const ny = zData.length, nx = zData[0].length;
      const div = document.createElement("div");
      div.id = `anim-plot-${ch}`;
      if (rightGrid) rightGrid.appendChild(div);

      const xArr = Array.from({length: nx}, (_, i) => i);
      const yArr = Array.from({length: ny}, (_, i) => i);

      Plotly.newPlot(div, [{
        z: zData, x: xArr, y: yArr, type: "contour",
        contours: {coloring: "heatmap"}, colorscale: ANIM_CH_CMAPS[ch] || "Viridis",
      }], {
        title: `${ANIM_CH_LABELS[ch] || ch} — Snap ${firstFrame.snap_idx}`,
        margin: {t: 30, b: 20, l: 30, r: 10},
        xaxis: {title: "X"}, yaxis: {title: "Y", scaleanchor: "x"},
        width: 380, height: 260,
      }, {responsive: true});

      _animPlots[ch] = div;
    }

    // Enable play button
    document.getElementById("suboff-anim-play").disabled = false;
    document.getElementById("suboff-anim-pause").disabled = false;
    _updateAnimProgress();

  } catch (e) {
    leftInfo.innerHTML = `<span class="text-danger">${e.message}</span>`;
    if (rightInfo) rightInfo.innerHTML = `<span class="text-danger">${e.message}</span>`;
    _animFrames = null;
  }
}

function suboffPlayAnimation() {
  if (!_animFrames || !_animFrames.frames.length) return;
  if (_animTimer) return; // already playing

  const fps = parseInt(document.getElementById("suboff-anim-speed").value) || 5;
  const interval = Math.max(50, 1000 / fps);

  _animTimer = setInterval(() => {
    _animIdx++;
    if (_animIdx >= _animFrames.frames.length) {
      _animIdx = 0; // loop
    }
    _renderAnimFrame();
  }, interval);
}

function suboffPauseAnimation() {
  if (_animTimer) {
    clearInterval(_animTimer);
    _animTimer = null;
  }
}

function _renderAnimFrame() {
  if (!_animFrames) return;
  const frame = _animFrames.frames[_animIdx];
  for (const ch of _animFrames.channels) {
    const div = _animPlots[ch];
    if (!div || !frame.channels[ch]) continue;
    const zData = frame.channels[ch].data;
    Plotly.animate(div, {
      data: [{z: zData}],
      layout: {title: `${ANIM_CH_LABELS[ch] || ch} — Snap ${frame.snap_idx}`},
    }, {
      transition: {duration: 0},
      frame: {duration: 0, redraw: true},
    });
  }
  _updateAnimProgress();
}

function _updateAnimProgress() {
  if (!_animFrames) return;
  const pct = ((_animIdx + 1) / _animFrames.frames.length) * 100;
  const rightProg = document.getElementById("suboff-anim-progress-right");
  const rightInfo = document.getElementById("suboff-anim-info-right");
  if (rightProg) rightProg.style.width = pct + "%";
  if (rightInfo) rightInfo.innerHTML =
    `<span class="text-muted">Frame ${_animIdx + 1}/${_animFrames.frames.length} (snap ${_animFrames?.frames[_animIdx]?.snap_idx ?? "?"})</span>`;
}

// ── Visualization (Plotly interactive contour) ──
async function suboffViz() {
  const leftEl = document.getElementById("suboff-viz-result");
  const rightEl = document.getElementById("suboff-viz-right");
  const statusText = "Loading slice data…";
  leftEl.textContent = statusText;
  if (rightEl) rightEl.innerHTML = statusText;
  const params = new URLSearchParams({
    data_dir: document.getElementById("suboff-data-dir").value,
    snap_idx: document.getElementById("suboff-viz-snap").value,
    n_points: parseInt(document.getElementById("suboff-infer-points").value) || 50000,
    slice_axis: document.getElementById("suboff-viz-axis").value,
    slice_idx: document.getElementById("suboff-viz-slice").value,
    device: document.getElementById("suboff-device").value,
  });
  try {
    const d = await _suboffFetch(`${SUBOFF_API}/viz-data?${params}`);
    const info = `Snapshot ${d.snapshot}, ${d.slice_axis}=${d.slice_idx} (${d.plane_name}), grid ${d.grid_size}³, ckpt ${d.checkpoint}, ${d.device}`;
    leftEl.innerHTML = `<span class="text-success"><i class="bi bi-check-circle"></i> Done</span> <span class="text-muted small">(see Visualization 结果 in right panel)</span>`;
    if (rightEl) {
      rightEl.innerHTML = `<span class="text-muted small">${info}</span><div id="suboff-plotly-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px"></div>`;
      const grid = document.getElementById("suboff-plotly-grid");
      const chNames = ["pressure", "vx", "vy", "vz"];
      const chLabels = {pressure:"Pressure (p)", vx:"Axial velocity (ux)", vy:"Lateral velocity (uy)", vz:"Vertical velocity (uz)"};
      const cmaps = {pressure:"Viridis", vx:"RdBu_r", vy:"PiYG", vz:"PiYG"};
      const ny = d.slices.pressure.shape[0], nx = d.slices.pressure.shape[1];
      const xArr = Array.from({length: nx}, (_, i) => i);
      const yArr = Array.from({length: ny}, (_, i) => i);

      for (const ch of chNames) {
        const s = d.slices[ch];
        // True contour
        const div1 = document.createElement("div");
        div1.id = `plot-${ch}-true`;
        grid.appendChild(div1);
        Plotly.newPlot(div1, [{
          z: s.true, x: xArr, y: yArr, type: "contour",
          contours: {coloring: "heatmap"}, colorscale: cmaps[ch],
        }], {
          title: `${chLabels[ch]} — True`, margin: {t:30,b:20,l:30,r:10},
          xaxis: {title: "X"}, yaxis: {title: "Y", scaleanchor:"x"},
          width: 380, height: 260,
        }, {responsive: true});

        // Error contour (if pred available)
        if (s.error) {
          const div2 = document.createElement("div");
          div2.id = `plot-${ch}-error`;
          grid.appendChild(div2);
          const errMax = Math.max(Math.abs(Math.min(...s.error.flat())), Math.abs(Math.max(...s.error.flat())));
          Plotly.newPlot(div2, [{
            z: s.error, x: xArr, y: yArr, type: "contour",
            contours: {coloring: "heatmap"}, colorscale: "RdBu_r",
            zmin: -errMax, zmax: errMax,
          }], {
            title: `${chLabels[ch]} — Error (pred−true)`, margin: {t:30,b:20,l:30,r:10},
            xaxis: {title: "X"}, yaxis: {title: "Y", scaleanchor:"x"},
            width: 380, height: 260,
          }, {responsive: true});
        }
      }
    }
  } catch (e) {
    leftEl.innerHTML = `<span class="text-danger">${e.message}</span>`;
    if (rightEl) rightEl.innerHTML = `<span class="text-danger">${e.message}</span>`;
  }
}
