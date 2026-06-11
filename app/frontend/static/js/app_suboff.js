// SUBOFF surrogate model frontend
const SUBOFF_API = "/api/ai/suboff";
let _suboffPollTimer = null;

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
    if (!d.exists) { info.innerHTML = `<span class="text-danger">Not found: ${dir}</span>`; return; }
    info.innerHTML = `<span class="text-success">${d.total_snapshots} snapshots</span>
      p:${d.channels.p||0} ux:${d.channels.ux||0} uy:${d.channels.uy||0} uz:${d.channels.uz||0}`;
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
        epochs: parseInt(document.getElementById("suboff-epochs").value),
        lr: parseFloat(document.getElementById("suboff-lr").value),
        n_points: parseInt(document.getElementById("suboff-npoints").value),
        data_dir: document.getElementById("suboff-data-dir").value,
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
        epochs: parseInt(document.getElementById("suboff-epochs").value),
        lr: parseFloat(document.getElementById("suboff-lr").value) * 0.1,
        n_points: parseInt(document.getElementById("suboff-npoints").value),
        data_dir: document.getElementById("suboff-data-dir").value,
      }),
    });
    status.innerHTML = `<span class="text-primary">Started ${j.job_id}</span>`;
    _suboffPollJob(j.job_id);
  } catch (e) { status.innerHTML = `<span class="text-danger">${e.message}</span>`; }
}

function _suboffPollJob(jobId) {
  if (_suboffPollTimer) clearInterval(_suboffPollTimer);
  const tbody = document.querySelector("#suboff-results-table tbody");
  let row = null;

  _suboffPollTimer = setInterval(async () => {
    try {
      const j = await _suboffFetch(`${SUBOFF_API}/train/${jobId}`);
      if (!row) {
        row = tbody.insertRow(0);
        row.className = "job-row";
        row.innerHTML = `<td>${jobId.slice(-8)}</td><td></td><td></td><td></td><td></td>`;
      }
      row.cells[1].textContent = j.status;
      row.cells[2].textContent = `${j.epoch}/${j.total}`;
      row.cells[3].textContent = j.loss != null ? j.loss.toFixed(2) : "-";
      row.cells[4].textContent = j.best_loss != null ? j.best_loss.toFixed(2) : "-";
      if (j.status === "failed") row.style.color = "red";
      if (j.status === "completed") row.style.color = "green";
      if (["completed", "failed"].includes(j.status)) {
        clearInterval(_suboffPollTimer);
        _suboffPollTimer = null;
        suboffRefreshStatus();
      }
    } catch (e) {
      clearInterval(_suboffPollTimer);
      _suboffPollTimer = null;
    }
  }, 2000);
}

// ── Inference ──
async function suboffPredict() {
  const result = document.getElementById("suboff-predict-result");
  result.textContent = "Predicting…";
  try {
    const p = await _suboffFetch(`${SUBOFF_API}/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n_points: parseInt(document.getElementById("suboff-infer-points").value) }),
    });
    result.innerHTML = `<span class="text-success">${p.time_ms}ms ${p.device}</span><br>
      vx: [${p.stats.vx.min.toFixed(3)}, ${p.stats.vx.max.toFixed(3)}] mean=${p.stats.vx.mean.toFixed(4)}<br>
      vy: [${p.stats.vy.min.toFixed(3)}, ${p.stats.vy.max.toFixed(3)}] mean=${p.stats.vy.mean.toFixed(4)}`;
  } catch (e) { result.innerHTML = `<span class="text-danger">${e.message}</span>`; }
}

// ── Error Analysis ──
async function suboffRunError() {
  const result = document.getElementById("suboff-error-result");
  result.textContent = "Analyzing…";
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
    result.innerHTML = html;
  } catch (e) { result.innerHTML = `<span class="text-danger">${e.message}</span>`; }
}

// ── Init ──
document.addEventListener("DOMContentLoaded", () => {
  // Auto-refresh when SUBOFF tab shown
  const origShowTab = window.showTab;
  window.showTab = function (name, el) {
    origShowTab(name, el);
    if (name === "suboff") {
      suboffRefreshStatus();
      suboffScanData();
    }
  };
});
