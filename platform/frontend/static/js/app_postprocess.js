// Post-processing
// ============================================================
function updatePPJobSelect() {
  const sel = document.getElementById('pp-job-select');
  if (!sel) return;
  const cur = sel.value;
  const jobs = Object.values(jobsMap).sort((a,b) => b.created_at.localeCompare(a.created_at));
  sel.innerHTML = `<option value="">${t('postprocess.select_hint')}</option>` +
    jobs.filter(j => j.status === 'completed' || j.status === 'failed')
        .map(j => `<option value="${j.job_id}"${j.job_id===cur?' selected':''}>${escHtml(j.name)} (${j.job_id})</option>`)
        .join('');
}

function onPPJobSelect() {
  const sel = document.getElementById('pp-job-select');
  ppSelectedJobId = sel.value || null;
  refreshPP();
}

async function refreshPP() {
  if (!ppSelectedJobId) return;
  const job = jobsMap[ppSelectedJobId];
  if (!job) return;

  // Summary
  try {
    const sum = await api('GET', `/api/postprocess/summary/${ppSelectedJobId}`);
    const dur = sum.duration_s !== null ? `${sum.duration_s}s` : '—';
    document.getElementById('pp-summary').innerHTML = `
      <table class="table table-sm mb-0">
        <tr><th>${t('postprocess.job_id')}</th><td>${sum.job_id}</td></tr>
        <tr><th>${t('postprocess.name')}</th><td>${escHtml(sum.job_name)}</td></tr>
        <tr><th>${t('postprocess.type')}</th><td>${sum.job_type}</td></tr>
        <tr><th>${t('postprocess.status')}</th><td><span class="dot dot-${sum.status}"></span> ${sum.status}</td></tr>
        <tr><th>${t('postprocess.duration')}</th><td>${dur}</td></tr>
        <tr><th>${t('postprocess.png_snapshots')}</th><td>${sum.png_files}</td></tr>
        <tr><th>${t('postprocess.csv_files')}</th><td>${sum.csv_files}</td></tr>
      </table>`;
  } catch(e) { document.getElementById('pp-summary').textContent = e.message; }

  if (ppCurrentTab === 'snapshots') await loadSnapshots();
  else if (ppCurrentTab === 'logs') loadLogs();
  else if (ppCurrentTab === 'files') loadFiles();
  else if (ppCurrentTab === 'metadata') loadMetadata();
  else if (ppCurrentTab === 'viewer') await loadViewerCheckpoints();
}

function showPPTab(name, el) {
  ppCurrentTab = name;
  document.querySelectorAll('#pp-tabs .nav-link').forEach(a => a.classList.remove('active'));
  el.classList.add('active');
  document.querySelectorAll('.pp-tab-panel').forEach(p => p.style.display = 'none');
  document.getElementById(`pp-${name}`).style.display = '';
  if (!ppSelectedJobId) return false;
  if (name === 'snapshots') loadSnapshots();
  else if (name === 'logs') loadLogs();
  else if (name === 'files') loadFiles();
  else if (name === 'metadata') loadMetadata();
  else if (name === 'viewer') loadViewerCheckpoints();
  return false;
}

async function loadSnapshots() {
  const grid = document.getElementById('snapshots-grid');
  grid.innerHTML = `<div class="col-12"><span class="spinner-border spinner-border-sm"></span> ${t('postprocess.loading')}</div>`;
  try {
    const r = await api('GET', `/api/jobs/${ppSelectedJobId}/images`);
    if (!r.images.length) {
      grid.innerHTML = `<div class="col-12 text-muted small">${t('postprocess.no_snapshots')}</div>`;
      return;
    }
    // Load images lazily – show placeholders, load on click
    grid.innerHTML = r.images.map(img => `
      <div class="col-sm-6 col-md-4">
        <div class="card p-1">
          <img src="/api/jobs/${ppSelectedJobId}/files/${img}" class="result-img img-thumb"
               loading="lazy" alt="${img}"
               onclick="openLightbox(this.src)" />
          <div class="small text-muted p-1" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${img}</div>
        </div>
      </div>`).join('');
  } catch(e) { grid.innerHTML = `<div class="col-12 alert alert-danger small">${e.message}</div>`; }
}

function loadLogs() {
  const box = document.getElementById('pp-log-box');
  const job = jobsMap[ppSelectedJobId];
  if (!job) { box.textContent = t('postprocess.job_not_found'); return; }
  box.textContent = job.logs.length ? job.logs.join('\n') : t('postprocess.no_log');
  box.scrollTop = box.scrollHeight;
}

async function loadFiles() {
  const el = document.getElementById('files-table');
  el.innerHTML = `<span class="spinner-border spinner-border-sm"></span> ${t('postprocess.loading')}`;
  try {
    const r = await api('GET', `/api/jobs/${ppSelectedJobId}/files`);
    if (!r.files.length) { el.innerHTML = `<p class="text-muted small">${t('postprocess.no_files')}</p>`; return; }
    const rows = r.files.map(f => `<tr>
      <td class="font-monospace small">${escHtml(f.path)}</td>
      <td class="small text-muted">${(f.size/1024).toFixed(1)} KB</td>
      <td class="small">${f.mime}</td>
      <td><a class="btn btn-sm btn-outline-primary py-0" href="/api/jobs/${ppSelectedJobId}/files/${f.path}" download>
        <i class="bi bi-download"></i>
      </a></td>
    </tr>`).join('');
    el.innerHTML = `<table class="table table-sm table-hover">
      <thead><tr><th>${t('postprocess.file_col')}</th><th>${t('postprocess.size_col')}</th><th>${t('postprocess.mime_col')}</th><th></th></tr></thead>
      <tbody>${rows}</tbody></table>`;
  } catch(e) { el.innerHTML = `<div class="alert alert-danger small">${e.message}</div>`; }
}

async function loadMetadata() {
  const box = document.getElementById('pp-metadata-box');
  try {
    const r = await api('GET', `/api/jobs/${ppSelectedJobId}/metadata`);
    box.textContent = JSON.stringify(r.metadata, null, 2);
  } catch(e) { box.textContent = e.message; }
}

// ============================================================
// Field Viewer (ParaView-style interactive post-processing)
// ============================================================

// State for the currently rendered field dataset
let _fvData = null;

async function loadViewerCheckpoints() {
  if (!ppSelectedJobId) return;
  const sel = document.getElementById('fv-checkpoint');
  if (!sel) return;
  try {
    const r = await api('GET', `/api/postprocess/checkpoints/${ppSelectedJobId}`);
    sel.innerHTML = '<option value="latest">latest</option>' +
      r.checkpoints.map(c => `<option value="${escHtml(c)}">${escHtml(c)}</option>`).join('');
    if (r.checkpoints.length === 0) {
      document.getElementById('fv-hint').textContent = t('postprocess.viewer_no_ckpt');
    } else {
      document.getElementById('fv-hint').textContent = t('postprocess.viewer_hint');
    }
  } catch(e) { /* silently ignore – may not be a 2D job */ }
}

async function renderFieldViewer() {
  if (!ppSelectedJobId) return;
  const btn = document.getElementById('fv-render-btn');
  const hint = document.getElementById('fv-hint');
  btn.disabled = true;
  hint.textContent = t('postprocess.loading');
  try {
    const field = document.getElementById('fv-field').value;
    const ckpt  = document.getElementById('fv-checkpoint').value;
    const r = await api('GET',
      `/api/postprocess/field-data/${ppSelectedJobId}?field=${field}&checkpoint=${encodeURIComponent(ckpt)}`
    );
    _fvData = r;
    _fvDrawCanvas(r);
    document.getElementById('fv-hint').style.display = 'none';
  } catch(e) {
    hint.style.display = '';
    hint.textContent = '⚠ ' + e.message;
  } finally { btn.disabled = false; }
}

// ---- Colormaps (256-entry RGB lookup tables) ----
const _CMAPS = {
  viridis: _buildCmap([
    [0.267,0.005,0.329],[0.283,0.141,0.459],[0.254,0.265,0.530],
    [0.207,0.372,0.553],[0.164,0.471,0.558],[0.128,0.566,0.551],
    [0.135,0.659,0.517],[0.267,0.749,0.441],[0.478,0.821,0.318],
    [0.741,0.873,0.150],[0.993,0.906,0.144]
  ]),
  plasma: _buildCmap([
    [0.050,0.030,0.528],[0.296,0.008,0.624],[0.494,0.013,0.657],
    [0.665,0.064,0.628],[0.807,0.163,0.548],[0.912,0.286,0.426],
    [0.973,0.421,0.303],[0.996,0.564,0.188],[0.981,0.716,0.147],
    [0.937,0.875,0.287],[0.940,0.975,0.131]
  ]),
  hot: _buildCmap([
    [0,0,0],[0.333,0,0],[0.667,0,0],[1,0,0],
    [1,0.333,0],[1,0.667,0],[1,1,0],[1,1,0.5],[1,1,1]
  ]),
  cool: _buildCmap([
    [0,1,1],[0.125,0.875,1],[0.25,0.75,1],[0.375,0.625,1],
    [0.5,0.5,1],[0.625,0.375,1],[0.75,0.25,1],[0.875,0.125,1],[1,0,1]
  ]),
  rdbu: _buildCmap([
    [0.647,0.082,0.094],[0.839,0.376,0.302],[0.957,0.647,0.510],
    [0.992,0.859,0.780],[0.969,0.969,0.969],[0.820,0.898,0.941],
    [0.573,0.773,0.871],[0.263,0.576,0.765],[0.129,0.400,0.675]
  ]),
  bwr: _buildCmap([
    [0,0,1],[0.5,0.5,1],[1,1,1],[1,0.5,0.5],[1,0,0]
  ]),
};

function _buildCmap(stops) {
  const lut = new Uint8ClampedArray(256 * 3);
  const N = stops.length - 1;
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const s = t * N;
    const lo = Math.floor(s), hi = Math.min(lo + 1, N);
    const f = s - lo;
    const [r0,g0,b0] = stops[lo];
    const [r1,g1,b1] = stops[hi];
    lut[i*3]   = Math.round((r0 + f*(r1-r0)) * 255);
    lut[i*3+1] = Math.round((g0 + f*(g1-g0)) * 255);
    lut[i*3+2] = Math.round((b0 + f*(b1-b0)) * 255);
  }
  return lut;
}

function _cmapColor(lut, t) {
  const i = Math.max(0, Math.min(255, Math.round(t * 255)));
  return [lut[i*3], lut[i*3+1], lut[i*3+2]];
}

function _fvDrawCanvas(r) {
  const cmapName = document.getElementById('fv-colormap').value;
  const lut = _CMAPS[cmapName] || _CMAPS.viridis;
  const showArrows = document.getElementById('fv-arrows').checked;
  const showStreamlines = document.getElementById('fv-streamlines').checked;

  const nx = r.nx, ny = r.ny;
  const fmin = r.field_min, fmax = r.field_max;
  const range = fmax - fmin || 1;

  // Canvas size – scale up to fill ~700px wide at most
  const SCALE = Math.max(1, Math.min(6, Math.floor(700 / nx)));
  const cw = nx * SCALE, ch = ny * SCALE;

  const canvas = document.getElementById('fv-canvas');
  canvas.width = cw; canvas.height = ch;
  canvas.style.display = 'inline-block';
  const ctx = canvas.getContext('2d');

  // Draw heatmap pixel-by-pixel
  const img = ctx.createImageData(cw, ch);
  const pix = img.data;
  for (let row = 0; row < ny; row++) {
    for (let col = 0; col < nx; col++) {
      const v = r.data[row * nx + col];
      const t = (v - fmin) / range;
      const [rv, gv, bv] = _cmapColor(lut, t);
      for (let sy = 0; sy < SCALE; sy++) {
        for (let sx = 0; sx < SCALE; sx++) {
          const px = ((row * SCALE + sy) * cw + col * SCALE + sx) * 4;
          pix[px]   = rv;
          pix[px+1] = gv;
          pix[px+2] = bv;
          pix[px+3] = 255;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);

  // Draw velocity vector arrows
  if (showArrows && r.ux && r.uy) {
    const STEP = Math.max(4, Math.round(Math.max(nx, ny) / 18));
    const maxU = Math.sqrt(
      r.ux.reduce((m, v) => Math.max(m, Math.abs(v)), 0) ** 2 +
      r.uy.reduce((m, v) => Math.max(m, Math.abs(v)), 0) ** 2
    ) || 1;
    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
    ctx.lineWidth = 1;
    for (let row = STEP; row < ny - STEP/2; row += STEP) {
      for (let col = STEP; col < nx - STEP/2; col += STEP) {
        const idx = row * nx + col;
        const ux = r.ux[idx], uy = r.uy[idx];
        const mag = Math.sqrt(ux*ux + uy*uy);
        if (mag < 1e-10) continue;
        const norm = mag / maxU;
        const len = norm * STEP * SCALE * 0.85;
        const cx0 = (col + 0.5) * SCALE;
        const cy0 = (row + 0.5) * SCALE;
        const dx = (ux / mag) * len;
        const dy = (uy / mag) * len;
        _drawArrow(ctx, cx0 - dx/2, cy0 - dy/2, cx0 + dx/2, cy0 + dy/2);
      }
    }
  }

  // Draw streamlines (Euler integration)
  if (showStreamlines && r.ux && r.uy) {
    ctx.strokeStyle = 'rgba(0,255,200,0.7)';
    ctx.lineWidth = 1.2;
    const SEEDS = 14;
    const DT = 0.5;
    const MAXSTEP = 400;
    for (let si = 0; si < SEEDS; si++) {
      for (let sj = 0; sj < SEEDS; sj++) {
        let px = (si + 0.5) / SEEDS * nx;
        let py = (sj + 0.5) / SEEDS * ny;
        ctx.beginPath();
        ctx.moveTo(px * SCALE, py * SCALE);
        for (let step = 0; step < MAXSTEP; step++) {
          const ix = Math.floor(px), iy = Math.floor(py);
          if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) break;
          const idx = iy * nx + ix;
          const ux = r.ux[idx], uy = r.uy[idx];
          const mag = Math.sqrt(ux*ux + uy*uy);
          if (mag < 1e-12) break;
          px += (ux / mag) * DT;
          py += (uy / mag) * DT;
          ctx.lineTo(px * SCALE, py * SCALE);
        }
        ctx.stroke();
      }
    }
  }

  // Draw color legend
  _drawLegend(lut, fmin, fmax);

  // Stats
  document.getElementById('fv-stats').textContent =
    `${t('postprocess.viewer_step')}: ${r.step}  |  ` +
    `${t('postprocess.viewer_grid')}: ${r.nx_orig}×${r.ny_orig}  |  ` +
    `${t('postprocess.viewer_min')}: ${fmin.toExponential(3)}  ` +
    `${t('postprocess.viewer_max')}: ${fmax.toExponential(3)}`;

  const fieldLabel = t(`postprocess.field_${r.field}`) || r.field;
  document.getElementById('fv-title').textContent = `${fieldLabel} (${cmapName})`;

  // Hover tooltip
  canvas.onmousemove = function(e) {
    const rect = canvas.getBoundingClientRect();
    const col = Math.floor((e.clientX - rect.left) / SCALE);
    const row = Math.floor((e.clientY - rect.top)  / SCALE);
    if (col < 0 || col >= nx || row < 0 || row >= ny) return;
    const v = r.data[row * nx + col];
    const tip = document.getElementById('fv-tooltip');
    tip.style.display = 'block';
    tip.style.left = (e.clientX + 12) + 'px';
    tip.style.top  = (e.clientY - 24) + 'px';
    tip.textContent = `(${col}, ${row})  ${v.toExponential(4)}`;
  };
  canvas.onmouseleave = () => {
    document.getElementById('fv-tooltip').style.display = 'none';
  };
}

function _drawArrow(ctx, x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.sqrt(dx*dx + dy*dy);
  if (len < 1) return;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  // Arrowhead
  const hw = Math.min(len * 0.35, 4);
  const angle = Math.atan2(dy, dx);
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - hw * Math.cos(angle - 0.45), y2 - hw * Math.sin(angle - 0.45));
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - hw * Math.cos(angle + 0.45), y2 - hw * Math.sin(angle + 0.45));
  ctx.stroke();
}

function _drawLegend(lut, vmin, vmax) {
  const legend = document.getElementById('fv-legend');
  legend.style.display = 'inline-block';
  const lctx = legend.getContext('2d');
  const H = legend.height;
  for (let i = 0; i < H; i++) {
    const t = 1 - i / (H - 1);
    const [r, g, b] = _cmapColor(lut, t);
    lctx.fillStyle = `rgb(${r},${g},${b})`;
    lctx.fillRect(0, i, 30, 1);
  }
  // Tick labels – draw on canvas as text
  lctx.fillStyle = '#333';
  lctx.font = '9px sans-serif';
  const ticks = [0, 0.25, 0.5, 0.75, 1.0];
  ticks.forEach(frac => {
    const y = Math.round((1 - frac) * (H - 1));
    const v = vmin + frac * (vmax - vmin);
    lctx.fillStyle = '#333';
    lctx.fillRect(0, y, 30, 1);
  });
}

// ============================================================
