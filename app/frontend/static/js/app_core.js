// ============================================================
// Global state
// ============================================================
const API = '';   // same origin
let ws = null;
const jobsMap = {};     // job_id → job dict
let selectedJobId = null;
let ppSelectedJobId = null;
let ppCurrentTab = 'snapshots';
let benchJobMap = {};   // bench_type → job_id
const UI_STORAGE_KEY = 'tensorlbm_ui_state_v1';
const TAB_SEQUENCE = ['dashboard', 'projects', 'templates', 'cad', 'preprocess', 'solve', 'postprocess', 'reports', 'benchmarks', 'compare', 'ai-flow', 'orchestration', 'agent', 'suboff'];
const uiState = {
  activeTab: 'dashboard',
  jobsSearch: '',
  jobsStatus: 'all',
  selectedJobId: null,
};

// ============================================================
// Initialisation
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  i18n.init().then(() => {
    loadUIState();
    initUIStateControls();
    bindKeyboardShortcuts();
    initPhysicsLayer();
    connectWS();
    loadStatus();
    loadJobs();
    loadAgentInfo();
    onSimTypeChange();
    setInterval(loadStatus, 15000);
    onCADHullTypeChange();
    showTab(uiState.activeTab, null);
  });
});

function loadUIState() {
  try {
    const raw = localStorage.getItem(UI_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object') {
      uiState.activeTab = TAB_SEQUENCE.includes(parsed.activeTab) ? parsed.activeTab : uiState.activeTab;
      uiState.jobsSearch = typeof parsed.jobsSearch === 'string' ? parsed.jobsSearch : uiState.jobsSearch;
      uiState.jobsStatus = ['all', 'queued', 'running', 'completed', 'failed'].includes(parsed.jobsStatus)
        ? parsed.jobsStatus
        : uiState.jobsStatus;
      uiState.selectedJobId = typeof parsed.selectedJobId === 'string' ? parsed.selectedJobId : null;
    }
  } catch (_) {}
}

function saveUIState() {
  try {
    localStorage.setItem(UI_STORAGE_KEY, JSON.stringify(uiState));
  } catch (_) {}
}

function initUIStateControls() {
  const search = document.getElementById('jobs-search');
  const status = document.getElementById('jobs-status-filter');
  if (search) search.value = uiState.jobsSearch;
  if (status) status.value = uiState.jobsStatus;
  if (uiState.selectedJobId) selectedJobId = uiState.selectedJobId;
}

function onJobsFilterChanged() {
  const search = document.getElementById('jobs-search');
  const status = document.getElementById('jobs-status-filter');
  uiState.jobsSearch = (search ? search.value : '').trim();
  uiState.jobsStatus = status ? status.value : 'all';
  saveUIState();
  renderJobsSidebar();
}

function bindKeyboardShortcuts() {
  document.addEventListener('keydown', (ev) => {
    const tag = (ev.target && ev.target.tagName || '').toLowerCase();
    const inInput = ['input', 'textarea', 'select'].includes(tag) || (ev.target && ev.target.isContentEditable);
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 'k') {
      ev.preventDefault();
      const input = document.getElementById('jobs-search');
      if (input) {
        input.focus();
        input.select();
      }
      return;
    }
    if (ev.altKey && !ev.shiftKey && !ev.ctrlKey && !ev.metaKey && /^[0-9]$/.test(ev.key)) {
      const idx = ev.key === '0' ? 9 : Number(ev.key) - 1;
      const tab = TAB_SEQUENCE[idx];
      if (!tab) return;
      const nav = document.querySelector(`.top-navbar nav a[data-tab="${tab}"]`);
      showTab(tab, nav || null);
      ev.preventDefault();
      return;
    }
    if (!inInput && ev.key === '/') {
      ev.preventDefault();
      const input = document.getElementById('jobs-search');
      if (input) input.focus();
    }
  });
}

// ============================================================
// WebSocket
// ============================================================
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => {
    document.getElementById('ws-status').innerHTML =
      '<span class="dot dot-completed"></span> ' + t('ws.connected');
  };
  ws.onclose = () => {
    document.getElementById('ws-status').innerHTML =
      '<span class="dot dot-failed"></span> ' + t('ws.disconnected');
    setTimeout(connectWS, 3000);
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'init') {
      msg.jobs.forEach(j => { jobsMap[j.job_id] = j; });
      renderJobsSidebar();
      updatePPJobSelect();
      updateStats();
    } else if (msg.type === 'job_update') {
      const j = msg.job;
      jobsMap[j.job_id] = j;
      renderJobsSidebar();
      updatePPJobSelect();
      updateStats();
      updateBenchStatus(j);
      if (ppSelectedJobId === j.job_id) refreshPP();
      if (aiFlowActiveJobId === j.job_id || j.job_type === 'ai_transformer_train') aiFlowHandleJob(j);
    }
  };
  // Ping to keep alive
  setInterval(() => { if (ws && ws.readyState === 1) ws.send('ping'); }, 30000);
}

// ============================================================
// REST helpers
// ============================================================
async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`${r.status}: ${txt}`);
  }
  return r.json();
}

async function loadStatus() {
  try {
    const s = await api('GET', '/api/status');
    const el = document.getElementById('platform-status');
    const cudaInfo = s.cuda_available
      ? `<span class="text-success"><i class="bi bi-gpu-card"></i> ${s.gpu_count} GPU(s): ${s.gpu_names.join(', ')}</span>`
      : `<span class="text-muted"><i class="bi bi-cpu"></i> ${t('dashboard.cpu_only')}</span>`;
    el.innerHTML = `
      <table class="table table-sm mb-0">
        <tr><th>${t('dashboard.version')}</th><td>${s.version}</td></tr>
        <tr><th>${t('dashboard.cuda')}</th><td>${cudaInfo}</td></tr>
        <tr><th>${t('dashboard.available_devices')}</th><td>${s.devices.join(', ')}</td></tr>
      </table>`;
    // Populate device dropdowns
    const devSelects = document.querySelectorAll('select[id$="-device"]');
    devSelects.forEach(sel => {
      const cur = sel.value;
      sel.innerHTML = s.devices.map(d => `<option value="${d}"${d===cur?' selected':''}>${d}</option>`).join('');
    });
    updateStats(s);
  } catch(e) { /* ignore */ }
}

async function loadJobs() {
  try {
    const jobs = await api('GET', '/api/jobs/');
    jobs.forEach(j => { jobsMap[j.job_id] = j; });
    renderJobsSidebar();
    updatePPJobSelect();
    updateStats();
  } catch(e) { /* ignore */ }
}

// ============================================================
// Stats
// ============================================================
function updateStats(s) {
  const jobs = Object.values(jobsMap);
  document.getElementById('stat-total').textContent = s ? s.total_jobs : jobs.length;
  document.getElementById('stat-running').textContent = s ? s.running_jobs : jobs.filter(j=>j.status==='running').length;
  document.getElementById('stat-completed').textContent = s ? s.completed_jobs : jobs.filter(j=>j.status==='completed').length;
  document.getElementById('stat-failed').textContent = s ? s.failed_jobs : jobs.filter(j=>j.status==='failed').length;
}

// ============================================================
// Jobs sidebar
// ============================================================
function renderJobsSidebar() {
  const list = document.getElementById('jobs-list');
  const allJobs = Object.values(jobsMap).sort((a,b) => b.created_at.localeCompare(a.created_at));
  const countEl = document.getElementById('jobs-count');
  if (countEl) countEl.textContent = String(allJobs.length);
  if (!allJobs.length) {
    list.innerHTML = `<div class="text-center text-muted py-4 small" id="no-jobs-msg">${t('sidebar.no_jobs')}</div>`;
    return;
  }
  const query = (uiState.jobsSearch || '').toLowerCase();
  const jobs = allJobs.filter(j => {
    if (uiState.jobsStatus !== 'all' && j.status !== uiState.jobsStatus) return false;
    if (!query) return true;
    const name = String(j.name || '').toLowerCase();
    const id = String(j.job_id || '').toLowerCase();
    return name.includes(query) || id.includes(query);
  });
  if (!jobs.length) {
    list.innerHTML = `<div class="text-center text-muted py-4 small">${t('sidebar.no_jobs_filtered')}</div>`;
    return;
  }
  if (selectedJobId && !jobsMap[selectedJobId]) selectedJobId = null;
  list.innerHTML = jobs.map(j => {
    const dot = `<span class="dot dot-${j.status}" style="margin-right:4px"></span>`;
    const badgeCls = {queued:'secondary',running:'warning',completed:'success',failed:'danger',cancelled:'dark'}[j.status]||'secondary';
    const dur = j.started_at && j.completed_at
      ? ` · ${((new Date(j.completed_at)-new Date(j.started_at))/1000).toFixed(1)}s` : '';
    return `<div class="job-card${selectedJobId===j.job_id?' selected':''}" onclick="selectJob('${j.job_id}')">
      <div class="job-name">${dot}${escHtml(j.name)}</div>
      <div class="job-meta">${j.job_id}${dur}</div>
      <div class="d-flex align-items-center justify-content-between mt-1">
        <span class="badge bg-${badgeCls} status-badge">${j.status}</span>
        <button class="btn btn-sm btn-outline-secondary py-0 px-1" onclick="reuseJobConfig(event,'${j.job_id}')" title="${t('solve.reuse_run')}">
          <i class="bi bi-arrow-repeat"></i>
        </button>
      </div>
    </div>`;
  }).join('');
}

function selectJob(id) {
  selectedJobId = id;
  uiState.selectedJobId = id;
  saveUIState();
  renderJobsSidebar();
  // Also update post-process if on that tab
  ppSelectedJobId = id;
  const sel = document.getElementById('pp-job-select');
  if (sel) sel.value = id;
  refreshPP();
  showTab('postprocess', document.querySelectorAll('.top-navbar nav a')[4]);
}

function reuseJobConfig(ev, id) {
  ev.stopPropagation();
  const job = jobsMap[id];
  if (!job || !SIM_TYPES[job.job_type]) {
    showToast(t('solve.reuse_not_supported'), 'warning');
    return;
  }
  const simSel = document.getElementById('sim-type');
  simSel.value = job.job_type;
  onSimTypeChange();
  const cfg = job.config || {};
  const physics = cfg.physics || {};
  if (physics.flow_type) document.getElementById('physics-flow-type').value = physics.flow_type;
  if (physics.turbulence_model) document.getElementById('physics-turbulence').value = physics.turbulence_model;
  if (physics.multiphase_model) document.getElementById('physics-multiphase').value = physics.multiphase_model;
  if (physics.boundary_condition) document.getElementById('physics-bc').value = physics.boundary_condition;
  if (physics.numerical_scheme) document.getElementById('physics-scheme').value = physics.numerical_scheme;
  const cs = physics.turbulence_params && physics.turbulence_params.smagorinsky_cs;
  if (cs !== undefined) document.getElementById('physics-cs').value = cs;
  applyCapabilityDefaults();
  currentSchema.fields.forEach(f => {
    if (cfg[f.name] === undefined) return;
    const el = document.getElementById(`field-${f.name}`);
    if (!el) return;
    el.value = cfg[f.name];
  });
  showTab('solve', document.querySelectorAll('.top-navbar nav a')[3]);
  showToast(`${t('solve.reuse_loaded')} ${id}`, 'success');
}

async function clearAllJobs() {
  const ids = Object.values(jobsMap)
    .filter(j => j.status !== 'running')
    .map(j => j.job_id);
  if (!ids.length) return;
  if (!window.confirm(t('sidebar.clear_confirm'))) return;
  for (const id of ids) {
    try { await api('DELETE', `/api/jobs/${id}`); delete jobsMap[id]; } catch(e) {}
  }
  if (selectedJobId && !jobsMap[selectedJobId]) {
    selectedJobId = null;
    uiState.selectedJobId = null;
  }
  saveUIState();
  renderJobsSidebar();
  updatePPJobSelect();
  updateStats();
}

// ============================================================
// Tab navigation
// ============================================================
function showTab(name, el) {
  if (!TAB_SEQUENCE.includes(name)) name = 'dashboard';
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`panel-${name}`).classList.add('active');
  document.querySelectorAll('.top-navbar nav a').forEach(a => a.classList.remove('active'));
  const activeEl = el || document.querySelector(`.top-navbar nav a[data-tab="${name}"]`);
  if (activeEl) activeEl.classList.add('active');
  uiState.activeTab = name;
  saveUIState();
  if (name === 'compare') refreshCompareJobList();
  if (name === 'ai-flow') aiFlowListModels();
  if (name === 'projects') {
    if (typeof projectsInit === 'function') projectsInit();
  }
  if (name === 'templates') {
    if (typeof templatesInit === 'function') templatesInit();
  }
  if (name === 'reports') {
    // Pre-fill the reports job-id field from the currently selected job
    const jobIdInput = document.getElementById('reports-job-id');
    if (jobIdInput && selectedJobId && !jobIdInput.value) {
      jobIdInput.value = selectedJobId;
    }
    if (typeof reportsTabInit === 'function') {
      const jid = (jobIdInput && jobIdInput.value.trim()) || selectedJobId;
      reportsTabInit(jid || null);
    }
  }
  if (name === 'orchestration') {
    orchLoadTemplates();
    orchLoadKpis();
  }
  if (name === 'agent') {
    setTimeout(() => {
      const el = document.getElementById('agent-input');
      if (el) el.focus();
    }, 50);
  }
  return false;
}

// ============================================================
// Compare runs
// ============================================================
function refreshCompareJobList() {
  const box = document.getElementById('compare-job-checkboxes');
  if (!box) return;
  const jobs = Object.values(jobsMap)
    .filter(j => j.status === 'completed')
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  if (!jobs.length) {
    box.innerHTML = `<span class="text-muted small">${t('compare.no_completed')}</span>`;
    return;
  }
  box.innerHTML = jobs.map(j => `
    <div class="form-check small">
      <input class="form-check-input compare-job-cb" type="checkbox" value="${j.job_id}" id="cmp-${j.job_id}">
      <label class="form-check-label" for="cmp-${j.job_id}">
        <span class="dot dot-${j.status}"></span>
        ${escHtml(j.name)} <span class="text-muted">(${j.job_id.slice(0, 8)}…)</span>
      </label>
    </div>`).join('');
}

async function runCompare() {
  const ids = Array.from(document.querySelectorAll('.compare-job-cb:checked')).map(cb => cb.value);
  const result = document.getElementById('compare-result');
  if (ids.length < 2) {
    result.innerHTML = `<div class="alert alert-warning">${t('compare.min_jobs')}</div>`;
    return;
  }
  if (ids.length > 10) {
    result.innerHTML = `<div class="alert alert-warning">${t('compare.max_jobs')}</div>`;
    return;
  }
  result.innerHTML = `<div class="text-muted small">${t('compare.loading')}</div>`;
  try {
    const qs = ids.map(id => `ids=${encodeURIComponent(id)}`).join('&');
    const r = await fetch(`/api/jobs/compare?${qs}`).then(x => x.json());
    if (!r.jobs || !r.jobs.length) {
      result.innerHTML = `<div class="alert alert-warning">${t('compare.no_data')}</div>`;
      return;
    }
    // Collect all metadata keys from "derived" + top-level numeric fields
    const rows = [];
    rows.push([t('compare.field_col'), ...r.jobs.map((_, i) => `${t('compare.run_col')} ${i + 1}`)]);
    rows.push([t('postprocess.name'), ...r.jobs.map(e => escHtml(e.job.name))]);
    rows.push([t('postprocess.type'), ...r.jobs.map(e => escHtml(e.job.job_type))]);
    rows.push([t('postprocess.status'), ...r.jobs.map(e => `<span class="badge bg-secondary">${e.job.status}</span>`)]);
    // Union of derived keys
    const derivedKeys = new Set();
    r.jobs.forEach(e => Object.keys(e.metadata.derived || {}).forEach(k => derivedKeys.add(k)));
    Array.from(derivedKeys).sort().forEach(k => {
      rows.push([
        `<i>derived.${k}</i>`,
        ...r.jobs.map(e => {
          const v = (e.metadata.derived || {})[k];
          return v === undefined ? '<span class="text-muted">—</span>'
                                 : (typeof v === 'number' ? v.toExponential(4) : escHtml(String(v)));
        }),
      ]);
    });
    // Scalar top-level metadata keys (numbers / strings)
    const scalarKeys = new Set();
    r.jobs.forEach(e => Object.entries(e.metadata).forEach(([k, v]) => {
      if (typeof v === 'number' || typeof v === 'string') scalarKeys.add(k);
    }));
    Array.from(scalarKeys).sort().forEach(k => {
      rows.push([
        k,
        ...r.jobs.map(e => {
          const v = e.metadata[k];
          if (v === undefined) return '<span class="text-muted">—</span>';
          return typeof v === 'number' ? (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e4
                                          ? v.toExponential(4) : v.toFixed(6))
                                       : escHtml(String(v));
        }),
      ]);
    });

    const thead = `<thead><tr><th>${rows[0][0]}</th>${r.jobs.map((_, i) =>
      `<th>${t('compare.run_col')} ${i + 1}</th>`).join('')}</tr></thead>`;
    const tbody = rows.slice(1).map(row =>
      `<tr><th class="small">${row[0]}</th>${row.slice(1).map(c =>
        `<td class="small">${c}</td>`).join('')}</tr>`).join('');
    let html = `<div class="table-responsive"><table class="table table-sm table-hover">${thead}<tbody>${tbody}</tbody></table></div>`;
    if (r.missing && r.missing.length) {
      html += `<div class="alert alert-warning small mt-2">${t('compare.missing_jobs')} ${r.missing.join(', ')}</div>`;
    }
    result.innerHTML = html;
  } catch (e) {
    result.innerHTML = `<div class="alert alert-danger small">${t('common.error')} ${escHtml(String(e))}</div>`;
  }
}

// ============================================================
