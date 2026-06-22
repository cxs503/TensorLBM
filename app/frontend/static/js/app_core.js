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
let dashboardLiveMetricsTimer = null;
let dashboardLiveMetricsSinceStep = 0;
let dashboardLiveMetricsCache = [];
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
      dashboardRenderSelectedJob();
      updatePPJobSelect();
      updateStats();
    } else if (msg.type === 'job_update') {
      const j = msg.job;
      jobsMap[j.job_id] = j;
      renderJobsSidebar();
      dashboardRenderSelectedJob();
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
    // Render workflow pipeline from embedded summary
    if (s.workflow_summary && s.workflow_summary.stages) {
      renderWorkflowPipeline(s.workflow_summary);
    }
  } catch(e) { /* ignore */ }
}

async function loadJobs() {
  try {
    const r = await api('GET', '/api/jobs/');
    const jobs = Array.isArray(r) ? r : (r.jobs || []);
    jobs.forEach(j => { jobsMap[j.job_id] = j; });
    renderJobsSidebar();
    dashboardRenderSelectedJob();
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
// Dashboard workflow operations
// ============================================================
function dashboardSetText(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

function dashboardGetSelectedJob() {
  return selectedJobId ? jobsMap[selectedJobId] || null : null;
}

function dashboardRenderSelectedJob() {
  const el = document.getElementById('dashboard-selected-job');
  if (!el) return;
  const job = dashboardGetSelectedJob();
  if (!job) {
    el.textContent = t('dashboard.no_job_selected');
    return;
  }
  el.innerHTML = `<strong>${escHtml(job.name || job.job_id)}</strong> · <code>${escHtml(job.job_id)}</code> · ${escHtml(job.status || 'unknown')}`;
}

function dashboardChooseJob(id) {
  selectedJobId = id;
  uiState.selectedJobId = id;
  saveUIState();
  renderJobsSidebar();
  dashboardRefreshSelectedJobOps();
}

function dashboardResetLiveMetrics() {
  dashboardLiveMetricsSinceStep = 0;
  dashboardLiveMetricsCache = [];
  dashboardSetText('dashboard-live-summary', '-');
  dashboardSetText('dashboard-live-metrics-result', '-');
}

function dashboardToggleLiveMetricsAuto() {
  if (dashboardLiveMetricsTimer) {
    clearInterval(dashboardLiveMetricsTimer);
    dashboardLiveMetricsTimer = null;
  }
  if (document.getElementById('dashboard-live-auto')?.checked) {
    dashboardLiveMetricsTimer = setInterval(() => {
      dashboardLoadLiveMetrics();
    }, 10000);
  }
}

async function dashboardLoadLiveMetrics() {
  const job = dashboardGetSelectedJob();
  if (!job) {
    dashboardResetLiveMetrics();
    return;
  }
  dashboardSetText('dashboard-live-summary', t('common.loading'));
  try {
    const r = await api(
      'GET',
      `/api/jobs/${encodeURIComponent(job.job_id)}/live-metrics?since_step=${dashboardLiveMetricsSinceStep}&limit=50`,
    );
    const incoming = Array.isArray(r.diagnostics) ? r.diagnostics : [];
    if (incoming.length) {
      dashboardLiveMetricsCache = dashboardLiveMetricsCache.concat(incoming).slice(-100);
      dashboardLiveMetricsSinceStep = Math.max(
        dashboardLiveMetricsSinceStep,
        ...incoming.map(item => Number(item.step || 0)),
      );
    }
    const latest = dashboardLiveMetricsCache[dashboardLiveMetricsCache.length - 1] || null;
    const summary = latest
      ? `${t('dashboard.live_metrics_records')}: ${r.total_diagnostics} · ${t('dashboard.live_metrics_latest_step')}: ${latest.step ?? '-'} · ${t('dashboard.live_metrics_status')}: ${r.status}`
      : `${t('dashboard.live_metrics_empty')} (${r.status})`;
    dashboardSetText('dashboard-live-summary', summary);
    dashboardSetText('dashboard-live-metrics-result', dashboardLiveMetricsCache.length ? dashboardLiveMetricsCache : []);
  } catch (e) {
    dashboardSetText('dashboard-live-summary', `${t('common.error')} ${e.message}`);
    dashboardSetText('dashboard-live-metrics-result', `${t('common.error')} ${e.message}`);
  }
}

async function dashboardApplyAutoStop() {
  const job = dashboardGetSelectedJob();
  if (!job) {
    showToast(t('dashboard.no_job_selected'), 'warning');
    return;
  }
  const body = {
    enabled: !!document.getElementById('dashboard-auto-stop-enabled')?.checked,
    residual_key: document.getElementById('dashboard-auto-stop-key')?.value || 'residual',
    rel_tol: Number(document.getElementById('dashboard-auto-stop-tol')?.value || 1e-4),
    patience: Number(document.getElementById('dashboard-auto-stop-patience')?.value || 5),
    min_steps: Number(document.getElementById('dashboard-auto-stop-min-steps')?.value || 20),
  };
  dashboardSetText('dashboard-auto-stop-result', t('common.submitting'));
  try {
    const r = await api('PATCH', `/api/jobs/${encodeURIComponent(job.job_id)}/auto-stop-config`, body);
    dashboardSetText('dashboard-auto-stop-result', r);
    showToast(t('dashboard.auto_stop_applied'), 'success');
  } catch (e) {
    dashboardSetText('dashboard-auto-stop-result', `${t('common.error')} ${e.message}`);
    showToast(`${t('common.error')} ${e.message}`, 'danger');
  }
}

async function dashboardSubmitHpc() {
  const job = dashboardGetSelectedJob();
  if (!job) {
    showToast(t('dashboard.no_job_selected'), 'warning');
    return;
  }
  const body = {
    partition: document.getElementById('dashboard-hpc-partition')?.value || null,
    nodes: Number(document.getElementById('dashboard-hpc-nodes')?.value || 1),
    cpus: Number(document.getElementById('dashboard-hpc-cpus')?.value || 4),
    mem: document.getElementById('dashboard-hpc-mem')?.value || null,
    walltime: document.getElementById('dashboard-hpc-walltime')?.value || null,
  };
  dashboardSetText('dashboard-hpc-result', t('common.submitting'));
  try {
    const r = await api('POST', `/api/jobs/${encodeURIComponent(job.job_id)}/submit-hpc`, body);
    dashboardSetText('dashboard-hpc-result', r);
    showToast(t('dashboard.hpc_submitted'), 'success');
  } catch (e) {
    dashboardSetText('dashboard-hpc-result', `${t('common.error')} ${e.message}`);
    showToast(`${t('common.error')} ${e.message}`, 'danger');
  }
}

async function dashboardLoadTimeline() {
  const box = document.getElementById('dashboard-timeline-table');
  if (!box) return;
  const status = document.getElementById('dashboard-timeline-status')?.value || '';
  const limit = Number(document.getElementById('dashboard-timeline-limit')?.value || 8);
  box.innerHTML = `<span class="text-muted">${t('common.loading')}</span>`;
  const params = new URLSearchParams({ limit: String(limit) });
  if (status) params.set('status', status);
  try {
    const r = await api('GET', `/api/jobs/timeline?${params.toString()}`);
    const rows = Array.isArray(r.timeline) ? r.timeline : [];
    if (!rows.length) {
      box.innerHTML = `<span class="text-muted">${t('dashboard.timeline_empty')}</span>`;
      return;
    }
    box.innerHTML = `
      <table class="table table-sm align-middle mb-0">
        <thead>
          <tr>
            <th>${t('dashboard.timeline_job')}</th>
            <th>${t('dashboard.timeline_status_col')}</th>
            <th>${t('dashboard.timeline_queue')}</th>
            <th>${t('dashboard.timeline_duration')}</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(row => `
            <tr>
              <td>
                <button class="btn btn-link btn-sm p-0 text-start" onclick="dashboardChooseJob('${escHtml(row.job_id)}')">${escHtml(row.name || row.job_id)}</button>
                <div class="text-muted" style="font-size:.72rem">${escHtml(row.job_type || '')}</div>
              </td>
              <td>${escHtml(row.status || '-')}</td>
              <td>${row.queue_wait_s == null ? '—' : `${Number(row.queue_wait_s).toFixed(1)}s`}</td>
              <td>${row.duration_s == null ? '—' : `${Number(row.duration_s).toFixed(1)}s`}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>`;
  } catch (e) {
    box.innerHTML = `<span class="text-danger">${escHtml(e.message)}</span>`;
  }
}

async function dashboardLoadNotifications() {
  try {
    const r = await api('GET', '/api/notifications/settings');
    const webhookEl = document.getElementById('dashboard-notify-webhook');
    if (webhookEl) webhookEl.value = r.webhook_url || '';
    const completeEl = document.getElementById('dashboard-notify-complete');
    if (completeEl) completeEl.checked = !!r.notify_on_complete;
    const failureEl = document.getElementById('dashboard-notify-failure');
    if (failureEl) failureEl.checked = !!r.notify_on_failure;
    const cancelEl = document.getElementById('dashboard-notify-cancel');
    if (cancelEl) cancelEl.checked = !!r.notify_on_cancel;
    const timeoutEl = document.getElementById('dashboard-notify-timeout');
    if (timeoutEl) timeoutEl.value = String(r.timeout_s || 10);
    dashboardSetText('dashboard-notify-result', '-');
  } catch (e) {
    dashboardSetText('dashboard-notify-result', `${t('common.error')} ${e.message}`);
  }
}

function dashboardNotificationBody() {
  return {
    webhook_url: document.getElementById('dashboard-notify-webhook')?.value || '',
    notify_on_complete: !!document.getElementById('dashboard-notify-complete')?.checked,
    notify_on_failure: !!document.getElementById('dashboard-notify-failure')?.checked,
    notify_on_cancel: !!document.getElementById('dashboard-notify-cancel')?.checked,
    timeout_s: Number(document.getElementById('dashboard-notify-timeout')?.value || 10),
  };
}

async function dashboardSaveNotifications() {
  dashboardSetText('dashboard-notify-result', t('common.submitting'));
  try {
    const r = await api('POST', '/api/notifications/settings', dashboardNotificationBody());
    dashboardSetText('dashboard-notify-result', r);
    showToast(t('dashboard.notify_saved'), 'success');
  } catch (e) {
    dashboardSetText('dashboard-notify-result', `${t('common.error')} ${e.message}`);
    showToast(`${t('common.error')} ${e.message}`, 'danger');
  }
}

async function dashboardTestWebhook() {
  const url = document.getElementById('dashboard-notify-webhook')?.value || '';
  if (!url) {
    showToast(t('dashboard.notify_webhook_required'), 'warning');
    return;
  }
  dashboardSetText('dashboard-notify-result', t('common.submitting'));
  try {
    const r = await api('POST', '/api/notifications/webhook-test', { url });
    dashboardSetText('dashboard-notify-result', r);
  } catch (e) {
    dashboardSetText('dashboard-notify-result', `${t('common.error')} ${e.message}`);
  }
}

async function dashboardRefreshSelectedJobOps() {
  dashboardRenderSelectedJob();
  if (!dashboardGetSelectedJob()) {
    dashboardResetLiveMetrics();
    dashboardSetText('dashboard-auto-stop-result', '-');
    dashboardSetText('dashboard-hpc-result', '-');
    return;
  }
  await dashboardLoadLiveMetrics();
}

function dashboardInit() {
  dashboardRenderSelectedJob();
  dashboardLoadTimeline();
  dashboardLoadNotifications();
  dashboardToggleLiveMetricsAuto();
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
  dashboardRenderSelectedJob();
  // Also update post-process if on that tab
  ppSelectedJobId = id;
  const sel = document.getElementById('pp-job-select');
  if (sel) sel.value = id;
  refreshPP();
  if (document.getElementById('panel-dashboard')?.classList.contains('active')) {
    dashboardRefreshSelectedJobOps();
  }
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
  dashboardRenderSelectedJob();
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
  if (name !== 'dashboard' && dashboardLiveMetricsTimer) {
    clearInterval(dashboardLiveMetricsTimer);
    dashboardLiveMetricsTimer = null;
  }
  if (name === 'dashboard') dashboardInit();
  if (name === 'compare') refreshCompareJobList();
  if (name === 'ai-flow') aiFlowListModels();
  if (name === 'projects') {
    if (typeof projectsInit === 'function') projectsInit();
  }
  if (name === 'preprocess') {
    loadMaterials();
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
    const r = await fetch(`/api/reports/compare/kpis?${qs}`).then(async x => {
      if (!x.ok) throw new Error(await x.text());
      return x.json();
    });
    if (!r.rows || !r.rows.length) {
      result.innerHTML = `<div class="alert alert-warning">${t('compare.no_data')}</div>`;
      return;
    }
    const rows = [];
    rows.push([t('compare.field_col'), ...r.rows.map((_, i) => `${t('compare.run_col')} ${i + 1}`)]);
    rows.push([t('postprocess.name'), ...r.rows.map(e => escHtml(e.name))]);
    rows.push([t('postprocess.type'), ...r.rows.map(e => escHtml(e.job_type))]);
    rows.push([t('postprocess.status'), ...r.rows.map(e => `<span class="badge bg-secondary">${escHtml(e.status)}</span>`)]);
    rows.push(['Report', ...r.rows.map(e => `<a class="btn btn-sm btn-outline-primary py-0" href="${escHtml(e.report_url)}" target="_blank">HTML</a>`)]);
    const metricKeys = new Set();
    r.rows.forEach(e => Object.keys(e.compare_metrics || {}).forEach(k => metricKeys.add(k)));
    Array.from(metricKeys).sort().forEach(k => {
      rows.push([
        `<i>${escHtml(k)}</i>`,
        ...r.rows.map(e => {
          const v = (e.compare_metrics || {})[k];
          const isBest = r.metric_summary[k] && r.metric_summary[k].best_job_id === e.job_id;
          return v === undefined ? '<span class="text-muted">—</span>'
                                 : `<span${isBest ? ' class="fw-bold text-success"' : ''}>${Math.abs(v) < 1e-3 || Math.abs(v) >= 1e4 ? v.toExponential(4) : v.toFixed(6)}</span>`;
        }),
      ]);
    });

    const thead = `<thead><tr><th>${rows[0][0]}</th>${r.rows.map((_, i) =>
      `<th>${t('compare.run_col')} ${i + 1}</th>`).join('')}</tr></thead>`;
    const tbody = rows.slice(1).map(row =>
      `<tr><th class="small">${row[0]}</th>${row.slice(1).map(c =>
        `<td class="small">${c}</td>`).join('')}</tr>`).join('');
    let html = `<div class="table-responsive"><table class="table table-sm table-hover">${thead}<tbody>${tbody}</tbody></table></div>`;
    if (r.missing && r.missing.length) {
      html += `<div class="alert alert-warning small mt-2">${t('compare.missing_jobs')} ${r.missing.join(', ')}</div>`;
    }
    if (r.metric_summary && Object.keys(r.metric_summary).length) {
      html += '<div class="table-responsive mt-3"><table class="table table-sm table-bordered small">';
      html += '<thead><tr><th>Metric</th><th>Min</th><th>Max</th><th>Mean</th><th>Best Job</th></tr></thead><tbody>';
      Object.entries(r.metric_summary).forEach(([key, stats]) => {
        html += `<tr><td>${escHtml(key)}</td><td>${stats.min.toFixed(6)}</td><td>${stats.max.toFixed(6)}</td><td>${stats.mean.toFixed(6)}</td><td class="font-monospace">${escHtml(String(stats.best_job_id).slice(-8))}</td></tr>`;
      });
      html += '</tbody></table></div>';
    }
    result.innerHTML = html;
  } catch (e) {
    result.innerHTML = `<div class="alert alert-danger small">${t('common.error')} ${escHtml(String(e))}</div>`;
  }
}

// ============================================================
// Workflow pipeline dashboard widget
// ============================================================

function renderWorkflowPipeline(wf) {
  const el = document.getElementById('wf-pipeline');
  if (!el || !wf) return;
  const stageIcons = {
    draft: 'bi-file-earmark',
    setup: 'bi-sliders',
    meshed: 'bi-grid-3x3',
    solved: 'bi-activity',
    post_processed: 'bi-bar-chart-line',
  };
  const stageCols = {
    draft: 'secondary',
    setup: 'info',
    meshed: 'primary',
    solved: 'success',
    post_processed: 'dark',
  };
  const parts = (wf.stages || []).map((stage, i) => {
    const cnt = (wf.counts || {})[stage] || 0;
    const icon = stageIcons[stage] || 'bi-circle';
    const col = stageCols[stage] || 'secondary';
    const label = t(`projects.stage_${stage}`) || stage;
    const arrow = i < (wf.stages.length - 1) ? '<i class="bi bi-arrow-right text-muted mx-1"></i>' : '';
    return `<div class="d-inline-flex align-items-center gap-1">
      <span class="badge bg-${col} fs-6 px-2 py-1"><i class="bi ${icon}"></i> ${escHtml(label)} <strong>${cnt}</strong></span>
    </div>${arrow}`;
  });
  el.innerHTML = parts.join('') +
    `<span class="text-muted small ms-3">${t('stat.total') || 'Total'}: <strong>${wf.total_cases || 0}</strong></span>`;
}

// ============================================================
// Fluid material database
// ============================================================

async function loadMaterials() {
  const tableEl = document.getElementById('material-table');
  if (!tableEl) return;
  const filterEl = document.getElementById('material-filter');
  const category = filterEl ? filterEl.value : '';
  tableEl.innerHTML = `<span class="text-muted">${t('common.loading')}</span>`;
  try {
    const url = '/api/preprocess/materials' + (category ? `?category=${encodeURIComponent(category)}` : '');
    const r = await api('GET', url);
    if (!r.materials || !r.materials.length) {
      tableEl.innerHTML = `<span class="text-muted">${t('preprocess.no_materials')}</span>`;
      return;
    }
    const rows = r.materials.map(m => `
<tr>
  <td><strong>${escHtml(m.name)}</strong>${m.notes ? `<div class="text-muted" style="font-size:.75rem">${escHtml(m.notes)}</div>` : ''}</td>
  <td class="text-center">${escHtml(m.category)}</td>
  <td class="text-end">${m.density_kg_m3}</td>
  <td class="text-end">${m.kinematic_viscosity_m2_s !== null ? m.kinematic_viscosity_m2_s.toExponential(3) : '—'}</td>
  <td class="text-end">${m.dynamic_viscosity_pa_s !== null ? m.dynamic_viscosity_pa_s.toExponential(3) : '—'}</td>
  <td class="text-end">${m.surface_tension_n_m !== null ? m.surface_tension_n_m : '—'}</td>
  <td class="text-end">${m.ref_temp_c} °C</td>
  <td>
    <button class="btn btn-xs btn-outline-secondary" style="font-size:.72rem;padding:.1rem .4rem"
      onclick="materialFillUnitConverter(${m.kinematic_viscosity_m2_s})">→ UC</button>
  </td>
</tr>`).join('');
    tableEl.innerHTML = `<div class="table-responsive">
<table class="table table-sm table-hover">
<thead class="table-light"><tr>
  <th data-i18n="common.name">Name</th>
  <th>Category</th>
  <th class="text-end"><span data-i18n="preprocess.material_density">Density</span></th>
  <th class="text-end">ν (m²/s)</th>
  <th class="text-end">μ (Pa·s)</th>
  <th class="text-end">σ (N/m)</th>
  <th class="text-end">T ref</th>
  <th></th>
</tr></thead><tbody>${rows}</tbody>
</table></div>`;
    i18n.apply(tableEl);
  } catch(e) {
    tableEl.innerHTML = `<div class="alert alert-danger small">${escHtml(String(e.message))}</div>`;
  }
}

/** Fill the Unit Converter ν field from a material selection. */
function materialFillUnitConverter(nu) {
  const el = document.getElementById('uc-nu');
  if (el && nu != null) {
    el.value = nu;
    showTab('preprocess', document.querySelector('[data-tab="preprocess"]'));
  }
}

// ============================================================
