/* UI integration for engineering gap-closure features.
 *
 * This module wires the new post-processing and solver workflow helpers into
 * the existing browser UI:
 * - Spectral analysis (probe spectrum)
 * - POD modal decomposition
 * - Iso-surface / iso-contour extraction
 * - Study-group comparison / report aggregation
 *
 * The code is intentionally defensive: if a section is absent in a given
 * page layout, the helpers fail softly and leave the rest of the UI intact.
 */

function _engineerSafeEl(id) {
  return document.getElementById(id);
}

function _engineerSetHtml(id, html) {
  const el = _engineerSafeEl(id);
  if (!el) return false;
  el.innerHTML = html;
  return true;
}

function _engineerSetText(id, text) {
  const el = _engineerSafeEl(id);
  if (!el) return false;
  el.textContent = text;
  return true;
}

function _engineerSelectedJobId() {
  const sel = _engineerSafeEl('pp-job-select');
  return sel && sel.value ? sel.value : '';
}

function _engineerJobSelectedOrWarn() {
  const jobId = _engineerSelectedJobId();
  if (!jobId) {
    throw new Error(t('postprocess.select_hint') || 'Select a job first.');
  }
  return jobId;
}

function _engineerCsvToNumberList(value) {
  return String(value || '')
    .split(',')
    .map((v) => parseFloat(v.trim()))
    .filter((v) => !Number.isNaN(v));
}

async function runProbeSpectrumAnalysis() {
  const jobId = _engineerSelectedJobId();
  const statusEl = _engineerSafeEl('probe-spectrum-status');
  const outputEl = _engineerSafeEl('probe-spectrum-output');
  const signalEl = _engineerSafeEl('probe-spectrum-signal');
  const dtEl = _engineerSafeEl('probe-spectrum-dt');
  const columnEl = _engineerSafeEl('probe-spectrum-column');
  const useJobEl = _engineerSafeEl('probe-spectrum-use-job');

  if (!outputEl) return;
  outputEl.innerHTML = '';
  if (statusEl) statusEl.textContent = t('postprocess.loading') || 'Loading…';

  try {
    const body = {
      dt: dtEl ? parseFloat(dtEl.value || '1.0') : 1.0,
      column: columnEl ? columnEl.value : 'cd',
      job_id: useJobEl && useJobEl.checked ? jobId : null,
      signal: null,
    };
    if (signalEl && (!useJobEl || !useJobEl.checked)) {
      body.signal = _engineerCsvToNumberList(signalEl.value);
    }
    const r = await api('POST', '/api/postprocess/probe-spectrum', body);
    outputEl.innerHTML = `
      <div class="small text-muted mb-2">${t('probe_spectrum_peaks') || 'Dominant Peaks'}: ${r.peak_frequencies.map(v => v.toFixed(4)).join(', ') || '—'}</div>
      <div class="small text-muted mb-2">${t('probe_spectrum_strouhal') || 'Strouhal Number'}: ${r.strouhal != null ? r.strouhal.toFixed(6) : '—'}</div>
      <div class="table-responsive">
        <table class="table table-sm table-bordered mb-0 small">
          <tr><th>f_nyquist</th><td>${Number(r.f_nyquist).toFixed(6)}</td></tr>
          <tr><th>n_samples</th><td>${r.n_samples}</td></tr>
          <tr><th>signal_rms</th><td>${Number(r.signal_rms).toExponential(4)}</td></tr>
        </table>
      </div>`;
    if (statusEl) statusEl.textContent = '✓';
    return r;
  } catch (e) {
    if (statusEl) statusEl.textContent = `⚠ ${e.message}`;
    outputEl.innerHTML = `<div class="alert alert-danger small py-1">${escHtml(String(e.message))}</div>`;
  }
}

async function runPODAnalysis() {
  const jobId = _engineerSelectedJobId();
  const statusEl = _engineerSafeEl('pod-status');
  const outputEl = _engineerSafeEl('pod-output');
  const fieldEl = _engineerSafeEl('pod-field');
  const modesEl = _engineerSafeEl('pod-n-modes');
  const coeffEl = _engineerSafeEl('pod-return-coefficients');

  if (!outputEl) return;
  outputEl.innerHTML = '';
  if (statusEl) statusEl.textContent = t('postprocess.loading') || 'Loading…';

  try {
    const body = {
      job_id: jobId,
      field_name: fieldEl ? fieldEl.value : 'ux',
      n_modes: modesEl ? parseInt(modesEl.value || '10', 10) : 10,
      return_coefficients: coeffEl ? !!coeffEl.checked : true,
      snapshots: null,
    };
    const r = await api('POST', '/api/postprocess/pod', body);
    const energy = Array.isArray(r.energy_fraction) ? r.energy_fraction.slice(0, 5).map(v => Number(v).toFixed(4)).join(', ') : '—';
    outputEl.innerHTML = `
      <div class="small text-muted mb-2">${t('pod_energy') || 'Cumulative energy'}: ${Array.isArray(r.cumulative_energy) ? Number(r.cumulative_energy[r.cumulative_energy.length - 1] || 0).toFixed(4) : '—'}</div>
      <div class="small text-muted mb-2">Energy fractions (preview): ${energy}</div>
      <div class="table-responsive">
        <table class="table table-sm table-bordered mb-0 small">
          <tr><th>n_snapshots</th><td>${r.n_snapshots}</td></tr>
          <tr><th>n_modes</th><td>${r.n_modes}</td></tr>
          <tr><th>spatial_shape</th><td>${Array.isArray(r.spatial_shape) ? r.spatial_shape.join('×') : '—'}</td></tr>
        </table>
      </div>`;
    if (statusEl) statusEl.textContent = '✓';
    return r;
  } catch (e) {
    if (statusEl) statusEl.textContent = `⚠ ${e.message}`;
    outputEl.innerHTML = `<div class="alert alert-danger small py-1">${escHtml(String(e.message))}</div>`;
  }
}

async function runIsoSurfaceExtraction() {
  const jobId = _engineerSelectedJobId();
  const statusEl = _engineerSafeEl('isosurface-status');
  const outputEl = _engineerSafeEl('isosurface-output');
  const fieldEl = _engineerSafeEl('isosurface-field');
  const valueEl = _engineerSafeEl('isosurface-value');
  const axisEl = _engineerSafeEl('isosurface-axis');

  if (!outputEl) return;
  outputEl.innerHTML = '';
  if (statusEl) statusEl.textContent = t('postprocess.loading') || 'Loading…';

  try {
    const field = fieldEl ? fieldEl.value : 'q_criterion';
    const isoValue = valueEl ? parseFloat(valueEl.value || '0.0') : 0.0;
    const axis = axisEl ? axisEl.value : 'z';
    const r = await api('GET', `/api/postprocess/isosurface/${encodeURIComponent(jobId)}?field=${encodeURIComponent(field)}&iso_value=${encodeURIComponent(isoValue)}&slice_axis=${encodeURIComponent(axis)}&max_segments=50000`);
    const countKey = r.mode === '3d' ? 'n_triangles' : 'n_segments';
    outputEl.innerHTML = `
      <div class="small text-muted mb-2">${t('isosurface_field') || 'Scalar field'}: ${escHtml(field)} · ${t('isosurface_value') || 'Iso-value'}: ${isoValue}</div>
      <div class="table-responsive">
        <table class="table table-sm table-bordered mb-0 small">
          <tr><th>mode</th><td>${escHtml(r.mode || '2d')}</td></tr>
          <tr><th>${escHtml(countKey)}</th><td>${Number(r[countKey] || 0)}</td></tr>
        </table>
      </div>`;
    if (statusEl) statusEl.textContent = '✓';
    return r;
  } catch (e) {
    if (statusEl) statusEl.textContent = `⚠ ${e.message}`;
    outputEl.innerHTML = `<div class="alert alert-danger small py-1">${escHtml(String(e.message))}</div>`;
  }
}

async function loadStudyReport() {
  const groupEl = _engineerSafeEl('study-compare-group');
  const reportEl = _engineerSafeEl('study-report');
  if (!reportEl || !groupEl || !groupEl.value.trim()) return;
  reportEl.innerHTML = `<span class="text-muted small">${t('postprocess.loading') || 'Loading…'}</span>`;
  try {
    const r = await api('GET', `/api/postprocess/study-compare/${encodeURIComponent(groupEl.value.trim())}`);
    const completed = r.n_completed || 0;
    const total = r.n_total || 0;
    const metricKeys = Object.keys(r.metric_summary || {});
    let html = `<div class="alert alert-info py-1 small mb-2">${total} jobs total, ${completed} completed.</div>`;
    if (metricKeys.length) {
      html += '<div class="table-responsive"><table class="table table-sm table-bordered small">';
      html += '<thead class="table-light"><tr><th>Metric</th><th>Min</th><th>Max</th><th>Mean</th><th>Best Job</th></tr></thead><tbody>';
      for (const mk of metricKeys) {
        const s = r.metric_summary[mk];
        html += `<tr><td>${escHtml(mk)}</td><td>${Number(s.min).toFixed(4)}</td><td>${Number(s.max).toFixed(4)}</td><td>${Number(s.mean).toFixed(4)}</td><td class="font-monospace">${escHtml(String(s.best_job_id).slice(-8))}</td></tr>`;
      }
      html += '</tbody></table></div>';
    }
    reportEl.innerHTML = html;
  } catch (e) {
    reportEl.innerHTML = `<div class="alert alert-danger small py-1">${escHtml(String(e.message))}</div>`;
  }
}

function initEngineeringFeatureTabs() {
  const tabMap = [
    ['probe-spectrum-run', runProbeSpectrumAnalysis],
    ['pod-run', runPODAnalysis],
    ['isosurface-extract', runIsoSurfaceExtraction],
    ['studycompare-load', loadStudyReport],
  ];
  for (const [id, fn] of tabMap) {
    const el = _engineerSafeEl(id);
    if (el) el.addEventListener('click', () => { void fn(); });
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initEngineeringFeatureTabs, { once: true });
} else {
  initEngineeringFeatureTabs();
}
