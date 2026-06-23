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
  // Support both the engineering panel (eng-study-compare-group / eng-study-report)
  // and the legacy study-compare sub-panel (study-compare-group / study-report).
  const groupEl = _engineerSafeEl('eng-study-compare-group') || _engineerSafeEl('study-compare-group');
  const reportEl = _engineerSafeEl('eng-study-report') || _engineerSafeEl('study-report');
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

const isoViewerState = {
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  mesh: null,
  resizeBound: false,
  animating: false,
};

function ensureIsoViewer() {
  if (isoViewerState.renderer) return isoViewerState;
  const canvas = _engineerSafeEl('iso-canvas');
  const container = _engineerSafeEl('iso-canvas-container');
  if (!canvas || !container || typeof THREE === 'undefined') {
    throw new Error('Three.js viewer is unavailable.');
  }

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
  });
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
  camera.position.set(2.5, 2.0, 2.5);

  const controls = new THREE.OrbitControls(camera, canvas);
  controls.enableDamping = true;

  scene.add(new THREE.AmbientLight(0xffffff, 1.2));
  const dir = new THREE.DirectionalLight(0xffffff, 1.4);
  dir.position.set(3, 4, 5);
  scene.add(dir);

  isoViewerState.renderer = renderer;
  isoViewerState.scene = scene;
  isoViewerState.camera = camera;
  isoViewerState.controls = controls;

  resizeIsoViewer();
  if (!isoViewerState.resizeBound) {
    isoViewerState.resizeBound = true;
    window.addEventListener('resize', resizeIsoViewer);
  }
  if (!isoViewerState.animating) {
    isoViewerState.animating = true;
    const renderLoop = () => {
      if (!isoViewerState.renderer) return;
      requestAnimationFrame(renderLoop);
      isoViewerState.controls.update();
      isoViewerState.renderer.render(isoViewerState.scene, isoViewerState.camera);
    };
    renderLoop();
  }
  return isoViewerState;
}

function resizeIsoViewer() {
  const container = _engineerSafeEl('iso-canvas-container');
  if (!container || !isoViewerState.renderer || !isoViewerState.camera) return;
  const width = Math.max(container.clientWidth || 1, 1);
  const height = Math.max(container.clientHeight || 1, 1);
  isoViewerState.renderer.setSize(width, height, false);
  isoViewerState.camera.aspect = width / height;
  isoViewerState.camera.updateProjectionMatrix();
}

function setIsoPlaceholder(message) {
  const el = _engineerSafeEl('iso-placeholder');
  if (!el) return;
  el.style.display = '';
  el.innerHTML = `<i class="bi bi-box me-2"></i>${escHtml(message)}`;
}

function hideIsoPlaceholder() {
  const el = _engineerSafeEl('iso-placeholder');
  if (el) el.style.display = 'none';
}

function clearIsoMesh() {
  if (!isoViewerState.scene || !isoViewerState.mesh) return;
  isoViewerState.scene.remove(isoViewerState.mesh);
  if (isoViewerState.mesh.geometry) isoViewerState.mesh.geometry.dispose();
  if (isoViewerState.mesh.material) isoViewerState.mesh.material.dispose();
  isoViewerState.mesh = null;
}

function mapIsoField(field) {
  return {
    velocity_magnitude: 'speed',
    pressure: 'rho',
    vorticity: 'q_criterion',
  }[field] || field;
}

function normaliseIsoValues(vertices, values) {
  if (Array.isArray(values) && values.length === vertices.length) return values.map((v) => Number(v) || 0);
  if (!vertices.length) return [];
  return vertices.map((vertex) => Number(vertex[2]) || 0);
}

function buildIsoGeometry(vertices, faces, values) {
  const geometry = new THREE.BufferGeometry();
  const positions = [];
  const colors = [];
  const scalarValues = normaliseIsoValues(vertices, values);
  const minVal = scalarValues.length ? Math.min(...scalarValues) : 0;
  const maxVal = scalarValues.length ? Math.max(...scalarValues) : 1;
  const span = Math.max(maxVal - minVal, 1e-9);
  const cold = new THREE.Color(0x2266cc);
  const hot = new THREE.Color(0xcc2222);

  faces.forEach((face) => {
    face.forEach((idx) => {
      const vertex = vertices[idx] || [0, 0, 0];
      positions.push(Number(vertex[0]) || 0, Number(vertex[1]) || 0, Number(vertex[2]) || 0);
      const value = scalarValues[idx] ?? 0;
      const blend = Math.min(Math.max((value - minVal) / span, 0), 1);
      const color = cold.clone().lerp(hot, blend);
      colors.push(color.r, color.g, color.b);
    });
  });

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.computeVertexNormals();
  return geometry;
}

function fitIsoCamera(mesh) {
  if (!isoViewerState.camera || !isoViewerState.controls) return;
  const box = new THREE.Box3().setFromObject(mesh);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 1);
  const distance = maxDim * 2.2;
  isoViewerState.camera.position.set(center.x + distance, center.y + distance * 0.7, center.z + distance);
  isoViewerState.camera.near = 0.01;
  isoViewerState.camera.far = Math.max(distance * 20, 100);
  isoViewerState.camera.updateProjectionMatrix();
  isoViewerState.controls.target.copy(center);
  isoViewerState.controls.update();
}

async function loadIsosurface() {
  const jobInput = _engineerSafeEl('iso-job-id');
  const fieldInput = _engineerSafeEl('iso-field');
  const valueInput = _engineerSafeEl('iso-value');
  const infoEl = _engineerSafeEl('iso-info');
  if (!jobInput || !fieldInput || !valueInput || !infoEl) return;

  if (!jobInput.value.trim()) {
    const selected = _engineerSelectedJobId();
    if (selected) jobInput.value = selected;
  }
  const jobId = jobInput.value.trim();
  if (!jobId) {
    setIsoPlaceholder(t('postprocess.iso_placeholder'));
    infoEl.textContent = t('postprocess.select_hint') || 'Select a completed job.';
    return;
  }

  const field = fieldInput.value;
  const isoValue = parseFloat(valueInput.value || '0');
  infoEl.textContent = t('postprocess.loading') || 'Loading…';
  setIsoPlaceholder(t('postprocess.iso_loading'));

  try {
    ensureIsoViewer();
    resizeIsoViewer();
    const apiField = mapIsoField(field);
    const data = await api(
      'GET',
      `/api/postprocess/isosurface/${encodeURIComponent(jobId)}?field=${encodeURIComponent(apiField)}&iso_value=${encodeURIComponent(isoValue)}&slice_axis=3d&max_segments=50000`,
    );
    const vertices = Array.isArray(data.vertices) ? data.vertices : [];
    const faces = Array.isArray(data.faces) ? data.faces : (Array.isArray(data.triangles) ? data.triangles : []);
    if (!vertices.length || !faces.length) {
      clearIsoMesh();
      setIsoPlaceholder(data.note || t('postprocess.iso_placeholder'));
      infoEl.textContent = data.note || t('postprocess.iso_no_data');
      return;
    }

    clearIsoMesh();
    const geometry = buildIsoGeometry(vertices, faces, data.values || []);
    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      metalness: 0.1,
      roughness: 0.55,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    isoViewerState.scene.add(mesh);
    isoViewerState.mesh = mesh;
    fitIsoCamera(mesh);
    hideIsoPlaceholder();
    infoEl.textContent = `${t('postprocess.iso_vertices')}: ${vertices.length} · ${t('postprocess.iso_faces')}: ${faces.length}`;
  } catch (e) {
    clearIsoMesh();
    const msg = String(e.message || e);
    setIsoPlaceholder(msg);
    infoEl.textContent = `${t('postprocess.engineering_status_error') || 'Error'}: ${msg}`;
  }
}
