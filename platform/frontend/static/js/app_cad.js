// CAD Modelling
// ============================================================

const CAD_HULL_DESCS = {
  wigley: 'wigley_desc',
  series60: 'series60_desc',
  kcs: 'kcs_desc',
  kvlcc2: 'kvlcc2_desc',
  npl: 'npl_desc',
};

function onCADHullTypeChange() {
  const ht = document.getElementById('cad-hull-type').value;
  const key = 'cad.' + (CAD_HULL_DESCS[ht] || '');
  document.getElementById('cad-hull-desc').textContent = key ? t(key) : '';
}

const cad3dState = {
  modelId: null,
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  mesh: null,
  clipOn: false,
  wireframe: false,
  raf: 0,
};

function cad3dEnsureViewer() {
  const host = document.getElementById('cad3d-canvas');
  if (!host || cad3dState.renderer) return;
  const w = host.clientWidth || 640;
  const h = host.clientHeight || 420;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111827);
  const camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 5000);
  camera.position.set(160, 100, 200);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(w, h);
  host.innerHTML = '';
  host.appendChild(renderer.domElement);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const key = new THREE.DirectionalLight(0xffffff, 0.8);
  key.position.set(2, 3, 1);
  scene.add(key);
  scene.add(new THREE.GridHelper(400, 24, 0x334155, 0x1f2937));
  cad3dState.scene = scene;
  cad3dState.camera = camera;
  cad3dState.renderer = renderer;
  cad3dState.controls = controls;
  const loop = () => {
    cad3dState.controls.update();
    cad3dState.renderer.render(cad3dState.scene, cad3dState.camera);
    cad3dState.raf = requestAnimationFrame(loop);
  };
  loop();
}

async function cad3dCreateOrUpdate() {
  cad3dEnsureViewer();
  const body = {
    source_type: 'parametric',
    units: 'lu',
    hull_type: document.getElementById('cad-hull-type').value,
    length: +document.getElementById('cad-length').value,
    beam: +document.getElementById('cad-beam').value,
    draft: +document.getElementById('cad-draft').value,
    n_long: 80,
    n_vert: 40,
  };
  const isUpdate = !!cad3dState.modelId;
  const path = isUpdate ? `/api/cad/3d/models/${cad3dState.modelId}` : '/api/cad/3d/models';
  const method = isUpdate ? 'PUT' : 'POST';
  try {
    const r = await api(method, path, body);
    cad3dState.modelId = r.model_id || cad3dState.modelId;
    document.getElementById('cad3d-model-id').textContent = cad3dState.modelId ? `ID: ${cad3dState.modelId}` : '';
    await cad3dLoadMesh();
  } catch (e) {
    showToast(`${t('common.error')} ${e.message}`, 'danger');
  }
}

async function cad3dLoadMesh() {
  if (!cad3dState.modelId) return;
  const r = await api('GET', `/api/cad/3d/models/${cad3dState.modelId}/mesh`);
  const verts = r.vertices;
  const faces = r.faces;
  const pos = new Float32Array(faces.length * 9);
  let p = 0;
  for (const f of faces) {
    for (const idx of f) {
      const v = verts[idx];
      pos[p++] = v[0];
      pos[p++] = v[1];
      pos[p++] = v[2];
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geometry.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color: 0x4f8ef7,
    side: THREE.DoubleSide,
    wireframe: cad3dState.wireframe,
    clippingPlanes: [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)],
    clipShadows: true,
  });
  if (cad3dState.mesh) cad3dState.scene.remove(cad3dState.mesh);
  const mesh = new THREE.Mesh(geometry, mat);
  cad3dState.mesh = mesh;
  cad3dState.scene.add(mesh);
  geometry.computeBoundingBox();
  const bb = geometry.boundingBox;
  const center = new THREE.Vector3();
  bb.getCenter(center);
  const size = new THREE.Vector3();
  bb.getSize(size);
  cad3dState.controls.target.copy(center);
  cad3dState.camera.position.set(center.x + size.x * 1.4, center.y + size.y * 1.4, center.z + size.z * 1.2 + 10);
  document.getElementById('cad3d-stats').innerHTML =
    `${t('cad.model3d_vertices')}: <strong>${r.stats.vertex_count}</strong>, ${t('cad.model3d_faces')}: <strong>${r.stats.face_count}</strong>`;
}

function cad3dToggleWireframe() {
  cad3dState.wireframe = !cad3dState.wireframe;
  if (cad3dState.mesh) cad3dState.mesh.material.wireframe = cad3dState.wireframe;
}

function cad3dToggleHull() {
  if (cad3dState.mesh) cad3dState.mesh.visible = !cad3dState.mesh.visible;
}

function cad3dToggleClip() {
  cad3dState.clipOn = !cad3dState.clipOn;
  if (cad3dState.renderer) cad3dState.renderer.localClippingEnabled = cad3dState.clipOn;
  if (cad3dState.mesh) {
    const m = cad3dState.mesh.material;
    m.clippingPlanes = cad3dState.clipOn ? [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)] : [];
  }
}

async function cad3dExport(fmt) {
  if (!cad3dState.modelId) {
    showToast(t('cad.model3d_build_first'), 'warning');
    return;
  }
  const resp = await fetch(API + `/api/cad/3d/models/${cad3dState.modelId}/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fmt }),
  });
  if (!resp.ok) {
    showToast(`Export failed: ${await resp.text()}`, 'danger');
    return;
  }
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${cad3dState.modelId}.${fmt === 'gltf' ? 'gltf' : fmt}`;
  a.click();
  URL.revokeObjectURL(url);
}

async function cadGeneratePreview() {
  const hull_type = document.getElementById('cad-hull-type').value;
  const length = +document.getElementById('cad-length').value;
  const beam = +document.getElementById('cad-beam').value;
  const draft = +document.getElementById('cad-draft').value;
  const n_stations = +document.getElementById('cad-stations').value;
  const area = document.getElementById('cad-preview-area');
  const caption = document.getElementById('cad-preview-caption');
  area.innerHTML = `<div class="py-5"><span class="spinner-border"></span> ${t('cad.generating')}</div>`;
  caption.textContent = '';
  try {
    const r = await api('POST', '/api/cad/preview', { hull_type, length, beam, draft, n_stations });
    area.innerHTML = `<img src="${r.image}" class="result-img" style="max-width:100%" />`;
    caption.textContent = `${hull_type.toUpperCase()}  L=${length}  B=${beam}  T=${draft}`;
    // Render stats
    const s = r.stats;
    document.getElementById('cad-stats').innerHTML = `
      <table class="table table-sm table-bordered mb-0">
        <tr><th>${t('cad.hull_stat_label')}</th><td>${escHtml(s.label)}</td></tr>
        <tr><th>C<sub>b</sub></th><td><strong>${s.Cb}</strong></td></tr>
        <tr><th>C<sub>wp</sub></th><td>${s.Cwp}</td></tr>
        <tr><th>C<sub>m</sub></th><td>${s.Cm}</td></tr>
        <tr><th>C<sub>p</sub></th><td>${s.Cp}</td></tr>
        <tr><th>L/B</th><td>${s['L/B']}</td></tr>
        <tr><th>B/T</th><td>${s['B/T']}</td></tr>
        <tr><th>Displacement (lu³)</th><td>${s.displacement_lu3}</td></tr>
      </table>`;
    // Sync solver hull params
    document.getElementById('cad-sol-length').value = length;
    document.getElementById('cad-sol-beam').value = beam;
    document.getElementById('cad-sol-draft').value = draft;
  } catch(e) {
    area.innerHTML = `<div class="alert alert-danger">${e.message}</div>`;
  }
}

async function cadGenerateMask() {
  const hull_type = document.getElementById('cad-hull-type').value;
  const nx = +document.getElementById('cad-nx').value;
  const ny = +document.getElementById('cad-ny').value;
  const nz = +document.getElementById('cad-nz').value;
  const length = +document.getElementById('cad-length').value;
  const beam = +document.getElementById('cad-beam').value;
  const draft = +document.getElementById('cad-draft').value;
  const el = document.getElementById('cad-mask-result');
  el.innerHTML = `<span class="spinner-border spinner-border-sm"></span> ${t('cad.computing')}`;
  try {
    const r = await api('POST', '/api/cad/hull-mask', { hull_type, nx, ny, nz, length, beam, draft });
    const s = r.stats;
    el.innerHTML = `
      <img src="${r.image}" class="result-img mb-2" style="max-width:100%" />
      <table class="table table-sm table-bordered mb-0 small">
        <tr><th>${t('cad.cb_num')}</th><td><strong>${s.Cb_numerical}</strong></td></tr>
        <tr><th>${t('cad.solid_cells')}</th><td>${s.solid_cells}</td></tr>
        <tr><th>${t('cad.fluid_cells')}</th><td>${s.fluid_cells}</td></tr>
        <tr><th>${t('cad.grid')}</th><td>${s.nx}×${s.ny}×${s.nz}</td></tr>
      </table>`;
  } catch(e) {
    el.innerHTML = `<div class="alert alert-danger small">${e.message}</div>`;
  }
}

async function cadComputeLBM() {
  const body = {
    length_m: +document.getElementById('cad-phys-L').value,
    speed_ms: +document.getElementById('cad-phys-U').value,
    nu_m2s: +document.getElementById('cad-phys-nu').value,
    lbm_length: +document.getElementById('cad-length').value,
    lbm_speed: 0.05,
    froude_target: +document.getElementById('cad-froude').value || null,
  };
  const el = document.getElementById('cad-lbm-result');
  try {
    const r = await api('POST', '/api/cad/lbm-parameters', body);
    const stableHtml = r.stable
      ? `<span class="badge bg-success">${t('cad.stable')}</span>`
      : `<span class="badge bg-danger">${t('cad.unstable')}</span>`;
    el.innerHTML = `
      <table class="table table-sm table-bordered mb-0 small">
        <tr><th>Re</th><td>${r.re_physical}</td></tr>
        <tr><th>Fr</th><td>${r.froude_number}</td></tr>
        <tr><th>dx (m)</th><td>${r.dx_m}</td></tr>
        <tr><th>dt (s)</th><td>${r.dt_s}</td></tr>
        <tr><th>τ</th><td>${r.lbm_tau} ${stableHtml}</td></tr>
        <tr><th>Ma</th><td>${r.mach_number}</td></tr>
      </table>`;
  } catch(e) {
    el.innerHTML = `<div class="alert alert-danger small">${e.message}</div>`;
  }
}

async function cadLaunchSolver() {
  const body = {
    hull_type: document.getElementById('cad-hull-type').value,
    nx: +document.getElementById('cad-sol-nx').value,
    ny: +document.getElementById('cad-sol-ny').value,
    nz: +document.getElementById('cad-sol-nz').value,
    hull_length: +document.getElementById('cad-sol-length').value,
    hull_beam: +document.getElementById('cad-sol-beam').value,
    hull_draft: +document.getElementById('cad-sol-draft').value,
    u_in: +document.getElementById('cad-sol-uin').value,
    re: +document.getElementById('cad-sol-re').value,
    smagorinsky_cs: +document.getElementById('cad-sol-cs').value,
    wave_amp: 0,
    wave_period: 200,
    n_steps: +document.getElementById('cad-sol-steps').value,
    output_interval: +document.getElementById('cad-sol-interval').value,
    device: document.getElementById('cad-sol-device').value,
    seed: 0,
  };
  const btn = document.getElementById('cad-launch-btn');
  const el = document.getElementById('cad-launch-result');
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span> ${t('cad.submitting')}`;
  try {
    const r = await api('POST', '/api/cad/send-to-solver', body);
    showToast(`${t('cad.job_submitted')} ${r.job_id}`, 'success');
    el.innerHTML = `<div class="alert alert-success small">Job ID: <code>${r.job_id}</code></div>`;
    showTab('postprocess', document.querySelectorAll('.top-navbar nav a')[4]);
  } catch(e) {
    el.innerHTML = `<div class="alert alert-danger small">${e.message}</div>`;
    showToast(`${t('common.error')} ${e.message}`, 'danger');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="bi bi-play-fill"></i> ${t('cad.submit_solver')}`;
  }
}

async function cadDownloadSTL() {
  const hull_type = document.getElementById('cad-hull-type').value;
  const length = +document.getElementById('cad-length').value;
  const beam = +document.getElementById('cad-beam').value;
  const draft = +document.getElementById('cad-draft').value;
  try {
    const resp = await fetch(API + '/api/cad/export-stl', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hull_type, length, beam, draft, n_long: 60, n_vert: 30 }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `${hull_type}_hull.stl`; a.click();
    URL.revokeObjectURL(url);
  } catch(e) {
    showToast(`STL export failed: ${e.message}`, 'danger');
  }
}

// ============================================================
// SUBOFF Submarine CAD
// ============================================================
async function suboffPreview() {
  const area = document.getElementById('suboff-preview-area');
  area.innerHTML = `<p class="text-muted py-4">${t('cad.generating')}</p>`;
  const body = {
    hull_type: document.getElementById('suboff-model-type').value,
    length: +document.getElementById('suboff-length').value,
    radius: +document.getElementById('suboff-radius').value,
    bow_frac: +document.getElementById('suboff-bow-frac').value,
    stern_frac: +document.getElementById('suboff-stern-frac').value,
  };
  try {
    const resp = await fetch(API + '/api/cad/suboff/preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    area.innerHTML = `<img src="${data.image}" class="img-fluid" alt="suboff preview"/>`;
    if (data.stats) {
      const stats = data.stats;
      document.getElementById('suboff-stats').innerHTML =
        `<dl class="row mb-0 small">
           <dt class="col-6" data-i18n="cad.suboff_l_d">L/D</dt><dd class="col-6">${(stats.l_d||0).toFixed(3)}</dd>
           <dt class="col-6" data-i18n="cad.suboff_cp">Cp</dt><dd class="col-6">${(stats.Cp||0).toFixed(4)}</dd>
           <dt class="col-6" data-i18n="cad.suboff_disp">Vol</dt><dd class="col-6">${(stats.volume||0).toFixed(1)} lu³</dd>
           <dt class="col-6" data-i18n="cad.suboff_wetted">Wetted</dt><dd class="col-6">${(stats.wetted_area||0).toFixed(1)} lu²</dd>
         </dl>`;
    }
  } catch(e) {
    area.innerHTML = `<p class="text-danger">${e.message}</p>`;
  }
}

async function suboffExportSTL() {
  const body = {
    hull_type: document.getElementById('suboff-model-type').value,
    length: +document.getElementById('suboff-length').value,
    radius: +document.getElementById('suboff-radius').value,
  };
  try {
    const resp = await fetch(API + '/api/cad/suboff/export-stl', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'suboff.stl'; a.click();
    URL.revokeObjectURL(url);
  } catch(e) {
    showToast(`STL export failed: ${e.message}`, 'danger');
  }
}

async function suboffGenerateMask() {
  const res = document.getElementById('suboff-mask-result');
  res.innerHTML = `<p class="text-muted">${t('cad.generating')}</p>`;
  const body = {
    hull_type: document.getElementById('suboff-model-type').value,
    length: +document.getElementById('suboff-length').value,
    radius: +document.getElementById('suboff-radius').value,
    nx: 80, ny: 30, nz: 30,
  };
  try {
    const resp = await fetch(API + '/api/cad/suboff/hull-mask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    res.innerHTML = `<img src="${data.image}" class="img-fluid mt-2" alt="mask"/>`;
  } catch(e) {
    res.innerHTML = `<p class="text-danger">${e.message}</p>`;
  }
}

// ============================================================
// Offshore Structure CAD
// ============================================================
const OFFSHORE_DESCS = {
  monopile: 'monopile_desc',
  jacket: 'jacket_desc',
  spar: 'spar_desc',
  semi_sub: 'semi_sub_desc',
};

function onOffshoreTypeChange() {
  const st = document.getElementById('offshore-struct-type').value;
  const key = 'offshore.' + (OFFSHORE_DESCS[st] || '');
  document.getElementById('offshore-struct-desc').textContent = t(key);
  ['monopile', 'jacket', 'spar', 'semisub'].forEach(k => {
    document.getElementById(`offshore-${k}-params`).classList.add('d-none');
  });
  const map = { monopile: 'monopile', jacket: 'jacket', spar: 'spar', semi_sub: 'semisub' };
  const el = document.getElementById(`offshore-${map[st]}-params`);
  if (el) el.classList.remove('d-none');
}

function _offshoreBody() {
  const st = document.getElementById('offshore-struct-type').value;
  const body = {
    struct_type: st,
    nx: +document.getElementById('offshore-nx').value,
    ny: +document.getElementById('offshore-ny').value,
    nz: +document.getElementById('offshore-nz').value,
  };
  if (st === 'monopile') body.diameter = +document.getElementById('offshore-diameter').value;
  if (st === 'jacket') {
    body.leg_diameter = +document.getElementById('offshore-leg-diameter').value;
    body.foot_spread = +document.getElementById('offshore-foot-spread').value;
    body.head_spread = +document.getElementById('offshore-head-spread').value;
  }
  if (st === 'spar') {
    body.hull_diameter = +document.getElementById('offshore-hull-diameter').value;
    body.keel_diameter = +document.getElementById('offshore-keel-diameter').value;
  }
  if (st === 'semi_sub') {
    body.column_diameter = +document.getElementById('offshore-column-diameter').value;
    body.pontoon_length = +document.getElementById('offshore-pontoon-length').value;
  }
  return body;
}

async function offshorePreview() {
  const area = document.getElementById('offshore-preview-area');
  area.innerHTML = `<p class="text-muted py-4">${t('offshore.generating')}</p>`;
  try {
    const resp = await fetch(API + '/api/cad/offshore/preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(_offshoreBody()),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    area.innerHTML = `<img src="${data.image}" class="img-fluid" alt="offshore preview"/>`;
  } catch(e) {
    area.innerHTML = `<p class="text-danger">${e.message}</p>`;
  }
}

async function offshoreGenerateMask() {
  const res = document.getElementById('offshore-mask-result');
  res.innerHTML = `<p class="text-muted">${t('offshore.generating')}</p>`;
  try {
    const resp = await fetch(API + '/api/cad/offshore/hull-mask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(_offshoreBody()),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    res.innerHTML = `<img src="${data.image}" class="img-fluid mt-2" alt="mask"/>`;
    if (data.stats) {
      const s = data.stats;
      document.getElementById('offshore-stats').innerHTML =
        `<dl class="row mb-0 small">
           <dt class="col-6" data-i18n="offshore.solid_cells">Solid</dt><dd class="col-6">${s.solid_cells}</dd>
           <dt class="col-6" data-i18n="offshore.fluid_cells">Fluid</dt><dd class="col-6">${s.fluid_cells}</dd>
           <dt class="col-6" data-i18n="offshore.grid">Grid</dt><dd class="col-6">${s.nx}×${s.ny}×${s.nz}</dd>
         </dl>`;
    }
  } catch(e) {
    res.innerHTML = `<p class="text-danger">${e.message}</p>`;
  }
}

async function offshoreExportSTL() {
  try {
    const resp = await fetch(API + '/api/cad/offshore/export-stl', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(_offshoreBody()),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const blob = await resp.blob();
    const st = document.getElementById('offshore-struct-type').value;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = `${st}.stl`; a.click();
    URL.revokeObjectURL(url);
  } catch(e) {
    showToast(`STL export failed: ${e.message}`, 'danger');
  }
}

// ============================================================
// Wageningen B-Series Propeller
// ============================================================
async function propComputeOpenWater() {
  const res = document.getElementById('prop-ow-result');
  res.innerHTML = `<span class="text-muted">${t('propeller.generating')}</span>`;
  const body = {
    J: +document.getElementById('prop-J').value,
    P_D: +document.getElementById('prop-PD').value,
    Ae_A0: +document.getElementById('prop-EAR').value,
    Z: +document.getElementById('prop-Z').value,
  };
  try {
    const resp = await fetch(API + '/api/cad/propeller/open-water', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const d = await resp.json();
    res.innerHTML =
      `<dl class="row mb-0">
         <dt class="col-6" data-i18n="propeller.KT">KT</dt><dd class="col-6">${d.KT.toFixed(4)}</dd>
         <dt class="col-6" data-i18n="propeller.KQ">KQ</dt><dd class="col-6">${d.KQ.toFixed(5)}</dd>
         <dt class="col-6" data-i18n="propeller.eta">η₀</dt><dd class="col-6">${d.eta_0.toFixed(4)}</dd>
       </dl>`;
  } catch(e) {
    res.innerHTML = `<span class="text-danger">${e.message}</span>`;
  }
}

async function propPlotCurves() {
  const area = document.getElementById('prop-curves-area');
  area.innerHTML = `<p class="text-muted py-4">${t('propeller.generating')}</p>`;
  const body = {
    P_D: +document.getElementById('prop-PD').value,
    Ae_A0: +document.getElementById('prop-EAR').value,
    Z: +document.getElementById('prop-Z').value,
    J_min: 0.01, J_max: 1.35, n_points: 60,
  };
  try {
    const resp = await fetch(API + '/api/cad/propeller/curves', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const d = await resp.json();
    area.innerHTML = `<img src="${d.image}" class="img-fluid" alt="open-water diagram"/>`;
  } catch(e) {
    area.innerHTML = `<p class="text-danger">${e.message}</p>`;
  }
}

async function propDesign() {
  const res = document.getElementById('prop-design-result');
  res.innerHTML = `<span class="text-muted">${t('propeller.generating')}</span>`;
  const body = {
    thrust_n: +document.getElementById('prop-thrust').value,
    Va_ms: +document.getElementById('prop-va').value,
    P_D: +document.getElementById('prop-PD').value,
    Ae_A0: +document.getElementById('prop-EAR').value,
    Z: +document.getElementById('prop-Z').value,
    n_rps: +document.getElementById('prop-nrps').value,
  };
  try {
    const resp = await fetch(API + '/api/cad/propeller/design', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const d = await resp.json();
    const rows = Object.entries(d).map(([k,v]) =>
      `<dt class="col-7">${k}</dt><dd class="col-5">${typeof v==='number'?v.toFixed(4):v}</dd>`
    ).join('');
    res.innerHTML = `<dl class="row mb-0">${rows}</dl>`;
  } catch(e) {
    res.innerHTML = `<span class="text-danger">${e.message}</span>`;
  }
}

// ============================================================
