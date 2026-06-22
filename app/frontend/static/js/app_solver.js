// Pre-processing
// ============================================================
async function runPolygonMask() {
  const nx = +document.getElementById('poly-nx').value;
  const ny = +document.getElementById('poly-ny').value;
  const lines = document.getElementById('poly-vertices').value.trim().split('\n');
  const vertices = lines.map(l => l.split(',').map(Number));
  const el = document.getElementById('poly-result');
  el.innerHTML = `<div class="spinner-border spinner-border-sm"></div> ${t('preprocess.generating')}`;
  try {
    const r = await api('POST', '/api/preprocess/polygon-mask', { nx, ny, vertices });
    el.innerHTML = `
      <img src="${r.image}" class="result-img mb-2" />
      <p class="small mb-0">${t('preprocess.obstacle_cells')}: <strong>${r.obstacle_cells}</strong> &nbsp; ${t('preprocess.fluid_cells')}: <strong>${r.fluid_cells}</strong></p>`;
  } catch(e) { el.innerHTML = `<div class="alert alert-danger small">${e.message}</div>`; }
}

async function runRandomPorosity() {
  const nx = +document.getElementById('poro-nx').value;
  const ny = +document.getElementById('poro-ny').value;
  const porosity = +document.getElementById('poro-phi').value;
  const corr_length = +document.getElementById('poro-corr').value;
  const seed = +document.getElementById('poro-seed').value;
  const el = document.getElementById('poro-result');
  el.innerHTML = `<div class="spinner-border spinner-border-sm"></div> ${t('preprocess.generating')}`;
  try {
    const r = await api('POST', '/api/preprocess/random-porosity-2d', {nx, ny, porosity, corr_length, seed});
    el.innerHTML = `
      <img src="${r.image}" class="result-img mb-2" />
      <p class="small mb-0">${t('preprocess.req_porosity')}: ${r.requested_porosity} &nbsp; ${t('preprocess.actual_porosity')}: <strong>${r.actual_porosity}</strong></p>`;
  } catch(e) { el.innerHTML = `<div class="alert alert-danger small">${e.message}</div>`; }
}

async function runUnitConvert() {
  const body = {
    phys_length_m: +document.getElementById('uc-L').value,
    phys_velocity_ms: +document.getElementById('uc-U').value,
    phys_nu_m2s: +document.getElementById('uc-nu').value,
    lbm_length: +document.getElementById('uc-lbm-L').value,
    lbm_velocity: +document.getElementById('uc-lbm-U').value,
  };
  const el = document.getElementById('uc-result');
  try {
    const r = await api('POST', '/api/preprocess/units', body);
    const stableHtml = r.stable
      ? `<span class="badge bg-success">${t('preprocess.stable_label')}</span>`
      : `<span class="badge bg-danger">${t('preprocess.unstable_label')}</span>`;
    el.innerHTML = `
      <div class="table-responsive">
        <table class="table table-sm table-bordered mb-0 small">
          <tr><th>${t('preprocess.reynolds')}</th><td>${r.reynolds_number}</td></tr>
          <tr><th>${t('preprocess.lbm_nu')}</th><td>${r.lbm_nu}</td></tr>
          <tr><th>${t('preprocess.lbm_tau')}</th><td>${r.lbm_tau} &nbsp; ${stableHtml}</td></tr>
          <tr><th>${t('preprocess.dx')}</th><td>${r.dx_m}</td></tr>
          <tr><th>${t('preprocess.dt')}</th><td>${r.dt_s}</td></tr>
          <tr><th>${t('preprocess.mach')}</th><td>${r.mach_number}</td></tr>
        </table>
      </div>`;
  } catch(e) { el.innerHTML = `<div class="alert alert-danger small">${e.message}</div>`; }
}

// ============================================================
// Solver – config schemas
// ============================================================
const SIM_TYPES = {
  cylinder_flow: {
    label: 'Cylinder Flow (2D)',
    desc: 'Flow past a 2D cylinder. Validates Strouhal number and drag coefficient against Williamson (1988).',
    endpoint: '/api/solve/cylinder-flow',
    fields: [
      {name:'nx', label:'Grid width (nx)', type:'number', default:320, min:20},
      {name:'ny', label:'Grid height (ny)', type:'number', default:100, min:10},
      {name:'u_in', label:'Inlet velocity', type:'number', default:0.08, step:0.01, min:0.001},
      {name:'re', label:'Reynolds number', type:'number', default:100, min:1},
      {name:'radius', label:'Cylinder radius (cells)', type:'number', default:12, min:1},
      {name:'n_steps', label:'Time steps', type:'number', default:1200, min:1},
      {name:'output_interval', label:'Output interval', type:'number', default:200, min:1},
      {name:'device', label:'Device', type:'device'},
      {name:'seed', label:'Seed', type:'number', default:0, min:0},
    ],
  },
  lid_driven_cavity: {
    label: 'Lid-Driven Cavity (2D)',
    desc: 'Square cavity driven by a sliding top wall. Compare against Ghia et al. (1982).',
    endpoint: '/api/solve/lid-driven-cavity',
    fields: [
      {name:'nx', label:'Grid size nx (ny=nx)', type:'number', default:128, min:8},
      {name:'u_lid', label:'Lid velocity', type:'number', default:0.1, step:0.01, min:0.001},
      {name:'re', label:'Reynolds number', type:'number', default:100, min:1},
      {name:'n_steps', label:'Time steps', type:'number', default:10000, min:1},
      {name:'output_interval', label:'Output interval', type:'number', default:2000, min:1},
      {name:'device', label:'Device', type:'device'},
      {name:'seed', label:'Seed', type:'number', default:0, min:0},
    ],
  },
  backward_facing_step: {
    label: 'Backward-Facing Step (2D)',
    desc: 'Channel with sudden expansion. Measures reattachment length x_r/h.',
    endpoint: '/api/solve/backward-facing-step',
    fields: [
      {name:'nx', label:'nx', type:'number', default:400, min:20},
      {name:'ny', label:'ny', type:'number', default:80, min:6},
      {name:'step_h', label:'Step height (cells)', type:'number', default:40, min:1},
      {name:'x_step', label:'Pre-step length (cells)', type:'number', default:80, min:1},
      {name:'u_in', label:'Inlet velocity', type:'number', default:0.05, step:0.01},
      {name:'re', label:'Reynolds number', type:'number', default:100, min:1},
      {name:'n_steps', label:'Time steps', type:'number', default:30000, min:1},
      {name:'output_interval', label:'Output interval', type:'number', default:5000, min:1},
      {name:'device', label:'Device', type:'device'},
    ],
  },
  turbulent_channel: {
    label: 'Turbulent Channel (2D LES)',
    desc: 'Body-force-driven channel with Smagorinsky LES. Validates log-law velocity profile.',
    endpoint: '/api/solve/turbulent-channel',
    fields: [
      {name:'nx', label:'nx', type:'number', default:256, min:16},
      {name:'ny', label:'ny', type:'number', default:64, min:8},
      {name:'re_tau', label:'Re_τ (friction Reynolds)', type:'number', default:100, min:1},
      {name:'u_tau', label:'Friction velocity u_τ', type:'number', default:0.005, step:0.001, min:0.0001},
      {name:'smagorinsky_cs', label:'Smagorinsky C_s', type:'number', default:0.1, step:0.01},
      {name:'n_steps', label:'Time steps', type:'number', default:50000, min:1},
      {name:'averaging_start', label:'Averaging start step', type:'number', default:20000, min:0},
      {name:'output_interval', label:'Output interval', type:'number', default:5000, min:1},
      {name:'device', label:'Device', type:'device'},
    ],
  },
  pipeline_flow: {
    label: 'Pipeline Flow (2D)',
    desc: 'Near-bed cylinder flow (e/D gap ratio). Measures Strouhal number.',
    endpoint: '/api/solve/pipeline-flow',
    fields: [
      {name:'nx', label:'nx', type:'number', default:400, min:20},
      {name:'ny', label:'ny', type:'number', default:160, min:10},
      {name:'diameter', label:'Cylinder diameter (cells)', type:'number', default:20, min:2},
      {name:'gap_ratio', label:'Gap ratio e/D', type:'number', default:0.5, step:0.1, min:0},
      {name:'u_in', label:'Inlet velocity', type:'number', default:0.05, step:0.01},
      {name:'re', label:'Reynolds number', type:'number', default:200, min:1},
      {name:'n_steps', label:'Time steps', type:'number', default:30000, min:1},
      {name:'output_interval', label:'Output interval', type:'number', default:5000, min:1},
      {name:'device', label:'Device', type:'device'},
    ],
  },
  dam_break: {
    label: 'Dam Break (2D multiphase)',
    desc: 'Collapsing fluid column benchmark. Validates front position against Martin & Moyce (1952).',
    endpoint: '/api/solve/dam-break',
    fields: [
      {name:'nx', label:'nx', type:'number', default:400, min:20},
      {name:'ny', label:'ny', type:'number', default:200, min:10},
      {name:'dam_width', label:'Dam width (cells)', type:'number', default:100, min:1},
      {name:'model', label:'Multiphase model', type:'select', default:'cg', options:['sc','scmp','cg','fe']},
      {name:'rho_heavy', label:'Heavy-phase density', type:'number', default:0.8, step:0.1},
      {name:'rho_light', label:'Light-phase density', type:'number', default:0.4, step:0.1},
      {name:'G', label:'Coupling constant G', type:'number', default:0.9, step:0.1},
      {name:'tau', label:'Relaxation time τ', type:'number', default:1.0, step:0.1, min:0.51},
      {name:'g', label:'Gravity (lu)', type:'number', default:5e-5, step:1e-5},
      {name:'n_steps', label:'Time steps', type:'number', default:4000, min:1},
      {name:'output_interval', label:'Output interval', type:'number', default:400, min:1},
      {name:'device', label:'Device', type:'device'},
    ],
  },
  sloshing_tank: {
    label: 'Sloshing Tank (2D multiphase)',
    desc: 'Harmonically forced tank. Compares sloshing frequency with Faltinsen (1978) model.',
    endpoint: '/api/solve/sloshing-tank',
    fields: [
      {name:'nx', label:'nx', type:'number', default:200, min:16},
      {name:'ny', label:'ny', type:'number', default:160, min:16},
      {name:'water_level', label:'Water level (cells)', type:'number', default:80, min:1},
      {name:'rho_water', label:'Water density', type:'number', default:0.8, step:0.1},
      {name:'rho_air', label:'Air density', type:'number', default:0.4, step:0.1},
      {name:'G', label:'Surface tension G', type:'number', default:0.9, step:0.1},
      {name:'tau', label:'τ', type:'number', default:1.0, step:0.1, min:0.51},
      {name:'g', label:'Gravity (lu)', type:'number', default:2e-5, step:1e-5},
      {name:'forcing_amp', label:'Forcing amplitude', type:'number', default:3e-5, step:1e-5},
      {name:'forcing_omega', label:'Forcing ω (0=natural)', type:'number', default:0, step:0.0001, min:0},
      {name:'n_steps', label:'Time steps', type:'number', default:6000, min:1},
      {name:'output_interval', label:'Output interval', type:'number', default:600, min:1},
      {name:'device', label:'Device', type:'device'},
    ],
  },
  sphere_flow: {
    label: 'Sphere Flow (3D D3Q19)',
    desc: '3D flow past a sphere. Measures drag coefficient.',
    endpoint: '/api/solve/sphere-flow',
    fields: [
      {name:'nx', label:'nx', type:'number', default:120, min:20},
      {name:'ny', label:'ny', type:'number', default:60, min:10},
      {name:'nz', label:'nz', type:'number', default:60, min:10},
      {name:'u_in', label:'Inlet velocity', type:'number', default:0.06, step:0.01},
      {name:'re', label:'Reynolds number', type:'number', default:50, min:1},
      {name:'radius', label:'Sphere radius (cells)', type:'number', default:8, min:1},
      {name:'n_steps', label:'Time steps', type:'number', default:500, min:1},
      {name:'output_interval', label:'Output interval', type:'number', default:100, min:1},
      {name:'device', label:'Device', type:'device'},
    ],
  },
  ship_hull: {
    label: 'Ship Hull – Wigley (3D)',
    desc: '3D Wigley hull resistance. Reports Cd and hull symmetry.',
    endpoint: '/api/solve/ship-hull',
    fields: [
      {name:'nx', label:'nx', type:'number', default:160, min:20},
      {name:'ny', label:'ny', type:'number', default:60, min:10},
      {name:'nz', label:'nz', type:'number', default:40, min:10},
      {name:'u_in', label:'Inlet velocity', type:'number', default:0.05, step:0.01},
      {name:'re', label:'Reynolds number', type:'number', default:200, min:1},
      {name:'hull_length', label:'Hull length (cells)', type:'number', default:80, min:10},
      {name:'hull_beam', label:'Hull beam (cells)', type:'number', default:8, min:1},
      {name:'hull_draft', label:'Hull draft (cells)', type:'number', default:12, min:1},
      {name:'smagorinsky_cs', label:'Smagorinsky C_s', type:'number', default:0.1, step:0.01},
      {name:'wave_amp', label:'Wave amplitude (0=none)', type:'number', default:0, step:0.5, min:0},
      {name:'n_steps', label:'Time steps', type:'number', default:2000, min:1},
      {name:'output_interval', label:'Output interval', type:'number', default:200, min:1},
      {name:'device', label:'Device', type:'device'},
    ],
  },
  porous_drainage: {
    label: 'Porous Drainage (2D)',
    desc: 'Two-phase drainage through a random porous medium.',
    endpoint: '/api/solve/porous-drainage',
    fields: [
      {name:'nx', label:'nx', type:'number', default:160, min:20},
      {name:'ny', label:'ny', type:'number', default:80, min:10},
      {name:'medium', label:'Medium type', type:'select', default:'random_cylinders', options:['random_cylinders','tube_array']},
      {name:'model', label:'Multiphase model', type:'select', default:'cg', options:['sc','cg']},
      {name:'porosity', label:'Porosity', type:'number', default:0.6, step:0.05, min:0.1, max:0.95},
      {name:'n_steps', label:'Time steps', type:'number', default:5000, min:1},
      {name:'output_interval', label:'Output interval', type:'number', default:1000, min:1},
      {name:'device', label:'Device', type:'device'},
      {name:'seed', label:'Seed', type:'number', default:0, min:0},
    ],
  },
  // ---- Advanced 3-D solvers (newly exposed) ----
  sphere_flow_d3q27: {
    label: 'Sphere Flow 3D – D3Q27 (27-vel)',
    desc: 'D3Q27 lattice (4th-order isotropy) flow past a sphere. Higher accuracy than D3Q19 in corner regions.',
    endpoint: '/api/solve/sphere-flow-d3q27',
    fields: [
      {name:'nx', label:'nx', type:'number', default:120, min:16},
      {name:'ny', label:'ny', type:'number', default:60, min:8},
      {name:'nz', label:'nz', type:'number', default:60, min:8},
      {name:'u_in', label:'Inlet velocity', type:'number', default:0.06, step:0.01},
      {name:'re', label:'Reynolds number', type:'number', default:50, min:1},
      {name:'radius', label:'Sphere radius (cells)', type:'number', default:8, min:1},
      {name:'n_steps', label:'Time steps', type:'number', default:500, min:1},
      {name:'output_interval', label:'Output interval', type:'number', default:100, min:1},
      {name:'device', label:'Device', type:'device'},
      {name:'seed', label:'Seed', type:'number', default:0, min:0},
    ],
  },
  thermal_cavity_3d: {
    label: '3D Thermal Cavity (Boussinesq)',
    desc: 'Differentially heated cube – D3Q19 velocity + D3Q7 temperature Boussinesq coupling. Reports Nusselt number.',
    endpoint: '/api/solve/thermal-cavity-3d',
    fields: [
      {name:'nx', label:'nx', type:'number', default:32, min:8},
      {name:'ny', label:'ny', type:'number', default:32, min:8},
      {name:'nz', label:'nz', type:'number', default:32, min:8},
      {name:'ra', label:'Rayleigh number (Ra)', type:'number', default:10000, min:1},
      {name:'pr', label:'Prandtl number (Pr)', type:'number', default:0.71, step:0.01, min:0.01},
      {name:'n_steps', label:'Time steps', type:'number', default:500, min:1},
      {name:'device', label:'Device', type:'device'},
    ],
  },
  porous_drainage_3d: {
    label: '3D Porous Drainage (Shan-Chen)',
    desc: 'Two-phase gas injection into a water-saturated 3D porous medium. Tracks gas saturation.',
    endpoint: '/api/solve/porous-drainage-3d',
    fields: [
      {name:'nx', label:'nx', type:'number', default:24, min:8},
      {name:'ny', label:'ny', type:'number', default:24, min:8},
      {name:'nz', label:'nz (depth)', type:'number', default:40, min:10},
      {name:'medium', label:'Medium type', type:'select', default:'random_spheres', options:['random_spheres','tube_array']},
      {name:'n_spheres', label:'No. of spheres', type:'number', default:8, min:1},
      {name:'G_12', label:'SC coupling G_12', type:'number', default:0.9, step:0.05, min:0.1},
      {name:'u_inlet', label:'Gas inlet velocity', type:'number', default:0.005, step:0.001, min:0.001},
      {name:'n_steps', label:'Time steps', type:'number', default:2000, min:1},
      {name:'output_interval', label:'Output interval', type:'number', default:500, min:1},
      {name:'device', label:'Device', type:'device'},
      {name:'seed', label:'Seed', type:'number', default:42, min:0},
    ],
  },
  hull_free_surface: {
    label: 'Hull Free-Surface (Color-Gradient)',
    desc: '3D ship-hull wave-making resistance. Color-Gradient two-phase LBM. Reports drag force.',
    endpoint: '/api/solve/hull-free-surface',
    fields: [
      {name:'nx', label:'nx (streamwise)', type:'number', default:80, min:20},
      {name:'ny', label:'ny (lateral)', type:'number', default:32, min:10},
      {name:'nz', label:'nz (vertical)', type:'number', default:32, min:10},
      {name:'hull_type', label:'Hull type', type:'select', default:'wigley', options:['wigley','series60','kcs']},
      {name:'fill_fraction', label:'Water fill fraction', type:'number', default:0.5, step:0.05, min:0.1, max:0.9},
      {name:'re', label:'Reynolds number', type:'number', default:100, min:1},
      {name:'u_in', label:'Inlet velocity (water)', type:'number', default:0.05, step:0.01},
      {name:'n_steps', label:'Time steps', type:'number', default:200, min:1},
      {name:'output_interval', label:'Output interval', type:'number', default:50, min:1},
      {name:'device', label:'Device', type:'device'},
    ],
  },
};

const MODEL_CAPABILITIES = {
  cylinder_flow: { flow_types:['single_phase'], turbulence:['none','smagorinsky_les'], multiphase:['none'], schemes:['bgk','trt'] },
  lid_driven_cavity: { flow_types:['single_phase'], turbulence:['none'], multiphase:['none'], schemes:['bgk','trt'] },
  backward_facing_step: { flow_types:['single_phase'], turbulence:['none','smagorinsky_les'], multiphase:['none'], schemes:['bgk','trt'] },
  turbulent_channel: { flow_types:['single_phase'], turbulence:['none','smagorinsky_les','dynamic_smagorinsky_les'], multiphase:['none'], schemes:['bgk'] },
  pipeline_flow: { flow_types:['single_phase'], turbulence:['none','smagorinsky_les'], multiphase:['none'], schemes:['bgk','trt'] },
  dam_break: { flow_types:['multiphase','free_surface'], turbulence:['none'], multiphase:['sc','scmp','cg','fe'], schemes:['bgk'] },
  sloshing_tank: { flow_types:['multiphase','free_surface'], turbulence:['none'], multiphase:['cg'], schemes:['bgk'] },
  sphere_flow: { flow_types:['single_phase'], turbulence:['none','smagorinsky_les'], multiphase:['none'], schemes:['bgk'] },
  ship_hull: { flow_types:['single_phase','free_surface'], turbulence:['none','smagorinsky_les','dynamic_smagorinsky_les'], multiphase:['none'], schemes:['bgk'] },
  porous_drainage: { flow_types:['multiphase'], turbulence:['none'], multiphase:['sc','cg'], schemes:['bgk'] },
  sphere_flow_d3q27: { flow_types:['single_phase'], turbulence:['none'], multiphase:['none'], schemes:['d3q27_bgk'] },
  thermal_cavity_3d: { flow_types:['thermal'], turbulence:['none'], multiphase:['none'], schemes:['bgk','boussinesq'] },
  porous_drainage_3d: { flow_types:['multiphase'], turbulence:['none'], multiphase:['sc'], schemes:['bgk'] },
  hull_free_surface: { flow_types:['multiphase','free_surface'], turbulence:['none'], multiphase:['cg'], schemes:['bgk'] },
};

const MODEL_PRESETS = {
  default: { flow_type:'single_phase', turbulence_model:'none', multiphase_model:'none', boundary_condition:'standard_bounce_back', numerical_scheme:'bgk', smagorinsky_cs:0.1 },
  pipeline_engineering: { flow_type:'single_phase', turbulence_model:'smagorinsky_les', multiphase_model:'none', boundary_condition:'zou_he', numerical_scheme:'trt', smagorinsky_cs:0.12 },
  free_surface: { flow_type:'free_surface', turbulence_model:'none', multiphase_model:'cg', boundary_condition:'standard_bounce_back', numerical_scheme:'bgk', smagorinsky_cs:0.1 },
  cavitation_like: { flow_type:'multiphase', turbulence_model:'none', multiphase_model:'fe', boundary_condition:'zou_he', numerical_scheme:'bgk', smagorinsky_cs:0.1 },
};

let currentSchema = null;

function initPhysicsLayer() {
  const selectOptions = (id, vals, labels) => {
    const el = document.getElementById(id);
    el.innerHTML = vals.map(v => `<option value="${v}">${escHtml(labels[v] || v)}</option>`).join('');
  };
  selectOptions('physics-flow-type', ['single_phase','multiphase','free_surface'], {
    single_phase: t('solve.flow_single'),
    multiphase: t('solve.flow_multi'),
    free_surface: t('solve.flow_free_surface'),
  });
  selectOptions('physics-turbulence', ['none','smagorinsky_les','dynamic_smagorinsky_les'], {
    none: t('solve.turb_none'),
    smagorinsky_les: t('solve.turb_smag'),
    dynamic_smagorinsky_les: t('solve.turb_dyn_smag'),
  });
  selectOptions('physics-multiphase', ['none','sc','scmp','cg','fe'], {
    none: t('solve.multi_none'),
    sc: 'Shan-Chen (SC)',
    scmp: 'Shan-Chen Multi (SCMP)',
    cg: 'Color-Gradient (CG)',
    fe: 'Free-Energy (FE)',
  });
  selectOptions('physics-bc', ['standard_bounce_back','zou_he','periodic'], {
    standard_bounce_back: t('solve.bc_bounce_back'),
    zou_he: t('solve.bc_zou_he'),
    periodic: t('solve.bc_periodic'),
  });
  selectOptions('physics-scheme', ['bgk','trt','mrt'], {bgk:'BGK',trt:'TRT',mrt:'MRT'});
  const preset = document.getElementById('physics-preset');
  const pkeys = Object.keys(MODEL_PRESETS);
  preset.innerHTML = pkeys.map(k => `<option value="${k}">${escHtml(t('solve.preset_' + k))}</option>`).join('');
}

function applyPhysicsPreset() {
  const key = document.getElementById('physics-preset').value;
  const p = MODEL_PRESETS[key];
  if (!p) return;
  document.getElementById('physics-flow-type').value = p.flow_type;
  document.getElementById('physics-turbulence').value = p.turbulence_model;
  document.getElementById('physics-multiphase').value = p.multiphase_model;
  document.getElementById('physics-bc').value = p.boundary_condition;
  document.getElementById('physics-scheme').value = p.numerical_scheme;
  document.getElementById('physics-cs').value = p.smagorinsky_cs;
  applyCapabilityDefaults();
}

function applyCapabilityDefaults() {
  const type = document.getElementById('sim-type').value;
  const c = MODEL_CAPABILITIES[type];
  if (!c) return;
  const applyAllowed = (id, allowed) => {
    const el = document.getElementById(id);
    const options = Array.from(el.options);
    options.forEach(o => { o.disabled = !allowed.includes(o.value); });
    if (!allowed.includes(el.value)) el.value = allowed[0];
  };
  applyAllowed('physics-flow-type', c.flow_types);
  applyAllowed('physics-turbulence', c.turbulence);
  applyAllowed('physics-multiphase', c.multiphase);
  applyAllowed('physics-scheme', c.schemes);
  const msg = document.getElementById('physics-compat-msg');
  msg.className = 'small text-muted';
  msg.textContent = `${t('solve.capability')}: ${c.flow_types.join('/')} | ${t('solve.turbulence_model')}: ${c.turbulence.join('/')} | ${t('solve.multiphase_model')}: ${c.multiphase.join('/')}`;
  const hint = document.getElementById('physics-range-hint');
  hint.textContent = currentSchema && currentSchema.hint ? currentSchema.hint : t('solve.range_hint_default');
}

function buildPhysicsPayload() {
  return {
    flow_type: document.getElementById('physics-flow-type').value,
    turbulence_model: document.getElementById('physics-turbulence').value,
    turbulence_params: { smagorinsky_cs: parseFloat(document.getElementById('physics-cs').value || '0') },
    multiphase_model: document.getElementById('physics-multiphase').value,
    multiphase_params: {},
    boundary_condition: document.getElementById('physics-bc').value,
    numerical_scheme: document.getElementById('physics-scheme').value,
    preset: document.getElementById('physics-preset').value,
  };
}

function onSimTypeChange() {
  const type = document.getElementById('sim-type').value;
  initPhysicsLayer();
  currentSchema = SIM_TYPES[type];
  if (!currentSchema) return;
  const label = t('sim.' + type + '.label') !== ('sim.' + type + '.label')
    ? t('sim.' + type + '.label')
    : currentSchema.label;
  const desc = t('sim.' + type + '.desc') !== ('sim.' + type + '.desc')
    ? t('sim.' + type + '.desc')
    : currentSchema.desc;
  document.getElementById('sim-form-title').textContent = label;
  document.getElementById('sim-description').textContent = desc;
  // Also translate optgroup labels
  const og2ds = document.getElementById('optgroup-2d-single');
  if (og2ds) og2ds.label = t('solve.group_2d_single');
  const og2dm = document.getElementById('optgroup-2d-multi');
  if (og2dm) og2dm.label = t('solve.group_2d_multi');
  const og3d = document.getElementById('optgroup-3d');
  if (og3d) og3d.label = t('solve.group_3d');
  // Translate option labels
  document.querySelectorAll('#sim-type option').forEach(opt => {
    const simKey = 'sim.' + opt.value + '.label';
    const translated = t(simKey);
    if (translated !== simKey) opt.textContent = translated;
  });
  renderConfigForm(currentSchema.fields);
  applyCapabilityDefaults();
}

function renderConfigForm(fields) {
  const form = document.getElementById('sim-form');
  const devices = getAvailableDevices();
  const rows = [];
  for (let i = 0; i < fields.length; i += 2) {
    const f1 = fields[i];
    const f2 = fields[i+1];
    rows.push(`<div class="row g-2 mb-2">
      <div class="col-md-6">${renderField(f1, devices)}</div>
      ${f2 ? `<div class="col-md-6">${renderField(f2, devices)}</div>` : '<div class="col-md-6"></div>'}
    </div>`);
  }
  form.innerHTML = rows.join('');
}

function renderField(f, devices) {
  const labelText = t('field.' + f.name) !== ('field.' + f.name)
    ? t('field.' + f.name)
    : f.label;
  if (f.type === 'device') {
    const opts = devices.map(d => `<option value="${d}">${d}</option>`).join('');
    return `<div><label class="form-label">${escHtml(labelText)}</label>
      <select class="form-select form-select-sm" id="field-${f.name}">${opts}</select></div>`;
  }
  if (f.type === 'select') {
    const opts = f.options.map(o => `<option value="${o}"${o===f.default?' selected':''}>${o}</option>`).join('');
    return `<div><label class="form-label">${escHtml(labelText)}</label>
      <select class="form-select form-select-sm" id="field-${f.name}">${opts}</select></div>`;
  }
  const extra = [];
  if (f.min !== undefined) extra.push(`min="${f.min}"`);
  if (f.max !== undefined) extra.push(`max="${f.max}"`);
  if (f.step !== undefined) extra.push(`step="${f.step}"`);
  return `<div><label class="form-label">${escHtml(labelText)}</label>
    <input type="number" class="form-control form-control-sm" id="field-${f.name}"
      value="${f.default}" ${extra.join(' ')} /></div>`;
}

function getAvailableDevices() {
  const sels = document.querySelectorAll('select[id$="-device"]');
  if (sels.length > 0) return Array.from(sels[0].options).map(o => o.value);
  return ['cpu'];
}

function resetDefaults() {
  if (!currentSchema) return;
  document.getElementById('physics-preset').value = 'default';
  applyPhysicsPreset();
  currentSchema.fields.forEach(f => {
    const el = document.getElementById(`field-${f.name}`);
    if (!el) return;
    if (f.type === 'select') el.value = f.default;
    else if (f.type !== 'device') el.value = f.default;
  });
}

async function submitJob() {
  if (!currentSchema) return;
  applyCapabilityDefaults();
  const body = {};
  for (const f of currentSchema.fields) {
    const el = document.getElementById(`field-${f.name}`);
    if (!el) continue;
    if (f.type === 'select' || f.type === 'device') body[f.name] = el.value;
    else body[f.name] = parseFloat(el.value);
  }
  body.physics = buildPhysicsPayload();
  const btn = document.getElementById('submit-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span> ${t('solve.submit_btn_loading')}`;
  try {
    const r = await api('POST', currentSchema.endpoint, body);
    showToast(`${t('solve.job_submitted')} ${r.job_id}`, 'success');
    showTab('postprocess', document.querySelectorAll('.top-navbar nav a')[4]);
  } catch(e) {
    showToast(`${t('solve.error')} ${e.message}`, 'danger');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="bi bi-play-fill"></i> ${t('solve.submit_btn')}`;
  }
}

// ============================================================
// Parameter validation
// ============================================================

/**
 * Collect current form values and POST to /api/solve/validate.
 * Shows a summary alert below the config form header.
 */
async function validateSolverParams() {
  if (!currentSchema) return;
  const body = { solver_type: currentSchema.endpoint.split('/').pop() };
  for (const f of currentSchema.fields) {
    const el = document.getElementById(`field-${f.name}`);
    if (!el) continue;
    const numNames = ['nx','ny','nz','re','u_in','u_lid','n_steps','output_interval'];
    if (numNames.includes(f.name)) {
      const v = parseFloat(el.value);
      if (!isNaN(v)) body[f.name] = v;
    }
  }
  const btn = document.getElementById('validate-btn');
  btn.disabled = true;
  const resultEl = document.getElementById('validate-result');
  resultEl.style.display = '';
  resultEl.innerHTML = `<span class="text-muted small">${t('solve.validating')}</span>`;
  try {
    const r = await api('POST', '/api/solve/validate', body);
    let html = '';
    if (r.valid) {
      html += `<div class="alert alert-success py-1 small mb-1"><i class="bi bi-check-circle-fill"></i> ${t('solve.validation_ok')}</div>`;
    }
    if (r.errors && r.errors.length) {
      html += `<div class="alert alert-danger py-1 small mb-1"><strong>${t('solve.validation_errors')}:</strong><ul class="mb-0 ps-3">${r.errors.map(e => `<li>${escHtml(e)}</li>`).join('')}</ul></div>`;
    }
    if (r.warnings && r.warnings.length) {
      html += `<div class="alert alert-warning py-1 small mb-1"><strong>${t('solve.validation_warnings')}:</strong><ul class="mb-0 ps-3">${r.warnings.map(w => `<li>${escHtml(w)}</li>`).join('')}</ul></div>`;
    }
    if (r.info && r.info.length) {
      html += `<div class="text-muted small">${r.info.map(i => `<span class="me-3"><i class="bi bi-info-circle"></i> ${escHtml(i)}</span>`).join('')}</div>`;
    }
    resultEl.innerHTML = html || '<span class="text-muted small">—</span>';
  } catch(e) {
    resultEl.innerHTML = `<div class="alert alert-danger py-1 small">${escHtml(String(e.message))}</div>`;
  } finally {
    btn.disabled = false;
  }
}

// ============================================================
// Pre-flight Engineering Validation Wizard
// ============================================================
/**
 * Collect current solver form values and POST to /api/preprocess/preflight.
 * Renders a structured checklist of pass/warning/fail checks plus
 * memory & n_steps recommendations in the validate-result panel.
 */
async function runPreflight() {
  if (!currentSchema) return;
  const solverKey = Object.keys(SIM_TYPES).find(k => SIM_TYPES[k] === currentSchema) || '';
  const body = { solver_type: solverKey };
  const numericFields = ['nx','ny','nz','re','u_in','u_lid','radius','n_steps','output_interval'];
  for (const f of currentSchema.fields) {
    const el = document.getElementById(`field-${f.name}`);
    if (!el) continue;
    if (numericFields.includes(f.name)) {
      const v = parseFloat(el.value);
      if (!isNaN(v)) body[f.name] = v;
    }
  }

  const btn = document.getElementById('preflight-btn');
  if (btn) btn.disabled = true;
  const resultEl = document.getElementById('validate-result');
  resultEl.style.display = '';
  resultEl.innerHTML = `<span class="text-muted small">${t('solve.preflight_running') || 'Running pre-flight checks…'}</span>`;

  try {
    const r = await api('POST', '/api/preprocess/preflight', body);
    const statusIcon = {ok:'check-circle-fill text-success', warning:'exclamation-triangle-fill text-warning', error:'x-circle-fill text-danger'};
    let html = '<div class="small">';
    html += `<strong>${t('solve.preflight_title') || 'Pre-flight Checks'}</strong>`;
    html += '<ul class="list-unstyled mt-1 mb-1">';
    for (const c of (r.checks || [])) {
      const ic = statusIcon[c.status] || 'info-circle';
      html += `<li><i class="bi bi-${ic} me-1"></i><strong>${escHtml(c.name.replace(/_/g,' '))}</strong>: ${escHtml(c.message)}</li>`;
    }
    html += '</ul>';
    if (r.memory_mb != null) {
      html += `<div class="text-muted"><i class="bi bi-memory me-1"></i>${t('solve.preflight_mem') || 'Est. memory'}: <strong>${r.memory_mb.toFixed(1)} MB</strong></div>`;
    }
    if (r.suggested_n_steps != null) {
      html += `<div class="text-info"><i class="bi bi-clock me-1"></i>${t('solve.preflight_steps') || 'Suggested n_steps'}: <strong>${r.suggested_n_steps}</strong></div>`;
    }
    if (r.recommendations && r.recommendations.length) {
      html += `<div class="mt-1"><strong>${t('solve.preflight_recs') || 'Recommendations'}:</strong><ul class="mb-0 ps-3">${r.recommendations.map(rec => `<li>${escHtml(rec)}</li>`).join('')}</ul></div>`;
    }
    html += '</div>';
    resultEl.innerHTML = html;
  } catch(e) {
    resultEl.innerHTML = `<div class="alert alert-danger py-1 small">${escHtml(String(e.message))}</div>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ============================================================
// Y+ Wall-Distance Calculator
// ============================================================
async function runYPlus() {
  const body = {
    re: +document.getElementById('yp-re').value,
    u_ms: +document.getElementById('yp-u').value,
    l_m: +document.getElementById('yp-l').value,
    nu_m2s: +document.getElementById('yp-nu').value,
    target_yplus: +document.getElementById('yp-yplus').value,
    n_cells: +document.getElementById('yp-ncells').value,
    geometry: document.getElementById('yp-geom').value,
  };
  const el = document.getElementById('yp-result');
  el.innerHTML = '<span class="text-muted small">Computing…</span>';
  try {
    const r = await api('POST', '/api/preprocess/yplus', body);
    const dyLbm = r.delta_y_lbm;
    const dyBadge = dyLbm < 0.5
      ? `<span class="badge bg-warning text-dark">Δy_LBM = ${dyLbm} — increase grid resolution</span>`
      : dyLbm > 5
        ? `<span class="badge bg-info text-dark">Δy_LBM = ${dyLbm} — wall-modelled resolution</span>`
        : `<span class="badge bg-success">Δy_LBM = ${dyLbm} — wall-resolved</span>`;
    el.innerHTML = `
      <div class="table-responsive">
        <table class="table table-sm table-bordered mb-1 small">
          <tr><th>${t('preprocess.yplus_result_dy')}</th><td>${r.delta_y_m.toExponential(3)}</td></tr>
          <tr><th>${t('preprocess.yplus_result_dy_lbm')}</th><td>${dyBadge}</td></tr>
          <tr><th>${t('preprocess.yplus_result_utau')}</th><td>${r.u_tau_ms.toExponential(3)}</td></tr>
          <tr><th>${t('preprocess.yplus_result_cf')}</th><td>${r.c_f.toExponential(4)}</td></tr>
          <tr><th>${t('preprocess.yplus_result_bl')}</th><td>${r.bl_thickness_m.toExponential(3)}</td></tr>
          <tr><th>${t('preprocess.yplus_result_cells_bl')}</th><td>${r.cells_inside_bl}</td></tr>
        </table>
        <p class="text-muted small mb-0"><em>Cf model:</em> ${escHtml(r.cf_model)}</p>
        ${r.note ? `<p class="text-muted small mb-0">${escHtml(r.note)}</p>` : ''}
      </div>`;
  } catch(e) { el.innerHTML = `<div class="alert alert-danger small">${escHtml(e.message)}</div>`; }
}

// ============================================================
// Parametric Sensitivity Study
// ============================================================
async function submitParametricStudy() {
  const el = document.getElementById('study-result');
  const solver_type = document.getElementById('study-solver').value;
  const parameter = document.getElementById('study-param').value;
  const rawValues = document.getElementById('study-values').value;
  const rawConfig = document.getElementById('study-base-config').value;

  let values, base_config;
  try {
    values = rawValues.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
    if (values.length < 2) throw new Error(t('solve.scan_min_values'));
    if (values.length > 20) throw new Error(t('solve.scan_max_values'));
  } catch(e) {
    el.innerHTML = `<div class="alert alert-danger small">${escHtml(e.message)}</div>`; return;
  }
  try {
    base_config = JSON.parse(rawConfig);
  } catch(e) {
    el.innerHTML = `<div class="alert alert-danger small">${t('solve.scan_invalid_json')}: ${escHtml(e.message)}</div>`; return;
  }

  el.innerHTML = `<div class="spinner-border spinner-border-sm"></div> ${t('solve.scan_submitting').replace('{n}', values.length)}`;
  try {
    const r = await api('POST', '/api/solve/parametric-study', { solver_type, base_config, parameter, values });
    el.innerHTML = `
      <div class="alert alert-success py-1 small mb-1">
        <i class="bi bi-check-circle-fill"></i> Study submitted: <strong>${r.job_ids.length} jobs</strong>
        — group <code>${r.study_group}</code><br>
        Parameter: <strong>${r.parameter}</strong> = [${r.values.join(', ')}]
      </div>
      <p class="small text-muted mb-0">Job IDs: ${r.job_ids.map(id => `<code>${id.slice(0,8)}…</code>`).join(', ')}</p>`;
  } catch(e) { el.innerHTML = `<div class="alert alert-danger small">${escHtml(e.message)}</div>`; }
}

// ============================================================
