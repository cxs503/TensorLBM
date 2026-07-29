// ── Cylinder Flow: Interactive Simulation + Speed Comparison ────────────────
// Section 1: Local JS D2Q9 interactive sim (auto-start, drag, vorticity, pressure)
// Section 2: JS CPU vs SDAA f32 speed comparison

// ── D2Q9 constants (shared by both sections) ────────────────────────────────
const EX  = [0, 1, 0,-1, 0, 1,-1,-1, 1];
const EY  = [0, 0, 1, 0,-1, 1,-1, 1,-1];
const WT  = [4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36];
const OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6];

// ════════════════════════════════════════════════════════════════════════════
// SECTION 1: Interactive Simulation (Local JS, auto-start)
// ════════════════════════════════════════════════════════════════════════════

let cylInterSim = null;      // Interactive JS simulation instance
let cylInterTimer = null;    // Auto-run timer
let cylInterPaused = false;  // paused state
let cylInterStarted = false; // whether sim has been started
let cylDragActive = false;   // dragging cylinder

// ── Turbo colormap (256 entries) ────────────────────────────────────────────
const CYL_TURBO = [];
(function _buildTurbo() {
  const stops = [
    [0.00, 48,  18,  59],
    [0.07, 69,  55,  130],
    [0.15, 104, 128, 179],
    [0.25, 134, 188, 184],
    [0.35, 164, 220, 170],
    [0.45, 190, 230, 150],
    [0.55, 215, 220, 130],
    [0.65, 235, 195, 100],
    [0.75, 248, 160,  60],
    [0.85, 252, 120,  20],
    [0.95, 240,  60,  10],
    [1.00, 122,   4,   3],
  ];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let lo = 0, hi = stops.length - 1;
    for (let s = 0; s < stops.length - 1; s++) {
      if (t >= stops[s][0] && t <= stops[s+1][0]) { lo = s; hi = s+1; break; }
    }
    const f = (t - stops[lo][0]) / (stops[hi][0] - stops[lo][0] || 1);
    CYL_TURBO.push([
      Math.round(stops[lo][1] + f * (stops[hi][1] - stops[lo][1])),
      Math.round(stops[lo][2] + f * (stops[hi][2] - stops[lo][2])),
      Math.round(stops[lo][3] + f * (stops[hi][3] - stops[lo][3])),
    ]);
  }
})();

// ── Viridis colormap (256 entries, diverging for vorticity) ────────────────
const CYL_VIRIDIS = [];
(function _buildViridis() {
  const stops = [
    [0.00, 68,   1,  84],
    [0.13, 72,  36, 117],
    [0.25, 65,  68, 135],
    [0.38, 53,  95, 141],
    [0.50, 42, 120, 142],
    [0.63, 33, 145, 140],
    [0.75, 53, 183, 121],
    [0.88, 109, 206,  89],
    [1.00, 253, 231,  37],
  ];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let lo = 0, hi = stops.length - 1;
    for (let s = 0; s < stops.length - 1; s++) {
      if (t >= stops[s][0] && t <= stops[s+1][0]) { lo = s; hi = s+1; break; }
    }
    const f = (t - stops[lo][0]) / (stops[hi][0] - stops[lo][0] || 1);
    CYL_VIRIDIS.push([
      Math.round(stops[lo][1] + f * (stops[hi][1] - stops[lo][1])),
      Math.round(stops[lo][2] + f * (stops[hi][2] - stops[lo][2])),
      Math.round(stops[lo][3] + f * (stops[hi][3] - stops[lo][3])),
    ]);
  }
})();

// ── Interactive D2Q9 LBM with drag support ──────────────────────────────────
class InteractiveCylinderLBM {
  constructor(nx, ny, uIn, re, radius, cx, cy) {
    this.nx = nx; this.ny = ny; this.uIn = uIn; this.re = re; this.radius = radius;
    this.cx = cx; this.cy = cy;
    this.nu = uIn * 2 * radius / re;
    this.tau = 3 * this.nu + 0.5;
    if (this.tau < 0.505) this.tau = 0.505;
    this.stepCount = 0;
    this.f = new Array(9);
    for (let k = 0; k < 9; k++) this.f[k] = new Float64Array(nx * ny);
    this.obstacle = new Uint8Array(nx * ny);
    this.wallMask = new Uint8Array(nx * ny);
    this._buildMasks();
    this._initEq();
    this.initialMass = 0;
    for (let k = 0; k < 9; k++) for (let i = 0; i < nx * ny; i++) this.initialMass += this.f[k][i];
  }

  _idx(i, j) { return j * this.nx + i; }

  _buildMasks() {
    const nx = this.nx, ny = this.ny;
    this.obstacle.fill(0);
    this.wallMask.fill(0);
    const r2 = this.radius * this.radius;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++)
      if ((i - this.cx) ** 2 + (j - this.cy) ** 2 <= r2) this.obstacle[this._idx(i, j)] = 1;
    for (let i = 0; i < nx; i++) {
      if (!this.obstacle[i]) this.wallMask[i] = 1;
      if (!this.obstacle[(ny - 1) * nx + i]) this.wallMask[(ny - 1) * nx + i] = 1;
    }
  }

  _feq(rho, ux, uy, k) {
    const eu = EX[k] * ux + EY[k] * uy;
    return rho * WT[k] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * (ux * ux + uy * uy));
  }

  _initEq() {
    for (let j = 0; j < this.ny; j++) for (let i = 0; i < this.nx; i++) {
      const idx = this._idx(i, j), ux = this.obstacle[idx] ? 0 : this.uIn;
      for (let k = 0; k < 9; k++) this.f[k][idx] = this._feq(1, ux, 0, k);
    }
  }

  step(n) {
    for (let s = 0; s < n; s++) {
      this._collide(); this._stream(); this._bounds();
      this.stepCount++;
      if (this.stepCount % 50 === 0) this._mass();
    }
  }

  _collide() {
    const nx = this.nx, ny = this.ny, tau = this.tau;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const idx = this._idx(i, j);
      let rho = 0, ux = 0, uy = 0;
      for (let k = 0; k < 9; k++) { rho += this.f[k][idx]; ux += EX[k] * this.f[k][idx]; uy += EY[k] * this.f[k][idx]; }
      ux /= rho; uy /= rho;
      for (let k = 0; k < 9; k++) this.f[k][idx] -= (this.f[k][idx] - this._feq(rho, ux, uy, k)) / tau;
    }
  }

  _stream() {
    const nx = this.nx, ny = this.ny;
    const fN = new Array(9);
    for (let k = 0; k < 9; k++) fN[k] = new Float64Array(nx * ny);
    for (let k = 0; k < 9; k++) {
      const ex = EX[k], ey = EY[k];
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const si = ((i - ex) % nx + nx) % nx, sj = ((j - ey) % ny + ny) % ny;
        fN[k][this._idx(i, j)] = this.f[k][this._idx(si, sj)];
      }
    }
    for (let k = 0; k < 9; k++) this.f[k].set(fN[k]);
  }

  _bounds() {
    const nx = this.nx, ny = this.ny, uIn = this.uIn;
    for (let j = 0; j < ny; j++) {
      let r1 = 0; for (let k = 0; k < 9; k++) r1 += this.f[k][this._idx(1, j)];
      for (let k = 0; k < 9; k++) this.f[k][this._idx(0, j)] = this._feq(r1, uIn, 0, k);
    }
    for (let j = 0; j < ny; j++) for (let k = 0; k < 9; k++) this.f[k][this._idx(nx - 1, j)] = this.f[k][this._idx(nx - 2, j)];
    this._bb(this.wallMask); this._bb(this.obstacle);
  }

  _bb(mask) {
    for (let j = 0; j < this.ny; j++) for (let i = 0; i < this.nx; i++) {
      const idx = this._idx(i, j); if (!mask[idx]) continue;
      const t = new Float64Array(9);
      for (let k = 0; k < 9; k++) t[k] = this.f[k][idx];
      for (let k = 0; k < 9; k++) this.f[k][idx] = t[OPP[k]];
    }
  }

  _mass() {
    let c = 0; for (let k = 0; k < 9; k++) for (let i = 0; i < this.nx * this.ny; i++) c += this.f[k][i];
    if (c < 1e-30) return;
    const f = this.initialMass / c;
    for (let k = 0; k < 9; k++) for (let i = 0; i < this.nx * this.ny; i++) this.f[k][i] *= f;
  }

  // ── Move cylinder (drag) ──────────────────────────────────────────────
  moveCylinder(cx, cy) {
    const margin = this.radius + 2;
    cx = Math.max(margin, Math.min(this.nx - margin, cx));
    cy = Math.max(margin, Math.min(this.ny - margin, cy));

    const oldObs = new Uint8Array(this.obstacle);
    this.cx = cx; this.cy = cy;
    this._buildMasks();

    // New solid cells → set to equilibrium with zero velocity
    for (let j = 0; j < this.ny; j++) for (let i = 0; i < this.nx; i++) {
      const idx = this._idx(i, j);
      if (this.obstacle[idx] && !oldObs[idx]) {
        for (let k = 0; k < 9; k++) this.f[k][idx] = this._feq(1, 0, 0, k);
      }
      // New fluid cells (were solid) → set to equilibrium with inlet velocity
      if (!this.obstacle[idx] && oldObs[idx]) {
        for (let k = 0; k < 9; k++) this.f[k][idx] = this._feq(1, this.uIn, 0, k);
      }
    }
  }

  // ── Get all fields ────────────────────────────────────────────────────
  getFields() {
    const nx = this.nx, ny = this.ny;
    const rho = new Float64Array(nx * ny);
    const ux = new Float64Array(nx * ny);
    const uy = new Float64Array(nx * ny);
    const speed = new Float64Array(nx * ny);
    const vorticity = new Float64Array(nx * ny);

    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const idx = this._idx(i, j);
      if (this.obstacle[idx]) {
        rho[idx] = 1; ux[idx] = 0; uy[idx] = 0; speed[idx] = 0; vorticity[idx] = 0;
        continue;
      }
      let r = 0, vx = 0, vy = 0;
      for (let k = 0; k < 9; k++) { r += this.f[k][idx]; vx += EX[k] * this.f[k][idx]; vy += EY[k] * this.f[k][idx]; }
      vx /= r; vy /= r;
      rho[idx] = r; ux[idx] = vx; uy[idx] = vy;
      speed[idx] = Math.sqrt(vx * vx + vy * vy);
    }

    // Vorticity: ∂uy/∂x − ∂ux/∂y (central difference)
    for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
      const idx = this._idx(i, j);
      vorticity[idx] = 0.5 * (uy[this._idx(i+1, j)] - uy[this._idx(i-1, j)])
                      - 0.5 * (ux[this._idx(i, j+1)] - ux[this._idx(i, j-1)]);
    }

    return { rho, ux, uy, speed, vorticity, obstacle: this.obstacle, nx, ny,
             step: this.stepCount, cx: this.cx, cy: this.cy, radius: this.radius };
  }
}

// ── Render a field on canvas with colormap ──────────────────────────────────
function cylRenderField(canvasId, data, obsArr, nx, ny, cx, cy, radius, colormap, diverging) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Resize canvas to match aspect ratio
  const aspect = nx / ny;
  const displayW = canvas.parentElement.clientWidth || 600;
  canvas.width = displayW;
  canvas.height = Math.round(displayW / aspect);

  // Compute min/max (excluding obstacle)
  let minV = Infinity, maxV = -Infinity;
  for (let i = 0; i < nx * ny; i++) {
    if (!obsArr[i]) {
      const v = data[i];
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
  }

  // For diverging colormaps (vorticity), center around 0
  if (diverging) {
    const absMax = Math.max(Math.abs(minV), Math.abs(maxV));
    minV = -absMax; maxV = absMax;
  }

  const range = maxV - minV || 1;

  const img = ctx.createImageData(nx, ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const idx = j * nx + i, p = idx * 4;
      if (obsArr[idx]) {
        img.data[p] = 50; img.data[p+1] = 50; img.data[p+2] = 50; img.data[p+3] = 255;
      } else {
        const val = (data[idx] - minV) / range;
        const ci = Math.max(0, Math.min(255, Math.round(val * 255)));
        const rgb = colormap[ci];
        img.data[p] = rgb[0]; img.data[p+1] = rgb[1]; img.data[p+2] = rgb[2]; img.data[p+3] = 255;
      }
    }
  }

  // Draw: flip Y axis (LBM j=0 is bottom)
  const tmp = document.createElement('canvas');
  tmp.width = nx; tmp.height = ny;
  tmp.getContext('2d').putImageData(img, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = true;
  ctx.save();
  ctx.translate(0, canvas.height);
  ctx.scale(1, -1);
  ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  // Cylinder outline
  if (cx !== undefined && cy !== undefined && radius !== undefined) {
    const cxPx = cx / nx * canvas.width;
    const cyPx = canvas.height - (cy / ny * canvas.height);
    const rPx = radius / nx * canvas.width;
    ctx.beginPath();
    ctx.arc(cxPx, cyPx, rPx, 0, 2 * Math.PI);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Colorbar
  const barW = 12, barH = canvas.height * 0.6;
  const barX = canvas.width - barW - 8, barY = (canvas.height - barH) / 2;
  for (let y = 0; y < barH; y++) {
    const t = 1 - y / barH;
    const ci = Math.round(t * 255);
    const rgb = colormap[Math.min(255, ci)];
    ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    ctx.fillRect(barX, barY + y, barW, 1);
  }
  ctx.fillStyle = '#fff';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(maxV.toFixed(3), barX + barW + 3, barY + 4);
  ctx.fillText(minV.toFixed(3), barX + barW + 3, barY + barH);
}

// ── Auto-init on tab switch ────────────────────────────────────────────────
function cylAutoInit() {
  if (cylInterStarted) return;
  cylInterStarted = true;
  cylInteractiveInit();
}

function cylInteractiveInit() {
  const nx = parseInt(document.getElementById('cyl-nx').value) || 200;
  const ny = parseInt(document.getElementById('cyl-ny').value) || 80;
  const uIn = parseFloat(document.getElementById('cyl-uin').value) || 0.08;
  const re = parseFloat(document.getElementById('cyl-re').value) || 100;
  const radius = parseFloat(document.getElementById('cyl-radius').value) || 10;

  // Create simulation
  cylInterSim = new InteractiveCylinderLBM(nx, ny, uIn, re, radius, nx * 0.25, ny * 0.5);
  cylInterSim.step(300);  // warmup

  // Render initial fields
  const fields = cylInterSim.getFields();
  cylRenderField('cyl-interactive-speed', fields.speed, fields.obstacle, nx, ny, fields.cx, fields.cy, fields.radius, CYL_TURBO, false);
  cylRenderField('cyl-interactive-vort', fields.vorticity, fields.obstacle, nx, ny, fields.cx, fields.cy, fields.radius, CYL_VIRIDIS, true);
  cylRenderField('cyl-interactive-rho', fields.rho, fields.obstacle, nx, ny, fields.cx, fields.cy, fields.radius, CYL_TURBO, false);

  document.getElementById('cyl-interactive-step').textContent = 'Step ' + cylInterSim.stepCount;
  document.getElementById('cyl-interactive-info').textContent = 'Running';
  document.getElementById('cyl-pause-btn').disabled = false;
  document.getElementById('cyl-resume-btn').disabled = true;
  document.getElementById('cyl-reset-btn').disabled = false;

  // Start auto-run
  cylInterPaused = false;
  if (cylInterTimer) clearInterval(cylInterTimer);
  cylInterTimer = setInterval(() => {
    if (!cylInterSim || cylInterPaused) return;
    const n = parseInt(document.getElementById('cyl-steps-per-frame').value) || 10;
    cylInterSim.step(n);
    const fields = cylInterSim.getFields();
    cylRenderField('cyl-interactive-speed', fields.speed, fields.obstacle, fields.nx, fields.ny, fields.cx, fields.cy, fields.radius, CYL_TURBO, false);
    cylRenderField('cyl-interactive-vort', fields.vorticity, fields.obstacle, fields.nx, fields.ny, fields.cx, fields.cy, fields.radius, CYL_VIRIDIS, true);
    cylRenderField('cyl-interactive-rho', fields.rho, fields.obstacle, fields.nx, fields.ny, fields.cx, fields.cy, fields.radius, CYL_TURBO, false);
    document.getElementById('cyl-interactive-step').textContent = 'Step ' + cylInterSim.stepCount;
  }, 80);
}

// ── Pause / Resume / Reset ──────────────────────────────────────────────────
function cylInteractivePause() {
  cylInterPaused = true;
  document.getElementById('cyl-pause-btn').disabled = true;
  document.getElementById('cyl-resume-btn').disabled = false;
  document.getElementById('cyl-interactive-info').textContent = 'Paused';
}

function cylInteractiveResume() {
  cylInterPaused = false;
  document.getElementById('cyl-pause-btn').disabled = false;
  document.getElementById('cyl-resume-btn').disabled = true;
  document.getElementById('cyl-interactive-info').textContent = 'Running';
}

function cylInteractiveReset() {
  if (!cylInterSim) return;
  const nx = cylInterSim.nx, ny = cylInterSim.ny;
  const uIn = cylInterSim.uIn, re = cylInterSim.re, radius = cylInterSim.radius;
  cylInterSim = new InteractiveCylinderLBM(nx, ny, uIn, re, radius, nx * 0.25, ny * 0.5);
  cylInterSim.step(300);
  const fields = cylInterSim.getFields();
  cylRenderField('cyl-interactive-speed', fields.speed, fields.obstacle, nx, ny, fields.cx, fields.cy, fields.radius, CYL_TURBO, false);
  cylRenderField('cyl-interactive-vort', fields.vorticity, fields.obstacle, nx, ny, fields.cx, fields.cy, fields.radius, CYL_VIRIDIS, true);
  cylRenderField('cyl-interactive-rho', fields.rho, fields.obstacle, nx, ny, fields.cx, fields.cy, fields.radius, CYL_TURBO, false);
  document.getElementById('cyl-interactive-step').textContent = 'Step ' + cylInterSim.stepCount;
  cylInterPaused = false;
  document.getElementById('cyl-pause-btn').disabled = false;
  document.getElementById('cyl-resume-btn').disabled = true;
  document.getElementById('cyl-interactive-info').textContent = 'Running';
}

// ── Drag interaction ────────────────────────────────────────────────────────
(function _setupDrag() {
  const canvas = document.getElementById('cyl-interactive-speed');
  if (!canvas) return;

  function _getLBMCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = (cylInterSim ? cylInterSim.nx : 200) / rect.width;
    const scaleY = (cylInterSim ? cylInterSim.ny : 80) / rect.height;
    const px = (e.clientX - rect.left) * scaleX;
    const py = (cylInterSim ? cylInterSim.ny : 80) - (e.clientY - rect.top) * scaleY;
    return { cx: px, cy: py };
  }

  canvas.addEventListener('mousedown', (e) => {
    if (!cylInterSim) return;
    const coords = _getLBMCoords(e);
    const dx = coords.cx - cylInterSim.cx, dy = coords.cy - cylInterSim.cy;
    if (Math.sqrt(dx*dx + dy*dy) < cylInterSim.radius * 1.5) {
      cylDragActive = true;
      canvas.style.cursor = 'grabbing';
      e.preventDefault();
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!cylDragActive || !cylInterSim) return;
    const coords = _getLBMCoords(e);
    cylInterSim.moveCylinder(coords.cx, coords.cy);
  });

  canvas.addEventListener('mouseup', () => {
    cylDragActive = false;
    canvas.style.cursor = 'grab';
  });

  canvas.addEventListener('mouseleave', () => {
    cylDragActive = false;
    canvas.style.cursor = 'grab';
  });

  // Touch support
  canvas.addEventListener('touchstart', (e) => {
    if (!cylInterSim) return;
    const touch = e.touches[0];
    const coords = _getLBMCoords(touch);
    const dx = coords.cx - cylInterSim.cx, dy = coords.cy - cylInterSim.cy;
    if (Math.sqrt(dx*dx + dy*dy) < cylInterSim.radius * 2) {
      cylDragActive = true;
      e.preventDefault();
    }
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    if (!cylDragActive || !cylInterSim) return;
    e.preventDefault();
    const touch = e.touches[0];
    const coords = _getLBMCoords(touch);
    cylInterSim.moveCylinder(coords.cx, coords.cy);
  }, { passive: false });

  canvas.addEventListener('touchend', () => { cylDragActive = false; });
})();


// ════════════════════════════════════════════════════════════════════════════
// SECTION 2: Speed Comparison (CPU f32 vs SDAA f32) — both backend lanes
// ════════════════════════════════════════════════════════════════════════════

let cylPollTimer = null;   // Backend poll timer
let cylCompareStarted = false;

// ── JS CPU LBM class removed — both lanes now run on backend ────────────

// ── Decode base64 for compare endpoint ──────────────────────────────────
function _decodeB64(b64Obj) {
  const raw = atob(b64Obj.b64);
  const buf = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) buf[i] = raw.charCodeAt(i);
  if (b64Obj.dtype === 'f16') {
    // Manual f16 → f32 conversion (Float16Array not widely supported)
    const n = buf.length / 2;
    const result = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const h = buf[2*i] | (buf[2*i+1] << 8);  // little-endian uint16
      const sign = (h >> 15) & 1;
      const exp  = (h >> 10) & 0x1f;
      const frac = h & 0x3ff;
      if (exp === 0) {
        result[i] = sign ? -frac / 16384 : frac / 16384;  // subnormal
      } else if (exp === 31) {
        result[i] = frac ? NaN : (sign ? -Infinity : Infinity);
      } else {
        result[i] = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
      }
    }
    return result;
  } else {
    return buf;
  }
}

// ── Render speed field for compare section ──────────────────────────────────
function cylRender(canvasId, data, obsArr, nx, ny, cx, cy, radius) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const aspect = nx / ny;
  canvas.width = 400; canvas.height = Math.round(400 / aspect);

  let minV=Infinity, maxV=-Infinity;
  for(let i=0;i<nx*ny;i++) if(!obsArr[i]){const v=data[i];if(v<minV)minV=v;if(v>maxV)maxV=v;}
  const range=maxV-minV||1;

  const img=ctx.createImageData(nx,ny);
  for(let j=0;j<ny;j++) for(let i=0;i<nx;i++){
    const idx=j*nx+i, p=idx*4;
    if(obsArr[idx]){img.data[p]=50;img.data[p+1]=50;img.data[p+2]=50;img.data[p+3]=255;}
    else{const val=(data[idx]-minV)/range; const ci=Math.max(0,Math.min(255,Math.round(val*255)));
      const rgb=CYL_TURBO[ci];
      img.data[p]=rgb[0];img.data[p+1]=rgb[1];img.data[p+2]=rgb[2];img.data[p+3]=255;}
  }
  const tmp=document.createElement('canvas');tmp.width=nx;tmp.height=ny;
  tmp.getContext('2d').putImageData(img,0,0);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.imageSmoothingEnabled=true;
  ctx.save();ctx.translate(0,canvas.height);ctx.scale(1,-1);
  ctx.drawImage(tmp,0,0,canvas.width,canvas.height);ctx.restore();

  const cxPx=cx/nx*canvas.width, cyPx=canvas.height-(cy/ny*canvas.height), rPx=radius/nx*canvas.width;
  ctx.beginPath();ctx.arc(cxPx,cyPx,rPx,0,2*Math.PI);
  ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
}

// ── Compare: start both backend simulations ────────────────────────────────
async function cylCompareStart() {
  const nx = parseInt(document.getElementById('cyl-cmp-nx').value) || 400;
  const ny = parseInt(document.getElementById('cyl-cmp-ny').value) || 160;
  const uIn = parseFloat(document.getElementById('cyl-cmp-uin').value) || 0.08;
  const re = parseFloat(document.getElementById('cyl-cmp-re').value) || 100;
  const radius = parseFloat(document.getElementById('cyl-cmp-radius').value) || 13;

  // Start backend simulations (cpu_f32 + sdaa_f32)
  document.getElementById('cyl-compare-info').textContent = 'Starting backend simulations…';
  try {
    const resp = await fetch('/api/cylinder-compare/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({nx, ny, u_in: uIn, re, radius}),
    });
    const data = await resp.json();
    document.getElementById('cyl-compare-info').textContent = 'Both simulations running';
    cylCompareStarted = true;
  } catch (e) {
    document.getElementById('cyl-compare-info').textContent = 'Backend error: ' + e.message;
    return;
  }

  // Poll backend results for both lanes
  if (cylPollTimer) clearInterval(cylPollTimer);
  cylPollTimer = setInterval(async () => {
    try {
      const resp = await fetch('/api/cylinder-compare/results');
      const data = await resp.json();

      // CPU f32 lane
      if (data.cpu_f32) {
        const d = data.cpu_f32;
        if (d.warming_up) {
          document.getElementById('cyl-cpu-step').textContent = 'Warming up…';
          document.getElementById('cyl-cpu-speed').textContent = '—';
        } else if (d.speed) {
          const speedArr = _decodeB64(d.speed);
          const obsArr = _decodeB64(d.obstacle);
          const dsNx = d.speed.shape[1], dsNy = d.speed.shape[0];
          const dsCx = d.nx * 0.25 / 2;
          const dsCy = d.ny * 0.5 / 2;
          const dsRadius = (d.radius || 13) / 2;
          cylRender('cyl-canvas-cpu', speedArr, obsArr, dsNx, dsNy, dsCx, dsCy, dsRadius);
          document.getElementById('cyl-cpu-step').textContent = 'Step ' + d.step;
          document.getElementById('cyl-cpu-speed').textContent = d.ms_per_step.toFixed(1) + ' ms/step';
          cylCompareData.cpuMs = d.ms_per_step;
        } else if (d.error) {
          document.getElementById('cyl-cpu-step').textContent = 'Error';
          document.getElementById('cyl-cpu-speed').textContent = d.error;
        }
      }

      // SDAA f32 lane
      if (data.sdaa_f32) {
        const d = data.sdaa_f32;
        if (d.warming_up) {
          document.getElementById('cyl-sdaa-step').textContent = 'Warming up…';
          document.getElementById('cyl-sdaa-speed').textContent = '—';
        } else if (d.speed) {
          const speedArr = _decodeB64(d.speed);
          const obsArr = _decodeB64(d.obstacle);
          const dsNx = d.speed.shape[1], dsNy = d.speed.shape[0];
          const dsCx = d.nx * 0.25 / 2;
          const dsCy = d.ny * 0.5 / 2;
          const dsRadius = (d.radius || 13) / 2;
          cylRender('cyl-canvas-sdaa', speedArr, obsArr, dsNx, dsNy, dsCx, dsCy, dsRadius);
          document.getElementById('cyl-sdaa-step').textContent = 'Step ' + d.step;
          document.getElementById('cyl-sdaa-speed').textContent = d.ms_per_step.toFixed(1) + ' ms/step';
          cylCompareData.sdaaMs = d.ms_per_step;
        } else if (d.error) {
          document.getElementById('cyl-sdaa-step').textContent = 'Error';
          document.getElementById('cyl-sdaa-speed').textContent = d.error;
        }
      }

      cylUpdateChart();
    } catch (e) { console.error('cylinder compare poll error:', e); }
  }, 200);
}

function cylCompareStop() {
  if (cylPollTimer) { clearInterval(cylPollTimer); cylPollTimer = null; }
  cylCompareStarted = false;
  fetch('/api/cylinder-compare/stop', {method:'POST'}).catch(()=>{});
  document.getElementById('cyl-compare-info').textContent = 'Stopped — click Start to restart';
}

// ── Speed comparison data ──────────────────────────────────────────────────
const cylCompareData = { cpuMs: 0, sdaaMs: 0 };

function cylUpdateChart() {
  const canvas = document.getElementById('cyl-compare-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const items = [
    { label: '某厂商', ms: cylCompareData.cpuMs, color: '#4488ff' },
    { label: 'SDAA', ms: cylCompareData.sdaaMs, color: '#44ff88' },
  ];

  const maxMs = Math.max(...items.map(i => i.ms || 1));
  const barW = 180, gap = 60, startX = 80;
  const chartH = H - 40, chartY = 20;

  ctx.fillStyle = '#fff'; ctx.font = '13px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('ms/step — lower is faster', W/2, 14);

  for (let i = 0; i < items.length; i++) {
    const ms = items[i].ms || 0;
    const barH = ms / maxMs * (chartH - 10);
    const x = startX + i * (barW + gap);
    const y = chartY + chartH - barH;

    ctx.fillStyle = items[i].color;
    ctx.fillRect(x, y, barW, barH);

    ctx.fillStyle = '#fff'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(items[i].label, x + barW/2, chartY + chartH + 14);
    ctx.fillText(ms.toFixed(1) + ' ms', x + barW/2, y - 6);

    if (i > 0 && ms > 0 && items[0].ms > 0) {
      const speedup = items[0].ms / ms;
      ctx.fillStyle = '#ff0'; ctx.font = '11px sans-serif';
      ctx.fillText(speedup.toFixed(1) + 'x', x + barW/2, y + barH/2);
    }
  }

  ctx.strokeStyle = '#666'; ctx.beginPath();
  ctx.moveTo(startX-5, chartY); ctx.lineTo(startX-5, chartY+chartH); ctx.stroke();
  ctx.fillStyle = '#aaa'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
  ctx.fillText('0', startX-8, chartY+chartH+4);
  ctx.fillText(maxMs.toFixed(0), startX-8, chartY+4);
}



