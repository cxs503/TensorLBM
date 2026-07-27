/**
 * app_generic.js – Phase 3 Generic CFD Frontend
 *
 * Self-contained vanilla-JS module for the generic STL-driven LBM simulation
 * workflow:
 *   1. STL upload + 3D preview (Three.js, drag-drop, mesh stats)
 *   2. Parameter panel (physics / solver / geometry / output)
 *   3. Run button → POST /api/simulations/generic-run (multipart FormData)
 *   4. Real-time results via WebSocket /api/simulations/generic-run/{job_id}/ws
 *      - Cd / Cl / St live curves (Chart.js)
 *      - Force decomposition (pressure + friction)
 *   5. Field visualization – GET .../fields/velocity → 2D slice canvas + colormap
 *   6. Results summary – GET .../results → final coefficients + error vs reference
 *
 * Style: consistent with app_core.js / app_geo3d.js (IIFE, window.* exports,
 * escHtml/showToast/api helpers from app_utils.js / app_core.js).
 */
/* global THREE, Chart */
(function () {
  'use strict';

  // ── Constants ────────────────────────────────────────────────────────────
  var GENERIC_API = '/api/simulations';
  var RUN_ENDPOINT = GENERIC_API + '/generic-run';

  // ── Module state ─────────────────────────────────────────────────────────
  var _stlFile = null;        // File object
  var _stlGeo = null;         // { vertices, faces, bbox, size, volume }
  var _renderer = null;
  var _scene = null;
  var _camera = null;
  var _controls = null;
  var _stlMesh = null;
  var _bboxHelper = null;
  var _currentJobId = null;
  var _jobWs = null;
  var _pollTimer = null;
  var _charts = {};           // { cd: Chart, cl: Chart, forces: Chart }
  var _fieldCanvas = null;
  var _fieldCtx = null;
  var _fieldData = null;
  var _fieldSlice = 0;
  var _fieldAxis = 'z';
  var _wsReconnectTimer = null;
  var _wsClosedByUser = false;

  // ── Helpers ──────────────────────────────────────────────────────────────

  function _el(id) { return document.getElementById(id); }

  function _setHTML(id, html) {
    var e = _el(id);
    if (e) e.innerHTML = html;
  }

  function _setText(id, txt) {
    var e = _el(id);
    if (e) e.textContent = txt;
  }

  function _num(id, fallback) {
    var e = _el(id);
    if (!e || e.value === '') return fallback;
    var v = parseFloat(e.value);
    return isNaN(v) ? fallback : v;
  }

  function _fmt(v, d) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    return Number(v).toFixed(d || 4);
  }

  function _fmtSci(v) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    var a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e4)) return v.toExponential(3);
    return Number(v).toFixed(4);
  }

  // ── Turbo colormap (approximation) for field visualization ───────────────

  function _turbo(t) {
    t = Math.max(0, Math.min(1, t));
    var r, g, b;
    // Piecewise approximation of the Turbo colormap
    if (t < 0.25) {
      r = 0.0; g = t / 0.25 * 0.6; b = 0.3 + t / 0.25 * 0.7;
    } else if (t < 0.5) {
      var u = (t - 0.25) / 0.25;
      r = u * 0.5; g = 0.6 + u * 0.4; b = 1.0 - u * 0.3;
    } else if (t < 0.75) {
      var u2 = (t - 0.5) / 0.25;
      r = 0.5 + u2 * 0.5; g = 1.0 - u2 * 0.4; b = 0.7 - u2 * 0.7;
    } else {
      var u3 = (t - 0.75) / 0.25;
      r = 1.0; g = 0.6 - u3 * 0.6; b = 0.0;
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  }

  // ── STL Parsing (binary + ASCII) ─────────────────────────────────────────

  function _isBinarySTL(buffer) {
    if (buffer.byteLength < 84) return false;
    var view = new DataView(buffer);
    var n = view.getUint32(80, true);
    return buffer.byteLength === 84 + n * 50;
  }

  function _parseSTL(buffer) {
    var header = new TextDecoder().decode(new Uint8Array(buffer, 0, Math.min(256, buffer.byteLength)));
    var isBinary = !/^solid/i.test(header.substring(0, 5)) || _isBinarySTL(buffer);
    return isBinary ? _parseBinarySTL(buffer) : _parseAsciiSTL(buffer);
  }

  function _parseBinarySTL(buffer) {
    var view = new DataView(buffer);
    var n = view.getUint32(80, true);
    var positions = new Float32Array(n * 9);
    var normals = new Float32Array(n * 9);
    var offset = 84;
    for (var i = 0; i < n; i++) {
      var nx = view.getFloat32(offset, true);
      var ny = view.getFloat32(offset + 4, true);
      var nz = view.getFloat32(offset + 8, true);
      offset += 12;
      for (var v = 0; v < 3; v++) {
        var b = i * 9 + v * 3;
        positions[b] = view.getFloat32(offset, true);
        positions[b + 1] = view.getFloat32(offset + 4, true);
        positions[b + 2] = view.getFloat32(offset + 8, true);
        normals[b] = nx; normals[b + 1] = ny; normals[b + 2] = nz;
        offset += 12;
      }
      offset += 2;
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    return geo;
  }

  function _parseAsciiSTL(buffer) {
    var text = new TextDecoder().decode(buffer);
    var lines = text.split('\n');
    var vertices = [];
    var normals = [];
    var cn = [0, 0, 1];
    for (var li = 0; li < lines.length; li++) {
      var line = lines[li].trim();
      if (line.indexOf('facet normal') === 0) {
        var p = line.split(/\s+/);
        cn = [parseFloat(p[2]), parseFloat(p[3]), parseFloat(p[4])];
      } else if (line.indexOf('vertex') === 0) {
        var pv = line.split(/\s+/);
        vertices.push(parseFloat(pv[1]), parseFloat(pv[2]), parseFloat(pv[3]));
        normals.push(cn[0], cn[1], cn[2]);
      }
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normals), 3));
    return geo;
  }

  // ── Three.js 3D Preview ──────────────────────────────────────────────────

  function _initThree() {
    var wrap = _el('generic-canvas-wrap');
    var canvas = _el('generic-canvas');
    if (!wrap || !canvas) return;
    if (_renderer) return;

    var W = wrap.clientWidth || 600;
    var H = wrap.clientHeight || 420;

    _renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    _renderer.setPixelRatio(window.devicePixelRatio);
    _renderer.setSize(W, H, false);

    _scene = new THREE.Scene();
    _scene.background = new THREE.Color(0x111827);

    _camera = new THREE.PerspectiveCamera(45, W / H, 0.001, 100000);
    _camera.position.set(0, 0, 5);

    _scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    var dir = new THREE.DirectionalLight(0xffffff, 0.85);
    dir.position.set(5, 10, 7);
    _scene.add(dir);
    var dir2 = new THREE.DirectionalLight(0x6688ff, 0.3);
    dir2.position.set(-5, -5, -7);
    _scene.add(dir2);

    var grid = new THREE.GridHelper(20, 20, 0x334155, 0x334155);
    _scene.add(grid);

    if (typeof THREE.OrbitControls !== 'undefined') {
      _controls = new THREE.OrbitControls(_camera, _renderer.domElement);
      _controls.enableDamping = true;
      _controls.dampingFactor = 0.08;
    }

    new ResizeObserver(function () {
      if (!_renderer || !wrap) return;
      var nW = wrap.clientWidth;
      var nH = wrap.clientHeight;
      _camera.aspect = nW / nH;
      _camera.updateProjectionMatrix();
      _renderer.setSize(nW, nH, false);
    }).observe(wrap);

    _animate();
  }

  function _animate() {
    requestAnimationFrame(_animate);
    if (_controls) _controls.update();
    if (_renderer && _scene && _camera) _renderer.render(_scene, _camera);
  }

  // ── Load STL file into 3D viewer + compute stats ─────────────────────────

  function _loadSTLFile(file) {
    if (!file) return;
    var name = (file.name || '').toLowerCase();
    if (!name.endsWith('.stl')) {
      showToast('Please select an .stl file', 'warning');
      return;
    }
    _stlFile = file;
    _setText('generic-stl-name', file.name);
    _setText('generic-stl-size', (file.size / 1024).toFixed(1) + ' KB');

    var reader = new FileReader();
    reader.onload = function (evt) {
      _initThree();
      if (_stlMesh) {
        _scene.remove(_stlMesh);
        _stlMesh.geometry.dispose();
        _stlMesh = null;
      }
      if (_bboxHelper) {
        _scene.remove(_bboxHelper);
        _bboxHelper = null;
      }
      try {
        var geo = _parseSTL(evt.target.result);
        geo.computeBoundingBox();
        geo.computeVertexNormals();

        var box = geo.boundingBox;
        var centre = new THREE.Vector3();
        box.getCenter(centre);
        var size = new THREE.Vector3();
        box.getSize(size);
        var maxDim = Math.max(size.x, size.y, size.z) || 1;

        // Centre + normalise to fit in view
        geo.translate(-centre.x, -centre.y, -centre.z);
        var scale = 2 / maxDim;

        var mat = new THREE.MeshPhongMaterial({
          color: 0x4488cc, specular: 0x224466, shininess: 60,
          side: THREE.DoubleSide, transparent: true, opacity: 0.92,
        });
        _stlMesh = new THREE.Mesh(geo, mat);
        _stlMesh.scale.setScalar(scale);
        _scene.add(_stlMesh);

        // Bounding box wireframe
        var bboxGeo = new THREE.BoxGeometry(size.x, size.y, size.z);
        var bboxEdges = new THREE.EdgesGeometry(bboxGeo);
        _bboxHelper = new THREE.LineSegments(
          bboxEdges,
          new THREE.LineBasicMaterial({ color: 0x22c55e, linewidth: 1 })
        );
        _bboxHelper.scale.setScalar(scale);
        _scene.add(_bboxHelper);

        _camera.position.set(0, 0, 4);
        if (_controls) _controls.reset();

        // Hide placeholder
        var ph = _el('generic-placeholder');
        if (ph) ph.style.display = 'none';

        // Compute mesh stats
        var nVerts = geo.attributes.position.count;
        var nFaces = Math.round(nVerts / 3);
        // Approximate volume via signed tetrahedron volume
        var volume = _computeMeshVolume(geo);

        _stlGeo = {
          vertices: nVerts,
          faces: nFaces,
          bbox: {
            min: [box.min.x, box.min.y, box.min.z],
            max: [box.max.x, box.max.y, box.max.z],
          },
          size: { x: size.x, y: size.y, z: size.z },
          volume: volume,
        };

        _setHTML('generic-mesh-info',
          '<table class="table table-sm table-borderless mb-0 small">' +
          '<tr><th>Vertices</th><td>' + nVerts.toLocaleString() + '</td></tr>' +
          '<tr><th>Triangles</th><td>' + nFaces.toLocaleString() + '</td></tr>' +
          '<tr><th>Bounding box</th><td>' +
            _fmt(size.x, 2) + ' × ' + _fmt(size.y, 2) + ' × ' + _fmt(size.z, 2) + '</td></tr>' +
          '<tr><th>Volume (approx)</th><td>' + _fmtSci(volume) + '</td></tr>' +
          '</table>'
        );

        // Auto-suggest domain
        _suggestDomain(size, volume);

        showToast('STL loaded: ' + nFaces.toLocaleString() + ' faces', 'success');
      } catch (e) {
        console.error('STL parse error:', e);
        showToast('Failed to parse STL: ' + e.message, 'danger');
        _setHTML('generic-mesh-info', '<span class="text-danger small">Parse error: ' + escHtml(e.message) + '</span>');
      }
    };
    reader.readAsArrayBuffer(file);
  }

  function _computeMeshVolume(geo) {
    // Signed volume of tetrahedra from origin
    var pos = geo.attributes.position.array;
    var n = pos.length / 9;
    var vol = 0;
    for (var i = 0; i < n; i++) {
      var a = i * 9;
      var v0x = pos[a],     v0y = pos[a + 1], v0z = pos[a + 2];
      var v1x = pos[a + 3], v1y = pos[a + 4], v1z = pos[a + 5];
      var v2x = pos[a + 6], v2y = pos[a + 7], v2z = pos[a + 8];
      vol += (v0x * (v1y * v2z - v1z * v2y) +
              v0y * (v1z * v2x - v1x * v2z) +
              v0z * (v1x * v2y - v1y * v2x)) / 6.0;
    }
    return Math.abs(vol);
  }

  // ── Auto-domain suggestion (blockage < 10%) ─────────────────────────────

  function _suggestDomain(size, volume) {
    // Flow direction = x; frontal area ≈ size.y * size.z
    var frontalArea = size.y * size.z;
    // Target blockage < 10% → domain cross-section > 10 * frontalArea
    var targetArea = frontalArea / 0.10;
    var dy = Math.sqrt(targetArea);
    var dz = dy;
    // Domain length: 5× object length upstream + 15× downstream (typical)
    var dx = size.x * 20;

    // Round to nice numbers
    dy = Math.ceil(dy / 10) * 10;
    dz = Math.ceil(dz / 10) * 10;
    dx = Math.ceil(dx / 10) * 10;

    var blockage = (frontalArea / (dy * dz) * 100).toFixed(2);

    _setHTML('generic-domain-suggest',
      '<div class="small">' +
      '<span class="text-muted">Suggested domain (blockage &lt; 10%):</span><br>' +
      '<code>NX=' + dx + '  NY=' + dy + '  NZ=' + dz + '</code><br>' +
      '<span class="text-muted">Blockage: ' + blockage + '%</span>' +
      '</div>'
    );

    // Auto-fill domain inputs
    _setVal('generic-nx', dx);
    _setVal('generic-ny', dy);
    _setVal('generic-nz', dz);
    _setText('generic-blockage-pct', blockage + '%');
  }

  function _setVal(id, v) {
    var e = _el(id);
    if (e) e.value = v;
  }

  // ── Parameter collection ─────────────────────────────────────────────────

  function _collectParams() {
    var collision = _el('generic-collision') ? _el('generic-collision').value : 'mrt';
    var outputFields = [];
    document.querySelectorAll('.generic-field-cb:checked').forEach(function (cb) {
      outputFields.push(cb.value);
    });

    return {
      physics: {
        re: _num('generic-re', 100),
        u_in: _num('generic-u-in', 0.05),
        density: _num('generic-density', 1.0),
        viscosity: _num('generic-viscosity', 0.001),
      },
      solver: {
        collision: collision,
        cs: _num('generic-cs', 0.1),
        steps: _num('generic-steps', 10000),
        warmup: _num('generic-warmup', 2000),
      },
      geometry: {
        nx: _num('generic-nx', 200),
        ny: _num('generic-ny', 100),
        nz: _num('generic-nz', 100),
        auto_domain: _el('generic-auto-domain') ? _el('generic-auto-domain').checked : true,
        blockage_target: _num('generic-blockage-target', 0.10),
      },
      output: {
        fields: outputFields,
        forces: _el('generic-out-forces') ? _el('generic-out-forces').checked : true,
        st: _el('generic-out-st') ? _el('generic-out-st').checked : true,
        interval: _num('generic-output-interval', 500),
      },
      device: _el('generic-device') ? _el('generic-device').value : 'cpu',
      name: _el('generic-job-name') ? _el('generic-job-name').value.trim() : '',
    };
  }

  // ── Run submission ──────────────────────────────────────────────────────

  function _submitRun() {
    if (!_stlFile) {
      showToast('Please upload an STL file first', 'warning');
      return;
    }
    var params = _collectParams();
    var formData = new FormData();
    formData.append('stl_file', _stlFile);
    formData.append('params', JSON.stringify(params));

    var btn = _el('generic-run-btn');
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Submitting…';
    }
    _setHTML('generic-run-status', '<span class="text-muted"><span class="spinner-border spinner-border-sm"></span> Submitting job…</span>');

    fetch(RUN_ENDPOINT, {
      method: 'POST',
      body: formData,
    })
      .then(function (r) {
        if (!r.ok) {
          return r.text().then(function (txt) {
            throw new Error(r.status + ': ' + (txt || r.statusText));
          });
        }
        return r.json();
      })
      .then(function (data) {
        _currentJobId = data.job_id;
        _setText('generic-job-id', data.job_id);
        _setHTML('generic-run-status',
          '<span class="badge bg-info">queued</span> Job ID: <code>' + escHtml(data.job_id) + '</code>');
        showToast('Job submitted: ' + data.job_id, 'success');
        // Enable cancel button
        var cancelBtn = _el('generic-cancel-btn');
        if (cancelBtn) cancelBtn.disabled = false;
        // Connect WebSocket for real-time results
        _connectJobWS(data.job_id);
        // Also start REST polling as fallback
        _startPolling(data.job_id);
      })
      .catch(function (e) {
        _setHTML('generic-run-status', '<span class="text-danger">Error: ' + escHtml(e.message) + '</span>');
        showToast('Submit failed: ' + e.message, 'danger');
      })
      .finally(function () {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = '<i class="bi bi-play-fill"></i> Run Simulation';
        }
      });
  }

  // ── Cancel job ───────────────────────────────────────────────────────────

  function _cancelJob() {
    if (!_currentJobId) return;
    if (!confirm('Cancel job ' + _currentJobId + '?')) return;
    fetch(RUN_ENDPOINT + '/' + encodeURIComponent(_currentJobId) + '/cancel', {
      method: 'POST',
    })
      .then(function (r) { return r.json().catch(function () { return { status: 'cancelled' }; }); })
      .then(function (data) {
        showToast('Job cancelled', 'warning');
        _setHTML('generic-run-status', '<span class="badge bg-secondary">cancelled</span>');
        _closeJobWS();
        _stopPolling();
      })
      .catch(function (e) {
        showToast('Cancel failed: ' + e.message, 'danger');
      });
  }

  // ── WebSocket for real-time results ───────────────────────────────────────

  function _connectJobWS(jobId) {
    _closeJobWS();
    _wsClosedByUser = false;
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    var url = proto + '//' + location.host + RUN_ENDPOINT + '/' + encodeURIComponent(jobId) + '/ws';
    try {
      _jobWs = new WebSocket(url);
    } catch (e) {
      console.warn('WS connect failed:', e);
      return;
    }

    _jobWs.onopen = function () {
      _setHTML('generic-ws-status', '<span class="dot dot-running"></span> Live');
    };
    _jobWs.onclose = function () {
      _setHTML('generic-ws-status', '<span class="dot dot-queued"></span> Disconnected');
      if (!_wsClosedByUser && _currentJobId) {
        _wsReconnectTimer = setTimeout(function () { _connectJobWS(_currentJobId); }, 3000);
      }
    };
    _jobWs.onerror = function () {
      _setHTML('generic-ws-status', '<span class="dot dot-failed"></span> Error');
    };
    _jobWs.onmessage = function (ev) {
      var msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      _handleWSMessage(msg);
    };
  }

  function _closeJobWS() {
    _wsClosedByUser = true;
    if (_wsReconnectTimer) { clearTimeout(_wsReconnectTimer); _wsReconnectTimer = null; }
    if (_jobWs) {
      _jobWs.onclose = null;
      try { _jobWs.close(); } catch (e) {}
      _jobWs = null;
    }
  }

  function _handleWSMessage(msg) {
    var type = msg.type || msg.event || '';
    var step = msg.step || msg.timestep || 0;

    // Status update
    if (type === 'status' || type === 'job_status') {
      _updateJobStatus(msg.status || msg.job_status || 'running', step, msg);
    }

    // Metrics update (Cd, Cl, St, forces)
    if (type === 'metrics' || type === 'force' || type === 'diagnostics' || msg.cd !== undefined) {
      _updateCharts(step, msg);
      _updateForceDecomposition(msg);
    }

    // Progress
    if (msg.progress !== undefined || msg.percent !== undefined) {
      var pct = msg.progress !== undefined ? msg.progress : msg.percent;
      _updateProgress(pct, step);
    }

    // Completion
    if (type === 'completed' || type === 'done' || msg.status === 'completed') {
      _updateJobStatus('completed', step, msg);
      _onJobComplete();
    }

    // Failure
    if (type === 'failed' || type === 'error' || msg.status === 'failed') {
      _updateJobStatus('failed', step, msg);
      _setHTML('generic-run-status', '<span class="badge bg-danger">failed</span> ' + escHtml(msg.error || msg.message || ''));
      _closeJobWS();
      _stopPolling();
    }
  }

  function _updateJobStatus(status, step, msg) {
    var badgeCls = { queued: 'secondary', running: 'warning', completed: 'success', failed: 'danger', cancelled: 'dark' }[status] || 'secondary';
    var html = '<span class="badge bg-' + badgeCls + '">' + escHtml(status) + '</span>';
    if (step) html += ' · step ' + step;
    if (msg.eta_s != null) html += ' · ETA ' + msg.eta_s + 's';
    _setHTML('generic-run-status', html);
  }

  function _updateProgress(pct, step) {
    var bar = _el('generic-progress-bar');
    if (bar) {
      bar.style.width = Math.min(100, pct) + '%';
      bar.setAttribute('aria-valuenow', pct);
    }
    var txt = _el('generic-progress-text');
    if (txt) txt.textContent = (pct.toFixed(1)) + '%' + (step ? ' · step ' + step : '');
  }

  // ── Chart.js live curves ─────────────────────────────────────────────────

  function _initCharts() {
    if (typeof Chart === 'undefined') return;
    if (_charts.cd) return; // already init

    var commonOpts = {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { title: { display: true, text: 'Step' }, grid: { color: '#e9ecef' } },
        y: { grid: { color: '#e9ecef' } },
      },
      plugins: { legend: { display: true, position: 'top', labels: { boxWidth: 12, font: { size: 10 } } } },
      elements: { point: { radius: 0 } },
    };

    // Cd chart (total + pressure + friction)
    _charts.cd = new Chart(_el('generic-chart-cd'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Cd (total)', data: [], borderColor: '#0d6efd', backgroundColor: 'rgba(13,110,253,.1)', borderWidth: 2, fill: false },
          { label: 'Cd (pressure)', data: [], borderColor: '#dc3545', borderWidth: 1.5, fill: false, borderDash: [4, 3] },
          { label: 'Cd (friction)', data: [], borderColor: '#198754', borderWidth: 1.5, fill: false, borderDash: [2, 2] },
        ],
      },
      options: Object.assign({}, commonOpts, {
        scales: Object.assign({}, commonOpts.scales, {
          y: Object.assign({}, commonOpts.scales.y, { title: { display: true, text: 'Cd' } }),
        }),
      }),
    });

    // Cl chart
    _charts.cl = new Chart(_el('generic-chart-cl'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Cl', data: [], borderColor: '#7c3aed', backgroundColor: 'rgba(124,58,237,.1)', borderWidth: 2, fill: false },
        ],
      },
      options: Object.assign({}, commonOpts, {
        scales: Object.assign({}, commonOpts.scales, {
          y: Object.assign({}, commonOpts.scales.y, { title: { display: true, text: 'Cl' } }),
        }),
      }),
    });

    // St chart (Strouhal)
    _charts.st = new Chart(_el('generic-chart-st'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'St', data: [], borderColor: '#d97706', backgroundColor: 'rgba(217,119,6,.1)', borderWidth: 2, fill: false },
        ],
      },
      options: Object.assign({}, commonOpts, {
        scales: Object.assign({}, commonOpts.scales, {
          y: Object.assign({}, commonOpts.scales.y, { title: { display: true, text: 'St' } }),
        }),
      }),
    });
  }

  function _updateCharts(step, msg) {
    if (!_charts.cd) _initCharts();
    if (!_charts.cd) return;

    var label = String(step);
    var maxPoints = 500;

    function push(chart, val) {
      if (val === null || val === undefined || isNaN(val)) return;
      chart.data.labels.push(label);
      chart.data.datasets[0].data.push(val);
      if (chart.data.labels.length > maxPoints) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
      }
      chart.update('none');
    }

    // Cd chart: total, pressure, friction
    var cd = msg.cd !== undefined ? msg.cd : msg.cd_total;
    var cdP = msg.cd_pressure !== undefined ? msg.cd_pressure : msg.cd_p;
    var cdF = msg.cd_friction !== undefined ? msg.cd_friction : msg.cd_f;

    if (cd !== undefined || cdP !== undefined || cdF !== undefined) {
      _pushMulti(_charts.cd, label, [cd, cdP, cdF], maxPoints);
    }

    // Cl
    var cl = msg.cl !== undefined ? msg.cl : msg.cl_total;
    if (cl !== undefined) push(_charts.cl, cl);

    // St
    var st = msg.st !== undefined ? msg.st : msg.strouhal;
    if (st !== undefined) push(_charts.st, st);
  }

  function _pushMulti(chart, label, vals, maxPoints) {
    chart.data.labels.push(label);
    for (var i = 0; i < vals.length; i++) {
      var v = vals[i];
      if (v === null || v === undefined || isNaN(v)) v = null;
      chart.data.datasets[i].data.push(v);
    }
    if (chart.data.labels.length > maxPoints) {
      chart.data.labels.shift();
      for (var j = 0; j < chart.data.datasets.length; j++) {
        chart.data.datasets[j].data.shift();
      }
    }
    chart.update('none');
  }

  function _updateForceDecomposition(msg) {
    var cdP = msg.cd_pressure !== undefined ? msg.cd_pressure : msg.cd_p;
    var cdF = msg.cd_friction !== undefined ? msg.cd_friction : msg.cd_f;
    var cdTot = msg.cd !== undefined ? msg.cd : msg.cd_total;
    if (cdP === undefined && cdF === undefined) return;

    var pPct = (cdTot && cdP) ? (cdP / cdTot * 100).toFixed(1) : '—';
    var fPct = (cdTot && cdF) ? (cdF / cdTot * 100).toFixed(1) : '—';
    _setHTML('generic-force-decomp',
      '<table class="table table-sm table-borderless mb-0 small">' +
      '<tr><th>Cd (pressure)</th><td>' + _fmt(cdP) + ' <span class="text-muted">(' + pPct + '%)</span></td></tr>' +
      '<tr><th>Cd (friction)</th><td>' + _fmt(cdF) + ' <span class="text-muted">(' + fPct + '%)</span></td></tr>' +
      '<tr><th>Cd (total)</th><td><strong>' + _fmt(cdTot) + '</strong></td></tr>' +
      '</table>'
    );
  }

  // ── REST polling fallback ────────────────────────────────────────────────

  function _startPolling(jobId) {
    _stopPolling();
    _pollTimer = setInterval(function () { _pollJob(jobId); }, 3000);
  }

  function _stopPolling() {
    if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
  }

  function _pollJob(jobId) {
    fetch(RUN_ENDPOINT + '/' + encodeURIComponent(jobId) + '/status')
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        if (!data) return;
        _updateJobStatus(data.status || 'running', data.step || 0, data);
        if (data.progress !== undefined) _updateProgress(data.progress, data.step || 0);
        if (data.metrics) {
          _updateCharts(data.step || 0, data.metrics);
          _updateForceDecomposition(data.metrics);
        }
        if (data.status === 'completed') _onJobComplete();
        if (data.status === 'failed' || data.status === 'cancelled') {
          _stopPolling();
          _closeJobWS();
        }
      })
      .catch(function () {});
  }

  // ── Job completion ────────────────────────────────────────────────────────

  function _onJobComplete() {
    _stopPolling();
    _closeJobWS();
    _setHTML('generic-ws-status', '<span class="dot dot-completed"></span> Done');
    var cancelBtn = _el('generic-cancel-btn');
    if (cancelBtn) cancelBtn.disabled = true;
    // Auto-load results
    _loadResults();
    _loadField('velocity');
    showToast('Simulation completed', 'success');
  }

  // ── Field visualization (2D slice) ───────────────────────────────────────

  function _loadField(fieldType) {
    if (!_currentJobId) {
      showToast('No active job. Run a simulation first.', 'warning');
      return;
    }
    _fieldAxis = _el('generic-field-axis') ? _el('generic-field-axis').value : 'z';
    _fieldSlice = _num('generic-field-slice', 0);

    var url = RUN_ENDPOINT + '/' + encodeURIComponent(_currentJobId) +
      '/fields/' + encodeURIComponent(fieldType) +
      '?axis=' + _fieldAxis + '&slice=' + _fieldSlice;
    _setHTML('generic-field-status', '<span class="text-muted"><span class="spinner-border spinner-border-sm"></span> Loading ' + escHtml(fieldType) + ' field…</span>');

    fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error(r.status + ': ' + r.statusText);
        var ct = (r.headers.get('content-type') || '').toLowerCase();
        if (ct.includes('image/')) return r.blob().then(function (b) { return { image: b }; });
        return r.json();
      })
      .then(function (data) {
        _fieldData = data;
        _renderField(data);
        _setHTML('generic-field-status', '<span class="text-success">Field loaded</span>');
      })
      .catch(function (e) {
        _setHTML('generic-field-status', '<span class="text-danger">Error: ' + escHtml(e.message) + '</span>');
        _setHTML('generic-field-info', '');
      });
  }

  function _renderField(data) {
    var canvas = _el('generic-field-canvas');
    if (!canvas) return;
    _fieldCtx = canvas.getContext('2d');

    // Case 1: server returned an image (PNG/JPEG)
    if (data.image) {
      var img = new Image();
      img.onload = function () {
        canvas.width = img.width;
        canvas.height = img.height;
        _fieldCtx.drawImage(img, 0, 0);
        _setHTML('generic-field-info', 'Image: ' + img.width + '×' + img.height);
      };
      img.src = URL.createObjectURL(data.image);
      return;
    }

    // Case 2: server returned a base64 image string
    if (data.image_b64 || data.png_b64) {
      var b64 = data.image_b64 || data.png_b64;
      var img2 = new Image();
      img2.onload = function () {
        canvas.width = img2.width;
        canvas.height = img2.height;
        _fieldCtx.drawImage(img2, 0, 0);
        _setHTML('generic-field-info', 'Image: ' + img2.width + '×' + img2.height);
      };
      img2.src = 'data:image/png;base64,' + b64;
      return;
    }

    // Case 3: server returned a 2D array of values
    var grid = data.grid || data.values || data.data;
    if (Array.isArray(grid) && grid.length > 0) {
      _renderFieldGrid(grid, data);
      return;
    }

    // Case 4: nested {u, v, w} arrays → magnitude
    if (data.u && Array.isArray(data.u)) {
      var ny = data.u.length;
      var nx = data.u[0].length;
      var mag = [];
      for (var i = 0; i < ny; i++) {
        mag.push([]);
        for (var j = 0; j < nx; j++) {
          var u = data.u[i][j] || 0;
          var v = (data.v && data.v[i]) ? data.v[i][j] || 0 : 0;
          var w = (data.w && data.w[i]) ? data.w[i][j] || 0 : 0;
          mag[i].push(Math.sqrt(u * u + v * v + w * w));
        }
      }
      _renderFieldGrid(mag, data);
      return;
    }

    _setHTML('generic-field-info', '<span class="text-muted">Unknown field format</span>');
  }

  function _renderFieldGrid(grid, meta) {
    var canvas = _el('generic-field-canvas');
    var ctx = canvas.getContext('2d');
    var ny = grid.length;
    var nx = grid[0].length;
    canvas.width = nx;
    canvas.height = ny;

    // Find min/max for normalisation
    var vmin = Infinity, vmax = -Infinity;
    for (var i = 0; i < ny; i++) {
      for (var j = 0; j < nx; j++) {
        var v = grid[i][j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
    }
    var range = (vmax - vmin) || 1;

    var imgData = ctx.createImageData(nx, ny);
    for (var i2 = 0; i2 < ny; i2++) {
      for (var j2 = 0; j2 < nx; j2++) {
        var val = grid[i2][j2];
        var t = (val - vmin) / range;
        var rgb = _turbo(t);
        var idx = (i2 * nx + j2) * 4;
        imgData.data[idx] = rgb[0];
        imgData.data[idx + 1] = rgb[1];
        imgData.data[idx + 2] = rgb[2];
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    _setHTML('generic-field-info',
      'Grid: ' + nx + '×' + ny + ' | Range: [' + _fmtSci(vmin) + ', ' + _fmtSci(vmax) + ']' +
      (meta.axis ? ' | Axis: ' + escHtml(meta.axis) : '') +
      (meta.slice !== undefined ? ' | Slice: ' + meta.slice : '')
    );
  }

  // ── Results summary ──────────────────────────────────────────────────────

  function _loadResults() {
    if (!_currentJobId) {
      showToast('No active job', 'warning');
      return;
    }
    var url = RUN_ENDPOINT + '/' + encodeURIComponent(_currentJobId) + '/results';
    _setHTML('generic-results', '<span class="text-muted"><span class="spinner-border spinner-border-sm"></span> Loading results…</span>');

    fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error(r.status + ': ' + r.statusText);
        return r.json();
      })
      .then(function (r) { _renderResults(r); })
      .catch(function (e) {
        _setHTML('generic-results', '<span class="text-danger">Error: ' + escHtml(e.message) + '</span>');
      });
  }

  function _renderResults(r) {
    var cdP = r.cd_pressure !== undefined ? r.cd_pressure : r.cd_p;
    var cdF = r.cd_friction !== undefined ? r.cd_friction : r.cd_f;
    var cdTot = r.cd !== undefined ? r.cd : r.cd_total;
    var cl = r.cl !== undefined ? r.cl : r.cl_total;
    var st = r.st !== undefined ? r.st : r.strouhal;

    var html = '<div class="row g-2">' +
      '<div class="col-6 col-md-4"><div class="card stat-card"><div class="stat-num text-primary">' + _fmt(cdTot) + '</div><div class="stat-label">Cd (total)</div></div></div>' +
      '<div class="col-6 col-md-4"><div class="card stat-card"><div class="stat-num text-danger">' + _fmt(cdP) + '</div><div class="stat-label">Cd (pressure)</div></div></div>' +
      '<div class="col-6 col-md-4"><div class="card stat-card"><div class="stat-num text-success">' + _fmt(cdF) + '</div><div class="stat-label">Cd (friction)</div></div></div>' +
      '<div class="col-6 col-md-4"><div class="card stat-card"><div class="stat-num" style="color:#7c3aed">' + _fmt(cl) + '</div><div class="stat-label">Cl</div></div></div>' +
      '<div class="col-6 col-md-4"><div class="card stat-card"><div class="stat-num" style="color:#d97706">' + _fmt(st) + '</div><div class="stat-label">St</div></div></div>' +
      '<div class="col-6 col-md-4"><div class="card stat-card"><div class="stat-num text-muted">' + (r.steps || r.n_steps || '—') + '</div><div class="stat-label">Steps</div></div></div>' +
      '</div>';

    // Error vs reference (if known)
    if (r.reference || r.error) {
      var ref = r.reference || {};
      var err = r.error || {};
      html += '<div class="card mt-3"><div class="card-header"><i class="bi bi-bullseye"></i> Error vs Reference</div>' +
        '<div class="card-body"><table class="table table-sm table-bordered mb-0 small">' +
        '<thead><tr><th>Metric</th><th>Computed</th><th>Reference</th><th>Error %</th></tr></thead><tbody>';
      var rows = [
        ['Cd', cdTot, ref.cd, err.cd],
        ['Cl', cl, ref.cl, err.cl],
        ['St', st, ref.st, err.st],
      ];
      rows.forEach(function (row) {
        var computed = row[1], reference = row[2], errorPct = row[3];
        var errDisplay = errorPct !== undefined ? _fmt(errorPct, 2) + '%' :
          (reference !== undefined && computed !== undefined && reference !== 0 ?
            _fmt(Math.abs((computed - reference) / reference * 100), 2) + '%' : '—');
        html += '<tr><td>' + row[0] + '</td><td>' + _fmt(computed) + '</td><td>' + _fmt(reference) + '</td><td>' + errDisplay + '</td></tr>';
      });
      html += '</tbody></table>';
      if (ref.source || ref.name) {
        html += '<div class="text-muted small mt-1">Reference: ' + escHtml(ref.source || ref.name) + '</div>';
      }
      html += '</div></div>';
    }

    // Raw metadata
    if (r.metadata || r.raw) {
      html += '<details class="mt-2"><summary class="small text-muted">Raw metadata</summary>' +
        '<pre class="small mt-1" style="white-space:pre-wrap;max-height:200px;overflow:auto">' +
        escHtml(JSON.stringify(r.metadata || r.raw, null, 2)) + '</pre></details>';
    }

    _setHTML('generic-results', html);
  }

  // ── Drag-and-drop STL ────────────────────────────────────────────────────

  function _bindDragDrop() {
    var dropZone = _el('generic-drop-zone');
    var fileInput = _el('generic-file-input');
    if (!dropZone || !fileInput) return;

    dropZone.addEventListener('click', function () { fileInput.click(); });
    dropZone.addEventListener('dragover', function (e) {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', function () {
      dropZone.classList.remove('drag-over');
    });
    dropZone.addEventListener('drop', function (e) {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      var file = e.dataTransfer.files[0];
      if (file) _loadSTLFile(file);
    });
    fileInput.addEventListener('change', function () {
      if (fileInput.files.length) _loadSTLFile(fileInput.files[0]);
    });
  }

  // ── Reset / clear ─────────────────────────────────────────────────────────

  function _resetAll() {
    _stlFile = null;
    _stlGeo = null;
    _currentJobId = null;
    _closeJobWS();
    _stopPolling();
    if (_stlMesh && _scene) { _scene.remove(_stlMesh); _stlMesh.geometry.dispose(); _stlMesh = null; }
    if (_bboxHelper && _scene) { _scene.remove(_bboxHelper); _bboxHelper = null; }
    var ph = _el('generic-placeholder');
    if (ph) ph.style.display = '';
    _setText('generic-stl-name', '—');
    _setText('generic-stl-size', '—');
    _setHTML('generic-mesh-info', '<span class="text-muted small">No STL loaded</span>');
    _setHTML('generic-domain-suggest', '');
    _setHTML('generic-run-status', '<span class="text-muted">Idle</span>');
    _setHTML('generic-ws-status', '<span class="dot dot-queued"></span> Not connected');
    _setHTML('generic-force-decomp', '<span class="text-muted">—</span>');
    _setHTML('generic-results', '<span class="text-muted">Run a simulation to see results.</span>');
    _setHTML('generic-field-info', '');
    _setHTML('generic-field-status', '');
    _updateProgress(0, 0);
    // Reset charts
    Object.keys(_charts).forEach(function (k) {
      if (_charts[k]) {
        _charts[k].data.labels = [];
        _charts[k].data.datasets.forEach(function (ds) { ds.data = []; });
        _charts[k].update('none');
      }
    });
    // Clear field canvas
    var canvas = _el('generic-field-canvas');
    if (canvas) { var ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, canvas.width, canvas.height); }
    showToast('Cleared', 'info');
  }

  // ── Tab enter handler ─────────────────────────────────────────────────────

  function _onTabEnter() {
    _initThree();
    _initCharts();
    _bindDragDrop();
  }

  // ── Public API (window exports) ──────────────────────────────────────────

  window.genericLoadField = _loadField;
  window.genericLoadResults = _loadResults;
  window.genericReset = _resetAll;

  // Expose init for TAB_ENTER_HANDLERS
  window.genericInit = _onTabEnter;

  // Bind run/cancel buttons after DOM ready
  document.addEventListener('DOMContentLoaded', function () {
    var runBtn = _el('generic-run-btn');
    if (runBtn) runBtn.addEventListener('click', _submitRun);
    var cancelBtn = _el('generic-cancel-btn');
    if (cancelBtn) cancelBtn.addEventListener('click', _cancelJob);
    var resetBtn = _el('generic-reset-btn');
    if (resetBtn) resetBtn.addEventListener('click', _resetAll);
    var loadFieldBtn = _el('generic-load-field-btn');
    if (loadFieldBtn) loadFieldBtn.addEventListener('click', function () {
      var ft = _el('generic-field-type') ? _el('generic-field-type').value : 'velocity';
      _loadField(ft);
    });
    var loadResultsBtn = _el('generic-load-results-btn');
    if (loadResultsBtn) loadResultsBtn.addEventListener('click', _loadResults);

    // Auto-domain toggle
    var autoDomainCb = _el('generic-auto-domain');
    if (autoDomainCb) {
      autoDomainCb.addEventListener('change', function () {
        var manual = _el('generic-domain-manual');
        if (manual) manual.style.display = autoDomainCb.checked ? 'none' : '';
      });
    }

    // Wrap showTab to init on tab enter
    var origShowTab = window.showTab;
    if (typeof origShowTab === 'function') {
      window.showTab = function (name, el) {
        origShowTab(name, el);
        if (name === 'generic') _onTabEnter();
      };
    }
  });

})();
