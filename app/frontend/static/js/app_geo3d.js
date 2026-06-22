/**
 * app_geo3d.js  – P3.1 Three.js 3D Geometry Preview
 *
 * Features
 * --------
 * - Load binary or ASCII STL from user file
 * - Orbit / pan / zoom via OrbitControls
 * - Translucent refinement-zone box overlay
 * - Shift+click probe placement with raycasting
 * - Mesh stats in status bar
 */

/* global THREE */

(function () {
  'use strict';

  // ── State ────────────────────────────────────────────────────────────────
  let _renderer = null;
  let _scene    = null;
  let _camera   = null;
  let _controls = null;
  let _stlMesh  = null;
  let _refBox   = null;
  let _probes   = [];
  let _raycaster = null;
  let _mouse     = null;
  let _probeMode = false; // true while Shift is held

  // ── Init ─────────────────────────────────────────────────────────────────
  function _initThree() {
    const wrap = document.getElementById('geo3d-canvas-wrap');
    const canvas = document.getElementById('geo3d-canvas');
    if (!wrap || !canvas) return;
    if (_renderer) return; // already initialised

    const W = wrap.clientWidth  || 800;
    const H = wrap.clientHeight || 580;

    _renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    _renderer.setPixelRatio(window.devicePixelRatio);
    _renderer.setSize(W, H, false);
    _renderer.shadowMap.enabled = true;

    _scene = new THREE.Scene();
    _scene.background = new THREE.Color(0x1a1a2e);

    _camera = new THREE.PerspectiveCamera(45, W / H, 0.001, 10000);
    _camera.position.set(0, 0, 5);

    // Lights
    _scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(5, 10, 7);
    _scene.add(dir);
    const dir2 = new THREE.DirectionalLight(0x8888ff, 0.3);
    dir2.position.set(-5, -5, -7);
    _scene.add(dir2);

    // Grid helper (subtle)
    const grid = new THREE.GridHelper(20, 20, 0x333355, 0x333355);
    _scene.add(grid);

    // Orbit controls
    if (typeof THREE.OrbitControls !== 'undefined') {
      _controls = new THREE.OrbitControls(_camera, _renderer.domElement);
      _controls.enableDamping = true;
      _controls.dampingFactor = 0.08;
    }

    // Raycaster for probes
    _raycaster = new THREE.Raycaster();
    _mouse = new THREE.Vector2();

    // Events
    canvas.addEventListener('click', _onCanvasClick);
    canvas.addEventListener('mousemove', _onMouseMove);
    window.addEventListener('keydown', (e) => { if (e.shiftKey) _probeMode = true; });
    window.addEventListener('keyup',   (e) => { if (!e.shiftKey) _probeMode = false; });

    // Resize observer
    new ResizeObserver(() => {
      const nW = wrap.clientWidth;
      const nH = wrap.clientHeight;
      _camera.aspect = nW / nH;
      _camera.updateProjectionMatrix();
      _renderer.setSize(nW, nH, false);
    }).observe(wrap);

    _animate();
  }

  function _animate() {
    requestAnimationFrame(_animate);
    if (_controls) _controls.update();
    _renderer.render(_scene, _camera);
  }

  // ── STL Loader ───────────────────────────────────────────────────────────

  /**
   * Parse an STL file (binary or ASCII) and return a THREE.BufferGeometry.
   * Handles both Uint8Array (binary) and string (ASCII).
   */
  function _parseSTL(buffer) {
    const text = new TextDecoder().decode(new Uint8Array(buffer, 0, Math.min(256, buffer.byteLength)));
    const isBinary = !/solid/.test(text.substring(0, 5)) || _isBinarySTL(buffer);
    return isBinary ? _parseBinarySTL(buffer) : _parseAsciiSTL(buffer);
  }

  function _isBinarySTL(buffer) {
    // Binary STL: header (80 bytes) + uint32 triangle count + 50*N bytes
    if (buffer.byteLength < 84) return false;
    const view = new DataView(buffer);
    const n = view.getUint32(80, true);
    return buffer.byteLength === 84 + n * 50;
  }

  function _parseBinarySTL(buffer) {
    const view = new DataView(buffer);
    const n = view.getUint32(80, true);
    const positions = new Float32Array(n * 9);
    const normals   = new Float32Array(n * 9);
    let offset = 84;
    for (let i = 0; i < n; i++) {
      const nx = view.getFloat32(offset,     true);
      const ny = view.getFloat32(offset + 4, true);
      const nz = view.getFloat32(offset + 8, true);
      offset += 12;
      for (let v = 0; v < 3; v++) {
        const b = i * 9 + v * 3;
        positions[b]     = view.getFloat32(offset,     true);
        positions[b + 1] = view.getFloat32(offset + 4, true);
        positions[b + 2] = view.getFloat32(offset + 8, true);
        normals[b] = nx; normals[b + 1] = ny; normals[b + 2] = nz;
        offset += 12;
      }
      offset += 2; // attribute byte count
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal',   new THREE.BufferAttribute(normals, 3));
    return geo;
  }

  function _parseAsciiSTL(buffer) {
    const text = new TextDecoder().decode(buffer);
    const lines = text.split('\n');
    const vertices = [];
    const normals  = [];
    let cn = [0, 0, 1];
    for (const raw of lines) {
      const line = raw.trim();
      if (line.startsWith('facet normal')) {
        const p = line.split(/\s+/);
        cn = [parseFloat(p[2]), parseFloat(p[3]), parseFloat(p[4])];
      } else if (line.startsWith('vertex')) {
        const p = line.split(/\s+/);
        vertices.push(parseFloat(p[1]), parseFloat(p[2]), parseFloat(p[3]));
        normals.push(...cn);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices), 3));
    geo.setAttribute('normal',   new THREE.BufferAttribute(new Float32Array(normals), 3));
    return geo;
  }

  // ── Public API ───────────────────────────────────────────────────────────

  window.geo3dLoadFile = function () {
    const input = document.getElementById('geo3d-file-input');
    if (!input || !input.files.length) {
      alert('Please select an STL file first.');
      return;
    }
    const file = input.files[0];
    const reader = new FileReader();
    reader.onload = (evt) => {
      _initThree();

      // Remove existing model
      if (_stlMesh) {
        _scene.remove(_stlMesh);
        _stlMesh.geometry.dispose();
        _stlMesh = null;
      }

      try {
        const geo = _parseSTL(evt.target.result);
        geo.computeBoundingBox();
        geo.computeVertexNormals();

        // Centre and normalise
        const box    = geo.boundingBox;
        const centre = new THREE.Vector3();
        box.getCenter(centre);
        const size   = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        geo.translate(-centre.x, -centre.y, -centre.z);
        const scale = 2 / maxDim;

        const mat = new THREE.MeshPhongMaterial({
          color: 0x4488cc,
          specular: 0x224466,
          shininess: 60,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.92,
        });
        _stlMesh = new THREE.Mesh(geo, mat);
        _stlMesh.scale.setScalar(scale);
        _scene.add(_stlMesh);

        // Frame camera
        _camera.position.set(0, 0, 4);
        if (_controls) _controls.reset();

        // Hide placeholder
        const ph = document.getElementById('geo3d-placeholder');
        if (ph) ph.style.display = 'none';

        // Stats
        const nTri = geo.attributes.position.count / 3;
        const info = document.getElementById('geo3d-mesh-info');
        if (info) info.textContent = `Triangles: ${Math.round(nTri).toLocaleString()}  |  Extent: ${size.x.toFixed(3)} × ${size.y.toFixed(3)} × ${size.z.toFixed(3)}`;

      } catch (e) {
        console.error('STL parse error:', e);
        alert('Failed to parse STL: ' + e.message);
      }
    };
    reader.readAsArrayBuffer(file);
  };

  window.geo3dUpdateRef = function () {
    _initThree();
    const cx = parseFloat(document.getElementById('geo3d-rx').value) || 0;
    const cy = parseFloat(document.getElementById('geo3d-ry').value) || 0;
    const cz = parseFloat(document.getElementById('geo3d-rz').value) || 0;
    const rw = parseFloat(document.getElementById('geo3d-rw').value) || 1;
    const rh = parseFloat(document.getElementById('geo3d-rh').value) || 1;
    const rd = parseFloat(document.getElementById('geo3d-rd').value) || 1;

    if (_refBox) { _scene.remove(_refBox); _refBox = null; }

    const geo = new THREE.BoxGeometry(rw, rh, rd);
    const mat = new THREE.MeshBasicMaterial({
      color: 0x00ff88,
      wireframe: false,
      transparent: true,
      opacity: 0.12,
      depthWrite: false,
    });
    const wire = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0x00ff88, linewidth: 1 })
    );
    _refBox = new THREE.Group();
    _refBox.add(new THREE.Mesh(geo, mat));
    _refBox.add(wire);
    _refBox.position.set(cx, cy, cz);
    _scene.add(_refBox);
  };

  window.geo3dToggleRef = function () {
    if (_refBox) _refBox.visible = document.getElementById('geo3d-show-ref').checked;
  };

  window.geo3dClearProbes = function () {
    for (const m of _probes) _scene.remove(m);
    _probes = [];
    const list = document.getElementById('geo3d-probe-list');
    if (list) list.innerHTML = '';
  };

  // ── Events ───────────────────────────────────────────────────────────────

  function _onMouseMove(evt) {
    const rect = evt.target.getBoundingClientRect();
    const coord = document.getElementById('geo3d-coord-display');
    if (coord) coord.textContent = `Mouse: (${(evt.clientX - rect.left).toFixed(0)}, ${(evt.clientY - rect.top).toFixed(0)})`;
  }

  function _onCanvasClick(evt) {
    if (!_probeMode) return;
    if (!_stlMesh || !_raycaster) return;

    const canvas = document.getElementById('geo3d-canvas');
    const rect   = canvas.getBoundingClientRect();
    _mouse.x = ((evt.clientX - rect.left) / rect.width)  * 2 - 1;
    _mouse.y = -((evt.clientY - rect.top)  / rect.height) * 2 + 1;

    _raycaster.setFromCamera(_mouse, _camera);
    const hits = _raycaster.intersectObject(_stlMesh);
    if (!hits.length) return;

    const pt = hits[0].point;

    // Sphere marker
    const geo = new THREE.SphereGeometry(0.04, 8, 8);
    const mat = new THREE.MeshBasicMaterial({ color: 0xff4444 });
    const sphere = new THREE.Mesh(geo, mat);
    sphere.position.copy(pt);
    _scene.add(sphere);
    _probes.push(sphere);

    // List entry
    const list = document.getElementById('geo3d-probe-list');
    if (list) {
      const li = document.createElement('div');
      li.textContent = `P${_probes.length}: (${pt.x.toFixed(4)}, ${pt.y.toFixed(4)}, ${pt.z.toFixed(4)})`;
      list.appendChild(li);
    }
  }

})();
