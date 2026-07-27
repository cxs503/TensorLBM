"""Side-by-side cylinder flow comparison: CPU f32 vs SDAA f32.

Both lanes run on backend via TensorLBM (PyTorch).
CPU f32: device='cpu', dtype=float32 — pure CPU baseline.
SDAA f32: device='sdaa:0', dtype=float32 — SDAA accelerated.

Each sim runs in its own daemon thread, stepping independently.
SDAA is faster per-step, so it naturally accumulates more steps
in the same wall-clock time — the frontend sees different step
counts, demonstrating the real speed advantage.

/results reads cached encoded fields under a brief lock — never
blocks on stepping.
"""

import base64
import time
import threading

import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream
from tensorlbm.boundaries import (
    apply_simple_channel_boundaries,
    cylinder_mask,
    make_channel_wall_mask,
)

from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter()

# ── Global simulation state ────────────────────────────────────────────────
_sims: dict[str, dict] = {}  # keys: "cpu_f32", "sdaa_f32"
_lock = threading.Lock()
_worker_threads: dict[str, threading.Thread] = {}


def _compute_vorticity(ux, uy):
    dux_dy = torch.zeros_like(ux)
    duy_dx = torch.zeros_like(uy)
    dux_dy[1:-1, :] = 0.5 * (ux[2:, :] - ux[:-2, :])
    duy_dx[:, 1:-1] = 0.5 * (uy[:, 2:] - uy[:, :-2])
    return duy_dx - dux_dy


def _encode(arr, downsample=2):
    """Encode tensor as base64 float16 + shape."""
    if downsample > 1:
        arr = arr[::downsample, ::downsample]
    np_arr = arr.detach().cpu().to(torch.float16).numpy()
    shape = list(np_arr.shape)
    b64 = base64.b64encode(np_arr.tobytes()).decode("ascii")
    return {"b64": b64, "shape": shape, "dtype": "f16"}

def _encode_u8(arr, downsample=2):
    if downsample > 1:
        arr = arr[::downsample, ::downsample]
    np_arr = arr.detach().cpu().to(torch.uint8).numpy()
    shape = list(np_arr.shape)
    b64 = base64.b64encode(np_arr.tobytes()).decode("ascii")
    return {"b64": b64, "shape": shape, "dtype": "u8"}


def _create_sim(nx, ny, u_in, re, radius, device_str, dtype_str):
    """Create simulation tensors (no warmup yet)."""
    device = torch.device(device_str)
    dtype = torch.float32 if dtype_str == "float32" else torch.bfloat16
    tau = 3.0 * u_in * 2.0 * radius / re + 0.5
    cx_obs = nx * 0.25
    cy_obs = ny * 0.5

    obstacle = cylinder_mask(nx, ny, cx_obs, cy_obs, radius, device=device)
    wall_mask = make_channel_wall_mask(ny, nx, obstacle, device=device)

    rho0 = torch.ones((ny, nx), device=device, dtype=dtype)
    ux0 = torch.full((ny, nx), u_in, device=device, dtype=dtype)
    uy0 = torch.zeros((ny, nx), device=device, dtype=dtype)
    ux0[obstacle] = 0.0
    f = equilibrium(rho0, ux0, uy0, device=device)

    initial_mass = float(f.sum().item())

    return {
        "f": f, "obstacle": obstacle, "wall_mask": wall_mask,
        "tau": tau, "u_in": u_in, "step": 0,
        "initial_mass": initial_mass, "device": device_str, "dtype": dtype_str,
        "nx": nx, "ny": ny, "radius": radius,
        "ms_per_step": 0.0, "steps_per_sec": 0.0,
        "running": True, "warming_up": True,
        "cached_fields": None,
    }


def _step_and_encode(sim, n=10):
    """Run n steps, measure timing, and encode fields."""
    f = sim["f"]
    obstacle = sim["obstacle"]
    wall_mask = sim["wall_mask"]
    tau = sim["tau"]
    u_in = sim["u_in"]
    device_str = sim["device"]

    t0 = time.perf_counter()
    for _ in range(n):
        f = collide_bgk(f, tau)
        f = stream(f)
        f = apply_simple_channel_boundaries(f, u_in=u_in, wall_mask=wall_mask, obstacle_mask=obstacle)
        sim["step"] += 1

    if device_str.startswith("sdaa"):
        torch.sdaa.synchronize()

    t1 = time.perf_counter()
    sim["ms_per_step"] = (t1 - t0) / n * 1000
    sim["steps_per_sec"] = n / (t1 - t0)

    # Mass correction every 50 steps
    if sim["step"] % 50 < n:
        current = float(f.sum().item())
        if current > 1e-30:
            f = f * (sim["initial_mass"] / current)

    sim["f"] = f

    # Encode fields
    rho, ux, uy = macroscopic(f)
    ux = ux.masked_fill(obstacle, 0.0)
    uy = uy.masked_fill(obstacle, 0.0)
    speed = torch.sqrt(ux * ux + uy * uy)
    vorticity = _compute_vorticity(ux, uy)

    fields = {
        "step": sim["step"],
        "ms_per_step": round(sim["ms_per_step"], 2),
        "steps_per_sec": round(sim["steps_per_sec"], 1),
        "speed": _encode(speed, 2),
        "vorticity": _encode(vorticity, 2),
        "rho": _encode(rho, 2),
        "obstacle": _encode_u8(obstacle, 2),
        "nx": sim["nx"], "ny": sim["ny"],
        "device": sim["device"], "dtype": sim["dtype"],
        "radius": sim["radius"],
    }
    return fields


def _sim_worker(sim_key):
    """Per-sim daemon thread: warmup then continuously step and cache fields.

    Each sim runs independently at its own pace. SDAA steps faster
    per-step, so it naturally accumulates more steps than CPU in the
    same wall-clock time — the speed difference is visible in the
    frontend step counter.
    """
    with _lock:
        sim = _sims.get(sim_key)
    if not sim or not sim.get("running"):
        return

    # Phase 1: Warmup (200 steps)
    f = sim["f"]
    obstacle = sim["obstacle"]
    wall_mask = sim["wall_mask"]
    tau = sim["tau"]
    u_in = sim["u_in"]
    device_str = sim["device"]

    for _ in range(200):
        f = collide_bgk(f, tau)
        f = stream(f)
        f = apply_simple_channel_boundaries(f, u_in=u_in, wall_mask=wall_mask, obstacle_mask=obstacle)
        sim["step"] += 1

    if device_str.startswith("sdaa"):
        torch.sdaa.synchronize()

    sim["f"] = f
    sim["warming_up"] = False

    # Encode initial fields after warmup
    try:
        rho, ux, uy = macroscopic(f)
        ux = ux.masked_fill(obstacle, 0.0)
        uy = uy.masked_fill(obstacle, 0.0)
        speed = torch.sqrt(ux * ux + uy * uy)
        vorticity = _compute_vorticity(ux, uy)
        fields = {
            "step": sim["step"],
            "ms_per_step": 0.0,
            "steps_per_sec": 0.0,
            "speed": _encode(speed, 2),
            "vorticity": _encode(vorticity, 2),
            "rho": _encode(rho, 2),
            "obstacle": _encode_u8(obstacle, 2),
            "nx": sim["nx"], "ny": sim["ny"],
            "device": sim["device"], "dtype": sim["dtype"],
            "radius": sim["radius"],
        }
        with _lock:
            _sims[sim_key]["cached_fields"] = fields
    except Exception as e:
        sim["running"] = False
        sim["error"] = str(e)
        return

    # Phase 2: Continuous stepping — each sim runs at its own speed
    n_steps = 10
    while sim.get("running"):
        try:
            fields = _step_and_encode(sim, n_steps)
            with _lock:
                _sims[sim_key]["cached_fields"] = fields
        except Exception as e:
            sim["running"] = False
            sim["error"] = str(e)
            break

        # Small sleep to avoid hogging CPU, but SDAA will still
        # naturally step faster because each step takes less time
        time.sleep(0.02)


@router.post("/start")
def start_sims(body: dict = None):
    """Start CPU f32 and SDAA f32 simulations. Each runs in its own thread."""
    global _worker_threads

    nx = (body or {}).get("nx", 400)
    ny = (body or {}).get("ny", 160)
    u_in = (body or {}).get("u_in", 0.08)
    re = (body or {}).get("re", 100.0)
    radius = (body or {}).get("radius", 13.0)

    # Stop previous workers if running
    for key, t in _worker_threads.items():
        with _lock:
            if key in _sims:
                _sims[key]["running"] = False
        if t.is_alive():
            t.join(timeout=5)
    _worker_threads.clear()

    with _lock:
        _sims.clear()
        configs = [
            ("cpu", "float32", "cpu_f32"),
            ("sdaa:0", "float32", "sdaa_f32"),
        ]
        for dev, dt, key in configs:
            try:
                _sims[key] = _create_sim(nx, ny, u_in, re, radius, dev, dt)
            except Exception as e:
                _sims[key] = {"running": False, "error": str(e), "device": dev, "dtype": dt,
                              "step": 0, "ms_per_step": 0, "steps_per_sec": 0, "nx": nx, "ny": ny,
                              "cached_fields": None, "warming_up": False}

    # Spawn per-sim worker threads — each runs independently
    for key, sim in _sims.items():
        if sim.get("running"):
            t = threading.Thread(target=_sim_worker, args=(key,), daemon=True)
            t.start()
            _worker_threads[key] = t

    return {"status": "started", "keys": list(_sims.keys()), "nx": nx, "ny": ny}


@router.get("/results")
def get_results():
    """Return latest cached fields + timing (non-blocking, lock held briefly)."""
    import json as _json

    results = {}
    with _lock:
        for key, sim in _sims.items():
            if sim.get("cached_fields"):
                results[key] = sim["cached_fields"]
            elif sim.get("running") and sim.get("warming_up"):
                results[key] = {"step": sim.get("step", 0), "ms_per_step": 0,
                                "warming_up": True,
                                "device": sim.get("device"), "dtype": sim.get("dtype"),
                                "nx": sim.get("nx"), "ny": sim.get("ny")}
            elif sim.get("running"):
                results[key] = {"step": sim.get("step", 0), "ms_per_step": 0,
                                "device": sim.get("device"), "dtype": sim.get("dtype"),
                                "nx": sim.get("nx"), "ny": sim.get("ny")}
            else:
                results[key] = {"error": sim.get("error", "not running"), "step": sim.get("step", 0),
                                "device": sim.get("device"), "dtype": sim.get("dtype")}

    return Response(content=_json.dumps(results, ensure_ascii=True, separators=(",", ":")),
                    media_type="application/json")


@router.post("/stop")
def stop_sims():
    """Stop all simulations and worker threads."""
    global _worker_threads

    with _lock:
        for sim in _sims.values():
            sim["running"] = False

    for key, t in _worker_threads.items():
        if t.is_alive():
            t.join(timeout=5)
    _worker_threads.clear()

    return {"status": "stopped"}
