"""WebSocket endpoint for device-specific cylinder flow simulation.

Runs LBM on specified device/dtype, auto-pushes fields every ~100ms.
Used for side-by-side speed comparison: JS-CPU vs SDAA-f32 vs SDAA-bf16.
"""

import asyncio
import base64
import json
import struct
import time

import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream
from tensorlbm.boundaries import (
    apply_simple_channel_boundaries,
    cylinder_mask,
    make_channel_wall_mask,
)

router = APIRouter()


def _encode_field(arr: torch.Tensor, downsample: int = 2) -> dict:
    """Encode a 2D tensor as base64 float16 + shape metadata."""
    if downsample > 1:
        arr = arr[::downsample, ::downsample]
    np_arr = arr.detach().cpu().to(torch.float16).numpy()
    shape = list(np_arr.shape)
    raw = np_arr.tobytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return {"b64": b64, "shape": shape, "dtype": "f16"}


def _encode_obstacle(arr: torch.Tensor, downsample: int = 2) -> dict:
    """Encode obstacle mask as base64 uint8."""
    if downsample > 1:
        arr = arr[::downsample, ::downsample]
    np_arr = arr.detach().cpu().to(torch.uint8).numpy()
    shape = list(np_arr.shape)
    raw = np_arr.tobytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return {"b64": b64, "shape": shape, "dtype": "u8"}


def _compute_vorticity(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    dux_dy = torch.zeros_like(ux)
    duy_dx = torch.zeros_like(uy)
    dux_dy[1:-1, :] = 0.5 * (ux[2:, :] - ux[:-2, :])
    duy_dx[:, 1:-1] = 0.5 * (uy[:, 2:] - uy[:, :-2])
    return duy_dx - dux_dy


@router.websocket("/ws/{device}/{dtype}")
async def ws_cylinder_device(ws: WebSocket, device: str, dtype: str):
    """WebSocket for cylinder flow on specific device/dtype.

    Protocol:
      Client → Server: {"action": "init", "nx":400, "ny":160, "u_in":0.08, "re":100, "radius":13}
      Server → Client: {"type":"fields", "speed":..., "vorticity":..., "rho":..., "obstacle":..., "step":N, "ms_per_step":..., "steps_per_sec":...}
      Client → Server: {"action": "stop"}
    """
    await ws.accept()

    # Parse dtype
    torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
    torch_device = torch.device(device)

    _sim_data = None  # (f, obstacle, wall_mask, tau, u_in, step_count, initial_mass)

    try:
        # Wait for init message
        msg_raw = await ws.receive_text()
        msg = json.loads(msg_raw)

        if msg.get("action") != "init":
            await ws.send_json({"type": "error", "message": "First message must be init"})
            return

        nx = msg.get("nx", 400)
        ny = msg.get("ny", 160)
        u_in = msg.get("u_in", 0.08)
        re = msg.get("re", 100.0)
        radius = msg.get("radius", 13.0)
        cx_obs = nx * 0.25
        cy_obs = ny * 0.5
        tau = 3.0 * u_in * 2.0 * radius / re + 0.5

        obstacle = cylinder_mask(nx, ny, cx_obs, cy_obs, radius, device=torch_device)
        wall_mask = make_channel_wall_mask(ny, nx, obstacle, device=torch_device)

        rho0 = torch.ones((ny, nx), device=torch_device, dtype=torch_dtype)
        ux0 = torch.full((ny, nx), u_in, device=torch_device, dtype=torch_dtype)
        uy0 = torch.zeros((ny, nx), device=torch_device, dtype=torch_dtype)
        ux0[obstacle] = 0.0
        f = equilibrium(rho0, ux0, uy0, device=torch_device)

        initial_mass = float(f.sum().item())
        step_count = 0

        # Warmup 200 steps
        loop = asyncio.get_event_loop()
        for _ in range(200):
            f = await loop.run_in_executor(None, collide_bgk, f, tau)
            f = await loop.run_in_executor(None, stream, f)
            f = await loop.run_in_executor(None, apply_simple_channel_boundaries, f, u_in, wall_mask, obstacle)
            step_count += 1

        _sim_data = (f, obstacle, wall_mask, tau, u_in, step_count, initial_mass, nx, ny)

        # Send initial fields
        await _send_fields(ws, _sim_data, torch_dtype, torch_device, 0)

        # Auto-run loop: push fields every ~100ms
        steps_per_push = 10
        while True:
            # Check for client messages (stop, etc.)
            try:
                client_msg = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                if json.loads(client_msg).get("action") == "stop":
                    break
            except (asyncio.TimeoutError, WebSocketDisconnect):
                pass

            # Run N steps
            f, obstacle, wall_mask, tau, u_in, step_count, initial_mass, nx, ny = _sim_data

            t0 = time.perf_counter()
            for _ in range(steps_per_push):
                f = await loop.run_in_executor(None, collide_bgk, f, tau)
                f = await loop.run_in_executor(None, stream, f)
                f = await loop.run_in_executor(None, apply_simple_channel_boundaries, f, u_in, wall_mask, obstacle)
                step_count += 1
            if device == "sdaa":
                torch.sdaa.synchronize()
            t1 = time.perf_counter()

            ms_per_step = (t1 - t0) / steps_per_push * 1000
            steps_per_sec = steps_per_push / (t1 - t0)

            # Mass correction every 50 steps
            if step_count % 50 < steps_per_push:
                current_mass = float(f.sum().item())
                if current_mass > 1e-30:
                    f = f * (initial_mass / current_mass)

            _sim_data = (f, obstacle, wall_mask, tau, u_in, step_count, initial_mass, nx, ny)

            # Push fields
            await _send_fields(ws, _sim_data, torch_dtype, torch_device, ms_per_step, steps_per_sec)

            # Small delay to not flood the client
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _send_fields(ws, sim_data, torch_dtype, torch_device, ms_per_step=0, steps_per_sec=0):
    f, obstacle, wall_mask, tau, u_in, step_count, initial_mass, nx, ny = sim_data

    rho, ux, uy = await asyncio.get_event_loop().run_in_executor(None, macroscopic, f)
    ux = ux.masked_fill(obstacle, 0.0)
    uy = uy.masked_fill(obstacle, 0.0)
    speed = torch.sqrt(ux * ux + uy * uy)
    vorticity = _compute_vorticity(ux, uy)

    downsample = 2
    payload = {
        "type": "fields",
        "step": step_count,
        "ms_per_step": round(ms_per_step, 2),
        "steps_per_sec": round(steps_per_sec, 1),
        "speed": _encode_field(speed, downsample),
        "vorticity": _encode_field(vorticity, downsample),
        "rho": _encode_field(rho, downsample),
        "obstacle": _encode_obstacle(obstacle, downsample),
        "nx": nx,
        "ny": ny,
    }
    await ws.send_text(json.dumps(payload, separators=(",", ":")))
