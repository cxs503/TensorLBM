"""Interactive 2D cylinder flow WebSocket endpoint.

Provides a real-time LBM simulation where the user can drag the cylinder
and see velocity, pressure, and vorticity fields update live.

WebSocket protocol:
  Client → Server messages (JSON):
    {"action": "init", "nx": 200, "ny": 80, "u_in": 0.08, "re": 100, "radius": 10}
    {"action": "step", "n": 10}
    {"action": "move", "cx": 50.0, "cy": 40.0}
    {"action": "set_params", "u_in": 0.1, "re": 200}
    {"action": "reset"}
    {"action": "get_fields"}

  Server → Client messages (JSON):
    {"type": "fields", "rho": [...], "ux": [...], "uy": [...], "speed": [...],
     "vorticity": [...], "obstacle": [...], "shape": [ny, nx], "step": N, ...}
    {"type": "params", "u_in": ..., "re": ..., "tau": ..., "nu": ..., "radius": ...}
    {"type": "moved", "cx": ..., "cy": ..., "radius": ...}
    {"type": "error", "message": "..."}
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Ensure tensorlbm is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tensorlbm.interactive_cylinder import InteractiveCylinderSim

router = APIRouter()

# Global simulation instance (single-user for now)
_sim: InteractiveCylinderSim | None = None


def _serialize_fields(fields: dict) -> str:
    """Serialize fields to compact JSON: arrays as base64 flat buffers."""
    import numpy as np
    import base64

    result = {"type": "fields"}
    for key, val in fields.items():
        if key == "shape":
            result[key] = val
        elif key in ("step", "cx", "cy", "radius", "u_in", "tau", "re"):
            result[key] = val
        elif isinstance(val, np.ndarray):
            if key == "obstacle":
                # Pack bool array as flat uint8 base64
                flat = val.astype(np.uint8).flatten()
                result[key] = base64.b64encode(flat.tobytes()).decode('ascii')
                result[key + "_shape"] = list(val.shape)
            else:
                # Pack float array as float32 base64
                flat = val.astype(np.float32).flatten()
                result[key] = base64.b64encode(flat.tobytes()).decode('ascii')
                result[key + "_shape"] = list(val.shape)
        elif isinstance(val, (list, int, float, bool)):
            result[key] = val
        else:
            result[key] = val
    return json.dumps(result, separators=(',', ':'))


@router.websocket("/ws")
async def cylinder_interactive_ws(ws: WebSocket) -> None:
    """WebSocket endpoint for interactive cylinder flow simulation."""
    await ws.accept()

    global _sim

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            action = msg.get("action", "")

            if action == "init":
                nx = msg.get("nx", 200)
                ny = msg.get("ny", 80)
                u_in = msg.get("u_in", 0.08)
                re = msg.get("re", 100.0)
                radius = msg.get("radius", 10.0)
                cx = msg.get("cx", nx * 0.25)
                cy = msg.get("cy", ny * 0.5)
                loop = asyncio.get_running_loop()
                _sim = await loop.run_in_executor(None, InteractiveCylinderSim,
                    nx, ny, u_in, re, radius, cx, cy, "cpu")
                await loop.run_in_executor(None, _sim.step, 300)
                fields = await loop.run_in_executor(None, _sim.get_fields, 2)
                payload = _serialize_fields(fields)
                await ws.send_text(payload)

            elif action == "step":
                if _sim is None:
                    await ws.send_json({"type": "error", "message": "No simulation initialised"})
                    continue
                n = msg.get("n", 10)
                # Run in a thread to not block the event loop
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, _sim.step, n)
                fields = await loop.run_in_executor(None, _sim.get_fields, 2)
                payload = _serialize_fields(fields)
                await ws.send_text(payload)

            elif action == "move":
                if _sim is None:
                    await ws.send_json({"type": "error", "message": "No simulation initialised"})
                    continue
                cx = msg.get("cx", _sim.cx)
                cy = msg.get("cy", _sim.cy)
                result = _sim.move_cylinder(cx, cy)
                # Step a few times after move to let flow adjust
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, _sim.step, 5)
                fields = await loop.run_in_executor(None, _sim.get_fields, 2)
                payload = _serialize_fields(fields)
                await ws.send_text(payload)

            elif action == "set_params":
                if _sim is None:
                    await ws.send_json({"type": "error", "message": "No simulation initialised"})
                    continue
                result = _sim.set_params(
                    u_in=msg.get("u_in"),
                    re=msg.get("re"),
                    radius=msg.get("radius"),
                )
                await ws.send_json({"type": "params", **result})

            elif action == "reset":
                if _sim is None:
                    await ws.send_json({"type": "error", "message": "No simulation initialised"})
                    continue
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, _sim.reset)
                await loop.run_in_executor(None, _sim.step, 300)
                fields = await loop.run_in_executor(None, _sim.get_fields, 2)
                payload = _serialize_fields(fields)
                await ws.send_text(payload)

            elif action == "get_fields":
                if _sim is None:
                    await ws.send_json({"type": "error", "message": "No simulation initialised"})
                    continue
                loop = asyncio.get_running_loop()
                fields = await loop.run_in_executor(None, _sim.get_fields, 2)
                payload = _serialize_fields(fields)
                await ws.send_text(payload)

            else:
                await ws.send_json({"type": "error", "message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@router.get("/status")
def cylinder_status():
    """Return current simulation status."""
    if _sim is None:
        return {"status": "not initialised"}
    return {
        "status": "running",
        "nx": _sim.nx,
        "ny": _sim.ny,
        "u_in": _sim.u_in,
        "re": _sim.re,
        "radius": _sim.radius,
        "cx": _sim.cx,
        "cy": _sim.cy,
        "tau": _sim.tau,
        "nu": _sim.nu,
        "step": _sim.step_count,
    }
