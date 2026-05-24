"""LLM-agent core for the TensorLBM platform.

This module implements a conversational *agent* that lets users drive the
whole simulation pipeline – geometry/modelling → solver → post-processing
and analysis – through natural-language chat (Chinese or English).

Design goals
------------
*   **Offline-safe**: the agent works with **zero external dependencies**.
    When no LLM endpoint is configured, an embedded rule-based intent
    parser handles the most common scenarios (cylinder flow, lid-driven
    cavity, dam break, sloshing tank, ship hull, …) and routes the
    request through the platform's existing job manager.
*   **LLM-optional**: if the environment variable
    ``TENSORLBM_LLM_API_KEY`` is set, the agent will use an
    OpenAI-compatible Chat-Completions endpoint
    (``TENSORLBM_LLM_BASE_URL`` / ``TENSORLBM_LLM_MODEL``) for richer,
    free-form replies.  The LLM only generates *natural-language* text –
    every action that touches the platform still goes through the
    deterministic tool layer below.  This avoids prompt-injection
    side-effects.
*   **Stateless**: every chat call accepts the full ``history`` array
    from the client.  No conversation state is kept in the backend.
*   **Safe**: the tool layer accepts *typed* parameters only, has fixed
    safety caps on grid size / step count, and never executes arbitrary
    code that the model produces.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from . import job_manager as _jm_types

logger = logging.getLogger("tensorlbm.agent")

# ---------------------------------------------------------------------------
# Safety caps
# ---------------------------------------------------------------------------

# Hard upper bounds applied by the agent to anything coming in through
# natural-language tools.  The user can still bypass these via the regular
# REST API – the limits are here to protect the platform against an LLM
# that hallucinates a 10000×10000 grid.
MAX_GRID_2D = 1024
MAX_GRID_3D = 256
MAX_STEPS = 200_000


def _clip(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """A single capability the agent can invoke."""

    name: str
    description: str
    parameters: dict[str, Any]          # JSON-schema-ish for docs
    handler: Callable[..., dict]        # actual implementation


_TOOLS: dict[str, Tool] = {}


def tool(name: str, description: str, parameters: dict[str, Any]) -> Callable:
    """Decorator that registers a tool in the agent's tool table."""

    def _decorator(fn: Callable[..., dict]) -> Callable[..., dict]:
        _TOOLS[name] = Tool(name=name, description=description,
                            parameters=parameters, handler=fn)
        return fn

    return _decorator


def list_tools() -> list[dict]:
    """Return a JSON-serialisable description of every registered tool."""
    return [
        {"name": t.name, "description": t.description, "parameters": t.parameters}
        for t in _TOOLS.values()
    ]


# ---------------------------------------------------------------------------
# Tool implementations – these are thin wrappers around the existing job
# manager so the agent reuses the same lifecycle/WebSocket plumbing as the
# UI.
# ---------------------------------------------------------------------------

def _submit(name: str, job_type: str, config: dict, fn: Callable) -> dict:
    from . import job_manager
    job_id = job_manager.submit(name=name, job_type=job_type, config=config, fn=fn)
    return {"job_id": job_id, "name": name, "job_type": job_type, "config": config}


@tool(
    name="submit_cylinder_flow",
    description=(
        "Run a 2D flow-past-a-cylinder LBM simulation. "
        "Use this for vortex-shedding, Strouhal number, drag/lift studies."
    ),
    parameters={
        "nx": "int (40–1024), grid width, default 320",
        "ny": "int (20–1024), grid height, default 100",
        "re": "float, Reynolds number, default 100",
        "u_in": "float, lattice inlet velocity (≤0.15), default 0.08",
        "radius": "float, cylinder radius in cells, default 12",
        "n_steps": "int, total time steps, default 1200",
        "output_interval": "int, output every N steps, default 200",
        "device": "str, torch device, default 'cpu'",
    },
)
def submit_cylinder_flow(
    nx: int = 320,
    ny: int = 100,
    re: float = 100.0,
    u_in: float = 0.08,
    radius: float = 12.0,
    n_steps: int = 1200,
    output_interval: int = 200,
    device: str = "cpu",
) -> dict:
    nx = _clip(nx, 40, MAX_GRID_2D)
    ny = _clip(ny, 20, MAX_GRID_2D)
    n_steps = _clip(n_steps, 1, MAX_STEPS)
    output_interval = _clip(output_interval, 1, n_steps)

    cfg = {
        "nx": nx, "ny": ny, "re": float(re), "u_in": float(u_in),
        "radius": float(radius), "n_steps": n_steps,
        "output_interval": output_interval, "device": device, "seed": 0,
    }

    def _run(job: _jm_types.Job) -> dict:
        from tensorlbm import CylinderFlowConfig, run_cylinder_flow
        c = dict(cfg)
        c["output_root"] = str(job.output_dir)
        c["overwrite"] = True
        run_dir = run_cylinder_flow(CylinderFlowConfig(**c))
        return {"run_dir": str(run_dir)}

    return _submit(f"Cylinder Flow Re={re}", "cylinder_flow", cfg, _run)


@tool(
    name="submit_lid_driven_cavity",
    description="Run a 2D lid-driven cavity benchmark (square domain, top wall slides).",
    parameters={
        "nx": "int, side length, default 128",
        "re": "float, Reynolds number, default 100",
        "u_lid": "float, lid velocity, default 0.1",
        "n_steps": "int, total steps, default 10000",
        "output_interval": "int, default 2000",
        "device": "str, default 'cpu'",
    },
)
def submit_lid_driven_cavity(
    nx: int = 128,
    re: float = 100.0,
    u_lid: float = 0.1,
    n_steps: int = 10000,
    output_interval: int = 2000,
    device: str = "cpu",
) -> dict:
    nx = _clip(nx, 16, MAX_GRID_2D)
    n_steps = _clip(n_steps, 1, MAX_STEPS)
    output_interval = _clip(output_interval, 1, n_steps)

    cfg = {
        "nx": nx, "u_lid": float(u_lid), "re": float(re),
        "n_steps": n_steps, "output_interval": output_interval,
        "device": device, "seed": 0,
    }

    def _run(job: _jm_types.Job) -> dict:
        from tensorlbm import LidDrivenCavityConfig, run_lid_driven_cavity
        c = dict(cfg)
        c["output_root"] = str(job.output_dir)
        c["overwrite"] = True
        run_dir = run_lid_driven_cavity(LidDrivenCavityConfig(**c))
        return {"run_dir": str(run_dir)}

    return _submit(f"Lid-Driven Cavity Re={re}", "lid_driven_cavity", cfg, _run)


@tool(
    name="submit_dam_break",
    description="Run a 2D multiphase dam-break simulation (free-surface).",
    parameters={
        "nx": "int, default 400",
        "ny": "int, default 200",
        "dam_width": "int, default 100",
        "model": "'sc'|'scmp'|'cg'|'fe' multiphase model, default 'cg'",
        "n_steps": "int, default 4000",
        "output_interval": "int, default 400",
        "device": "str, default 'cpu'",
    },
)
def submit_dam_break(
    nx: int = 400,
    ny: int = 200,
    dam_width: int = 100,
    model: str = "cg",
    n_steps: int = 4000,
    output_interval: int = 400,
    device: str = "cpu",
) -> dict:
    nx = _clip(nx, 32, MAX_GRID_2D)
    ny = _clip(ny, 16, MAX_GRID_2D)
    n_steps = _clip(n_steps, 1, MAX_STEPS)
    output_interval = _clip(output_interval, 1, n_steps)
    if model not in ("sc", "scmp", "cg", "fe"):
        model = "cg"

    cfg = {
        "nx": nx, "ny": ny, "dam_width": int(dam_width),
        "model": model, "rho_heavy": 0.8, "rho_light": 0.4,
        "G": 0.9, "tau": 1.0, "g": 5e-5,
        "n_steps": n_steps, "output_interval": output_interval,
        "device": device,
    }

    def _run(job: _jm_types.Job) -> dict:
        from tensorlbm import DamBreakConfig, run_dam_break
        c = dict(cfg)
        c["output_root"] = str(job.output_dir)
        c["overwrite"] = True
        run_dir = run_dam_break(DamBreakConfig(**c))
        return {"run_dir": str(run_dir)}

    return _submit(f"Dam Break [{model.upper()}]", "dam_break", cfg, _run)


@tool(
    name="submit_sloshing_tank",
    description="Run a 2D sloshing-tank multiphase simulation (Faltinsen benchmark).",
    parameters={
        "nx": "int, default 200",
        "ny": "int, default 160",
        "water_level": "int, default 80",
        "n_steps": "int, default 6000",
        "output_interval": "int, default 600",
        "device": "str, default 'cpu'",
    },
)
def submit_sloshing_tank(
    nx: int = 200,
    ny: int = 160,
    water_level: int = 80,
    n_steps: int = 6000,
    output_interval: int = 600,
    device: str = "cpu",
) -> dict:
    nx = _clip(nx, 32, MAX_GRID_2D)
    ny = _clip(ny, 32, MAX_GRID_2D)
    n_steps = _clip(n_steps, 1, MAX_STEPS)
    output_interval = _clip(output_interval, 1, n_steps)

    cfg = {
        "nx": nx, "ny": ny, "water_level": int(water_level),
        "rho_water": 0.8, "rho_air": 0.4, "G": 0.9, "tau": 1.0,
        "g": 2e-5, "forcing_amp": 3e-5, "forcing_omega": 0.0,
        "n_steps": n_steps, "output_interval": output_interval,
        "device": device,
    }

    def _run(job: _jm_types.Job) -> dict:
        from tensorlbm import SloshingTankConfig, run_sloshing_tank
        c = dict(cfg)
        c["output_root"] = str(job.output_dir)
        c["overwrite"] = True
        run_dir = run_sloshing_tank(SloshingTankConfig(**c))
        return {"run_dir": str(run_dir)}

    return _submit("Sloshing Tank", "sloshing_tank", cfg, _run)


@tool(
    name="submit_ship_hull",
    description="Run a 3D Wigley/KCS/Series60-style ship-hull flow simulation.",
    parameters={
        "nx": "int, default 160", "ny": "int, default 60", "nz": "int, default 40",
        "re": "float, default 200",
        "u_in": "float, default 0.05",
        "n_steps": "int, default 2000",
        "output_interval": "int, default 200",
        "device": "str, default 'cpu'",
    },
)
def submit_ship_hull(
    nx: int = 160,
    ny: int = 60,
    nz: int = 40,
    re: float = 200.0,
    u_in: float = 0.05,
    n_steps: int = 2000,
    output_interval: int = 200,
    device: str = "cpu",
) -> dict:
    nx = _clip(nx, 40, MAX_GRID_3D)
    ny = _clip(ny, 16, MAX_GRID_3D)
    nz = _clip(nz, 16, MAX_GRID_3D)
    n_steps = _clip(n_steps, 1, MAX_STEPS)
    output_interval = _clip(output_interval, 1, n_steps)

    cfg = {
        "nx": nx, "ny": ny, "nz": nz, "u_in": float(u_in), "re": float(re),
        "hull_length": 80.0, "hull_beam": 8.0, "hull_draft": 12.0,
        "smagorinsky_cs": 0.1, "wave_amp": 0.0, "wave_period": 200.0,
        "n_steps": n_steps, "output_interval": output_interval,
        "device": device, "seed": 0,
    }

    def _run(job: _jm_types.Job) -> dict:
        from tensorlbm import ShipHullFlowConfig, run_ship_hull_flow
        c = dict(cfg)
        c["output_root"] = str(job.output_dir)
        c["overwrite"] = True
        c.setdefault("wave_k", 0.05)
        c.setdefault("water_depth", 0.0)
        run_dir = run_ship_hull_flow(ShipHullFlowConfig(**c))
        return {"run_dir": str(run_dir)}

    return _submit(f"Ship Hull (Wigley) Re={re}", "ship_hull", cfg, _run)


@tool(
    name="submit_pipeline_flow",
    description="Run a 2D near-bed pipeline (cylinder above wall) flow simulation.",
    parameters={
        "nx": "int, default 400", "ny": "int, default 160",
        "diameter": "float, default 20",
        "gap_ratio": "float (e/D), default 0.5",
        "re": "float, default 200",
        "n_steps": "int, default 30000",
        "output_interval": "int, default 5000",
        "device": "str, default 'cpu'",
    },
)
def submit_pipeline_flow(
    nx: int = 400,
    ny: int = 160,
    diameter: float = 20.0,
    gap_ratio: float = 0.5,
    re: float = 200.0,
    u_in: float = 0.05,
    n_steps: int = 30000,
    output_interval: int = 5000,
    device: str = "cpu",
) -> dict:
    nx = _clip(nx, 40, MAX_GRID_2D)
    ny = _clip(ny, 20, MAX_GRID_2D)
    n_steps = _clip(n_steps, 1, MAX_STEPS)
    output_interval = _clip(output_interval, 1, n_steps)

    cfg = {
        "nx": nx, "ny": ny, "diameter": float(diameter),
        "gap_ratio": float(gap_ratio), "u_in": float(u_in),
        "re": float(re), "n_steps": n_steps,
        "output_interval": output_interval, "device": device, "seed": 0,
    }

    def _run(job: _jm_types.Job) -> dict:
        from tensorlbm import PipelineFlowConfig, run_pipeline_flow
        c = dict(cfg)
        c["output_root"] = str(job.output_dir)
        c["overwrite"] = True
        run_dir = run_pipeline_flow(PipelineFlowConfig(**c))
        return {"run_dir": str(run_dir)}

    return _submit(
        f"Pipeline Flow Re={re} e/D={gap_ratio}",
        "pipeline_flow", cfg, _run,
    )


@tool(
    name="submit_turbulent_channel",
    description="Run a 2D LES turbulent channel flow benchmark.",
    parameters={
        "nx": "int, default 256", "ny": "int, default 64",
        "re_tau": "float friction-Re_tau, default 100",
        "n_steps": "int, default 50000",
        "output_interval": "int, default 5000",
        "device": "str, default 'cpu'",
    },
)
def submit_turbulent_channel(
    nx: int = 256,
    ny: int = 64,
    re_tau: float = 100.0,
    n_steps: int = 50000,
    output_interval: int = 5000,
    device: str = "cpu",
) -> dict:
    nx = _clip(nx, 32, MAX_GRID_2D)
    ny = _clip(ny, 16, MAX_GRID_2D)
    n_steps = _clip(n_steps, 1, MAX_STEPS)
    output_interval = _clip(output_interval, 1, n_steps)

    cfg = {
        "nx": nx, "ny": ny, "re_tau": float(re_tau), "u_tau": 0.005,
        "smagorinsky_cs": 0.1, "n_steps": n_steps,
        "averaging_start": min(n_steps // 2, 20000),
        "output_interval": output_interval, "device": device, "seed": 0,
    }

    def _run(job: _jm_types.Job) -> dict:
        from tensorlbm import TurbulentChannelConfig, run_turbulent_channel
        c = dict(cfg)
        c["output_root"] = str(job.output_dir)
        c["overwrite"] = True
        run_dir = run_turbulent_channel(TurbulentChannelConfig(**c))
        return {"run_dir": str(run_dir)}

    return _submit(
        f"Turbulent Channel Re_tau={re_tau}",
        "turbulent_channel", cfg, _run,
    )


@tool(
    name="get_job_status",
    description="Return the current status, output directory and result of a job.",
    parameters={"job_id": "str, job identifier returned by a submit_* tool"},
)
def get_job_status(job_id: str) -> dict:
    from . import job_manager
    job = job_manager.get_job(job_id)
    if job is None:
        return {"error": f"Job {job_id} not found"}
    return job.to_dict()


@tool(
    name="list_recent_jobs",
    description="List up to N recent jobs known to the platform (newest first).",
    parameters={"limit": "int, default 10"},
)
def list_recent_jobs(limit: int = 10) -> dict:
    from . import job_manager
    limit = _clip(limit, 1, 100)
    jobs = job_manager.list_jobs()[:limit]
    return {"count": len(jobs), "jobs": jobs}


@tool(
    name="analyze_job",
    description=(
        "Produce a human-readable analysis summary of a completed job, "
        "including run metadata, number of snapshots and CSV outputs, and "
        "the solver's result dict."
    ),
    parameters={"job_id": "str, job identifier"},
)
def analyze_job(job_id: str) -> dict:
    from . import job_manager
    job = job_manager.get_job(job_id)
    if job is None:
        return {"error": f"Job {job_id} not found"}
    out_dir = job.output_dir
    meta: dict = {}
    candidates = list(out_dir.rglob("run_metadata.json"))
    if candidates:
        try:
            meta = json.loads(candidates[0].read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
    png_count = len(list(out_dir.rglob("*.png")))
    csv_count = len(list(out_dir.rglob("*.csv")))
    snapshots = sorted(
        str(p.relative_to(out_dir)) for p in out_dir.rglob("snapshot_*.png")
    )
    return {
        "job_id": job_id,
        "name": job.name,
        "job_type": job.job_type,
        "status": job.status.value,
        "png_files": png_count,
        "csv_files": csv_count,
        "snapshots": snapshots[:10],
        "metadata": meta,
        "result": job.result,
    }


@tool(
    name="velocity_profile",
    description=(
        "Extract a 1-D velocity profile (u, v) along a slice from the latest "
        "checkpoint of a completed job."
    ),
    parameters={
        "job_id": "str",
        "direction": "'x' or 'y' slice direction, default 'y'",
        "position": "float in [0,1] – fractional location, default 0.5",
    },
)
def velocity_profile(
    job_id: str,
    direction: str = "y",
    position: float = 0.5,
) -> dict:
    from . import job_manager
    job = job_manager.get_job(job_id)
    if job is None:
        return {"error": f"Job {job_id} not found"}
    if job.status.value != "completed":
        return {"error": f"Job is not completed yet (status: {job.status.value})"}
    try:
        from tensorlbm import load_checkpoint, macroscopic
        ckpts = sorted(job.output_dir.rglob("checkpoint_*.pt"),
                       key=lambda p: p.stem)
        if not ckpts:
            return {"error": "No checkpoint files found in job output"}
        f, step = load_checkpoint(ckpts[-1])
        _rho, ux, uy = macroscopic(f)
        ny, nx = ux.shape
        position = float(max(0.0, min(1.0, position)))
        if direction == "y":
            idx = max(0, min(int(position * nx), nx - 1))
            u = ux[:, idx].cpu().tolist()
            v = uy[:, idx].cpu().tolist()
            coords = [i / (ny - 1) for i in range(ny)]
        else:
            idx = max(0, min(int(position * ny), ny - 1))
            u = ux[idx, :].cpu().tolist()
            v = uy[idx, :].cpu().tolist()
            coords = [i / (nx - 1) for i in range(nx)]
        return {
            "job_id": job_id, "step": step,
            "direction": direction, "position": position,
            "coords": coords, "u": u, "v": v,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Rule-based intent parser
# ---------------------------------------------------------------------------

# Scenario keywords (Chinese + English).  Order matters – more specific
# matches must come first.
_SCENARIOS: list[tuple[str, list[str]]] = [
    ("pipeline_flow", ["pipeline", "near-bed", "管道", "近底", "海底管"]),
    ("ship_hull", ["ship", "hull", "wigley", "kcs", "船", "船体", "船舶"]),
    ("turbulent_channel", ["turbulent channel", "channel flow", "湍流通道", "槽道"]),
    ("sloshing_tank", ["slosh", "晃荡", "晃动", "晃荡水舱"]),
    ("dam_break", ["dam break", "dam-break", "溃坝", "破堤"]),
    ("lid_driven_cavity", ["lid-driven", "lid driven", "cavity", "顶盖驱动", "方腔"]),
    ("cylinder_flow", [
        "cylinder", "vortex shedding", "von karman", "kármán",
        "圆柱", "绕流", "卡门",
    ]),
]


_INTENT_LIST = [
    "list jobs", "list job", "jobs list", "show jobs", "all jobs",
    "任务列表", "所有任务", "查看任务",
]
_INTENT_STATUS = [
    "status", "progress", "进度", "状态",
]
_INTENT_ANALYZE = [
    "analyze", "summary", "summarize", "result", "分析", "总结", "结果",
]
_INTENT_PROFILE = [
    "velocity profile", "profile", "速度剖面", "剖面",
]
_INTENT_HELP = [
    "help", "what can you do", "capabilities", "帮助", "能做什么", "支持哪些",
]


def _extract_int(text: str, keys: list[str]) -> int | None:
    """Find an integer after any of the given keys in text."""
    for k in keys:
        m = re.search(rf"{k}\s*[=:]?\s*(-?\d+)", text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _extract_float(text: str, keys: list[str]) -> float | None:
    for k in keys:
        m = re.search(
            rf"{k}\s*[=:]?\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            text, re.IGNORECASE,
        )
        if m:
            return float(m.group(1))
    return None


def _extract_job_id(text: str, history_actions: list[dict]) -> str | None:
    # Explicit "job_id=..." or UUID-like patterns
    m = re.search(
        r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1)
    m = re.search(r"job[_\s-]?id[\s=:]{1,3}([0-9a-fA-F-]{6,})", text)
    if m:
        return m.group(1)
    # Fall back to last submitted job in this conversation
    for action in reversed(history_actions):
        if "job_id" in action.get("result", {}):
            return action["result"]["job_id"]
    return None


def _detect_scenario(text: str) -> str | None:
    low = text.lower()
    for scenario, keys in _SCENARIOS:
        for k in keys:
            if k.lower() in low:
                return scenario
    return None


def _parse_intent(text: str, history_actions: list[dict]) -> dict:
    """Map user text to a single tool invocation plan.

    Returns a dict ``{"tool": str, "args": {...}, "reason": str}`` or
    ``{"tool": None, "reason": "..."}`` when nothing matched.
    """
    low = text.lower()

    # 1. Help / capabilities ------------------------------------------------
    if any(k in low for k in _INTENT_HELP):
        return {"tool": "_help", "args": {}, "reason": "help request"}

    # 2. List jobs ----------------------------------------------------------
    if any(k in low for k in _INTENT_LIST):
        return {"tool": "list_recent_jobs", "args": {}, "reason": "list jobs"}

    # 3. Status / progress -------------------------------------------------
    if any(k in low for k in _INTENT_STATUS):
        jid = _extract_job_id(text, history_actions)
        if jid:
            return {"tool": "get_job_status", "args": {"job_id": jid},
                    "reason": "job status"}

    # 4. Velocity profile --------------------------------------------------
    if any(k in low for k in _INTENT_PROFILE):
        jid = _extract_job_id(text, history_actions)
        if jid:
            direction = "x" if re.search(r"\bx\b|水平|横向", text) else "y"
            return {"tool": "velocity_profile",
                    "args": {"job_id": jid, "direction": direction},
                    "reason": "velocity profile"}

    # 5. Analyze job -------------------------------------------------------
    if any(k in low for k in _INTENT_ANALYZE):
        jid = _extract_job_id(text, history_actions)
        if jid:
            return {"tool": "analyze_job", "args": {"job_id": jid},
                    "reason": "analyze job"}

    # 6. New simulation ----------------------------------------------------
    scenario = _detect_scenario(text)
    if scenario is None:
        return {"tool": None, "reason": "no scenario keyword matched"}

    args: dict[str, Any] = {}
    re_v = _extract_float(text, ["re", "reynolds", "雷诺数", "雷诺"])
    if re_v is not None:
        args["re"] = re_v
    re_tau = _extract_float(text, ["re_tau", "retau", "re-tau", "摩擦雷诺数"])
    if re_tau is not None and scenario == "turbulent_channel":
        args["re_tau"] = re_tau
    n_steps = _extract_int(text, ["n_steps", "steps", "iterations",
                                  "迭代", "步数", "时间步"])
    if n_steps is not None:
        args["n_steps"] = n_steps
    nx = _extract_int(text, ["nx", "网格宽", "grid width"])
    ny = _extract_int(text, ["ny", "网格高", "grid height"])
    if nx is not None:
        args["nx"] = nx
    if ny is not None and scenario != "lid_driven_cavity":
        args["ny"] = ny
    device = None
    if re.search(r"\bgpu\b|cuda|显卡", low):
        device = "cuda:0"
    elif re.search(r"\bcpu\b", low):
        device = "cpu"
    if device:
        args["device"] = device
    # Model choice for dam break / multiphase
    if scenario == "dam_break":
        m = re.search(r"\b(sc|scmp|cg|fe)\b", low)
        if m:
            args["model"] = m.group(1)

    tool_name = f"submit_{scenario}"
    return {"tool": tool_name, "args": args, "reason": f"scenario={scenario}"}


# ---------------------------------------------------------------------------
# LLM call (optional)
# ---------------------------------------------------------------------------

def _llm_enabled() -> bool:
    return bool(os.environ.get("TENSORLBM_LLM_API_KEY"))


def _llm_chat(messages: list[dict], system: str, timeout: float = 20.0) -> str | None:
    """Call an OpenAI-compatible chat completion endpoint, return text or None.

    Failures (network, auth, malformed response) are logged and return
    ``None`` so the agent gracefully falls back to its deterministic
    summary text.
    """
    api_key = os.environ.get("TENSORLBM_LLM_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get(
        "TENSORLBM_LLM_BASE_URL", "https://api.openai.com/v1",
    ).rstrip("/")
    model = os.environ.get("TENSORLBM_LLM_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.3,
        "max_tokens": 600,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
            TimeoutError, ValueError) as exc:
        logger.warning("LLM call failed: %s", exc)
        return None


_AGENT_SYSTEM_PROMPT = (
    "You are an assistant embedded in the TensorLBM web platform. The user "
    "drives Lattice-Boltzmann simulations (modelling → solving → "
    "post-processing) by chatting with you in Chinese or English. The "
    "platform has already executed any tool the user requested and given "
    "you the result as JSON. Your job is to explain that result clearly "
    "in 2–6 sentences, summarise the key numbers (Reynolds number, grid, "
    "steps, output directory, drag/lift, …) and suggest the next step "
    "the user might want to take (e.g. 'view snapshots', 'compute the "
    "velocity profile', 'increase Reynolds number'). Be concise and "
    "answer in the language the user used."
)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    reply: str
    actions: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    used_llm: bool = False
    intent: dict | None = None


def _format_action_summary(tool_name: str, result: dict) -> str:
    """Build a deterministic natural-language summary of a tool result."""
    if tool_name == "_help":
        return _help_text()
    if "error" in result:
        return f"❌ {tool_name} failed: {result['error']}"
    if tool_name.startswith("submit_"):
        jid = result.get("job_id", "?")
        cfg = result.get("config", {})
        bits = []
        for k in ("nx", "ny", "nz", "re", "re_tau", "n_steps",
                 "output_interval", "device", "model"):
            if k in cfg:
                bits.append(f"{k}={cfg[k]}")
        params = ", ".join(bits)
        return (
            f"✅ Submitted **{result.get('name', tool_name)}** "
            f"(job_id `{jid}`) with {params}. "
            f"It is now running in the background; you can ask me for its "
            f"status, analyze the result, or extract a velocity profile."
        )
    if tool_name == "get_job_status":
        return (
            f"Job `{result.get('job_id', '?')}` is **{result.get('status', '?')}** "
            f"({result.get('job_type', '?')}). Output dir: "
            f"`{result.get('output_dir', '?')}`."
        )
    if tool_name == "list_recent_jobs":
        jobs = result.get("jobs", [])
        if not jobs:
            return "No jobs on the platform yet."
        lines = [f"Found {len(jobs)} recent job(s):"]
        for j in jobs[:10]:
            lines.append(
                f"  • `{j['job_id']}` — {j['name']} — **{j['status']}**"
            )
        return "\n".join(lines)
    if tool_name == "analyze_job":
        meta = result.get("metadata", {}) or {}
        res = result.get("result", {}) or {}
        lines = [
            f"📊 Analysis of `{result['job_id']}` ({result['name']}):",
            f"  • Status: **{result['status']}**",
            f"  • Snapshots (PNG): {result['png_files']}",
            f"  • CSV files: {result['csv_files']}",
        ]
        if res:
            lines.append(f"  • Solver result: `{json.dumps(res)[:200]}`")
        if meta:
            # Pull a few interesting fields if present
            for k in ("re", "re_tau", "u_in", "n_steps", "nx", "ny",
                     "strouhal", "cd_mean", "cl_rms"):
                if k in meta:
                    lines.append(f"  • {k}: `{meta[k]}`")
        return "\n".join(lines)
    if tool_name == "velocity_profile":
        n = len(result.get("u", []))
        return (
            f"Extracted a {result['direction']}-direction velocity profile from "
            f"job `{result['job_id']}` at step {result['step']} "
            f"({n} sample points, position={result['position']:.2f})."
        )
    return f"{tool_name} → {json.dumps(result)[:200]}"


def _help_text() -> str:
    return (
        "👋 I'm the TensorLBM agent. I can drive the whole pipeline for you:\n"
        "  • **Modelling** – pick a scenario (cylinder flow, lid-driven "
        "cavity, dam break, sloshing tank, turbulent channel, near-bed "
        "pipeline, ship hull, …).\n"
        "  • **Solving** – I submit the job to the platform's job manager; "
        "you'll see it appear in the sidebar and progress over WebSocket.\n"
        "  • **Analysis** – I can list/inspect running jobs, summarise "
        "completed runs, and extract velocity profiles from checkpoints.\n\n"
        "Try saying things like:\n"
        "  • *Run a cylinder flow at Re=200 with 2000 steps*\n"
        "  • *用顶盖驱动方腔做一个 Re=400 的算例*\n"
        "  • *Show the status of the latest job*\n"
        "  • *Analyze job <id> and give me a velocity profile*"
    )


def _suggestions_for(tool_name: str, result: dict) -> list[str]:
    if tool_name.startswith("submit_") and "job_id" in result:
        return [
            f"Show the status of job {result['job_id']}",
            f"Analyze job {result['job_id']} when it finishes",
            f"Extract a velocity profile from job {result['job_id']}",
        ]
    if tool_name == "get_job_status" and result.get("status") == "completed":
        return [
            f"Analyze job {result.get('job_id', '')}",
            f"Velocity profile of {result.get('job_id', '')}",
        ]
    if tool_name == "list_recent_jobs":
        return ["Run a new cylinder flow at Re=200",
                "Analyze the most recent job"]
    return ["Help – what can you do?"]


def chat(message: str, history: list[dict] | None = None) -> AgentResponse:
    """Single-turn conversational entry point used by the REST router.

    ``history`` is the chronological list of previous ``{role, content,
    actions?}`` dicts coming from the frontend.  It is used **only** to
    recover the last submitted ``job_id`` so the user can refer to "the
    job" implicitly; no further state is kept.
    """
    history = history or []
    history_actions: list[dict] = []
    for turn in history:
        for a in turn.get("actions", []) or []:
            history_actions.append(a)

    intent = _parse_intent(message, history_actions)
    tool_name = intent.get("tool")
    actions: list[dict] = []

    if tool_name is None:
        reply = (
            "I couldn't infer a clear simulation request from that. "
            "Try mentioning a scenario keyword (cylinder flow, cavity, dam "
            "break, sloshing tank, ship hull, turbulent channel, pipeline) "
            "or ask for 'help' to see everything I can do."
        )
        suggestions = ["Help – list all my capabilities",
                       "Run a cylinder flow at Re=100",
                       "List recent jobs"]
        return AgentResponse(reply=reply, actions=actions,
                             suggestions=suggestions, intent=intent)

    if tool_name == "_help":
        return AgentResponse(
            reply=_help_text(),
            actions=[],
            suggestions=["Run a cylinder flow at Re=100",
                         "List recent jobs",
                         "运行一个顶盖驱动方腔 Re=400"],
            intent=intent,
        )

    t = _TOOLS.get(tool_name)
    if t is None:
        return AgentResponse(
            reply=f"Internal error: unknown tool {tool_name!r}",
            intent=intent,
        )

    try:
        result = t.handler(**intent["args"])
    except TypeError as exc:
        return AgentResponse(
            reply=f"⚠️ I couldn't call `{tool_name}` with the inferred "
            f"parameters: {exc}",
            intent=intent,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent tool %s raised", tool_name)
        return AgentResponse(
            reply=f"⚠️ The tool `{tool_name}` failed: {exc}",
            intent=intent,
        )

    actions.append({"tool": tool_name, "args": intent["args"], "result": result})
    base_summary = _format_action_summary(tool_name, result)
    suggestions = _suggestions_for(tool_name, result)

    used_llm = False
    reply = base_summary
    if _llm_enabled():
        llm_messages = [
            {"role": h.get("role", "user"), "content": h.get("content", "")}
            for h in history if h.get("content")
        ]
        llm_messages.append({"role": "user", "content": message})
        llm_messages.append({
            "role": "user",
            "content": (
                f"[platform executed tool {tool_name} with args "
                f"{json.dumps(intent['args'])} and result "
                f"{json.dumps(result)[:1500]}]"
            ),
        })
        text = _llm_chat(llm_messages, _AGENT_SYSTEM_PROMPT)
        if text:
            reply = text.strip()
            used_llm = True

    return AgentResponse(
        reply=reply,
        actions=actions,
        suggestions=suggestions,
        used_llm=used_llm,
        intent=intent,
    )
