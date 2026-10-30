#!/usr/bin/env python3
"""SDAA sweep for permeability3d benchmark.

Runs the φ=0.343档 on 'sdaa' device, computes all verdict numbers
independently, and writes artifacts/sdaa_permeability3d.json.

No business code is modified.  Physics kernels are imported from the
local worktree src/tensorlbm.

Time budget: ≤20 min wall clock.  On SDAA, N=64 converges in ~3 min,
N=96 is ~10× slower.  We run N=64 fully converged and N=96 in smoke
mode (reduced steps) to stay within budget.  N=128/160 are skipped
with honest disclosure.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ------------------------------------------------------------------
# Path setup: use local worktree src
# ------------------------------------------------------------------
WORKTREE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = WORKTREE_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402

# ------------------------------------------------------------------
# Triton workaround record
# ------------------------------------------------------------------
TRITON_INTERPRET_SET = False
try:
    import triton  # noqa: F401
    if "TRITON_INTERPRET" not in os.environ:
        os.environ["TRITON_INTERPRET"] = "1"
        TRITON_INTERPRET_SET = True
except ImportError:
    pass  # triton not needed for this case

# ------------------------------------------------------------------
# Import tensorlbm from local src
# ------------------------------------------------------------------
import tensorlbm  # noqa: E402

assert tensorlbm.__file__.startswith(str(SRC_DIR)), (
    f"tensorlbm imported from {tensorlbm.__file__}, expected prefix {SRC_DIR}"
)

from tensorlbm.boundaries3d import bounce_back_cells_3d  # noqa: E402
from tensorlbm.d3q19 import W_EXACT64, equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.d3q19 import C as D3Q19_C  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

# ------------------------------------------------------------------
# Constants (copied from run.py, same values)
# ------------------------------------------------------------------
TAU = 1.0
NU = (TAU - 0.5) / 3.0  # = 1/6
RE_TARGET = 0.05
TOL = 0.03  # verdict tolerance on finest tier

ZH_TABLE: list[tuple[float, float]] = [
    (0.027, 2.0077),
    (0.064, 2.8102),
    (0.125, 4.292),
    (0.216, 7.4423),
    (0.343, 15.402),
    (0.45, 28.09),
    (0.5236, 41.99),
]
K_TABLE = dict(ZH_TABLE)
PHI = 0.343
K_TABLE_PHI = K_TABLE[PHI]

# Tiers we attempt: N=64 (full), N=96 (smoke/reduced)
# N=128/160 are too slow on SDAA for 20-min budget
LADDER_N_FULL = [64]
LADDER_N_SMOKE = [96]
LADDER_N_SKIPPED = [128, 160]

SAMPLE_EVERY = 200
DRIFT_SPAN = 10
DRIFT_TOL = 1e-5
STEADY_REPEATS = 3
POST_STEADY = 8000
MEAS_SAMPLES = 20
MAX_STEPS_CAP = 250_000
MAX_STEPS_COEF = 20
FORCE_SCALE = 10.0  # fs=10 formal protocol

# Smoke mode parameters (reduced for time budget)
SMOKE_SAMPLE_EVERY = 100
SMOKE_DRIFT_TOL = 1e-3
SMOKE_POST_STEADY = 300
SMOKE_MAX_STEPS = 2000


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def nominal_radius(n: int, phi: float) -> float:
    return n * (3.0 * phi / (4.0 * math.pi)) ** (1.0 / 3.0)


def k_ref_lu(n: int, phi: float, k_table: float) -> float:
    a = nominal_radius(n, phi)
    return n**3 / (6.0 * math.pi * a * k_table)


def sphere_array_mask_3d(n: int, phi: float, device: torch.device) -> torch.Tensor:
    import numpy as np

    a_real = nominal_radius(n, phi)
    idx = np.arange(n, dtype=np.float64)
    dd = np.abs(idx - n / 2.0)
    dd = np.minimum(dd, n - dd)
    dist2 = dd[:, None, None] ** 2 + dd[None, :, None] ** 2 + dd[None, None, :] ** 2
    mask = dist2 <= a_real * a_real
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def apply_body_force_3d(f: torch.Tensor, a_x: float) -> torch.Tensor:
    device = f.device
    c = D3Q19_C.to(device).to(f.dtype)
    w = W_EXACT64.to(device=device, dtype=f.dtype).view(19, 1, 1, 1)
    rho = f.sum(dim=0)
    cx = c[:, 0].view(19, 1, 1, 1)
    return f + w * 3.0 * rho.unsqueeze(0) * cx * a_x


def run_one_case(n: int, phi: float, device: torch.device, smoke: bool = False) -> dict:
    """Run a single (phi, N) case and return all derived numbers."""
    k_table = K_TABLE[phi]
    nu_case = (TAU - 0.5) / 3.0
    a_nom = nominal_radius(n, phi)
    k_lu = k_ref_lu(n, phi, k_table)
    a_body = RE_TARGET * nu_case * nu_case / (2.0 * a_nom * k_lu) * FORCE_SCALE

    if smoke:
        max_steps = SMOKE_MAX_STEPS
        sample_every = SMOKE_SAMPLE_EVERY
        drift_tol = SMOKE_DRIFT_TOL
        post_steady = SMOKE_POST_STEADY
    else:
        max_steps = min(MAX_STEPS_COEF * n * n, MAX_STEPS_CAP)
        sample_every = SAMPLE_EVERY
        drift_tol = DRIFT_TOL
        post_steady = POST_STEADY

    solid = sphere_array_mask_3d(n, phi, device)
    fluid = ~solid
    n_solid = int(solid.sum().item())
    phi_actual = n_solid / (n**3)

    rho0 = torch.ones(n, n, n, dtype=torch.float32, device=device)
    u0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, u0, u0, u0)
    mass0 = float(f.sum().item())
    del rho0, u0

    solid_b = solid.unsqueeze(0)

    def _step(f_t: torch.Tensor) -> torch.Tensor:
        f_t = collide_bgk3d(f_t, TAU)
        f_t = stream3d(f_t)
        f_t = torch.where(solid_b, f_t, apply_body_force_3d(f_t, a_body))
        return bounce_back_cells_3d(f_t, solid)

    hist_step: list[int] = []
    hist_uxf: list[float] = []
    hist_uyf: list[float] = []
    hist_uzf: list[float] = []
    hist_umax: list[float] = []
    hist_mass: list[float] = []
    recent_drifts: list[float] = []
    steady_step: int | None = None
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        f = _step(f)
        if step % sample_every == 0:
            _, ux, uy, uz = macroscopic3d(f)
            uxf = float(ux[fluid].mean().item())
            uyf = float(uy[fluid].mean().item())
            uzf = float(uz[fluid].mean().item())
            umax = float(torch.sqrt(ux * ux + uy * uy + uz * uz)[fluid].max().item())
            mass = float(f.sum().item())
            hist_step.append(step)
            hist_uxf.append(uxf)
            hist_uyf.append(uyf)
            hist_uzf.append(uzf)
            hist_umax.append(umax)
            hist_mass.append(mass)
            if steady_step is None and len(hist_uxf) >= DRIFT_SPAN + 1:
                drift = abs(hist_uxf[-1] - hist_uxf[-1 - DRIFT_SPAN]) / abs(hist_uxf[-1])
                recent_drifts.append(drift)
                if len(recent_drifts) >= STEADY_REPEATS and all(
                    v < drift_tol for v in recent_drifts[-STEADY_REPEATS:]
                ):
                    steady_step = step - DRIFT_SPAN * sample_every
        if steady_step is not None and step >= steady_step + post_steady:
            break

    wall = time.perf_counter() - t0
    steps_run = hist_step[-1] if hist_step else 0
    steady_detected = steady_step is not None

    def _win_mean(vals: list[float], n_win: int) -> float | None:
        return sum(vals[-n_win:]) / len(vals[-n_win:]) if len(vals) >= n_win else None

    uxf_meas = _win_mean(hist_uxf, MEAS_SAMPLES)
    uyf_meas = _win_mean(hist_uyf, MEAS_SAMPLES)
    uzf_meas = _win_mean(hist_uzf, MEAS_SAMPLES)
    umax_meas = _win_mean(hist_umax, MEAS_SAMPLES)

    if uxf_meas is None:
        # Fallback: use last available sample
        uxf_meas = hist_uxf[-1] if hist_uxf else 0.0
        uyf_meas = hist_uyf[-1] if hist_uyf else 0.0
        uzf_meas = hist_uzf[-1] if hist_uzf else 0.0
        umax_meas = hist_umax[-1] if hist_umax else 0.0

    # Primary channel
    k_sim = nu_case * (1.0 - phi) * uxf_meas / a_body
    k_ref = k_lu
    err = abs(k_sim / k_ref - 1.0)

    # K_sim (inverse map)
    K_sim = a_body * n**3 / (6.0 * math.pi * nu_case * a_nom * (1.0 - phi) * uxf_meas)

    # Ma
    Ma_max = umax_meas

    # Mass drift
    mass_drift = max(abs(m - mass0) / mass0 for m in hist_mass) if hist_mass else 0.0

    return {
        "N": n,
        "phi": phi,
        "phi_actual": phi_actual,
        "a_nom": a_nom,
        "a_body": a_body,
        "k_ref": k_ref,
        "k_sim": k_sim,
        "K_sim": K_sim,
        "K_table": k_table,
        "err": err,
        "Ma_max": Ma_max,
        "steady_detected": steady_detected,
        "steps_run": steps_run,
        "wall_seconds": wall,
        "uxf_mean_win4000": uxf_meas,
        "mass_drift_max_rel": mass_drift,
        "smoke": smoke,
    }


def main():
    device = torch.device("sdaa:0")
    print(f"Device: {device}")
    device_name = torch.sdaa.get_device_name(0)
    print(f"Device name: {device_name}")
    device_props = torch.sdaa.get_device_properties(0)
    print(f"Device props: {device_props}")

    results = []
    total_wall = 0.0

    # Run full tiers
    for n in LADDER_N_FULL:
        print(f"\n--- Running N={n} (full) ---")
        r = run_one_case(n, PHI, device, smoke=False)
        results.append(r)
        total_wall += r["wall_seconds"]
        print(f"  err={r['err']:.6f}  K_sim={r['K_sim']:.6f}  "
              f"steady={r['steady_detected']}  wall={r['wall_seconds']:.1f}s")

    # Run smoke tiers
    for n in LADDER_N_SMOKE:
        print(f"\n--- Running N={n} (smoke) ---")
        r = run_one_case(n, PHI, device, smoke=True)
        results.append(r)
        total_wall += r["wall_seconds"]
        print(f"  err={r['err']:.6f}  K_sim={r['K_sim']:.6f}  "
              f"steady={r['steady_detected']}  wall={r['wall_seconds']:.1f}s")

    # Skipped tiers
    skipped_info = [{"N": n, "reason": "too slow on SDAA for 20-min budget"} for n in LADDER_N_SKIPPED]

    # ---- Gate evaluation ----
    errs = [r["err"] for r in results]
    # With only 1 full tier + 1 smoke tier, monotone check is limited
    mono = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)) if len(errs) > 1 else None
    finest_le_tol = errs[-1] <= TOL if errs else None

    # Library entry gate
    lib_entry_pass = True

    # Direct observable gate
    direct_obs_pass = all(r["steady_detected"] for r in results) if results else False

    # Monotone convergence — only meaningful with ≥2 tiers
    mono_pass = mono if mono is not None else False

    gates = [
        {
            "gate": "library_entry",
            "pass": lib_entry_pass,
            "evidence": [
                {"v": "from tensorlbm.solver3d import collide_bgk3d, stream3d",
                 "f": "scripts/sdaa_sweep_permeability3d.py", "k": "import"},
                {"v": "from tensorlbm.boundaries3d import bounce_back_cells_3d",
                 "f": "scripts/sdaa_sweep_permeability3d.py", "k": "import"},
            ],
        },
        {
            "gate": "direct_observable",
            "pass": direct_obs_pass,
            "evidence": [
                {"v": r["err"], "N": r["N"], "err_pct": f"{r['err']*100:.4f}%",
                 "smoke": r["smoke"]}
                for r in results
            ],
        },
        {
            "gate": "monotone_convergence",
            "pass": mono_pass,
            "mode": "abs_decrease",
            "evidence": [
                {"v": errs[i], "N": results[i]["N"], "smoke": results[i]["smoke"]}
                for i in range(len(errs))
            ] if len(errs) > 1 else [{"note": "insufficient tiers for monotone check"}],
        },
        {
            "gate": "finest_le_tol_3pct",
            "pass": finest_le_tol if finest_le_tol is not None else False,
            "evidence": [
                {"v": errs[-1], "N": results[-1]["N"], "tol": TOL} if errs else [],
            ],
        },
    ]

    all_pass = all(g["pass"] for g in gates)

    # ---- Build output ----
    output = {
        "case": "permeability3d",
        "device": "sdaa:0",
        "device_name": device_name,
        "device_properties": str(device_props),
        "torch_version": torch.__version__,
        "tensorlbm_file": tensorlbm.__file__,
        "triton_interpret_set": TRITON_INTERPRET_SET,
        "phi": PHI,
        "k_table": K_TABLE_PHI,
        "tol": TOL,
        "tiers": [
            {
                "N": r["N"],
                "err": r["err"],
                "err_pct": round(r["err"] * 100, 4),
                "K_sim": r["K_sim"],
                "K_table": r["K_table"],
                "k_sim": r["k_sim"],
                "k_ref": r["k_ref"],
                "Ma_max": r["Ma_max"],
                "steady_detected": r["steady_detected"],
                "steps_run": r["steps_run"],
                "wall_seconds": round(r["wall_seconds"], 2),
                "phi_actual": r["phi_actual"],
                "mass_drift_max_rel": r["mass_drift_max_rel"],
                "smoke": r["smoke"],
            }
            for r in results
        ],
        "tiers_skipped": skipped_info,
        "gates": gates,
        "all_pass": all_pass,
        "total_wall_seconds": round(total_wall, 2),
        "created_utc": utc_now(),
    }

    # Write artifact
    artifacts_dir = WORKTREE_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "sdaa_permeability3d.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nArtifact written to {out_path}")
    print(f"Total wall time: {total_wall:.1f}s")
    print(f"All gates pass: {all_pass}")
    for g in gates:
        status = "PASS" if g["pass"] else "FAIL"
        print(f"  {g['gate']}: {status}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
