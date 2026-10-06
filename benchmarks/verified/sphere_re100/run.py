"""W8-A runner — sphere Re=100 with dense vs sparse BFL kernels.

Same simulation loop as the Wave-7 W7-B runner (run_diag.py): engine
kernels + library public entrances only; the BFL wall is either the
library dense kernel (bouzidi_bounce_back_d3q19) or the W8-A sparse
kernels (bfl_boundary_link_indices precompute + bfl_bounce_back_sparse /
bfl_force_ledger_sparse), selected with --kernel.

Driver-side memory scheduling (values identical, disclosed in prereg.md):
the sparse route restores solid cells from a gathered (19, n_solid) block
instead of holding a full pre-collision clone, and frees the dense
mask/q_field right after the link precompute.

grep rule honored: no collide/stream/equilibrium/bounce/zou_he/far_field
definitions in this file — only library calls.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import threading
import time

import torch

from tensorlbm.bfl_common import (  # noqa: E402
    bfl_bounce_back_sparse,
    bfl_boundary_link_indices,
    bfl_force_ledger_sparse,
)
from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19  # noqa: E402
from tensorlbm.boundaries3d import (  # noqa: E402
    bounce_back_cells_3d,
    far_field_bc_3d,
)
from tensorlbm.external_open_boundary import (  # noqa: E402
    non_equilibrium_far_field_bc_3d,
)
from tensorlbm.general_sim import (  # noqa: E402
    CollisionModel,
    ForceMethod,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    OutputFormat,
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
)
from tensorlbm.interpolated_bc import compute_q_sphere  # noqa: E402
from tensorlbm.solver3d import (  # noqa: E402
    collide_bgk3d,
    collide_mrt3d_low_memory,
    collide_trt3d,
    correct_mass3d,
    stream3d,
    stream3d_roll,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_instrument import cv_box_force_exact  # noqa: E402

DEV = os.environ.get("W8A_DEV") or ("cuda:0" if torch.cuda.is_available() else "sdaa:0")
PROGRESS = {"step": 0, "phase": "init"}


def build_engine_config(D, lat, up, down):
    pad = (up, down, lat, lat, lat, lat)
    return GeneralSimConfig(
        name=f"w8a_sphere_D{D}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SPHERE,
            sphere_radius=0.5,
            sphere_center=(0.0, 0.0, 0.0),
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=1.0e-6,
            inlet_velocity=1.0e-4,
            reference_length=1.0,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.MRT,
            resolution=D,
            domain_padding=pad,
            max_steps=10,
            snapshot_interval=10**9,
            force_sample_interval=50,
            device=DEV,
            wall_treatment=WallTreatment.BOUNCE_BACK,
            mass_correction=True,
            mass_correction_interval=200,
            force_method=ForceMethod.MOMENTUM_EXCHANGE,
            mem_variant="wet_node",
        ),
        output=OutputConfig(
            directory="out",
            formats=[OutputFormat.NPY],
            save_macroscopic=False,
            save_forces=True,
        ),
    )


def magic_s_q(tau: float, lam: float = 3.0 / 16.0) -> float:
    """Ginzburg/Pan magic relation (tau-0.5)*(1/s_q-0.5)=lam -> s_q."""
    return 1.0 / (0.5 + lam / (tau - 0.5))


def make_collider(kind: str, tau: float):
    if kind == "mrt":
        return lambda f: collide_mrt3d_low_memory(f, tau=tau)
    if kind == "mrt_magic":
        return lambda f: collide_mrt3d_low_memory(f, tau=tau, s_q=magic_s_q(tau))
    if kind == "trt":
        return lambda f: collide_trt3d(f, tau_plus=tau, lambda_trt=3.0 / 16.0)
    if kind == "bgk":
        return lambda f: collide_bgk3d(f, tau=tau)
    raise ValueError(kind)


def cd_ref_family(re: float) -> float:
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


class MemSampler:
    """Background nvidia-smi sampler (engineering probe only)."""

    def __init__(self, physical_gpu: str, interval: float = 2.0):
        self.physical_gpu = physical_gpu
        self.interval = interval
        self.max_mib = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import subprocess

        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        self.physical_gpu,
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                v = int(out.stdout.strip().splitlines()[0])
                if v > self.max_mib:
                    self.max_mib = v
                self.samples += 1
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


def run_case(
    out_path, kernel, D, steps, lat, up, down, tau, ulb, treat, sample, cv_tail=0, mem_stats=False
):
    cfg = build_engine_config(D, lat, up, down)
    engine = GeneralSimEngine(cfg)
    engine.setup()
    nu_lb = (tau - 0.5) / 3.0
    re_eff = ulb * D / nu_lb
    R_lb = cfg.geometry.sphere_radius / (cfg.physics.reference_length / cfg.solver.resolution)
    dpS = 0.5 * ulb**2 * math.pi * R_lb**2
    solid = engine.solid
    nz, ny, nx = solid.shape
    dx = cfg.physics.reference_length / cfg.solver.resolution
    cx_lb = (cfg.geometry.sphere_center[0] - engine.domain_phys[0]) / dx
    cy_lb = (cfg.geometry.sphere_center[1] - engine.domain_phys[2]) / dx
    cz_lb = (cfg.geometry.sphere_center[2] - engine.domain_phys[4]) / dx
    dev = torch.device(DEV)

    sampler = MemSampler(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) if mem_stats else None
    if sampler is not None:
        sampler.__enter__()

    bfl_mask, bfl_q = compute_q_sphere(nx, ny, nz, cx_lb, cy_lb, cz_lb, R_lb, dev)
    n_links = int(bfl_mask[1:].sum().item())
    q_active = bfl_q[1:][bfl_mask[1:]]
    if kernel == "sparse":
        links = bfl_boundary_link_indices(bfl_mask, bfl_q, lattice="D3Q19")
        if links.n_links != n_links or links.n_wrapped != 0:
            raise RuntimeError(
                f"link precompute mismatch: {links.n_links} vs {n_links}, wrapped={links.n_wrapped}"
            )
        q_min = float(links.link_q.min().item())
        q_max = float(links.link_q.max().item())
        del bfl_mask, bfl_q, q_active
        torch.cuda.empty_cache()
    else:
        links = None
        q_min = float(q_active.min().item())
        q_max = float(q_active.max().item())
        del q_active

    peak_after_setup = float(torch.cuda.max_memory_allocated()) / 2**30 if mem_stats else None
    if mem_stats:
        torch.cuda.reset_peak_memory_stats()

    bc_config = {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)
    collide = make_collider("mrt", tau)
    stream_fn = stream3d_roll if os.environ.get("W8A_STREAM", "roll") == "roll" else stream3d

    f = engine.f.clone()
    engine.f = None  # memory trim: release engine's distribution reference
    target_mass = float(f.sum().item())
    initial_mass = target_mass
    hist = []
    finite = True
    diverged_at = None
    sample_set = set(range(sample, steps + 1, sample))
    sm = solid.unsqueeze(0).expand_as(f)
    solid_idx = solid.reshape(-1).nonzero(as_tuple=True)[0]
    t0 = time.time()
    PROGRESS["phase"] = "loop"
    for step in range(1, steps + 1):
        PROGRESS["step"] = step
        if kernel == "sparse":
            # NoDynamics bookkeeping: gather pre-collision values at solid
            # cells only (values identical to the full-clone restore).
            f_solid_pre = f.reshape(19, -1)[:, solid_idx]
            f = collide(f)
            f.reshape(19, -1)[:, solid_idx] = f_solid_pre
            del f_solid_pre
        else:
            f_pre = f.clone()
            f = collide(f)
            for q in range(f.shape[0]):
                f[q] = torch.where(sm[q], f_pre[q], f[q])
            del f_pre
        f = bounce_back_cells_3d(f, solid)
        f_pre_stream = f.clone()
        f = stream_fn(f)
        if treat == "noneq":
            f = non_equilibrium_far_field_bc_3d(
                f,
                u_in=ulb,
                faces=("x-", "x+", "y-", "y+", "z-", "z+"),
            )
        else:
            f = far_field_fn(f, ulb)
        if kernel == "sparse":
            force3, _ = bfl_force_ledger_sparse(f, f_pre_stream, links)
            f = bfl_bounce_back_sparse(f, f_pre_stream, links)
        else:
            f, force3 = bouzidi_bounce_back_d3q19(
                f, f_pre_stream, bfl_mask, bfl_q, return_force=True
            )
        if step in sample_set:
            cd = float(force3[0].item()) / dpS
            cl = float(force3[1].item()) / dpS
            cs = float(force3[2].item()) / dpS
            entry = {
                "step": step,
                "cd": cd,
                "cl": cl,
                "cs": cs,
                "mass": float(f.sum().item()),
            }
            if cv_tail > 0 and step > steps - cv_tail:
                cv, _ = cv_box_force_exact(f_pre_stream, solid)
                entry["cv_mom_cd"] = cv[0] / dpS
                entry["cv_mom_cl"] = cv[1] / dpS
            hist.append(entry)
        del f_pre_stream
        if step % 200 == 0:
            f = correct_mass3d(f, target_mass)
        if step % 500 == 0 and not bool(torch.isfinite(f).all()):
            finite = False
            diverged_at = step
            break
        if step % 2000 == 0:
            el = time.time() - t0
            print(f"step {step}/{steps} ({el:.0f}s, {el / step * 1000:.0f} ms/step)", flush=True)

    peak_loop = float(torch.cuda.max_memory_allocated()) / 2**30 if mem_stats else None
    if sampler is not None:
        sampler.__exit__()

    meta = {
        "kernel": kernel,
        "route": "bfl",
        "collision": "mrt",
        "treat": treat,
        "D": D,
        "lat": lat,
        "up": up,
        "down": down,
        "tau": tau,
        "u_lb": ulb,
        "nu_lb": nu_lb,
        "re_eff": re_eff,
        "steps": steps if finite else diverged_at,
        "diverged": not finite,
        "domain_lu": [nz, ny, nx],
        "dpS": dpS,
        "R_lb": R_lb,
        "center_lu": [cz_lb, cy_lb, cx_lb],
        "blockage": math.pi * R_lb**2 / (ny * nz),
        "cd_ref_at_re_eff": cd_ref_family(re_eff),
        "mass_drift_ppm": (
            (float(f.sum().item()) - initial_mass) / initial_mass * 1e6 if finite else None
        ),
        "bfl_links": n_links,
        "q_min": q_min,
        "q_max": q_max,
        "stream": "roll" if stream_fn is stream3d_roll else "gather",
        "wall_s": time.time() - t0,
    }
    if mem_stats:
        meta["mem_peak_after_setup_gib"] = peak_after_setup
        meta["mem_peak_loop_gib"] = peak_loop
        meta["mem_peak_reserved_gib"] = float(torch.cuda.max_memory_reserved()) / 2**30
        if sampler is not None:
            meta["mem_peak_nvidia_smi_mib"] = sampler.max_mib
            meta["mem_nvidia_smi_samples"] = sampler.samples

    with open(out_path, "w") as fh:
        json.dump({"meta": meta, "history": hist}, fh)
    print("written", out_path)
    if hist:
        tail = [h["cd"] for h in hist[-8:]]
        print("last-8 cd mean:", sum(tail) / len(tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--kernel", choices=["dense", "sparse"], default="sparse")
    ap.add_argument("--D", type=int, default=40)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lat", type=float, default=2.0)
    ap.add_argument("--up", type=float, default=1.25)
    ap.add_argument("--down", type=float, default=2.25)
    ap.add_argument("--tau", type=float, default=0.56)
    ap.add_argument("--ulb", type=float, default=0.05)
    ap.add_argument("--treat", choices=["hard", "noneq"], default="hard")
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--cv_tail", type=int, default=0)
    ap.add_argument("--mem_stats", action="store_true")
    args = ap.parse_args()
    PROGRESS["phase"] = "run"
    try:
        run_case(
            args.out,
            args.kernel,
            args.D,
            args.steps,
            args.lat,
            args.up,
            args.down,
            args.tau,
            args.ulb,
            args.treat,
            args.sample,
            args.cv_tail,
            args.mem_stats,
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower() and not isinstance(
            exc, torch.cuda.OutOfMemoryError
        ):
            raise
        import traceback

        rec = {
            "oom": True,
            "kernel": args.kernel,
            "D": args.D,
            "lat": args.lat,
            "up": args.up,
            "down": args.down,
            "tau": args.tau,
            "steps_requested": args.steps,
            "progress": dict(PROGRESS),
            "torch_peak_allocated_gib": float(torch.cuda.max_memory_allocated()) / 2**30,
            "torch_peak_reserved_gib": float(torch.cuda.max_memory_reserved()) / 2**30,
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "traceback_tail": traceback.format_exc().strip().splitlines()[-8:],
            "error": str(exc)[:2000],
        }
        with open(args.out + ".oom.json", "w") as fh:
            json.dump(rec, fh, indent=2)
        print(json.dumps(rec, indent=2))
        sys.exit(3)


if __name__ == "__main__":
    main()
