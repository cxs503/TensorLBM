#!/usr/bin/env python3
"""sphere_re100 D3Q27 twin gate runner (preregistered driver, W11-D case 2).

Protocol lineage: machine-pulled remote main verified/sphere_re100
(commit 20cd23472; w8a/run.py + w8a/result.json under diag/remote_pull/).
Loop structure mirrors w8a/run.py sparse route step-for-step; the lattice
is swapped D3Q19->D3Q27 and the collision MRT-low-memory->BGK27
(controller-designated; library has no low-memory MRT27 -- disclosed in
prereg.md).  Memory-mirrored implementations P1-P5 (feq per-direction,
in-place collide, solid gather/scatter bounce-back, in-place far-field,
in-place mass correction) are bitwise-identical to the library originals
(diag/ab27.json, torch.equal on two field families) and cap the per-step
live-set at 3 full (27,N) copies (library forms reach 4-6 -> OOM at D60big
on 32GB cards).  Engine is used for geometry/domain only (its D3Q19 f is
discarded; init rebuilt with equilibrium27 per the archived init state:
rho=1, ux=u_lb, ux=0 on solid).

Defaults are the archived tier parameters (D40: 280x280x320, tau 0.56,
12000 steps; D60: 420x420x480, tau 0.59, 24000 steps); when
W27_ARCHIVE_DIR points at diag/remote_pull/verified__sphere_re100 the
driver machine-checks domain/tau/steps/re_eff/u_lb against the pulled
archive JSONs (t1/t2 tier files) and refuses to run on mismatch
(zero hand-copied judgment numbers).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time

import torch

from tensorlbm.bfl_common import (
    bfl_bounce_back_sparse,
    bfl_boundary_link_indices,
    bfl_force_ledger_sparse,
    compute_q_sphere_common,
)
from tensorlbm.boundaries3d import sphere_mask  # noqa: F401  (probe import sanity)
from tensorlbm.d3q27 import (
    OPPOSITE,
    C,
    W,
    equilibrium27,
    macroscopic27,
    stream27_roll,
)
from tensorlbm.general_sim import (
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

DEV = os.environ.get("W27_DEV", "cuda:0")
PROGRESS = {"step": 0, "phase": "init"}


# ---------------- P1-P5 mirrored implementations (ab27.json-verified) ------
def feq27_perq_into(feq: torch.Tensor, rho, ux, uy, uz) -> None:
    w = W.to(device=rho.device, dtype=rho.dtype)
    c = C.to(rho.device)
    u_sq = ux * ux + uy * uy + uz * uz
    one5 = 1.5 * u_sq
    for q in range(27):
        cx, cy, cz = float(c[q][0]), float(c[q][1]), float(c[q][2])
        cu = cx * ux
        cu = cu + cy * uy
        cu = cu + cz * uz
        t = 3.0 * cu
        t = 1.0 + t
        t3 = 4.5 * cu
        t3 = t3 * cu
        t = t + t3
        t = t - one5
        wr = float(w[q]) * rho
        torch.mul(wr, t, out=feq[q])


def collide27_inplace(f: torch.Tensor, tau: float) -> torch.Tensor:
    rho, ux, uy, uz = macroscopic27(f)
    feq = torch.empty_like(f)
    feq27_perq_into(feq, rho, ux, uy, uz)
    tmp = torch.sub(f, feq)
    del feq
    tmp.div_(tau)
    f.sub_(tmp)
    return f


def bb27_solid_scatter(f: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
    flat = f.reshape(27, -1)
    idx = solid.reshape(-1).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return f
    saved = flat[:, idx]
    opp = OPPOSITE.to(f.device)
    for q in range(27):
        flat[q, idx] = saved[opp[q]]
    return f


def far_field27_inplace(f: torch.Tensor, u_in: float) -> torch.Tensor:
    nz, ny, nx = f.shape[1:]
    dt = f.dtype

    def mk(shape, val):
        return torch.full(shape, val, dtype=dt, device=f.device)

    feq_in = equilibrium27(mk((nz, ny, 1), 1.0), mk((nz, ny, 1), u_in),
                           mk((nz, ny, 1), 0.0), mk((nz, ny, 1), 0.0), device=f.device)
    f[:, :, :, 0] = feq_in[:, :, :, 0]
    del feq_in
    f[:, :, :, -1] = f[:, :, :, -2]
    feq_y = equilibrium27(mk((nz, 1, nx), 1.0), mk((nz, 1, nx), u_in),
                          mk((nz, 1, nx), 0.0), mk((nz, 1, nx), 0.0), device=f.device)
    f[:, 0, :, :] = feq_y[:, 0, :, :]
    f[:, -1, :, :] = feq_y[:, -1, :, :]
    del feq_y
    feq_z = equilibrium27(mk((1, ny, nx), 1.0), mk((1, ny, nx), u_in),
                          mk((1, ny, nx), 0.0), mk((1, ny, nx), 0.0), device=f.device)
    f[:, :, 0, :] = feq_z[:, :, 0, :]
    f[:, :, -1, :] = feq_z[:, :, -1, :]
    del feq_z
    return f


def correct_mass27_inplace(f: torch.Tensor, target_mass: float) -> torch.Tensor:
    current = f.sum()
    if current.abs() < 1e-30:
        return f
    f.mul_(target_mass / current)
    return f


def cv_box_force_exact_27(f_pre_stream, solid, margin=6):
    """CV momentum-box force, parameterized port of archived cv_instrument
    (bitwise-equal to the archived function at D3Q19 -- mini27_D12.json)."""
    dev = f_pre_stream.device
    zs, ys, xs = torch.nonzero(solid, as_tuple=True)
    m = margin
    nz, ny, nx = solid.shape
    z0 = max(int(zs.min()) - m, 2); z1 = min(int(zs.max()) + m, nz - 3)
    y0 = max(int(ys.min()) - m, 2); y1 = min(int(ys.max()) + m, ny - 3)
    x0 = max(int(xs.min()) - m, 2); x1 = min(int(xs.max()) + m, nx - 3)
    inbox = torch.zeros_like(solid)
    inbox[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = True
    c = C.to(dev)
    T = torch.zeros(3, device=dev, dtype=torch.float64)
    for i in range(1, 27):
        di, dj, dk = int(c[i][0]), int(c[i][1]), int(c[i][2])
        nb_in = torch.roll(inbox, (-dk, -dj, -di), dims=(0, 1, 2))
        leaving = inbox & ~nb_in
        entering = (~inbox) & nb_in
        if not (leaving.any() or entering.any()):
            continue
        s_out = float(f_pre_stream[i][leaving].sum().item())
        s_in = float(f_pre_stream[i][entering].sum().item())
        T[0] += (s_in - s_out) * di
        T[1] += (s_in - s_out) * dj
        T[2] += (s_in - s_out) * dk
    return [float(T[0]), float(T[1]), float(T[2])], (x0, x1, y0, y1, z0, z1)


def cd_ref_family(re: float) -> float:
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


# ---------------- engine config (geometry only) ----------------------------
def build_engine_config(D, lat, up, down):
    pad = (up, down, lat, lat, lat, lat)
    return GeneralSimConfig(
        name=f"w11d_sphere27_D{D}",
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


class MemSampler:
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
                    ["nvidia-smi", "-i", self.physical_gpu,
                     "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
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


def check_against_archive(D, domain_lu, tau, steps, re_eff, ulb):
    """Machine cross-check of tier parameters vs pulled archive (no hand-copy)."""
    d = os.environ.get("W27_ARCHIVE_DIR")
    if not d or D not in (40, 60):
        return None
    tier = "t1_d40big_12k.json" if D == 40 else "t2_d60big_24k.json"
    p = os.path.join(d, tier)
    if not os.path.exists(p):
        return None
    meta = json.load(open(p))["meta"]
    a_dom = meta["domain_lu"]
    checks = {
        "domain_lu": a_dom == domain_lu,
        "tau": abs(meta["tau"] - tau) < 1e-12,
        "steps": meta["steps"] == steps,
        "u_lb": abs(meta["u_lb"] - ulb) < 1e-12,
        "re_eff": abs(meta["re_eff"] - re_eff) < 1e-9,
        "lat": abs(meta["lat"] - (domain_lu[0] / D - 1) / 2) < 1e-12,
    }
    if not all(checks.values()):
        raise RuntimeError(f"archive mismatch for D{D}: {checks}")
    return {"archive_file": tier, "checks": checks}


def run_case(out_path, D, steps, lat, up, down, tau, ulb, sample, cv_tail=0, mem_stats=False):
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

    arch = check_against_archive(D, [nz, ny, nx], tau, steps, re_eff, ulb)

    # guard: engine lattice-u conversion must equal the protocol u_lb
    f19 = engine.f
    from tensorlbm.d3q19 import macroscopic3d as mac19

    _, ux19, _, _ = mac19(f19)
    ux_in_mean = float(ux19[~solid].mean().item())
    del f19, ux19
    engine.f = None
    torch.cuda.empty_cache()
    if abs(ux_in_mean - ulb) > 1e-6:
        raise RuntimeError(f"engine init ux {ux_in_mean} != protocol u_lb {ulb}")

    sampler = MemSampler(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) if mem_stats else None
    if sampler is not None:
        sampler.__enter__()

    bfl_mask, bfl_q = compute_q_sphere_common(nx, ny, nz, cx_lb, cy_lb, cz_lb, R_lb, dev, lattice="D3Q27")
    n_links = int(bfl_mask[1:].sum().item())
    q_active = bfl_q[1:][bfl_mask[1:]]
    links = bfl_boundary_link_indices(bfl_mask, bfl_q, lattice="D3Q27")
    if links.n_links != n_links or links.n_wrapped != 0:
        raise RuntimeError(
            f"link precompute mismatch: {links.n_links} vs {n_links}, wrapped={links.n_wrapped}"
        )
    q_min = float(links.link_q.min().item())
    q_max = float(links.link_q.max().item())
    del bfl_mask, bfl_q, q_active
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # init: equilibrium27 at (rho=1, ux=u_lb, ux=0 on solid) -- archived init state
    f = torch.empty((27, nz, ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full((nz, ny, nx), float(ulb), dtype=torch.float32, device=dev)
    ux0[solid] = 0.0
    feq27_perq_into(
        f, torch.ones((nz, ny, nx), dtype=torch.float32, device=dev), ux0,
        torch.zeros_like(ux0), torch.zeros_like(ux0),
    )
    del ux0
    target_mass = float(f.sum().item())
    initial_mass = target_mass

    hist = []
    finite = True
    diverged_at = None
    sample_set = set(range(sample, steps + 1, sample))
    solid_idx = solid.reshape(-1).nonzero(as_tuple=True)[0]
    t0 = time.time()
    PROGRESS["phase"] = "loop"
    for step in range(1, steps + 1):
        PROGRESS["step"] = step
        f_solid_pre = f.reshape(27, -1)[:, solid_idx]
        f = collide27_inplace(f, tau)
        f.reshape(27, -1)[:, solid_idx] = f_solid_pre
        del f_solid_pre
        f = bb27_solid_scatter(f, solid)
        f_pre_stream = f.clone()
        f = stream27_roll(f)
        f = far_field27_inplace(f, ulb)
        force3, _ = bfl_force_ledger_sparse(f, f_pre_stream, links)
        f = bfl_bounce_back_sparse(f, f_pre_stream, links)
        if step in sample_set:
            cd = float(force3[0].item()) / dpS
            cl = float(force3[1].item()) / dpS
            cs = float(force3[2].item()) / dpS
            entry = {"step": step, "cd": cd, "cl": cl, "cs": cs,
                     "mass": float(f.sum().item())}
            if cv_tail > 0 and step > steps - cv_tail:
                cv, _ = cv_box_force_exact_27(f_pre_stream, solid)
                entry["cv_mom_cd"] = cv[0] / dpS
                entry["cv_mom_cl"] = cv[1] / dpS
            hist.append(entry)
        del f_pre_stream
        if step % 200 == 0:
            f = correct_mass27_inplace(f, target_mass)
        if step % 500 == 0 and not bool(torch.isfinite(f).all()):
            finite = False
            diverged_at = step
            break
        if step % 2000 == 0:
            el = time.time() - t0
            print(f"step {step}/{steps} ({el:.0f}s, {el / step * 1000:.0f} ms/step)", flush=True)

    peak_loop = float(torch.cuda.max_memory_allocated()) / 2**30 if (mem_stats and dev.type == "cuda") else None
    if sampler is not None:
        sampler.__exit__()

    meta = {
        "case": "sphere_re100_d3q27",
        "lattice": "D3Q27",
        "collision": "bgk27",
        "kernel": "sparse",
        "route": "bfl",
        "treat": "hard",
        "mirrored_impl": ["feq27_perq_into", "collide27_inplace", "bb27_solid_scatter",
                          "far_field27_inplace", "correct_mass27_inplace"],
        "mirrored_ab_evidence": "diag/ab27.json (torch.equal P1-P5)",
        "D": D,
        "lat": lat,
        "up": up,
        "down": down,
        "tau": tau,
        "u_lb": ulb,
        "nu_lb": nu_lb,
        "re_eff": re_eff,
        "steps": steps if finite else diverged_at,
        "steps_requested": steps,
        "diverged": not finite,
        "domain_lu": [nz, ny, nx],
        "dpS": dpS,
        "R_lb": R_lb,
        "center_lu": [cz_lb, cy_lb, cx_lb],
        "blockage": math.pi * R_lb**2 / (ny * nz),
        "cd_ref_at_re_eff": cd_ref_family(re_eff),
        "engine_init_ux_check": ux_in_mean,
        "archive_twin_check": arch,
        "mass_drift_ppm": (
            (float(f.sum().item()) - initial_mass) / initial_mass * 1e6 if finite else None
        ),
        "bfl_links": n_links,
        "q_min": q_min,
        "q_max": q_max,
        "stream": "roll27",
        "force_frame": "laboratory",
        "wall_s": time.time() - t0,
    }
    if mem_stats and dev.type == "cuda":
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
    ap.add_argument("--D", type=int, default=40)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lat", type=float, default=3.0)
    ap.add_argument("--up", type=float, default=3.0)
    ap.add_argument("--down", type=float, default=4.0)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--ulb", type=float, default=0.05)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--cv-tail", type=int, default=0)
    ap.add_argument("--mem-stats", action="store_true")
    args = ap.parse_args()
    if args.steps is None:
        args.steps = 12000 if args.D == 40 else 24000
    if args.tau is None:
        args.tau = 0.56 if args.D == 40 else 0.59
    PROGRESS["phase"] = "run"
    try:
        run_case(args.out, args.D, args.steps, args.lat, args.up, args.down,
                 args.tau, args.ulb, args.sample, args.cv_tail, args.mem_stats)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower() and not isinstance(
            exc, torch.cuda.OutOfMemoryError
        ):
            raise
        import traceback

        rec = {
            "oom": True,
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
