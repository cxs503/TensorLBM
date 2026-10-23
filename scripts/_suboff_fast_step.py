#!/usr/bin/env python3
"""Experimental optimized step for suboff_re1000 on SDAA.

Variants (cumulative):
  baseline : orig int64-gather stream + orig NoDynamics/BB + isfinite/step
  +cat     : streaming via concatenation shifts (bit-exact)
  +fuse    : NoDynamics + half-way BB fused into one torch.where (bit-exact)
  +fin50   : divergence guard every 50 steps (behaviour change, annotated)
  +fmrt    : single-gemm MRT (R = Minv diag(s) M)  [NOT bit-exact]

Bit-exactness of each variant vs baseline is measured (max|diff| after N steps).
"""
from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

REPO = Path("/root/TensorLBM_feat2")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks/pending/suboff_re1000"))

import torch  # noqa: E402
import torch_sdaa  # noqa: E402,F401

from tensorlbm import solver3d as s3  # noqa: E402
from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.lbm_step_correct import lbm_step_correct  # noqa: E402
from tensorlbm.solver3d import _get_d3q19_mrt_matrices, _mrt3d_s_vec  # noqa: E402

_ORIG_STREAM3D = s3.stream3d
_SHIFTS = []
_OPP = None


def build_shifts(device):
    global _SHIFTS
    c = C.to(torch.device(device))
    _SHIFTS = [(int(c[q, 0]), int(c[q, 1]), int(c[q, 2])) for q in range(19)]


def _rollcat(d, k, dim):
    n = d.shape[dim]
    k %= n
    if k == 0:
        return d
    pre = [slice(None)] * dim
    post = [slice(None)] * (d.dim() - dim - 1)
    if k > 0:
        a = d[tuple(pre + [slice(n - k, n)] + post)]
        b = d[tuple(pre + [slice(0, n - k)] + post)]
    else:
        a = d[tuple(pre + [slice(-k, n)] + post)]
        b = d[tuple(pre + [slice(0, -k)] + post)]
    return torch.cat([a, b], dim=dim)


def stream3d_cat(f):
    Q = f.shape[0]
    out = torch.empty_like(f)
    out[0].copy_(f[0])
    for q in range(1, Q):
        sx, sy, sz = _SHIFTS[q]
        d = f[q]
        if sz:
            d = _rollcat(d, sz, 0)
        if sy:
            d = _rollcat(d, sy, 1)
        if sx:
            d = _rollcat(d, sx, 2)
        out[q] = d
    return out


class FusedMRT:
    def __init__(self, tau, s_e=1.19, s_eps=1.4, s_q=1.2):
        self.tau, self.s_e, self.s_eps, self.s_q = tau, s_e, s_eps, s_q
        self._R = None

    def R(self, f):
        if self._R is None or self._R.device != f.device:
            M, Minv = _get_d3q19_mrt_matrices(f.device, f.dtype)
            s_vec = _mrt3d_s_vec(self.s_e, self.s_eps, self.s_q, self.s_e,
                                 1.0 / self.tau, dtype=f.dtype, device=f.device)
            self._R = Minv @ (s_vec.unsqueeze(1) * M)
        return self._R

    def __call__(self, f, tau=None, **kw):
        R = self.R(f)
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        ff = f.reshape(19, -1)
        diff = (f - feq).reshape(19, -1)
        return (ff - R @ diff).reshape(f.shape)


def build(L, device):
    import run as runmod  # type: ignore

    from tensorlbm.general_sim import (
        CollisionModel, ForceMethod, GeneralSimConfig, GeneralSimEngine,
        GeometryConfig, GeometrySource, LatticeModel, OutputConfig,
        PhysicsConfig, SolverConfig, WallTreatment,
    )

    build_shifts(device)
    viscosity = runmod.U_PHYS * runmod.SUBOFF_LENGTH_M / 1000.0
    out_dir = Path(f"/root/prof_out/fast_L{L}")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = GeneralSimConfig(
        name=f"fast_suboff_L{L}",
        geometry=GeometryConfig(source=GeometrySource.PARAMETRIC_SUBOFF,
                                suboff_length=runmod.SUBOFF_LENGTH_M,
                                suboff_radius=runmod.SUBOFF_RADIUS_M),
        physics=PhysicsConfig(density=1000.0, viscosity=viscosity,
                              inlet_velocity=runmod.U_PHYS,
                              reference_length=runmod.SUBOFF_LENGTH_M),
        solver=SolverConfig(lattice=LatticeModel.D3Q19, collision=CollisionModel.MRT,
                            resolution=L, domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
                            max_steps=100000, warmup_steps=None,
                            snapshot_interval=10_000_000, force_sample_interval=10,
                            device=device, wall_treatment=WallTreatment.AUTO,
                            force_method=ForceMethod.PRESSURE_FRICTION,
                            pressure_extrap="none", p0_method="near_wall",
                            friction_formula="standard", mass_correction=True,
                            mass_correction_interval=200, smagorinsky_cs=0.05),
        output=OutputConfig(directory=str(out_dir), formats=[],
                            save_macroscopic=False, save_forces=True),
    )
    eng = GeneralSimEngine(config)
    eng.setup()
    return eng


def make_step(collide_fn, fuse, collide_kwargs):
    def step(f, tau, solid, u_in, far_field_fn, cm_fn, target_mass, step_no, mi, **ck):
        global _OPP
        if _OPP is None:
            _OPP = OPPOSITE.to(f.device)
        if fuse:
            f_pre_opp = f[_OPP].clone()
            f = collide_fn(f, tau=tau, **collide_kwargs)
            f = torch.where(solid.unsqueeze(0), f_pre_opp, f)
            f = s3.stream3d(f)
        else:
            f = lbm_step_correct(f, collide_fn, tau, solid, u_in, far_field_fn,
                                 correct_mass_fn=None, target_mass=None,
                                 step=step_no, wall_treatment="bb", **collide_kwargs)
        if fuse:
            f = far_field_fn(f, u_in)
        if cm_fn is not None and target_mass is not None and step_no % mi == 0:
            f = cm_fn(f, target_mass)
        return f

    return step


def run_loop(eng, nsteps, step_fn, fin_int, sample=True, collide_fn=None, ckw=None):
    sol = eng.config.solver
    tau, nu_lb, u_in = eng.uc.tau, eng.uc.nu_lb, eng.uc.u_lb
    if collide_fn is None:
        collide_fn, ckw = eng._get_collide_fn()
    from tensorlbm import boundaries3d as b3
    far_field_fn = functools.partial(b3.far_field_bc_3d, bc_config=eng._build_bc_config())
    from tensorlbm.solver3d import correct_mass3d
    cm_fn = correct_mass3d if sol.mass_correction else None
    dpS = eng._compute_dpS()
    solid = eng.solid if eng.solid is not None else torch.zeros_like(eng.f[0], dtype=torch.bool)
    for i in range(1, nsteps + 1):
        eng.step_count += 1
        eng.f = step_fn(eng.f, tau, solid, u_in, far_field_fn, cm_fn,
                        eng._initial_mass, eng.step_count,
                        sol.mass_correction_interval, **(ckw or {}))
        if sample and eng.config.output.save_forces and i % sol.force_sample_interval == 0:
            eng._sample_forces(dpS, nu_lb)
        if fin_int and i % fin_int == 0:
            if not torch.isfinite(eng.f).all():
                break
    return eng.f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--verify-steps", type=int, default=3)
    ap.add_argument("--profile", action="store_true", help="profile last variant")
    args = ap.parse_args()

    eng = build(args.resolution, args.device)
    torch.sdaa.synchronize()
    f0 = eng.f.clone()
    sc = eng.step_count

    def reset():
        eng.f = f0.clone()
        eng.step_count = sc

    print(f"=== L={args.resolution} f.shape={tuple(eng.f.shape)} "
          f"cells={eng.f[0].numel()} solid={int(eng.solid.sum())} ===", flush=True)

    base_cf, base_ckw = eng._get_collide_fn()
    base_step = make_step(base_cf, False, base_ckw)

    def timeit(stepfn, fin_int):
        reset()
        s3.stream3d = _ORIG_STREAM3D
        run_loop(eng, 2, stepfn, 0, sample=False)
        torch.sdaa.synchronize()
        reset()
        t = time.time()
        run_loop(eng, args.steps, stepfn, fin_int, sample=True)
        torch.sdaa.synchronize()
        return (time.time() - t) / args.steps * 1000

    # baseline field for bit-exactness
    reset()
    s3.stream3d = _ORIG_STREAM3D
    run_loop(eng, args.verify_steps, base_step, 0, sample=False)
    torch.sdaa.synchronize()
    f_base = eng.f.clone()

    results = {}
    base_ms = timeit(base_step, 1)
    results["baseline (gather stream)"] = base_ms
    print(f"{'baseline':36s} {base_ms:8.1f} ms/step", flush=True)

    cf_orig, ckw_orig = eng._get_collide_fn()

    def variant(label, use_cat, fuse, fin_int, coll):
        # choose collision fn/kwargs
        if coll == "fused":
            cfn, ckw = FusedMRT(eng.uc.tau), {}
        else:
            cfn, ckw = cf_orig, ckw_orig
        st = make_step(cfn, fuse, ckw)
        # patch stream
        s3.stream3d = stream3d_cat if use_cat else _ORIG_STREAM3D
        ms = timeit(st, fin_int)
        reset()
        s3.stream3d = stream3d_cat if use_cat else _ORIG_STREAM3D
        run_loop(eng, args.verify_steps, st, 0, sample=False,
                 collide_fn=cfn, ckw=ckw)
        torch.sdaa.synchronize()
        same = torch.equal(eng.f, f_base)
        md = (eng.f - f_base).abs().max().item()
        results[label] = ms
        print(f"{label:36s} {ms:8.1f} ms/step  speedup={base_ms/ms:5.2f}x  "
              f"bitexact={same} maxdiff={md:.2e}", flush=True)
        s3.stream3d = _ORIG_STREAM3D

    variant("+cat stream", True, False, 1, "orig")
    variant("+cat +fused nodyn/bb", True, True, 1, "orig")
    variant("+cat +fused +isfinite/50", True, True, 50, "orig")
    variant("+cat +fused +isfinite/50 +fmrt", True, True, 50, "fused")

    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        cfn, ckw = FusedMRT(eng.uc.tau), {}
        st = make_step(cfn, True, ckw)
        s3.stream3d = stream3d_cat
        reset()
        run_loop(eng, 2, st, 0, sample=False, collide_fn=cfn, ckw=ckw)
        torch.sdaa.synchronize()
        reset()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.SDAA],
                     record_shapes=True) as prof:
            run_loop(eng, 5, st, 50, sample=False, collide_fn=cfn, ckw=ckw)
            torch.sdaa.synchronize()
        print("\n=== OPTIMIZED: TOP SDAA ops ===", flush=True)
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=25),
              flush=True)
        print("\n=== OPTIMIZED: TOP CPU ops ===", flush=True)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25),
              flush=True)
        s3.stream3d = _ORIG_STREAM3D

    print(f"\n=== SUMMARY L={args.resolution} ===", flush=True)
    for k, v in results.items():
        print(f"  {k:36s} {v:8.1f} ms/step  speedup={base_ms/v:5.2f}x", flush=True)


if __name__ == "__main__":
    main()