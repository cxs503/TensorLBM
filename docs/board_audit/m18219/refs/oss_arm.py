#!/usr/bin/env python3
"""OSS arm — live execution of lbmpy's two-component Shan-Chen implementation.

Provenance (zero hand-copy, source-read)
----------------------------------------
Upstream: **lbmpy 2.0** (lssfau/lbmpy), sdist ``lbmpy-2.0.tar.gz`` fetched from the
site pip mirror (``pip download --no-deps --no-binary :all: lbmpy``), sha256
``c29aec635e2b29ef33ed716852d30e19fb027d2b704717e62ac0998b4120dca9``.
Reference file: ``tests/shan_chen/test_shan_chen_two_component.py``
(sha256 ``7e88e89825a34a56e79723fa670a1114926d6282766bf98b918849577fa63415``,
GPL-3.0, vendored verbatim as ``oss_reference/test_shan_chen_two_component.py``).
The code below is that test's driver, unchanged in structure and algebra; the only
edits are:

  E1  parameters (``N``, ``g_ab``, ``steps``, ``omega``, pseudopotential) are
      arguments instead of literals  (upstream ``:6-22``, ``:55-59``, ``:73-81``);
  E2  ``psi_mode`` selects the pseudopotential map: ``"exp"`` = upstream
      ``rho0*(1 - exp(-dens/rho0))`` with ``rho0 = 1``  (``:53-54``),
      ``"linear"`` = ``psi = rho`` (Palabos ``PsiIsRho`` convention, used by the
      TensorLBM MCMP kernel; needed for the operator-identity arm);
  E3  the converged state, the maxima and the measured profile are written to
      ``.npz`` instead of being compared to ``np.allclose`` only  (``:139-166``);
  E4  the upstream paper-reference profile (``:139-166``, generated from
      lbm-principles-practice/code ``chapter9/shanchen.cpp``) is kept verbatim and
      returned, so the faithfulness of *this* arm can be asserted by the caller.

Everything else -- the explicit D2Q9 stencil force
``F_a = -sum_b g_ab psi(rho_a) * sum_d w_d psi(rho_b[+d]) d``  (``:55-59``),
the barycentric velocity with the half-force correction  (``:73-81``),
``force_model=ForceModel.GUO``  (``:85-95``), the stream-pull kernels and the
collide/stream/sync loop  (``:112-153``) -- is the upstream implementation.

Run:  PYTHONPATH=<lbmpy sdist>/src python oss_arm.py --out artifacts/oss_cases.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import sympy as sp

import pystencils as ps
import lbmpy
from lbmpy.enums import Stencil
from lbmpy import LBMConfig, ForceModel, create_lb_update_rule
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
from lbmpy.creationfunctions import create_stream_pull_with_output_kernel

# upstream `:139-166`: reference profile of the lbm-principles-practice C++ code
# (chapter9/shanchen.cpp, nsteps=1000, nfluids=2, gA=0), N=64, g_ab=6, psi=exp.
REF_A = np.array([0.213948, 0.0816724, 0.0516763, 0.0470179, 0.0480882, 0.0504771,
                  0.0531983, 0.0560094, 0.0588071, 0.0615311, 0.064102, 0.0664467,
                  0.0684708, 0.070091, 0.0712222, 0.0718055, 0.0718055, 0.0712222,
                  0.070091, 0.0684708, 0.0664467, 0.064102, 0.0615311, 0.0588071,
                  0.0560094, 0.0531983, 0.0504771, 0.0480882, 0.0470179, 0.0516763,
                  0.0816724, 0.213948, 0.517153, 0.833334, 0.982884, 1.0151, 1.01361,
                  1.0043, 0.993178, 0.981793, 0.970546, 0.959798, 0.949751, 0.940746,
                  0.933035, 0.926947, 0.922713, 0.920548, 0.920548, 0.922713, 0.926947,
                  0.933035, 0.940746, 0.949751, 0.959798, 0.970546, 0.981793, 0.993178,
                  1.0043, 1.01361, 1.0151, 0.982884, 0.833334, 0.517153])

PSI_MODES = {
    # upstream `:53-54`
    "exp": lambda dens: 1.0 * (1.0 - sp.exp(-dens / 1.0)),
    # Palabos PsiIsRho / TensorLBM MCMP convention
    "linear": lambda dens: dens,
}


def build_kernels(N: int, g_ab: float, psi_mode: str, omega: float):
    """Assemble the upstream MCMP collision/stream/force kernels (no run)."""
    psi = PSI_MODES[psi_mode]

    stencil = lbmpy.LBStencil(Stencil.D2Q9)
    weights = stencil.weights
    dim = stencil.D

    dh = ps.create_data_handling((N,) * dim, periodicity=True,
                                 default_target=ps.Target.CPU)

    src_a = dh.add_array("src_a", values_per_cell=stencil.Q)
    dst_a = dh.add_array_like("dst_a", "src_a")
    src_b = dh.add_array("src_b", values_per_cell=stencil.Q)
    dst_b = dh.add_array_like("dst_b", "src_b")

    rho_a = dh.add_array("rho_a")
    rho_b = dh.add_array("rho_b")
    u_a = dh.add_array("u_a", values_per_cell=stencil.D)
    u_b = dh.add_array("u_b", values_per_cell=stencil.D)
    u_bary = dh.add_array_like("u_bary", u_a.name)
    f_a = dh.add_array("f_a", values_per_cell=stencil.D)
    f_b = dh.add_array_like("f_b", f_a.name)

    zero_vec = sp.Matrix([0] * stencil.D)

    # upstream `:55-59` (component a) and `:73-81` (component b).
    # g_aa = g_bb = 0 (only the cross term is active), g_ab = g_ba.
    force_a = zero_vec
    for factor, rho in zip([0.0, g_ab], [rho_a, rho_b]):
        force_a += sum((psi(rho[d]) * w_d * sp.Matrix(d)
                        for d, w_d in zip(stencil, weights)), zero_vec) * psi(rho_a.center) * -1 * factor

    force_b = zero_vec
    for factor, rho in zip([g_ab, 0.0], [rho_a, rho_b]):
        force_b += sum((psi(rho[d]) * w_d * sp.Matrix(d)
                        for d, w_d in zip(stencil, weights)), zero_vec) * psi(rho_b.center) * -1 * factor

    f_expressions = ps.AssignmentCollection([
        ps.Assignment(f_a.center_vector, force_a),
        ps.Assignment(f_b.center_vector, force_b),
    ])

    # upstream `:73-81`: barycentric velocity, then the +F/2 correction
    u_temp = ps.AssignmentCollection(ps.Assignment(
        u_bary.center_vector,
        (rho_a.center * u_a.center_vector - f_a.center_vector / 2
         + rho_b.center * u_b.center_vector - f_b.center_vector / 2)
        / (rho_a.center + rho_b.center)))

    u_corr = ps.AssignmentCollection(ps.Assignment(
        u_bary.center_vector,
        u_bary.center_vector
        + (f_a.center_vector / 2 + f_b.center_vector / 2) / (rho_a.center + rho_b.center)))

    # upstream `:85-100`: force_model = GUO (Silva 2020 moment-space Guo forcing)
    lbm_config_a = LBMConfig(stencil=stencil, relaxation_rate=omega, compressible=True,
                             velocity_input=u_bary, density_input=rho_a,
                             force_model=ForceModel.GUO, force=f_a, kernel_type="collide_only")
    lbm_config_b = LBMConfig(stencil=stencil, relaxation_rate=omega, compressible=True,
                             velocity_input=u_bary, density_input=rho_b,
                             force_model=ForceModel.GUO, force=f_b, kernel_type="collide_only")

    collision_a = create_lb_update_rule(lbm_config=lbm_config_a,
                                        optimization={"symbolic_field": src_a})
    collision_b = create_lb_update_rule(lbm_config=lbm_config_b,
                                        optimization={"symbolic_field": src_b})

    stream_a = create_stream_pull_with_output_kernel(collision_a.method, src_a, dst_a,
                                                     {"density": rho_a, "velocity": u_a})
    stream_b = create_stream_pull_with_output_kernel(collision_b.method, src_b, dst_b,
                                                     {"density": rho_b, "velocity": u_b})

    cfg = ps.CreateKernelConfig(target=dh.default_target)
    kernels = {
        "stream_a": ps.create_kernel(stream_a, config=cfg).compile(),
        "stream_b": ps.create_kernel(stream_b, config=cfg).compile(),
        "collision_a": ps.create_kernel(collision_a, config=cfg).compile(),
        "collision_b": ps.create_kernel(collision_b, config=cfg).compile(),
        "force": ps.create_kernel(f_expressions, config=cfg).compile(),
        "u_temp": ps.create_kernel(u_temp, config=cfg).compile(),
        "u_corr": ps.create_kernel(u_corr, config=cfg).compile(),
        "init_a": ps.create_kernel(macroscopic_values_setter(
            collision_a.method, velocity=(0, 0),
            pdfs=src_a.center_vector, density=rho_a.center), ghost_layers=0).compile(),
        "init_b": ps.create_kernel(macroscopic_values_setter(
            collision_b.method, velocity=(0, 0),
            pdfs=src_b.center_vector, density=rho_b.center), ghost_layers=0).compile(),
    }
    names = dict(src_a=src_a.name, dst_a=dst_a.name, src_b=src_b.name, dst_b=dst_b.name,
                 rho_a=rho_a.name, rho_b=rho_b.name, u_a=u_a.name, u_b=u_b.name,
                 u_bary=u_bary.name, f_a=f_a.name, f_b=f_b.name)
    return dh, kernels, names


def run(N: int = 64, g_ab: float = 6.0, psi_mode: str = "exp", steps: int = 1000,
        omega: float = 1.0) -> dict:
    """Run the upstream spinodal-decomposition case; return the measured state."""
    t0 = time.time()
    dh, k, nm = build_kernels(N, g_ab, psi_mode, omega)
    build_s = time.time() - t0

    sync_pdfs = dh.synchronization_function([nm["src_a"], nm["src_b"]])
    sync_rhos = dh.synchronization_function([nm["rho_a"], nm["rho_b"]])

    # upstream `:122-126`: two-slab initial condition (interface normal to axis 0)
    dh.fill(nm["rho_a"], 0.1, slice_obj=ps.make_slice[:, :0.5])
    dh.fill(nm["rho_a"], 0.9, slice_obj=ps.make_slice[:, 0.5:])
    dh.fill(nm["rho_b"], 0.9, slice_obj=ps.make_slice[:, :0.5])
    dh.fill(nm["rho_b"], 0.1, slice_obj=ps.make_slice[:, 0.5:])
    dh.fill(nm["u_a"], 0.0)
    dh.fill(nm["u_b"], 0.0)
    dh.fill(nm["f_a"], 0.0)
    dh.fill(nm["f_b"], 0.0)
    dh.run_kernel(k["u_temp"])
    dh.run_kernel(k["init_a"])
    dh.run_kernel(k["init_b"])

    t1 = time.time()
    for _ in range(steps):
        sync_rhos()
        dh.run_kernel(k["force"])
        dh.run_kernel(k["u_corr"])
        dh.run_kernel(k["collision_a"])
        dh.run_kernel(k["collision_b"])
        sync_pdfs()
        dh.run_kernel(k["stream_a"])
        dh.run_kernel(k["stream_b"])
        dh.run_kernel(k["u_temp"])
        dh.swap(nm["src_a"], nm["dst_a"])
        dh.swap(nm["src_b"], nm["dst_b"])
    solve_s = time.time() - t1

    rho_a = np.array(dh.gather_array(nm["rho_a"]))
    rho_b = np.array(dh.gather_array(nm["rho_b"]))
    u_bary = np.array(dh.gather_array(nm["u_bary"]))
    # upstream `:167`: profile along the split axis, taken at the first index
    return dict(
        N=N, g_ab=g_ab, psi_mode=psi_mode, steps=steps, omega=omega,
        prof_a=rho_a[0, :].copy(), prof_b=rho_b[0, :].copy(),
        rho_a=rho_a.transpose(), rho_b=rho_b.transpose(),
        umax=float(np.max(np.abs(u_bary))),
        wall_s_build=build_s, wall_s_solve=solve_s,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/oss_cases.npz")
    ap.add_argument("--cases", default='[{"N":64,"g_ab":6.0,"psi_mode":"exp"},'
                                       '{"N":128,"g_ab":6.0,"psi_mode":"exp"},'
                                       '{"N":64,"g_ab":5.0,"psi_mode":"exp"},'
                                       '{"N":64,"g_ab":6.0,"psi_mode":"linear"}]')
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()

    cases = json.loads(args.cases)
    payload, meta = {}, []
    for c in cases:
        tag = "N%d_g%g_%s" % (c["N"], c["g_ab"], c["psi_mode"])
        res = run(N=c["N"], g_ab=c["g_ab"], psi_mode=c["psi_mode"], steps=args.steps)
        for key in ("prof_a", "prof_b", "rho_a", "rho_b"):
            payload[tag + "_" + key] = res[key]
        rec = {k: v for k, v in res.items() if k not in ("rho_a", "rho_b", "prof_a", "prof_b")}
        if c["psi_mode"] == "exp" and c["N"] == 64 and c["g_ab"] == 6.0 and args.steps == 1000:
            # faithfulness check of this arm against the upstream paper reference
            rec["ref_profile_absdiff"] = float(np.max(np.abs(res["prof_a"] - REF_A)))
            rec["ref_profile_rel"] = float(np.max(np.abs(res["prof_a"] - REF_A))
                                           / (REF_A.max() - REF_A.min()))
        meta.append(rec)
        print(tag, json.dumps(rec), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **payload)
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as fh:
        json.dump({"lbmpy_version": lbmpy.__version__, "cases": meta}, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()