#!/usr/bin/env python3
"""XVAL-LBM-POINTWISE-REFERENCE-REGRESSION driver.

Runs the predeclared arms (see ``predeclared.json``) and writes ``summary.json``,
``table.csv`` and per-arm raw metrics under ``runs/``.

Usage
-----
    PYTHONPATH=src python xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/run_xval.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import lbmpy_pointwise_ref as ref  # noqa: E402

TAUS = [0.51, 0.55, 0.7, 1.0, 3.0]
AMPLITUDES = [1e-5, 1e-3, 0.05, 0.15]
SEEDS = [0, 1, 2, 3, 4]
MACH = {a: 3.0 * a / np.sqrt(1.0 / 3.0) for a in AMPLITUDES}  # max|u|/cs estimate, kept symbolic

GATE_FP64 = 1e-10
GATE_D3Q19 = 1e-6
GATE_CONTROL = 1e-6


# ---------------------------------------------------------------------------
# kernels under test (TensorLBM side)
# ---------------------------------------------------------------------------
def kernel_fns():
    from tensorlbm import solver
    from tensorlbm.collision_d3q19_advanced import collide_bgk_d3q19

    return {
        "d2q9_collide_bgk": solver.collide_bgk,
        "d2q9_collide_bgk_matmul": solver.collide_bgk_matmul,
        "d2q9_collide_trt": solver.collide_trt,
        "d2q9_collide_mrt": solver.collide_mrt,
        "d2q9_collide_rlbm": solver.collide_rlbm,
        "d3q19_collide_bgk": collide_bgk_d3q19,
    }


def _torch(f):
    import torch

    return torch.from_numpy(np.ascontiguousarray(f))


def run_kernel(kernel, f, *, tau=0.7, s_e=1.64, s_eps=1.54, s_q=1.7, lam=3.0 / 16.0):
    """Call a TensorLBM kernel on a numpy field, return numpy in the same order."""
    t = _torch(f)
    name = kernel.__name__
    if name in {"collide_bgk", "collide_bgk_matmul", "collide_rlbm", "collide_bgk_d3q19"}:
        out = kernel(t, tau)
    elif name == "collide_trt":
        out = kernel(t, tau, lam)
    elif name == "collide_mrt":
        out = kernel(t, tau, s_e=s_e, s_eps=s_eps, s_q=s_q)
    else:  # pragma: no cover - defensive
        raise ValueError(f"no dispatch for {name}")
    return out.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def arm_records():
    """(arm_id, kernel_key, reference_spec) tuples -- the predeclared matrix."""
    recs = []
    for tau in TAUS:
        recs.append(("A1_srt", "d2q9_collide_bgk", dict(lattice="D2Q9", model="srt", tau=tau)))
        recs.append(("A2_srt_matmul", "d2q9_collide_bgk_matmul", dict(lattice="D2Q9", model="srt", tau=tau)))
        recs.append(("A3_trt", "d2q9_collide_trt", dict(lattice="D2Q9", model="trt", tau_plus=tau)))
        recs.append((
            "A5_mrt", "d2q9_collide_mrt",
            dict(lattice="D2Q9", model="mrt_from_matrix", moment_matrix=REPO_MRT, tau=tau,
                 relaxation_rates=[1.0, 1.64, 1.54, 1.0, 1.7, 1.0, 1.7, 1.0 / tau, 1.0 / tau]),
        ))
        recs.append((
            "A4_mrt_degenerate", "d2q9_collide_mrt",
            dict(lattice="D2Q9", model="mrt_from_matrix", moment_matrix=REPO_MRT, tau=tau,
                 relaxation_rates=[1.0 / tau] * 9),
        ))
    for tau in (0.7, 1.0):
        recs.append(("A6_d3q19_srt", "d3q19_collide_bgk", dict(lattice="D3Q19", model="srt", tau=tau)))
        # added after the first pass: lbmpy's *discrete* hydrodynamic Maxwellian,
        # to separate an equilibrium-convention gap from a kernel defect
        recs.append(("A6b_d3q19_srt_discrete_eq", "d3q19_collide_bgk",
                     dict(lattice="D3Q19", model="srt", tau=tau, continuous_equilibrium=False)))
    return recs


from tensorlbm.solver import _M_D2Q9_DATA  # noqa: E402

REPO_MRT = np.array(_M_D2Q9_DATA, dtype=float)


def kernel_kwargs(arm_id, tau, ref_spec):
    if arm_id == "A4_mrt_degenerate":
        return dict(tau=tau, s_e=1.0 / tau, s_eps=1.0 / tau, s_q=1.0 / tau)
    return dict(tau=tau)


def lattice_of(ref_spec):
    return ref_spec["lattice"]


def tau_of(ref_spec):
    return ref_spec.get("tau", ref_spec.get("tau_plus", 1.0))


_REF_CACHE: dict[str, "ref.PointwiseReference"] = {}


def _reference(ref_spec):
    key = json.dumps(ref_spec, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if key not in _REF_CACHE:
        _REF_CACHE[key] = ref.PointwiseReference(**ref_spec)
    return _REF_CACHE[key]


def evaluate_arm(arm_id, kernel_key, ref_spec, kernels, *, amplitude, seed, dtype=np.float64):
    lat = lattice_of(ref_spec)
    f = ref.make_test_field(lat, tau=tau_of(ref_spec), seed=seed, amplitude=amplitude, dtype=dtype)
    reference = _reference(ref_spec)
    f_ref = reference.apply(f)
    tau = tau_of(ref_spec)
    got = run_kernel(kernels[kernel_key], f, **kernel_kwargs(arm_id, tau, ref_spec))
    m = ref.compare(got, f_ref)
    mom_ref = ref.moments(f_ref, lat)
    mom_got = ref.moments(got, lat)
    m["moment_err_rho"] = float(np.abs(mom_got["rho"] - mom_ref["rho"]).max() / np.abs(mom_ref["rho"]).mean())
    m["moment_err_j"] = float(np.abs(mom_got["j"] - mom_ref["j"]).max() / max(np.abs(mom_ref["j"]).max(), 1e-300))
    m["conservation_residual"] = float(
        abs(got.sum() - f.sum()) / max(abs(f.mean()), 1e-300)
    )
    return m, f, f_ref, got


# ---------------------------------------------------------------------------
# positive controls: these MUST be detected, otherwise the harness is vacuous
# ---------------------------------------------------------------------------
def control_unpermuted(ref_spec, f):
    """C1: evaluate the reference on the *unpermuted* field -> must blow up.

    This proves the direction permutation is load-bearing: if it were dropped
    the comparison would fail by O(1), i.e. the agreement in A1..A5 is not an
    artefact of a comparison that is insensitive to direction order.
    """
    from tensorlbm import solver

    reference = _reference(ref_spec)
    lb = np.asarray(f, dtype=np.float64)  # deliberately NOT permuted
    out = reference._func(*[lb[i] for i in range(lb.shape[0])])
    d_lb = np.stack([out[sym] for sym in reference._d_syms], axis=0).astype(np.float64)
    got = run_kernel(solver.collide_bgk, f, tau=tau_of(ref_spec))
    # compare in lbmpy's own index order, i.e. skip both permutations
    got_lb = np.empty_like(d_lb)
    got_lb[reference.perm] = got
    return ref.compare(got_lb, d_lb)["eps_scaled_inf"]


def control_tau_shift(kernels, ref_spec, f, *, rel=1e-6):
    """C2: reference at tau*(1+rel) vs kernel at tau -> must be ~rel."""
    shifted = dict(ref_spec)
    tau = tau_of(ref_spec)
    new_tau = tau * (1.0 + rel)
    if "tau" in shifted:
        shifted["tau"] = new_tau
    else:
        shifted["tau_plus"] = new_tau
    if ref_spec["model"] == "mrt_from_matrix":
        rates = list(ref_spec["relaxation_rates"])
        rates[-1] = rates[-2] = 1.0 / new_tau
        shifted["relaxation_rates"] = rates
    f_ref = _reference(shifted).apply(f)
    got = run_kernel(kernels["d2q9_collide_bgk" if "tau" in ref_spec else "d2q9_collide_trt"],
                     f, tau=tau)
    return ref.compare(got, f_ref)["eps_scaled_inf"]


# ---------------------------------------------------------------------------
# regularised BGK: from-definition reference (lbmpy has no D2Q9 counterpart)
# ---------------------------------------------------------------------------
def rlbm_from_definition(f, tau):
    """Independent numpy statement of the Latt & Chopard regularised-BGK update.

    Written from the definition (project the non-equilibrium part onto the
    second-order Hermite subspace, then relax), reusing *no* TensorLBM code
    except the lattice constants read from the package.
    """
    lat = "D2Q9"
    vel = ref.tensorlbm_velocities(lat).astype(np.float64)
    w = ref._weights(lat)
    ndim = 2
    shape = f.shape[1:]
    rho = f.sum(axis=0)
    j = np.tensordot(vel, f, axes=(0, 0))
    u = j / rho[None]
    wq = w.reshape((len(w),) + (1,) * ndim)
    cu = np.tensordot(vel, u, axes=(1, 0))
    u2 = (u ** 2).sum(axis=0)[None]
    feq = rho[None] * wq * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u2)
    fneq = f - feq
    pi = np.einsum("qa,qb,q...->ab...", vel, vel, fneq)  # (2,2,*spatial)
    cs2 = 1.0 / 3.0
    h = np.zeros((len(vel), ndim, ndim))
    for a in range(ndim):
        for b in range(ndim):
            h[:, a, b] = vel[:, a] * vel[:, b] - (cs2 if a == b else 0.0)
    fneq_reg = (4.5 * w.reshape((len(w),) + (1,) * ndim)) * np.einsum("qab,ab...->q...", h, pi)
    return feq + (1.0 - 1.0 / tau) * fneq_reg


def neq_third_order_norm(f_post, f_eq, lattice="D2Q9"):
    """max |sum_i c_ix c_iy c_ix f_neq| -- must vanish for a regularised update.

    A regularised collision rebuilds the non-equilibrium part from the
    second-order Hermite subspace only, so every third-order non-equilibrium
    moment must be zero up to round-off.  This is the from-definition invariant
    that lbmpy cannot provide for this kernel (no OSS counterpart).
    """
    vel = ref.tensorlbm_velocities(lattice).astype(np.float64)
    d = f_post - f_eq
    t3 = np.einsum("qa,qb,qc,q...->...abc", vel, vel, vel, d)
    return float(np.abs(t3).max())


# ---------------------------------------------------------------------------
# physics arm: periodic shear wave, 200 steps
# ---------------------------------------------------------------------------
def shear_wave(kernel, tau, *, ny=64, nx=8, steps=200, u0=0.01):
    """Decay of a periodic shear wave; returns the measured/analytic viscosity."""
    lat = "D2Q9"
    vel = ref.tensorlbm_velocities(lat)
    y = np.arange(ny)
    k = 2.0 * np.pi / ny
    ux = u0 * np.sin(k * y)[None, :] * np.ones((nx, 1))
    uy = np.zeros_like(ux)
    u = np.stack([ux, uy])
    rho = np.ones((nx, ny))
    f = ref.equilibrium_field(lat, rho, u)  # (Q, nx, ny)
    e0 = _kinetic_energy(f, vel)
    for _ in range(steps):
        f = run_kernel(kernel, f, tau=tau)
        f = _stream_np(f, vel)
    e1 = _kinetic_energy(f, vel)
    nu_meas = -np.log(e1 / e0) / (2.0 * k ** 2 * steps)
    return {
        "nu_measured": float(nu_meas),
        "nu_theory": float((tau - 0.5) / 3.0),
        "err_pct": float(abs(nu_meas - (tau - 0.5) / 3.0) / ((tau - 0.5) / 3.0) * 100.0),
        "E0": float(e0),
        "E1": float(e1),
        "steps": steps,
        "u0": u0,
    }


def _kinetic_energy(f, vel):
    """0.5 * sum_x |j|^2 / rho -- the quantity whose log decays at 2 nu k^2."""
    rho = f.sum(axis=0)
    j = np.tensordot(vel, f, axes=(0, 0))
    return float(0.5 * ((j ** 2).sum(axis=0) / rho).sum())


def _stream_np(f, vel):
    """Periodic streaming by one lattice unit per direction (numpy, D2Q9 order)."""
    out = np.empty_like(f)
    for q, v in enumerate(vel):
        out[q] = np.roll(f[q], shift=int(v[1]), axis=1)
        out[q] = np.roll(out[q], shift=int(v[0]), axis=0)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    runs_dir = HERE / "runs"
    runs_dir.mkdir(exist_ok=True)
    kernels = kernel_fns()

    provenance = {
        "oss": ref.lbmpy_provenance(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": __import__("torch").__version__,
        "predeclared": json.loads((HERE / "predeclared.json").read_text()),
    }

    basis_check = ref.moment_basis_check(REPO_MRT, "D2Q9")

    table = []
    raw = {}
    for arm_id, kernel_key, ref_spec in arm_records():
        lat = lattice_of(ref_spec)
        for amp in AMPLITUDES:
            for seed in SEEDS:
                m, f, f_ref, got = evaluate_arm(
                    arm_id, kernel_key, ref_spec, kernels, amplitude=amp, seed=seed
                )
                rec = {
                    "arm": arm_id, "kernel": kernel_key, "lattice": lat,
                    "tau": tau_of(ref_spec), "amplitude": amp, "seed": seed,
                    **m,
                }
                table.append(rec)
                raw[f"{arm_id}|{kernel_key}|tau{tau_of(ref_spec)}|a{amp}|s{seed}"] = rec

    # aggregate: worst case per (arm, tau)
    agg = {}
    for rec in table:
        key = (rec["arm"], rec["tau"])
        cur = agg.setdefault(key, {"arm": rec["arm"], "tau": rec["tau"],
                                   "eps_scaled_inf_max": 0.0, "eps_scaled_inf_vals": [],
                                   "eps_rel_rms_max": 0.0})
        cur["eps_scaled_inf_max"] = max(cur["eps_scaled_inf_max"], rec["eps_scaled_inf"])
        cur["eps_rel_rms_max"] = max(cur["eps_rel_rms_max"], rec["eps_rel_rms"])
        cur["eps_scaled_inf_vals"].append(rec["eps_scaled_inf"])
    agg_rows = []
    for key in sorted(agg):
        cur = agg[key]
        gate = GATE_D3Q19 if key[0].startswith("A6") else GATE_FP64
        agg_rows.append({
            "arm": cur["arm"], "tau": cur["tau"],
            "eps_scaled_inf_max": cur["eps_scaled_inf_max"],
            "eps_scaled_inf_median": float(np.median(cur["eps_scaled_inf_vals"])),
            "eps_rel_rms_max": cur["eps_rel_rms_max"],
            "gate": gate,
            "pass": bool(cur["eps_scaled_inf_max"] < gate),
        })

    # dtype sensitivity: what the same pointwise gate looks like once the
    # populations only carry fp32 information (declared in variables_changed)
    fp32_rows = []
    for tau in (0.7, 1.0):
        spec = dict(lattice="D2Q9", model="srt", tau=tau)
        f64 = ref.make_test_field("D2Q9", tau=tau, seed=0, amplitude=0.05)
        reference = _reference(spec)
        m64 = ref.compare(run_kernel(kernels["d2q9_collide_bgk"], f64, tau=tau),
                          reference.apply(f64))
        got32 = run_kernel(kernels["d2q9_collide_bgk"], f64, tau=tau).astype(np.float32).astype(np.float64)
        ref32 = reference.apply(f64).astype(np.float32).astype(np.float64)
        m32 = ref.compare(got32, ref32)
        fp32_rows.append({"tau": tau,
                          "eps_scaled_inf_fp64": m64["eps_scaled_inf"],
                          "eps_scaled_inf_fp32_rounding": m32["eps_scaled_inf"]})

    # regularised BGK: from-definition reference (A7)
    rlbm_rows = []
    for tau in (0.51, 0.7, 1.0, 3.0):
        for amp in (1e-3, 0.05):
            f = ref.make_test_field("D2Q9", tau=tau, seed=0, amplitude=amp)
            got = run_kernel(kernels["d2q9_collide_rlbm"], f, tau=tau)
            exp = rlbm_from_definition(f, tau)
            m = ref.compare(got, exp)
            vel = ref.tensorlbm_velocities("D2Q9").astype(np.float64)
            rho = f.sum(0); u = np.tensordot(vel, f, axes=(0, 0)) / rho[None]
            wq = ref._weights("D2Q9").reshape((9, 1, 1))
            cu = np.tensordot(vel, u, axes=(1, 0)); u2 = (u ** 2).sum(0)[None]
            feq = rho[None] * wq * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * u2)
            third = neq_third_order_norm(got, feq)
            rlbm_rows.append({"tau": tau, "amplitude": amp, **m,
                              "gate": GATE_FP64, "pass": bool(m["eps_scaled_inf"] < GATE_FP64),
                              "neq_third_order_max": third,
                              "neq_third_order_rel": float(third / max(np.abs(got).max(), 1e-300))})

    # positive controls
    controls = []
    f_ctrl = ref.make_test_field("D2Q9", tau=0.7, seed=0, amplitude=0.05)
    c1 = control_unpermuted(dict(lattice="D2Q9", model="srt", tau=0.7), f_ctrl)
    controls.append({"control": "C1_no_permutation", "eps_scaled_inf": c1,
                     "detected": bool(c1 > GATE_CONTROL)})
    for rel in (1e-6, 1e-3):
        c2 = control_tau_shift(kernels, dict(lattice="D2Q9", model="srt", tau=0.7), f_ctrl, rel=rel)
        controls.append({"control": f"C2_tau_rel_{rel:g}", "eps_scaled_inf": c2,
                         "detected": bool(c2 > GATE_CONTROL)})
        c3 = control_tau_shift(kernels, dict(lattice="D2Q9", model="trt", tau_plus=0.7), f_ctrl, rel=rel)
        controls.append({"control": f"C3_trt_tau_plus_rel_{rel:g}", "eps_scaled_inf": c3,
                         "detected": bool(c3 > GATE_CONTROL)})

    # physics arm: shear-wave decay, kernel vs lbmpy reference operator
    phys = []
    for tau in (0.55, 0.7, 1.0):
        phys.append({"side": "tensorlbm_kernel", "tau": tau,
                     **shear_wave(kernels["d2q9_collide_bgk"], tau)})
    # same marching loop, but the collision step replaced by the lbmpy reference
    ref_op = ref.PointwiseReference(lattice="D2Q9", model="srt", tau=0.7)
    class _RefKernel:
        __name__ = "collide_bgk"

        def __call__(self, t, tau):
            import torch

            arr = ref_op.apply(t.detach().cpu().numpy().astype(np.float64))
            return torch.from_numpy(np.ascontiguousarray(arr))

    phys.append({"side": "lbmpy_reference_operator", "tau": 0.7,
                 **shear_wave(_RefKernel(), 0.7)})

    summary = {
        "anchor": "XVAL-LBM-POINTWISE-REFERENCE-REGRESSION",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - t0, 2),
        "provenance": provenance,
        "mrt_basis_maxdiff": basis_check,
        "mrt_basis_identical": bool(basis_check < 1e-12),
        "aggregate": agg_rows,
        "rlbm_from_definition": rlbm_rows,
        "fp32_sensitivity": fp32_rows,
        "controls": controls,
        "physics_shear_wave": phys,
        "n_points": len(table),
        "all_arms_pass": bool(all(r["pass"] for r in agg_rows) and all(r["pass"] for r in rlbm_rows)),
        "arms_failing": [f"{r['arm']}@tau={r['tau']}" for r in agg_rows if not r["pass"]]
                       + [f"A7_rlbm@tau={r['tau']}" for r in rlbm_rows if not r["pass"]],
        "all_controls_detected": bool(all(c["detected"] for c in controls)),
    }
    (HERE / "summary.json").write_text(ref.dumps(summary))
    (runs_dir / "raw_points.json").write_text(ref.dumps(raw))
    with (HERE / "table.csv").open("w") as fh:
        cols = ["arm", "kernel", "lattice", "tau", "amplitude", "seed",
                "eps_scaled_inf", "eps_rel_pointwise_max", "eps_rel_rms",
                "moment_err_rho", "moment_err_j", "conservation_residual"]
        fh.write(",".join(cols) + "\n")
        for rec in table:
            fh.write(",".join(str(rec[c]) for c in cols) + "\n")

    print(f"wall {summary['wall_seconds']} s, {len(table)} points")
    print(f"MRT basis maxdiff {basis_check:.3e}")
    for row in agg_rows:
        print(f"{row['arm']:<22} tau={row['tau']:<5} eps_inf={row['eps_scaled_inf_max']:.3e} "
              f"gate={row['gate']:.0e} {'PASS' if row['pass'] else 'FAIL'}")
    for row in rlbm_rows:
        print(f"A7_rlbm(from definition) tau={row['tau']:<5} eps_inf={row['eps_scaled_inf']:.3e} "
              f"{'PASS' if row['pass'] else 'FAIL'}")
    for c in controls:
        print(f"{c['control']:<28} eps={c['eps_scaled_inf']:.3e} detected={c['detected']}")
    for p in phys:
        print(f"shear wave {p['side']:<24} tau={p['tau']} nu={p['nu_measured']:.6e} "
              f"theory={p['nu_theory']:.6e} err={p['err_pct']:.3f}%")
    return 0 if (summary["all_arms_pass"] and summary["all_controls_detected"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())