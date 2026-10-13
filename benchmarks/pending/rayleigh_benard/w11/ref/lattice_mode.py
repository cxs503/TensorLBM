#!/usr/bin/env python
"""Wave-11 Track C: EXACT linear lattice-mode operator for the RBC stepper.

First-principles prediction of the measurable lattice growth rate, per
horizontal wavenumber, for the rb_lbm.py stepper (which reuses verified
thermal_cavity machinery verbatim). The one-step map of the stepper is
linearized about its exact fixed point (hydro f_bar = W9 constant; thermal
T_bar(j) linear, exactly stationary per probe P0.5), complexified in the
x-Fourier mode exp(i k x), and assembled as a dense (14*ny) x (14*ny) matrix
by basis-vector images. Bilinear terms (delta T * delta u, u* . F,
(c.u*)(c.F)) are dropped; every retained term mirrors the library formula:

  collide_bgk_force   u* = u + F/(2 rho);  f_new = f - (f - feq(rho,u*))/tau
                      + w_i (1-1/(2 tau)) [3(c_i.F - u*.F) + 9 (c_i.u*)(c_i.F)]
  equilibrium         w rho (1 + 3 cu + 4.5 cu^2 - 1.5 u^2)
  temperature_*       g_eq = w T (1 + 3 c.u);  g_new = g - (g-g_eq)/tau_T
  pre_streaming_bounce_back  wall rows <- f_pre[opp]
  stream              periodic gather both axes
  apply_rb_temp_bc    rows 0 / ny-1 (node pin or half anti-symmetric mirror)

Step order in rb_lbm.py (sample point = top of loop, before macroscopic):
  macros(f) -> theta -> F_y = g_beta*theta -> thermal collide -> thermal
  stream -> thermal BC rows -> hydro collide (Guo, F) -> BB(wall rows,
  f_pre = pre-collision) -> stream.

Outputs per (construction, ny, Ra): lambda_max (growth per lattice step),
gamma_pred = ln(lambda)*d^2/alpha  [thermal time t* = d^2/alpha per 1/alpha=6
steps], directly comparable to the continuum eigensolver gamma_ref.
"""

import json
import math
from pathlib import Path

import numpy as np

C9 = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int64
)
W9 = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
C5 = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int64)
W5 = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])

TAU = 0.855
TAU_T = 1.0
NU = (TAU - 0.5) / 3.0
ALPHA = (TAU_T - 0.5) / 3.0
T_HOT = 1.0
T_COLD = 0.0


class LatticeRB:
    """Linear one-step map per horizontal wavenumber k."""

    def __init__(self, ny, k, g_beta=0.0, construction="node"):
        self.ny = ny
        self.k = k
        self.g_beta = g_beta
        self.construction = construction
        if construction == "node":
            self.d = ny - 1
            self.Tbar = T_HOT + (T_COLD - T_HOT) * np.arange(ny) / (ny - 1)
        elif construction == "half":
            self.d = ny - 2
            self.Tbar = T_HOT - (np.arange(ny) - 0.5) / self.d
        else:
            raise ValueError(construction)
        self.px9 = np.exp(-1j * k * C9[:, 0].astype(float))
        self.px5 = np.exp(-1j * k * C5[:, 0].astype(float))
        jj = np.arange(ny)
        self.jm9 = (jj[None, :] - C9[:, 1][:, None]) % ny  # (9, ny) gather idx
        self.jm5 = (jj[None, :] - C5[:, 1][:, None]) % ny

    def apply(self, f, g):
        """f (9,ny) complex, g (5,ny) complex -> next (f, g)."""
        ny = self.ny
        rho = f.sum(axis=0)
        ux = (C9[:, 0][:, None] * f).sum(axis=0)
        uy = (C9[:, 1][:, None] * f).sum(axis=0)
        th = g.sum(axis=0)
        fy = self.g_beta * th
        # thermal collision (linearized around Tbar, u=0)
        cu5 = C5[:, 0][:, None] * ux[None, :] + C5[:, 1][:, None] * uy[None, :]
        geq = W5[:, None] * (th[None, :] + 3.0 * self.Tbar[None, :] * cu5)
        gs = g - (g - geq) / TAU_T
        # thermal stream (x: phase, y: periodic gather)
        gst = self.px5[:, None] * np.take_along_axis(gs, self.jm5, axis=1)
        # thermal BC rows
        t1 = gst[:, 1].sum()
        tn2 = gst[:, -2].sum()
        if self.construction == "node":
            tb = 0.0
            tt = 0.0
        else:
            tb = -t1
            tt = -tn2
        neqb = gst[3, 1] - W5[3] * t1
        neqt = gst[4, -2] - W5[4] * tn2
        g2 = gst.copy()
        g2[:, 0] = W5 * tb
        g2[3, 0] += neqb
        g2[0, 0] -= neqb
        g2[:, -1] = W5 * tt
        g2[4, -1] += neqt
        g2[0, -1] -= neqt
        # hydro collision, Guo force linearized (F = (0, fy))
        usx = ux.copy()
        usy = uy + 0.5 * fy
        cu9 = C9[:, 0][:, None] * usx[None, :] + C9[:, 1][:, None] * usy[None, :]
        feq = W9[:, None] * (rho[None, :] + 3.0 * cu9)
        fc = f - (f - feq) / TAU
        fi = W9[:, None] * (1.0 - 0.5 / TAU) * 3.0 * C9[:, 1][:, None] * fy[None, :]
        fpost = fc + fi
        # pre-streaming bounce-back on wall rows (f_pre = pre-collision f)
        fbb = fpost.copy()
        fbb[:, 0] = f[OPP, 0]
        fbb[:, -1] = f[OPP, -1]
        # hydro stream
        fn = self.px9[:, None] * np.take_along_axis(fbb, self.jm9, axis=1)
        return fn, g2

    def matrix(self):
        n = 14 * self.ny
        M = np.zeros((n, n), dtype=complex)
        for idx in range(n):
            e = np.zeros(n, dtype=complex)
            e[idx] = 1.0
            f = e[: 9 * self.ny].reshape(9, self.ny)
            g = e[9 * self.ny :].reshape(5, self.ny)
            fn, gn = self.apply(f, g)
            M[:, idx] = np.concatenate([fn.ravel(), gn.ravel()])
        return M

    def sigma_dense(self):
        ev = np.linalg.eigvals(self.matrix())
        return float(ev[np.argmax(ev.real)].real), ev

    def sigma_arpack(self, tol=1e-12):
        from scipy.sparse.linalg import LinearOperator, eigs

        n = 14 * self.ny
        M = self.matrix()

        def mv(x):
            f = x[: 9 * self.ny].reshape(9, self.ny)
            g = x[9 * self.ny :].reshape(5, self.ny)
            fn, gn = self.apply(f, g)
            return np.concatenate([fn.ravel(), gn.ravel()])

        op = LinearOperator((n, n), matvec=mv, dtype=complex)
        vals = eigs(op, k=4, which="LR", tol=tol, maxiter=20000, v0=np.ones(n, dtype=complex))
        vals = np.asarray(vals)
        return float(vals[np.argmax(vals.real)].real), vals


def analytic_thermal_sigma(ny, k, construction):
    """Ideal tridiagonal prediction (tau_T = 1 full relaxation):
    sigma = ln[(1+cos k+cos q)/3], q = pi/d_T."""
    q = math.pi / ((ny - 1) if construction == "node" else (ny - 2))
    return math.log((1.0 + math.cos(k) + math.cos(q)) / 3.0)


def _fit_slope(t, a, floor=1e-300):
    m = np.abs(a) > floor
    if m.sum() < 8:
        return None
    p = np.polyfit(t[m], np.log(np.abs(a[m])), 1)
    return float(p[0])


def trajectory_slope(op, seed_f, seed_g, shape_Th, steps, sample_every=10):
    """Iterate the exact linear one-step map from the seeded state, project
    the A_T amplitude exactly as rb_lbm.py does, fit with the frozen
    3-window plateau protocol. Ghost modes (wall-row collision invariants,
    lambda exactly 1) contribute at most a constant offset, which the window
    spread criterion exposes."""
    f = seed_f.copy()
    g = seed_g.copy()
    t_rec, a_rec = [], []
    nrm = float(np.sum(shape_Th**2))
    for step in range(0, steps + 1):
        if step % sample_every == 0:
            th = g.sum(axis=0)
            a_rec.append(complex(np.sum(th * shape_Th) / nrm))
            t_rec.append(step)
        if step == steps:
            break
        f, g = op.apply(f, g)
    t = np.array(t_rec, dtype=float)
    a = np.array(a_rec)
    n = len(t)
    slopes = {}
    for tag, (f0, f1) in (("w1", (0.15, 0.90)), ("w2", (0.05, 0.75)), ("w3", (0.30, 1.00))):
        i0, i1 = int(f0 * n), min(int(f1 * n), n - 1)
        sl = _fit_slope(t[i0 : i1 + 1], a[i0 : i1 + 1].real)
        if sl is not None:
            slopes[tag] = sl
    vals = [v for v in slopes.values()]
    out = {
        "n_samples": n,
        "slopes": slopes,
        "spread_rel": (max(vals) - min(vals)) / abs(np.mean(vals)) if vals else None,
        "slope": slopes.get("w1"),
    }
    return out


def seed_from_npz(ny, npz, tag, amp=1e-4):
    k = 2.0 * math.pi / (2 * ny)
    d = ny - 2
    vs = ALPHA / d
    Th = npz[f"ny{ny}/{tag}/Theta_node"]
    Ph = npz[f"ny{ny}/{tag}/Phi_node"]
    Phd = npz[f"ny{ny}/{tag}/Phider_node"]
    con = "half"
    tb = T_HOT - (np.arange(ny) - 0.5) / d
    ux = -1j * amp * vs * np.asarray(Phd)
    uy = -amp * vs * k * np.asarray(Ph)
    cu9 = C9[:, 0][:, None] * ux[None, :] + C9[:, 1][:, None] * uy[None, :]
    f0 = W9[:, None] * 3.0 * cu9
    th = amp * np.asarray(Th)
    cu5 = C5[:, 0][:, None] * ux[None, :] + C5[:, 1][:, None] * uy[None, :]
    g0 = W5[:, None] * (th[None, :] + 3.0 * tb[None, :] * cu5)
    return f0 + 0j, g0 + 0j, np.asarray(Th)


def analytic_seed(ny, amp=1e-2, construction="node"):
    k = 2.0 * math.pi / (2 * ny)
    y = np.arange(ny, dtype=float)
    if construction == "node":
        Th = np.sin(math.pi * y / (ny - 1))
        tb = T_HOT - y / (ny - 1)
    else:
        Th = np.sin(math.pi * (y - 0.5) / (ny - 2))
        tb = T_HOT - (y - 0.5) / (ny - 2)
    g0 = W5[:, None] * (amp * Th)[None, :] * 1.0
    f0 = np.zeros((9, ny), dtype=complex)
    return f0, g0, Th


def _selftest():
    ok = True
    msgs = []
    ny = 32
    k = 2.0 * math.pi / (2 * ny)
    for con in ("node", "half"):
        op = LatticeRB(ny, k, 0.0, con)
        f0, g0, Th = analytic_seed(ny, 1e-2, con)
        r = trajectory_slope(op, f0, g0, Th, steps=4000, sample_every=20)
        ana = analytic_thermal_sigma(ny, k, con)
        rel = abs(r["slope"] - ana) / abs(ana)
        ok &= rel < 5e-3
        msgs.append(
            f"{con}: traj slope={r['slope']:.8e} analytic={ana:.8e} "
            f"rel={rel:.2e} spread={r['spread_rel']:.2e}"
        )
    return ("SELFTEST_OK" if ok else "SELFTEST_FAIL"), msgs


def main():
    import sys

    import rb_eig

    st, msgs = _selftest()
    print(st)
    for m in msgs:
        print("  " + m)
    if st != "SELFTEST_OK":
        sys.exit(1)

    out = {"tau": TAU, "tau_T": TAU_T, "nu": NU, "alpha": ALPHA, "pr": NU / ALPHA, "selftest": msgs}
    pdir = Path("/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/ref")
    N_SP = 64
    PR = NU / ALPHA
    arms = (0.85, 0.95, 1.0, 1.05, 1.15, 1.3)
    npzs = {ny: dict(np.load(pdir / f"eig_ny{ny}.npz")) for ny in (64, 128, 256)}
    preds = []
    rac_lat_out = []
    for con in ("half",):
        for ny in (64, 128, 256):
            nx = 2 * ny
            k = 2.0 * math.pi / nx
            dv = ny - 2
            a = k * dv
            rac_a = rb_eig.ra_critical(
                a, PR, lambda aa, rr, pp: rb_eig.sigma_spectral(aa, rr, pp, N_SP)
            )
            lev = {"construction": con, "ny": ny, "a": a, "Ra_c_continuum_at_a": rac_a, "arms": []}
            sig_pts = []
            for frac in arms:
                Ra = frac * rac_a
                d = (ny - 1) if con == "node" else (ny - 2)
                gb = Ra * NU * ALPHA / d**3
                op = LatticeRB(ny, k, gb, con)
                f0, g0, Th = seed_from_npz(ny, npzs[ny], f"arm{frac}")
                gref = rb_eig.sigma_spectral(a, Ra, PR, N_SP)
                # steps for >= ~2.5 e-folds
                s_est = abs(gref) * ALPHA / (dv * dv) + 1e-9
                steps = min(int(2.8 / s_est), 900000)
                steps = max(steps, 4000)
                r = trajectory_slope(
                    op, f0, g0, Th, steps=steps, sample_every=max(10, steps // 600)
                )
                g_pred = r["slope"] * dv * dv / ALPHA
                lev["arms"].append(
                    {
                        "frac": frac,
                        "Ra": Ra,
                        "steps": steps,
                        "slope_step": r["slope"],
                        "spread_rel": r["spread_rel"],
                        "gamma_pred": g_pred,
                        "gamma_ref": gref,
                        "rel_err": g_pred / gref - 1.0,
                    }
                )
                if frac in (0.95, 1.0, 1.05, 1.15):
                    sig_pts.append((Ra, r["slope"]))
                print(
                    f"ny={ny} con={con} arm={frac:.2f} steps={steps} "
                    f"g_pred={g_pred:+.6f} g_ref={gref:+.6f} "
                    f"rel={g_pred / gref - 1.0:+.5%} spread={r['spread_rel']:.1e}"
                )
            xs = np.array([p[0] for p in sig_pts])
            ys = np.array([p[1] for p in sig_pts])
            pf = np.polyfit(xs, ys, 1)
            rac_lat = -pf[1] / pf[0]
            lev["Ra_c_lattice_linearfit"] = float(rac_lat)
            lev["Ra_c_lattice_drift"] = float(rac_lat / rac_a - 1.0)
            rac_lat_out.append(lev)
            print(f"  -> Ra_c_lattice={rac_lat:.4f} drift={rac_lat / rac_a - 1.0:+.6%}")
    out["levels"] = rac_lat_out
    out["arms"] = list(arms)
    (pdir / "lattice_predictions.json").write_text(json.dumps(out, indent=1))
    print("LATTICE_PREDICT_DONE")


if __name__ == "__main__":
    main()
