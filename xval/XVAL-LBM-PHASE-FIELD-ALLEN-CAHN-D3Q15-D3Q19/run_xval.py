"""XVAL-LBM-PHASE-FIELD-ALLEN-CAHN-D3Q15-D3Q19 cross-validation driver.

OSS arm = lssfau/walberla apps/benchmarks/PhaseFieldAllenCahn (oss_solutions.id=363,
tag phase-field-allen-cahn-d3q15-d3q19).  waLBerla generates that app with lbmpy and
the same library is installed here, so the formulation is read off lbmpy at run time:
  beta = 12 sigma / W, kappa = 3/2 sigma W, omega_phi = 1/(1/2+3M)   (AllenCahnParameters)
  mu = 4 beta phi (phi-1/2)(phi-1) - kappa lap(phi)                  (chemical_potential_symbolic)
  F_i = (1/3)(1-4(phi-1/2)^2)/W * n_i                                (interface_tracking_force)
  lap / grad = lbmpy isotropic operators ON THE LB STENCIL (D3Q15 AC, D3Q19 hydro)
TensorLBM arm = tensorlbm.allen_cahn_lbm.step_allen_cahn_field (psi in [-1,1],
seven-point laplacian, Korteweg force mu grad psi).
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np, sympy as sp, torch
from pystencils import Field
from lbmpy.stencils import LBStencil, Stencil
from lbmpy.phasefield_allen_cahn.derivatives import (isotropic_gradient_symbolic,
                                                     laplacian_symbolic)
from lbmpy.phasefield_allen_cahn.kernel_equations import (chemical_potential_symbolic,
                                                          interface_tracking_force)
from lbmpy.phasefield_allen_cahn.parameter_calculation import AllenCahnParameters

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts"); os.makedirs(ART, exist_ok=True)
torch.set_default_dtype(torch.float64)
PROBE = Field.create_generic("phi", 3, dtype="float64")
def _acc_of(off):
    a = PROBE.center
    for ax in range(3):
        if off[ax]:
            a = a.neighbor(ax, int(off[ax]))
    return a


def _entries(st):
    e = st.stencil_entries
    return tuple(e()) if callable(e) else tuple(e)


STENCILS = {"D3Q15": LBStencil(Stencil.D3Q15), "D3Q19": LBStencil(Stencil.D3Q19)}


# --------------------------------------------------------- lbmpy extraction
def coefficient_table(expr):
    table = {}
    for term in sp.Add.make_args(sp.expand(expr)):
        accs = [a for a in term.atoms() if isinstance(a, Field.Access)]
        if not accs:
            assert sp.simplify(term) == 0, term
            continue
        assert len(accs) == 1, term
        acc = accs[0]
        table[tuple(int(o) for o in acc.offsets)] = float(sp.expand(term / acc))
    return table


def lbmpy_operators(name):
    st = STENCILS[name]
    lap = sp.expand(laplacian_symbolic(PROBE, st))
    grads = [sp.expand(g) for g in isotropic_gradient_symbolic(PROBE, st)]
    tabs = {"laplacian": coefficient_table(lap),
            "gradient": [coefficient_table(g) for g in grads]}
    def _acc(off):
        a = PROBE.center
        for ax in range(3):
            if off[ax]:
                a = a.neighbor(ax, int(off[ax]))
        return a
    acc = {tuple(map(int, e)): _acc(tuple(map(int, e))) for e in _entries(st)}
    acc[(0, 0, 0)] = PROBE.center
    rb = lambda tab: sum(sp.Float(c) * acc[off] for off, c in tab.items())
    assert sp.simplify(lap - rb(tabs["laplacian"])) == 0, "laplacian not reproduced"
    for g, t in zip(grads, tabs["gradient"]):
        assert sp.simplify(g - rb(t)) == 0, "gradient not reproduced"
    tabs["stencil"] = [list(map(int, e)) for e in _entries(st)]
    tabs["json"] = {"laplacian": {",".join(map(str, k)): v for k, v in tabs["laplacian"].items()},
                    "gradient": [{",".join(map(str, k)): v for k, v in t.items()}
                                 for t in tabs["gradient"]]}
    return tabs


def oss_parameters(sigma, W, mobility):
    p = AllenCahnParameters(density_heavy=1.0, density_light=1.0,
                            dynamic_viscosity_heavy=0.1, dynamic_viscosity_light=0.1,
                            surface_tension=sigma, mobility=mobility,
                            gravitational_acceleration=0.0, interface_thickness=W)
    pm = p.parameter_map()
    return {"sigma": sigma, "W": W, "mobility": mobility,
            "beta": float(sp.simplify(p.beta.subs(pm))),
            "kappa": float(sp.simplify(p.kappa.subs(pm))),
            "omega_phi": float(p.omega_phi), "tau_phi": 0.5 + 3.0 * mobility,
            "beta_symbolic": str(p.beta), "kappa_symbolic": str(p.kappa)}


def weights_d3q15(st):
    c = np.array(_entries(st), float)
    n2 = (c ** 2).sum(1)
    w = np.where(n2 == 0, 2.0 / 9.0, np.where(n2 == 1, 1.0 / 9.0, 1.0 / 72.0))
    assert abs(w.sum() - 1) < 1e-14
    assert np.abs((w[:, None] * c ** 2).sum(0) - 1.0 / 3.0).max() < 1e-14
    return c, w


def apply_op(field, table, shape=None):
    shape = field.shape if shape is None else shape
    out = torch.zeros(shape, dtype=torch.float64)
    for off, coef in table.items():
        if coef:
            out += coef * torch.roll(field, shifts=(int(off[2]), int(off[1]), int(off[0])),
                                     dims=(0, 1, 2))
    return out


# ----------------------------------------------------------------- geometry
def _grid(n):
    a = torch.arange(n, dtype=torch.float64)
    return torch.meshgrid(a, a, a, indexing="ij")


def sphere_phi(n, R0, W):
    zz, yy, xx = _grid(n)
    c = 0.5 * (n - 1)
    r = torch.sqrt((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2)
    return 0.5 * (1.0 + torch.tanh(2.0 * (R0 - r) / W))       # phi in [0,1]


def planar_phi(n, W, t=8):
    x = torch.arange(n, dtype=torch.float64) - 0.5 * n
    return (0.5 * (1.0 + torch.tanh(2.0 * x / W))).view(1, 1, n).expand(t, t, n).contiguous()


def radius_from_volume(phi01):
    return (3.0 * float(phi01.sum()) / (4.0 * math.pi)) ** (1.0 / 3.0)


# ------------------------------------------------------------ OSS AC D3Q15
class OSSAllenCahnD3Q15:
    def __init__(self, W, sigma, mobility):
        self.W, self.p = float(W), oss_parameters(sigma, W, mobility)
        c, w = weights_d3q15(STENCILS["D3Q15"])
        self.c, self.w = torch.tensor(c), torch.tensor(w)
        ops = lbmpy_operators("D3Q15")
        self.lap_tab, self.grad_tabs = ops["laplacian"], ops["gradient"]

    def gradient(self, phi):
        return [apply_op(phi, t) for t in self.grad_tabs]

    def chemical_potential(self, phi):
        return (4.0 * self.p["beta"] * phi * (phi - 0.5) * (phi - 1.0)
                - self.p["kappa"] * apply_op(phi, self.lap_tab))

    def interface_force(self, phi):
        g = self.gradient(phi)
        nrm = torch.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2 + 1e-32)
        frac = (1.0 - 4.0 * (phi - 0.5) ** 2) / self.W
        return [(1.0 / 3.0) * frac * gi / nrm for gi in g]

    def equilibrium(self, phi, u=None):
        w = self.w.view(-1, 1, 1, 1)
        if u is None:
            return w * phi.unsqueeze(0)
        cu = sum(self.c[:, i].view(-1, 1, 1, 1) * u[i].unsqueeze(0) for i in range(3))
        usq = sum(x ** 2 for x in u)
        return w * phi.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq.unsqueeze(0))

    def initialise(self, phi):
        F = self.interface_force(phi)          # lbmpy: u_symp = u - F/2, rho -> phi
        return self.equilibrium(phi, [-0.5 * Fi for Fi in F])

    def step(self, h):
        phi = h.sum(0)
        F = self.interface_force(phi)
        om = self.p["omega_phi"]
        h_post = h - om * (h - self.equilibrium(phi))
        cf = sum(self.c[:, i].view(-1, 1, 1, 1) * F[i].unsqueeze(0) for i in range(3))
        h_post = h_post + (1.0 - 0.5 * om) * 3.0 * self.w.view(-1, 1, 1, 1) * cf
        out = torch.empty_like(h_post)
        for q in range(self.c.shape[0]):
            out[q] = torch.roll(h_post[q], shifts=(int(self.c[q, 2]), int(self.c[q, 1]),
                                                   int(self.c[q, 0])), dims=(0, 1, 2))
        return out


# ------------------------------------------------------- TensorLBM (arm B)
def tlbm_class():
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
    import tensorlbm.allen_cahn_lbm as acm

    class TLBMAllenCahn:
        def __init__(self, W, sigma, mobility):
            self.W, self.mobility = float(W), float(mobility)
            self.params = acm.free_energy_parameters(sigma, self.W)
            self._step, self._lap, self._force = (acm.step_allen_cahn_field,
                                                  acm.laplacian_3d, acm.force_mu_grad_phi)

        def chemical_potential(self, psi):
            b, k = self.params.beta, self.params.kappa
            return b * psi * (psi * psi - 1.0) - k * self._lap(psi)

        def force(self, psi):
            return self._force(psi, self.chemical_potential(psi))

        def step(self, psi):
            return self._step(psi, beta=self.params.beta, kappa=self.params.kappa,
                              mobility=self.mobility)
    return TLBMAllenCahn


# ----------------------------------------------------------------- metrics
def width_from_profile(v, ref_w):
    """1/4-3/4 crossings of a monotone [0,1] profile; returns measured thickness."""
    v = np.asarray(v, float)
    if v[0] < v[-1]:
        v = v[::-1]
    def cross(level):
        for j in range(1, v.size):
            if v[j - 1] >= level >= v[j]:
                return j - 1 + (v[j - 1] - level) / (v[j - 1] - v[j])
        raise RuntimeError("level not crossed")
    return abs(cross(0.25) - cross(0.75)) / math.atanh(0.5)


def shrinkage(step, phi01, steps, every=50):
    hist = [(0, radius_from_volume(phi01))]
    phi = phi01
    for k in range(1, steps + 1):
        phi = step(phi)
        if k % every == 0:
            hist.append((k, radius_from_volume(phi if phi.min() >= -1e-12 else 0.5 * (phi + 1.0))))
    t = np.array([h[0] for h in hist], float); R = np.array([h[1] for h in hist], float)
    A = np.vstack([t, np.ones_like(t)]).T
    slope, icpt = np.linalg.lstsq(A, R ** 2, rcond=None)[0]
    pred = A @ np.array([slope, icpt])
    ss = 1.0 - ((R ** 2 - pred) ** 2).sum() / ((R ** 2 - (R ** 2).mean()) ** 2).sum()
    return {"slope_dR2_dt": float(slope), "D_eff": float(-slope / 4.0), "fit_r2": float(ss),
            "R0": float(R[0]), "R_end": float(R[-1]),
            "hist": [[int(a), float(b)] for a, b in zip(t, R)]}


def selftest():
    out = {}
    for name in ("D3Q15", "D3Q19"):
        ops = lbmpy_operators(name)
        out[name] = {"sum_lap": float(sum(ops["laplacian"].values())),
                     "sum_grad_x": float(sum(ops["gradient"][0].values())),
                     "lap_centre": ops["laplacian"].get((0, 0, 0), 0.0),
                     "n_lap_terms": len(ops["laplacian"])}
        assert abs(out[name]["sum_lap"]) < 1e-14
        assert abs(out[name]["sum_grad_x"]) < 1e-14
    n = 24
    zz, yy, xx = _grid(n)
    q = 0.3 * xx ** 2 + 0.7 * yy ** 2 + 0.2 * zz ** 2
    ops = lbmpy_operators("D3Q15")
    out["D3Q15_quadratic"] = {
        "lap_err": float((apply_op(q, ops["laplacian"]) - (2 * 1.2)).abs().max()),
        "gradx_err": float((apply_op(q, ops["gradient"][0]) - 0.6 * xx).abs().max()),
        "grady_err": float((apply_op(q, ops["gradient"][1]) - 1.4 * yy).abs().max())}
    p = oss_parameters(0.02, 4.0, 0.2)
    mu_sym = chemical_potential_symbolic(PROBE, STENCILS["D3Q15"], sp.Float(p["beta"]),
                                         sp.Float(p["kappa"]))
    P = PROBE.center
    poly = sp.expand(mu_sym.subs({PROBE.neighbor(0, off): 0 for off in
                                  STENCILS["D3Q15"].stencil_entries if off != (0, 0, 0)}))
    mine = 4 * sp.Float(p["beta"]) * P * (P - sp.Rational(1, 2)) * (P - 1)
    out["mu_polynomial_matches_lbmpy"] = bool(sp.simplify(poly - mine) == 0)
    assert out["mu_polynomial_matches_lbmpy"]
    F_sym = interface_tracking_force(
        PROBE, STENCILS["D3Q15"],
        AllenCahnParameters(1.0, 1.0, 0.1, 0.1, 0.02, 0.2, 0.0, 4),
        fd_stencil=STENCILS["D3Q15"])
    arm = OSSAllenCahnD3Q15(4.0, 0.02, 0.2)
    phi0 = planar_phi(32, 4.0, 4)
    Fm = arm.interface_force(phi0)
    subs = {PROBE.neighbor(0, off): float(phi0[0, 0, 16 + int(off[0])])
            for off in STENCILS["D3Q15"].stencil_entries}
    subs[PROBE.center] = float(phi0[0, 0, 16])
    fs = float(sp.N(F_sym[0].subs(subs)))
    out["interface_force_matches_lbmpy"] = {
        "symbolic": fs, "numeric": float(Fm[0][0, 0, 16]),
        "rel_diff": abs(fs - float(Fm[0][0, 0, 16])) / max(abs(fs), 1e-30)}
    return out


def main():
    out = {"anchor": "XVAL-LBM-PHASE-FIELD-ALLEN-CAHN-D3Q15-D3Q19",
           "oss_reference": "lssfau/walberla apps/benchmarks/PhaseFieldAllenCahn "
                            "(oss_solutions.id=363, phase-field-allen-cahn-d3q15-d3q19)",
           "selftest": selftest(),
           "oss_formulation": {
               "parameters_sigma0.02_W4_M0.2": oss_parameters(0.02, 4.0, 0.2),
               "parameters_sigma0.02_W3_M0.2": oss_parameters(0.02, 3.0, 0.2),
               "operators": {n: {"stencil": lbmpy_operators(n)["stencil"],
                                 "laplacian": lbmpy_operators(n)["json"]["laplacian"],
                                 "gradient": lbmpy_operators(n)["json"]["gradient"]}
                             for n in ("D3Q15", "D3Q19")}}}
    json.dump(out["oss_formulation"], open(os.path.join(ART, "oss_fd_operators.json"), "w"), indent=1)

    sigma, M = 0.02, 0.2
    D_an = M * sigma
    rows = []
    TLBM = tlbm_class()

    # M1 equilibrium thickness (planar, 300 steps)
    m1 = {}
    for W in (4.0, 3.0):
        phi0 = planar_phi(48, W, 8)
        armA = OSSAllenCahnD3Q15(W, sigma, M)
        h = armA.initialise(phi0)
        for _ in range(300):
            h = armA.step(h)
        pA = h.sum(0)[0, 0].numpy()
        armB = TLBM(W, sigma, M)
        psi = 2.0 * phi0 - 1.0
        for _ in range(300):
            psi = armB.step(psi)
        pB = (0.5 * (psi[0, 0] + 1.0)).numpy()
        wA, wB = width_from_profile(pA, W), width_from_profile(pB, W)
        m1[str(W)] = {"W_label": W, "A_width_lu": wA, "B_width_lu": wB,
                      "A_ref_lu": W, "B_ref_lu": 2.0 * W,
                      "A_err_pct": 100 * (wA - W) / W, "B_err_pct": 100 * (wB - 2 * W) / (2 * W),
                      "A_range": [float(pA.min()), float(pA.max())],
                      "B_range": [float(pB.min()), float(pB.max())]}
        rows.append({"metric": "M1_width_lu", "config": f"W={W}", "arm": "OSS_D3Q15",
                     "value": wA, "ref": W, "err_pct": m1[str(W)]["A_err_pct"]})
        rows.append({"metric": "M1_width_lu", "config": f"W={W}", "arm": "TensorLBM",
                     "value": wB, "ref": 2 * W, "err_pct": m1[str(W)]["B_err_pct"]})
    out["M1_equilibrium_width"] = m1

    # M2 curvature-driven shrinking
    configs = [{"name": "base", "n": 64, "R0": 12.0, "W": 4.0, "steps": 1200},
               {"name": "W3", "n": 64, "R0": 12.0, "W": 3.0, "steps": 1200},
               {"name": "coarse", "n": 48, "R0": 9.0, "W": 4.0, "steps": 900}]
    m2 = {}
    for cfg in configs:
        n, R0, W, steps = cfg["n"], cfg["R0"], cfg["W"], cfg["steps"]
        phi0 = sphere_phi(n, R0, W)
        armA = OSSAllenCahnD3Q15(W, sigma, M)
        h = armA.initialise(phi0)
        rA = shrinkage(lambda x: armA.step(x), h.sum(0), steps)
        armB = TLBM(W, sigma, M)
        psi0 = 2.0 * phi0 - 1.0
        rB = shrinkage(armB.step, psi0, steps)
        m2[cfg["name"]] = {"cfg": cfg, "A": rA, "B": rB, "D_analytic": D_an,
                           "A_err_pct": 100 * (rA["D_eff"] - D_an) / D_an,
                           "B_err_pct": 100 * (rB["D_eff"] - D_an) / D_an,
                           "A_vs_B_pct": 100 * (rA["D_eff"] - rB["D_eff"]) / rB["D_eff"]}
        print(cfg["name"], {k: round(m2[cfg["name"]][k], 6) for k in
                            ("A_err_pct", "B_err_pct", "A_vs_B_pct")}, flush=True)
        rows.append({"metric": "M2_D_eff", "config": cfg["name"], "arm": "OSS_D3Q15",
                     "value": rA["D_eff"], "ref": D_an, "err_pct": m2[cfg["name"]]["A_err_pct"]})
        rows.append({"metric": "M2_D_eff", "config": cfg["name"], "arm": "TensorLBM",
                     "value": rB["D_eff"], "ref": D_an, "err_pct": m2[cfg["name"]]["B_err_pct"]})
    out["M2_shrinkage"] = m2

    # M3 interface-force normal integral vs sigma K = 2 sigma / R (D3Q19 operators)
    n, R0, W = 64, 12.0, 4.0
    phi = sphere_phi(n, R0, W)
    ops19 = lbmpy_operators("D3Q19")
    p = oss_parameters(sigma, W, M)
    lap19 = apply_op(phi, ops19["laplacian"])
    mu = 4.0 * p["beta"] * phi * (phi - 0.5) * (phi - 1.0) - p["kappa"] * lap19
    g19 = [apply_op(phi, t) for t in ops19["gradient"]]
    nrm = torch.sqrt(sum(x ** 2 for x in g19) + 1e-32)
    rhoL, rhoH = 1.0, 2.0
    rho = rhoL + phi * (rhoH - rhoL)
    Fsurf = [mu * gi for gi in g19]
    Fpress = [-(1.0 / 3.0) * (rho - 1.0) * (rhoH - rhoL) * gi for gi in g19]
    line, half = n // 2, n // 2
    def line_int(F):
        Fr = sum(F[i][line, line, :] * g19[i][line, line, :] / nrm[line, line, :]
                 for i in range(3))
        return float(Fr[line:half].sum())
    psi = 2.0 * phi - 1.0
    fx, fy, fz = TLBM(W, sigma, M).force(psi)
    FrB = (fx[line, line, :] * g19[0][line, line, :] / nrm[line, line, :]
           + fy[line, line, :] * g19[1][line, line, :] / nrm[line, line, :]
           + fz[line, line, :] * g19[2][line, line, :] / nrm[line, line, :])
    ref = 2.0 * sigma / R0
    m3 = {"2sigma_over_R": ref, "oss_surface_tension": line_int(Fsurf),
          "oss_surface_tension_err_pct": 100 * (line_int(Fsurf) + ref) / ref,
          "oss_pressure": line_int(Fpress),
          "tensorbm_korteweg": float(FrB[line:half].sum()),
          "tensorbm_korteweg_err_pct": 100 * (float(FrB[line:half].sum()) + ref) / ref}
    out["M3_force_integral"] = m3
    for nm, val in (("OSS_D3Q19_surface_tension", m3["oss_surface_tension"]),
                    ("TensorLBM_Korteweg", m3["tensorbm_korteweg"])):
        rows.append({"metric": "M3_force_integral", "config": "sphere_R12", "arm": nm,
                     "value": val, "ref": -ref,
                     "err_pct": 100 * (val + ref) / ref})

    out["table"] = rows
    json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    with open(os.path.join(HERE, "table.csv"), "w") as f:
        f.write("metric,config,arm,value,ref,err_pct\n")
        for r in rows:
            f.write(f"{r['metric']},{r['config']},{r['arm']},{r['value']:.8g},{r['ref']:.8g},"
                    f"{r['err_pct']:.4f}\n")
    print("OK -> summary.json, table.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        print(json.dumps(selftest(), indent=1, default=str))
    else:
        main()
