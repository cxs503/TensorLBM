"""XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES

Cross-validation of TensorLBM's static immersed-marker obstacle approach
against the mature OSS implementation joconnor22/LIFE (Cylinder example:
rigid circle in channel flow, Re_D = 100).

Arms
----
A1  OSS ground truth : the actual LIFE C++ binary (examples/Cylinder),
                       parsed from Results/TotalForces.out.
A2  full transcription: LIFE LBM core + LIFE IBM pipeline, transcribed
                       line-for-line into numpy (independent code path).
A3  TensorLBM native : the SAME LBM core + TensorLBM's native IBM
                       (direct forcing F=(U-u)/dt, Peskin 2-pt hat kernel,
                       Guo application) -- isolates the IBM-approach delta.

Every LIFE rule is cited to the OSS file:line in predeclared.json (zero
hand-copy).  The LBM core is shared verbatim between A2 and A3 so that the
A2-vs-A3 gap is purely the IBM approach.

KPIs (window t = 10..20 s, i.e. steps 40000..80000):
  K1 mean Cx, K2 rms Cy, K3 Strouhal St (FFT of Cy), K4 max Cx.
Criterion (owner): A2 vs A1 < 15%.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# D2Q9 (src/Grid.cpp:1246-1248)
# ---------------------------------------------------------------------------
C = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
              (1, 1), (-1, 1), (-1, -1), (1, -1)], dtype=float)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7])
CS2 = 1.0 / 3.0
CX = C[:, 0]
CY = C[:, 1]

# ---------------------------------------------------------------------------
# Case parameters (examples/Cylinder/params.h)
# ---------------------------------------------------------------------------
NX, NY = 441, 83
H = 0.41
DX = H / (NY - 1)
DT = 0.00025
NU_P = 0.001
RHO_P = 1.0
U_IN_P = 1.0
OMEGA = 1.0 / (NU_P * DT / (CS2 * DX ** 2) + 0.5)
NSTEPS = 80000
RAMP = 2.0
DM = RHO_P * DX ** 3
FORCE_SCALE = DM / DT ** 2            # Dm*Dx/Dt^2 * 1/Dx  (Objects.cpp:519)
ND = 0.5 * RHO_P * U_IN_P ** 2 * 0.1   # ref_L = 0.1 (params.h)
CYL_X, CYL_Y = 40, 40                 # (0.2,0.2) m in lattice units
R_LATT = 10.0                         # 0.05 m / DX
NUM_NODES = int(np.floor(2.0 * np.pi * R_LATT))  # IBMBody.cpp:66


def life_delta_3pt(r):
    """LIFE 3-point regularised Dirac delta (inc/Utils.h:220-231)."""
    ra = np.abs(r)
    inner = (1.0 + np.sqrt(np.clip(1.0 - 3.0 * ra ** 2, 0, None))) / 3.0
    outer = (5.0 - 3.0 * ra - np.sqrt(np.clip(-3.0 * (1.0 - ra) ** 2 + 1.0, 0, None))) / 6.0
    return np.where(ra <= 0.5, inner, np.where(ra <= 1.5, outer, 0.0))


def hat_delta(r):
    """Peskin 2-point hat kernel (src/tensorlbm/ibm.py::ibm_delta_hat)."""
    return np.clip(1.0 - np.abs(r), 0.0, None)


def build_markers():
    th = 2.0 * np.pi * np.arange(NUM_NODES) / NUM_NODES
    return CYL_X + R_LATT * np.cos(th), CYL_Y + R_LATT * np.sin(th)


def build_support_dense(mx, my, delta_fn):
    """5x5 window about round(x), keep |dx|<1.5 & |dy|<1.5 (IBMNode.cpp:140-180).
    Returns dense (N,9) site-id and weight matrices (0 = empty)."""
    n = mx.size
    S = np.zeros((n, 9), dtype=int)
    Sw = np.zeros((n, 9))
    for k in range(n):
        inear = int(round(mx[k]))
        jnear = int(round(my[k]))
        s = 0
        for i in range(inear - 2, inear + 3):
            dx = abs(mx[k] - i)
            wx = float(delta_fn(np.array([dx]))[0])
            if wx == 0.0:
                continue
            for j in range(jnear - 2, jnear + 3):
                dy = abs(my[k] - j)
                w = wx * float(delta_fn(np.array([dy]))[0])
                if w == 0.0 or not (0 <= i < NX and 0 <= j < NY):
                    continue
                S[k, s] = i * NY + j
                Sw[k, s] = w
                s += 1
    return S, Sw


def compute_ds(mx, my):
    """min distance to any other marker of the same body (IBMNode.cpp:186-205)."""
    n = mx.size
    ds = np.empty(n)
    for k in range(n):
        d = np.hypot(mx - mx[k], my - my[k])
        d[k] = np.inf
        ds[k] = d.min()
    return ds


def compute_epsilon(mx, my, S, Sw, ds):
    """Gram A[i,j]=sum_s w_i(s)*d(x_j-x_i(s))*d(y_j-y_i(s))*ds_j; A eps=1
    (Objects.cpp:235-320)."""
    n = mx.size
    A = np.zeros((n, n))
    for i in range(n):
        for s in range(9):
            if Sw[i, s] == 0.0:
                continue
            si = S[i, s] // NY
            sj = S[i, s] % NY
            wi = Sw[i, s]
            for j in range(n):
                wj = float(life_delta_3pt(np.array([abs(mx[j] - si)]))[0]) * \
                    float(life_delta_3pt(np.array([abs(my[j] - sj)]))[0])
                A[i, j] += wi * wj
        A[i, :] *= ds
    return np.linalg.solve(A, np.ones(n))


class LifeLBM:
    def __init__(self):
        self.f = np.zeros((9, NY, NX))
        self.f_n = np.zeros((9, NY, NX))
        self.u = np.zeros((2, NY, NX))
        self.u_n = np.zeros((2, NY, NX))
        self.rho = np.ones((NY, NX))
        self.rho_n = np.ones((NY, NX))
        self.force_xy = np.zeros((2, NY, NX))
        self.force_ibm = np.zeros((2, NY, NX))
        R = H / 2.0
        self.u_in = np.zeros(NY)
        for j in range(NY):
            y = j * DX - R
            self.u_in[j] = 1.5 * (U_IN_P * DT / DX) * (1.0 - (y / R) ** 2)
        self.f = self._feq(self.rho, self.u)
        self.f_n = self.f.copy()

    def _feq(self, rho, u):
        ux, uy = u
        cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
        return (rho[None] * W[:, None, None] *
                (1.0 + 3.0 * cu +
                 4.5 * (ux ** 2 * (CX ** 2 - CS2)[:, None, None] +
                       uy ** 2 * (CY ** 2 - CS2)[:, None, None]) +
                 9.0 * CX[:, None, None] * CY[:, None, None] * ux * uy))

    def _lattice_force(self, u, force):
        ux, uy = u
        Fx, Fy = force
        cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
        return 3.0 * W[:, None, None] * (
            Fx[None] * (CX[:, None, None] - ux + 3.0 * CX[:, None, None] * cu) +
            Fy[None] * (CY[:, None, None] - uy + 3.0 * CY[:, None, None] * cu))

    def _macro(self):
        self.rho = self.f.sum(0)
        self.u[0] = (np.einsum('v,vji->ji', CX, self.f) + 0.5 * self.force_xy[0]) / self.rho
        self.u[1] = (np.einsum('v,vji->ji', CY, self.f) + 0.5 * self.force_xy[1]) / self.rho

    def _reg_reconstruct(self, i, j, n, nd, feq):
        cnd = CX if nd == 0 else CY
        f = self.f
        for v in np.where(cnd == n)[0]:
            o = OPP[v]
            f[v, i, j] = feq[v, i, j] + (f[o, i, j] - feq[o, i, j])
        fneq = f - feq
        Sxx = np.einsum('v,vji->ji', CX ** 2, fneq)
        Syy = np.einsum('v,vji->ji', CY ** 2, fneq)
        Sxy = np.einsum('v,vji->ji', CX * CY, fneq)
        stress = ((CX ** 2 - CS2) * Sxx + (CY ** 2 - CS2) * Syy + 2 * CX * CY * Sxy)
        f[:, i, j] = feq[:, i, j] + (W / (2.0 * CS2 ** 2)) * stress[:, i, j]

    def _corner(self, i, j, feq):
        nx = 1 if i == 0 else -1
        ny = 1 if j == 0 else -1
        self.rho_n[j, i] = 2 * self.rho[j + ny, i + nx] - self.rho[j + 2 * ny, i + 2 * nx]
        f = self.f
        for v in range(9):
            if C[v, 0] == nx or C[v, 1] == ny:
                if nx * C[v, 0] + ny * C[v, 1] == 0:
                    f[v, j, i] = feq[v, j, i]
                else:
                    o = OPP[v]
                    f[v, j, i] = feq[v, j, i] + (f[o, j, i] - feq[o, j, i])
        fneq = f - feq
        Sxx = np.einsum('v,vji->ji', CX ** 2, fneq)
        Syy = np.einsum('v,vji->ji', CY ** 2, fneq)
        Sxy = np.einsum('v,vji->ji', CX * CY, fneq)
        stress = ((CX ** 2 - CS2) * Sxx + (CY ** 2 - CS2) * Syy + 2 * CX * CY * Sxy)
        f[:, j, i] = feq[:, j, i] + (W / (2.0 * CS2 ** 2)) * stress[:, j, i]

    def step(self, t, ibm):
        self.f, self.f_n = self.f_n, self.f
        self.u, self.u_n = self.u_n, self.u
        self.rho, self.rho_n = self.rho_n, self.rho
        feq = self._feq(self.rho_n, self.u_n)
        lf = self._lattice_force(self.u_n, self.force_xy + self.force_ibm)
        fstar = self.f_n + OMEGA * (feq - self.f_n) + (1.0 - 0.5 * OMEGA) * lf
        f_new = np.empty_like(fstar)
        for v in range(9):
            f_new[v] = np.roll(fstar[v], shift=(int(CX[v]), int(CY[v])), axis=(1, 2))
        self.f = f_new
        self._macro()
        ramp = (1.0 - np.cos(np.pi * DT * t / RAMP)) / 2.0 if DT * t <= RAMP else 1.0
        self.u_n[0, :, 0] = self.u_in * ramp
        self.u_n[1, :, 0] = 0.0
        f = self.f
        fplus = f[CX == -1].sum(0)
        fzero = f[CX == 0].sum(0)
        self.rho_n[:, 0] = (2 * fplus[:, 0] + fzero[:, 0]) / (1.0 - self.u_n[0, :, 0])
        self._reg_reconstruct(0, slice(None), 1, 0, feq)
        fplus = f[CX == 1].sum(0)
        fzero = f[CX == 0].sum(0)
        self.u_n[0, :, NX - 1] = -1.0 * (1.0 - (2 * fplus[:, NX - 1] + fzero[:, NX - 1]) / self.rho_n[:, NX - 1])
        self._reg_reconstruct(NX - 1, slice(None), -1, 0, feq)
        self.u_n[:, 0, :] = 0.0
        fplus = f[CY == -1].sum(0)
        fzero = f[CY == 0].sum(0)
        self.rho_n[0, :] = 2 * fplus[0, :] + fzero[0, :]
        self._reg_reconstruct(slice(None), 0, 1, 1, feq)
        self.u_n[:, NY - 1, :] = 0.0
        fplus = f[CY == 1].sum(0)
        fzero = f[CY == 0].sum(0)
        self.rho_n[NY - 1, :] = 2 * fplus[NY - 1, :] + fzero[NY - 1, :]
        self._reg_reconstruct(slice(None), NY - 1, -1, 1, feq)
        for (ci, cj) in [(0, 0), (NX - 1, 0), (0, NY - 1), (NX - 1, NY - 1)]:
            self._corner(ci, cj, feq)
        ibm(self, t)


def make_ibm(kind):
    mx, my = build_markers()
    delta_fn = life_delta_3pt if kind == "life" else hat_delta
    S, Sw = build_support_dense(mx, my, delta_fn)
    if kind == "life":
        ds = compute_ds(mx, my)
        eps = compute_epsilon(mx, my, S, Sw, ds)
        scale = eps * ds
    else:
        scale = np.ones(NUM_NODES)

    def ibm(g, t):
        if kind == "life":
            irho = (Sw * g.rho.ravel()[S]).sum(1)
            imom0 = (Sw * (g.rho * g.u[0]).ravel()[S]).sum(1)
            imom1 = (Sw * (g.rho * g.u[1]).ravel()[S]).sum(1)
            force = -2.0 * (DT / DX) * np.stack([imom0, imom1])
        else:
            um0 = (Sw * g.u[0].ravel()[S]).sum(1)
            um1 = (Sw * g.u[1].ravel()[S]).sum(1)
            force = -np.stack([um0, um1])
        g.force_ibm[:] = 0.0
        np.add.at(g.force_ibm[0].ravel(), S, np.repeat(force[0], 9) * Sw.ravel())
        np.add.at(g.force_ibm[1].ravel(), S, np.repeat(force[1], 9) * Sw.ravel())
        sites = np.unique(S[Sw > 0])
        for sid in sites:
            J, I = sid // NX, sid % NX
            rho_t = g.f[:, J, I].sum()
            g.rho[J, I] = rho_t
            g.u[0][J, I] = (np.einsum('v,v', CX, g.f[:, J, I]) + 0.5 * (g.force_xy[0][J, I] + g.force_ibm[0][J, I])) / rho_t
            g.u[1][J, I] = (np.einsum('v,v', CY, g.f[:, J, I]) + 0.5 * (g.force_xy[1][J, I] + g.force_ibm[1][J, I])) / rho_t
        Fx = -np.sum(force[0] * scale) * FORCE_SCALE
        Fy = -np.sum(force[1] * scale) * FORCE_SCALE
        return Fx / ND, Fy / ND

    return ibm


def run_arm(arm_name, ibm, outdir, nsteps=NSTEPS):
    g = LifeLBM()
    cx, cy, ts = [], [], []
    for t in range(1, nsteps + 1):
        c = ibm(g, t)
        if t % 80 == 0:
            cx.append(c[0])
            cy.append(c[1])
            ts.append(t * DT)
        if t % 4000 == 0:
            print(f"{arm_name} t={t}/{nsteps}", flush=True)
    np.savez(os.path.join(outdir, f"{arm_name}_forces.npz"),
             t=np.array(ts), cx=np.array(cx), cy=np.array(cy))
    return ts, cx, cy


def kpis(ts, cx, cy):
    m = np.array(ts) >= 10.0
    cxw, cyw = np.array(cx)[m], np.array(cy)[m]
    dt = DT * 80
    f = np.fft.rfft(cyw - cyw.mean())
    freq = np.fft.rfftfreq(len(cyw), d=dt)
    k = np.argmax(np.abs(f[1:])) + 1
    st = freq[k] * 0.1 / U_IN_P
    return {"mean_Cx": float(cxw.mean()), "rms_Cy": float(np.sqrt((cyw ** 2).mean())),
            "St": float(st), "max_Cx": float(cxw.max())}


def parse_a1(path):
    d = np.loadtxt(path)
    return d[:, 1], d[:, 4], d[:, 5]


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    arm = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {"case": "LIFE Cylinder (rigid circle, Re_D=100)",
           "Nx": NX, "Ny": NY, "Dx": DX, "Dt": DT, "omega": OMEGA,
           "Re": U_IN_P * 0.1 / NU_P, "num_markers": NUM_NODES}
    if arm in ("all", "a1"):
        a1 = os.path.join(outdir, "A1_life_binary", "TotalForces.out")
        if os.path.exists(a1):
            ts, cx, cy = parse_a1(a1)
            res["A1"] = kpis(ts, cx, cy)
            np.savez(os.path.join(outdir, "A1_forces.npz"), t=ts, cx=cx, cy=cy)
    if arm in ("all", "a2"):
        ts, cx, cy = run_arm("A2", make_ibm("life"), outdir)
        res["A2"] = kpis(ts, cx, cy)
    if arm in ("all", "a3"):
        ts, cx, cy = run_arm("A3", make_ibm("native"), outdir)
        res["A3"] = kpis(ts, cx, cy)
    for a, b in [("A2", "A1"), ("A3", "A1")]:
        if a in res and b in res:
            res[f"{a}_vs_{b}_pct"] = {k: 100 * abs(res[a][k] - res[b][k]) / abs(res[b][k])
                                      for k in res[b] if res[b][k] != 0}
    with open(os.path.join(outdir, f"results_{arm}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
