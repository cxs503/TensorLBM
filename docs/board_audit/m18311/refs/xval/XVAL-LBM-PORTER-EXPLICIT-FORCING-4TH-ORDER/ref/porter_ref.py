"""Reference arm: literal transcription of the documented reference algorithm.

Independent of ``src/tensorlbm/porter_forcing.py`` on purpose (separate lattice
constants, separate per-direction loops, no shared helper).  Written from the
four documented steps (see the module docstring of the library implementation and
``docs/oss-inventory-openLBMPM.md`` §II.2):

  F_i = (F.(c_i-u_eq)) f_i^eq/(rho cs^2)
  f <- f - 0.5 F_i
  f <- f + (f_eq - f - 0.5 F_i)/tau + F_i
  u   = [sum_k sum_i f_ik c_k + 0.5 (F_0+F_1)]/rho_tot
  u_eq= sum_i[(sum_k f_ik c_k + 0.5 F_i)/tau_i]/sum_i(rho_i/tau_i)

Everything is vectorised only over nodes; the nine directions are looped so the
code mirrors the pseudocode line by line.
"""

from __future__ import annotations

import numpy as np

cs2 = 1.0 / 3.0

# D2Q9 direction table, same numbering as the host library
ndir = 9
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], float)
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], float)
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def _feq(rho, ux, uy):
    out = np.empty((9,) + rho.shape)
    usq = ux**2 + uy**2
    for i in range(ndir):
        cu = ex[i] * ux + ey[i] * uy
        out[i] = w[i] * rho * (1.0 + cu / cs2 + (cu * cu - cs2 * usq) / (2.0 * cs2 * cs2))
    return out


def _momentum(f):
    jx = np.zeros(f.shape[1:])
    jy = np.zeros(f.shape[1:])
    for i in range(ndir):
        jx = jx + f[i] * ex[i]
        jy = jy + f[i] * ey[i]
    return jx, jy


def _force_pdf(Fx, Fy, feq, rho, ueqx, ueqy):
    Fpdf = np.empty_like(feq)
    usq = ueqx * ueqx + ueqy * ueqy
    for i in range(ndir):
        proj = (ex[i] - ueqx) * Fx + (ey[i] - ueqy) * Fy
        cu = ex[i] * ueqx + ey[i] * ueqy
        shape = w[i] * (1.0 + cu / cs2 + (cu * cu - cs2 * usq) / (2.0 * cs2 * cs2))
        Fpdf[i] = proj * shape / cs2
    return Fpdf


def _stream(f, solid):
    ny, nx = f.shape[1], f.shape[2]
    jj = np.arange(ny)[:, None]
    ii = np.arange(nx)[None, :]
    new = np.zeros_like(f)
    for i in range(ndir):
        dy, dx = int(ey[i]), int(ex[i])
        sj = jj - dy
        si = (ii - dx) % nx
        inside = (sj >= 0) & (sj < ny)
        sjc = np.clip(sj, 0, ny - 1)
        src = f[i][sjc, si]
        src_solid = solid[sjc, si]
        # where the upstream node is solid the population is reflected in place
        new[i] = np.where(inside, np.where(src_solid, f[opp[i]], src), 0.0)
    new[:, solid] = 0.0
    return new


def _advance(fs, rhs, solid, Fs, taus, ueqx, ueqy, feqs, Fpdfs, variant="doc_literal"):
    """One collision + streaming step for the list of components.

    Two readings of the documented sequence are provided because the inventory
    record is ambiguous about whether the ``-0.5 F`` inside the relaxation term
    is the *already applied* pre-collision subtraction or a second one:

    ``doc_literal``  f <- f - 0.5F ; f <- f + (f_eq - f - 0.5F)/tau + F
    ``doc_seq``      f <- f - 0.5F ; f <- f + (f_eq - f)/tau + F
    """
    for s in range(len(fs)):
        f = fs[s]
        Fpdf = Fpdfs[s]
        # step 1: half force before collision
        f = f - 0.5 * Fpdf
        # step 2: collide, add the full force
        if variant == "doc_literal":
            f = f + (feqs[s] - f - 0.5 * Fpdf) / taus[s] + Fpdf
        elif variant == "doc_seq":
            f = f + (feqs[s] - f) / taus[s] + Fpdf
        elif variant == "guo":
            f = f + 0.5 * Fpdf  # undo the pre-collision half force
            f = f + (feqs[s] - f) / taus[s] + (1.0 - 0.5 / taus[s]) * Fpdf
        else:
            raise ValueError(variant)
        fs[s] = f
    out = [_stream(fs[s], solid) for s in range(len(fs))]
    return out


def run(ny, nx, taus, rho_init, gx, gy, steps, check_every=200, tol=1e-9, variant="doc_literal", ux0=None):
    """Run a forced two-component channel flow.

    Parameters are all lattice units.  ``rho_init`` is a list of per-component
    density fields; ``gx``/``gy`` are per-component body accelerations.
    Returns ``(ux, uy, rho_tot, n_steps, history)``.
    """
    rho = [np.array(r, float) for r in rho_init]
    solid = np.zeros((ny, nx), bool)
    solid[0, :] = True
    solid[-1, :] = True
    for r in rho:  # solid rows hold no fluid
        r[0, :] = 0.0
        r[-1, :] = 0.0

    fs = []
    u_init = np.zeros((ny, nx)) if ux0 is None else np.asarray(ux0, float).reshape(ny, nx)
    for s in range(len(rho)):
        fs.append(_feq(rho[s], u_init, np.zeros((ny, nx))))

    history = []
    prev = None
    n_done = steps
    for t in range(1, steps + 1):
        # forces
        Fxs = [1.0 * gx[s] * rho[s] for s in range(len(rho))]
        Fys = [1.0 * gy[s] * rho[s] for s in range(len(rho))]
        jxs, jys = [], []
        for s in range(len(rho)):
            jx, jy = _momentum(fs[s])
            jxs.append(jx)
            jys.append(jy)
        rho_tot = sum(rho)
        # piece 3: physical velocity
        ux = (sum(jxs) + 0.5 * sum(Fxs)) / np.maximum(rho_tot, 1e-30)
        uy = (sum(jys) + 0.5 * sum(Fys)) / np.maximum(rho_tot, 1e-30)
        # piece 4: viscosity weighted equilibrium velocity
        numx = sum((jxs[s] + 0.5 * Fxs[s]) / taus[s] for s in range(len(rho)))
        numy = sum((jys[s] + 0.5 * Fys[s]) / taus[s] for s in range(len(rho)))
        den = sum(rho[s] / taus[s] for s in range(len(rho)))
        ueqx = numx / np.maximum(den, 1e-30)
        ueqy = numy / np.maximum(den, 1e-30)

        feqs = [_feq(rho[s], ueqx, ueqy) for s in range(len(rho))]
        Fpdfs = [_force_pdf(Fxs[s], Fys[s], feqs[s], rho[s], ueqx, ueqy) for s in range(len(rho))]
        fs = _advance(fs, rho, solid, None, taus, ueqx, ueqy, feqs, Fpdfs, variant=variant)
        # density of each component from its own streamed populations
        rho = [fs[s].sum(axis=0) for s in range(len(rho))]

        if t % check_every == 0:
            amp = np.max(np.abs(ux))
            amp = max(amp, 1e-30)
            if prev is not None:
                delta = np.max(np.abs(ux - prev)) / amp
                history.append((t, float(delta), float(amp)))
                if delta < tol:
                    n_done = t
                    break
            prev = ux.copy()
    return ux, uy, rho_tot, n_done, history