"""Faithful, guardrailed port of the OSS Shan-Chen Young-Laplace calibrator.

Anchor: XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION (round 1).
OSS: ``lbm-principles-practice/code`` (MIT), file ``chapter9/shanchen.cpp``.

Zero hand-copy: every ported practice below is taken from the *inventoried*
source of that file produced by the OSS-inventory campaign
(``TensorPlatform/docs/oss-inventory/REPORT_lbm-principles-practice-code.md``,
§4.1, §4.3, §九), which cites the upstream lines.  The upstream C++ file itself
is not vendored in this work tree, so this module is a re-implementation of the
**documented** practices, not a line-by-line transcription.  The port is
validated a posteriori by reproducing the inventoried numbers
(§4.3: rho_l/rho_g = 9.05, gamma = 0.020728 at R_init=16; §九 table 1:
gamma_LSQR = 0.020268, r^2 = 0.988 over R = 12/16/20).

Ported practices (report line cites -> upstream lines)
-----------------------------------------------------
* pseudopotential ``psi(rho) = rho0 [1 - exp(-rho/rho0)]``, ``rho0 = 1``
  (report §4.1 -> ``chapter9/shanchen.cpp:74-76``, ``:37``).
* Shan-Chen force ``F = -gA psi(x) sum_{i!=0} w_i c_i psi(x+c_i)``, eight
  nearest neighbours, rest direction excluded (report §4.1 -> ``:79-109``,
  ``:90``), periodic wrap (``:91-92``).
* Guo 2nd-order forcing ``w_i (1 - 0.5*omega) [(3(c-u) + 9 c(c.u)) . F]``
  (report §4.1 -> ``:261-263``), BGK relaxation ``omega = 1/tau`` (``:252-254``).
* Barycentric velocity with the half-force shift ``u = (sum_i c_i f_i + F/2)/rho``
  (report §4.1 -> ``:142-161``, ``:151-152``).
* Pressure closure ``p = sum_s rho_s/3 + sum_{s,ss} g[s][ss] psi_s psi_ss/6``
  (report §4.1 -> ``:126-139``, interaction term at ``:134``); for the
  single-component case this is ``p = rho/3 + gA psi^2/6``.
* Calibrator ``calculateDeltaPressure()``: 4x4-node box averages for the gas
  (corner) and liquid (drop centre) pressure, interface located along
  ``y = ny/2`` by *linear interpolation* of the ``rho = rho_av`` crossing with
  ``rho_av = (rho_gas + rho_liq)/2``, output ``gamma = dp * r``
  (report §4.1/§4.3 -> ``:326-377``, ``:333-353``, ``:357-367``, ``:369``).

Deliberate deviation (declared): the upstream radius estimator is
``double rad;`` **uninitialised** at ``chapter9/shanchen.cpp:358`` and prints an
undefined ``gamma`` when the scan does not break (measured: R=24 ->
``r = 5207, gamma = -4.86``, report §4.3).  Here the estimator raises
:class:`RadiusEstimatorFailed` instead of returning a garbage number
("吸收该标定器必须补护栏", report §4.3).  That guardrail is the only semantic
change; it never fires for a resolved droplet.
"""

from __future__ import annotations

import numpy as np

CS2 = 1.0 / 3.0
# D2Q9, zero direction first (the convention the report attributes to
# chapter9/shanchen.cpp:44-46); the rest direction is excluded from the
# interaction sum, exactly as in the upstream computeSCForces loop.
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float64)
W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4, dtype=np.float64)


class RadiusEstimatorFailed(RuntimeError):
    """Upstream UB guardrail: no rho = rho_av crossing on the scan line."""


def psi_exp(rho: np.ndarray, rho0: float = 1.0) -> np.ndarray:
    """``psi(rho) = rho0 (1 - exp(-rho/rho0))`` (shanchen.cpp:74-76)."""
    return rho0 * (1.0 - np.exp(-rho / rho0))


def feq(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    usq = ux * ux + uy * uy
    return W[:, None, None] * rho[None] * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])


def _roll(field: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """``field(x + d)`` with periodic wrap (array axes are (y, x))."""
    return np.roll(field, shift=(-dy, -dx), axis=(0, 1))


class OSSShanChen:
    """SCMP droplet solver following the documented OSS loop order.

    One step: forces from rho -> velocity with half-force shift -> Guo-forced
    BGK collision -> periodic streaming (shanchen.cpp:398-... main loop).
    """

    def __init__(self, n: int = 64, tau: float = 1.0, gA: float = -4.7,
                 rho0: float = 1.0, rhol: float = 2.1, rhog: float = 0.15,
                 nfluid: int = 1):
        self.n = int(n)
        self.tau = float(tau)
        self.omega = 1.0 / self.tau
        self.gA = float(gA)
        self.rho0 = float(rho0)
        self.rhol = float(rhol)
        self.rhog = float(rhog)
        self.nfluid = int(nfluid)
        self.f = np.zeros((9, self.n, self.n), dtype=np.float64)
        self.steps = 0

    # -- initialization: sharp droplet, equilibrium populations, u = 0 --------
    def init_droplet(self, R: float) -> None:
        """rho = rhol inside radius R, rhog outside (shanchen.cpp:201-211);
        f = feq(rho, 0, 0) (shanchen.cpp:232-246)."""
        y = np.arange(self.n, dtype=np.float64)[:, None]
        x = np.arange(self.n, dtype=np.float64)[None, :]
        c = 0.5 * (self.n - 1)
        r = np.sqrt((x - c) ** 2 + (y - c) ** 2)
        rho = np.where(r <= R, self.rhol, self.rhog)
        self.f = feq(rho, np.zeros_like(rho), np.zeros_like(rho))
        self.steps = 0

    def rho(self) -> np.ndarray:
        return self.f.sum(axis=0)

    # -- force (shanchen.cpp:79-109) -----------------------------------------
    def force(self) -> tuple[np.ndarray, np.ndarray]:
        psi = psi_exp(self.rho(), self.rho0)
        sx = np.zeros_like(psi)
        sy = np.zeros_like(psi)
        for i in range(1, 9):  # rest direction excluded (:90)
            nb = _roll(psi, int(CX[i]), int(CY[i]))
            sx += W[i] * CX[i] * nb
            sy += W[i] * CY[i] * nb
        return -self.gA * psi * sx, -self.gA * psi * sy

    def velocity(self, Fx: np.ndarray, Fy: np.ndarray):
        """Barycentric velocity + half-force shift (shanchen.cpp:142-161)."""
        rho = self.rho()
        mx = (CX[:, None, None] * self.f).sum(axis=0) + 0.5 * Fx
        my = (CY[:, None, None] * self.f).sum(axis=0) + 0.5 * Fy
        rho_s = np.clip(rho, 1e-12, None)
        return rho, mx / rho_s, my / rho_s

    def step(self) -> None:
        Fx, Fy = self.force()
        rho, ux, uy = self.velocity(Fx, Fy)
        fe = feq(rho, ux, uy)
        cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
        cf = CX[:, None, None] * Fx[None] + CY[:, None, None] * Fy[None]
        uf = ux * Fx + uy * Fy
        src = W[:, None, None] * (1.0 - 0.5 * self.omega) * (
            3.0 * (cf - uf[None]) + 9.0 * cu * cf)
        fpost = (1.0 - self.omega) * self.f + self.omega * fe + src
        # periodic streaming (pull convention == push, shanchen.cpp:271-272 wrap)
        out = np.empty_like(fpost)
        for i in range(9):
            out[i] = np.roll(fpost[i], shift=(int(CY[i]), int(CX[i])), axis=(0, 1))
        self.f = out
        self.steps += 1

    def run(self, nsteps: int) -> None:
        for _ in range(int(nsteps)):
            self.step()


# -- calibrator (shanchen.cpp:326-377) ---------------------------------------

def closure_pressure(rho: np.ndarray, gA: float, rho0: float = 1.0) -> np.ndarray:
    """``p = rho/3 + gA psi^2/6`` (single-component case of ``:126-139``)."""
    ps = psi_exp(rho, rho0)
    return rho * CS2 + gA * ps * ps / 6.0


def box_mean(field: np.ndarray, y0: int, x0: int, size: int = 4) -> float:
    """4x4-node box average with periodic wrap (``:333-353``)."""
    n = field.shape[0]
    ys = [(y0 + i) % n for i in range(size)]
    xs = [(x0 + i) % n for i in range(size)]
    return float(field[np.ix_(ys, xs)].mean())


def radius_midline(rho: np.ndarray, rho_av: float, y: int | None = None,
                   two_sided: bool = False) -> float:
    """Radius from the linearised ``rho = rho_av`` crossing (``:357-367``).

    Scans outward from the drop centre along ``y = ny/2``.  If the scan finds no
    crossing we raise (upstream prints an undefined ``gamma``, ``:358``).
    """
    ny, nx = rho.shape
    y = ny // 2 if y is None else y
    c = nx // 2
    row = rho[y]

    def _scan(sign: int) -> float:
        prev = row[c]
        for j in range(1, nx // 2):
            cur = row[c + sign * j]
            if (prev - rho_av) * (cur - rho_av) <= 0.0 and cur != prev:
                frac = (rho_av - prev) / (cur - prev)
                return float(j - 1 + frac)
            prev = cur
        raise RadiusEstimatorFailed(
            "no rho = rho_av crossing on the centre line (upstream "
            "uninitialised-rad UB at shanchen.cpp:358)")

    rp = _scan(+1)
    if not two_sided:
        return rp
    try:
        rm = _scan(-1)
    except RadiusEstimatorFailed:
        return rp
    return 0.5 * (rp + rm)


def calibrate(f: np.ndarray, gA: float, rho0: float = 1.0, box: int = 4,
              two_sided: bool = False) -> dict[str, float]:
    """Return ``rho_g, rho_l, dp, r, gamma`` exactly as ``:369-374`` prints."""
    rho = f.sum(axis=0)
    p = closure_pressure(rho, gA, rho0)
    n = rho.shape[0]
    c = n // 2
    off = box // 2
    rho_l = box_mean(rho, c - off, c - off, box)
    rho_g = box_mean(rho, 0, 0, box)
    p_l = box_mean(p, c - off, c - off, box)
    p_g = box_mean(p, 0, 0, box)
    rho_av = 0.5 * (rho_l + rho_g)
    r = radius_midline(rho, rho_av, two_sided=two_sided)
    dp = p_l - p_g
    return {"rho_l": rho_l, "rho_g": rho_g, "rho_ratio": rho_l / rho_g,
            "p_l": p_l, "p_g": p_g, "dp": dp, "r": r, "gamma": dp * r,
            "rho_av": rho_av}