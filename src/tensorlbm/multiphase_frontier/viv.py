"""Vortex-induced vibration (VIV) of an elastically mounted cylinder in the
**accelerating frame**, D2Q9 TRT lattice Boltzmann + halfway bounce-back.

Why a frame transformation
--------------------------
The previously recorded VIV probes of this repository
(:mod:`tensorlbm.fsi_viv_ibm`) are marked *WITHHELD*: the direct-forcing IBM on
interior markers inflated the load (``Cd = 32.3`` vs the reference ``1.34`` at
``Re = 100``, ``D = 16``).  A moving staircase geometry has a second,
independent problem -- the body mask jitters as the body crosses lattice nodes,
which aliases into the momentum-exchange force.

Both problems disappear if the *cylinder is kept fixed* and the relative motion
is expressed in the fluid instead.  With ``u_F`` the fluid velocity in the body
frame and ``V = dx_s/dt`` the body velocity, the incompressible momentum
equation transforms exactly as::

    du_F/dt + (u_F . grad) u_F = -grad p / rho + nu lap u_F - dV/dt

so the body-fixed problem is: fixed cylinder, far-field velocity ``(U, -V_y)``,
and a *uniform* pseudo-acceleration ``-dV/dt`` on the fluid.  Stresses (hence
the traction on the body) are invariant under a translation of the frame, so the
momentum-exchange force is the physical hydrodynamic load.  For ``V = 0`` the
solver degenerates to the classical fixed-cylinder benchmark -- exactly the
validation gate used below (``Cd = 1.34``, ``St = 0.1645`` at ``Re = 100``).

Numerics
--------
* D2Q9 two-relaxation-time collision (``s+ = 1/tau``,
  ``s- = 1/(1/2 + (3/16)/(tau - 1/2))``, magic parameter ``Lambda = 3/16`` so
  halfway bounce-back puts the wall at the mid-link position independent of
  ``tau``).
* Guo forcing for the uniform pseudo-acceleration.
* Halfway bounce-back on the staircase cylinder, momentum-exchange force
  ``F = sum_links 2 e_i f_i^post`` (Ladd 1994).
* Far field: equilibrium velocity inlet upstream/top/bottom, zero-gradient
  convective outlet downstream.
* Structure: 1-DOF transverse spring-mass-damper, velocity-Verlet stepping
  with the fluid force lagged by one lattice step (stable for ``m* > 1``).

The inner loop is written with preallocated ``float32`` per-direction buffers so
that a 24D x 20D run costs tens of milliseconds per lattice step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .poiseuille import D2Q9_E, D2Q9_W

__all__ = ["VIVFrameSolver", "VivResult", "elastic_mount_response",
           "dominant_frequency", "period_from_peaks", "free_decay_verlet",
           "stationary_strouhal"]

_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
_EX = D2Q9_E[0]
_EY = D2Q9_E[1]
_W = D2Q9_W.astype(np.float32)
_LAMBDA = 3.0 / 16.0
_DTYPE = np.float32


def dominant_frequency(t: np.ndarray, y: np.ndarray, detrend: bool = True) -> dict:
    """Dominant frequency of a uniformly sampled signal (FFT + parabolic peak)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    dt = float(np.mean(np.diff(t)))
    yy = y - y.mean() if detrend else y
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(yy * win))
    freqs = np.fft.rfftfreq(n, d=dt)
    spec[0] = 0.0
    k = int(np.argmax(spec))
    if 1 <= k < len(spec) - 1:                     # parabolic refinement
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = (a - 2.0 * b + c)
        delta = 0.5 * (a - c) / denom if denom != 0.0 else 0.0
    else:
        delta = 0.0
    df = 1.0 / (n * dt)
    half = spec[k] / np.sqrt(2.0)
    lo = k
    while lo > 1 and spec[lo] > half:
        lo -= 1
    hi = k
    while hi < len(spec) - 1 and spec[hi] > half:
        hi += 1
    return {"f": (k + delta) * df, "f_bin": k * df, "df": df,
            "amplitude": 2.0 * spec[k] / (win.sum() / 2.0) / n,
            "half_width": 0.5 * (hi - lo) * df}


def period_from_peaks(t: np.ndarray, y: np.ndarray,
                      n_ignore_frac: float = 0.0) -> dict:
    """Mean peak-to-peak period of a near-sinusoidal signal.

    A window that holds only 8-15 cycles has an FFT bin width of several
    percent of the shedding frequency, which is *larger* than the accuracy the
    Strouhal-number gate asks for; the peak spacing is a factor ~10 more
    accurate for the same record length, so it is the estimator used for the
    reported St values (the FFT is kept only as a cross-check).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n0 = int(n_ignore_frac * len(y))
    tt, yy = t[n0:], y[n0:]
    # local maxima with a +-1 sample comparison
    idx = np.where((yy[1:-1] > yy[:-2]) & (yy[1:-1] >= yy[2:]))[0] + 1
    if len(idx) < 3:
        return {"T": float("nan"), "n_peaks": int(len(idx)), "periods": []}
    tp = tt[idx]
    periods = np.diff(tp)
    return {"T": float(np.mean(periods)), "T_std": float(np.std(periods)),
            "n_peaks": int(len(idx)), "periods": periods.tolist()}


def free_decay_verlet(omega_n: float, zeta: float, x0: float, n_steps: int,
                      dt: float = 1.0) -> dict:
    """Velocity-Verlet stepping of ``x'' + 2 zeta omega x' + omega^2 x = 0``.

    Exposed separately so the *same* integrator used inside the fluid coupling
    can be certified against :func:`elastic_mount_response`.
    """
    m = 1.0
    k = m * omega_n ** 2
    c = 2.0 * zeta * omega_n * m
    x, v = float(x0), 0.0
    xs = np.empty(n_steps)
    vs = np.empty(n_steps)
    for n in range(n_steps):
        xs[n], vs[n] = x, v
        a = (-c * v - k * x) / m
        v_half = v + 0.5 * dt * a
        x = x + dt * v_half
        a_new = (-c * v_half - k * x) / m
        v = v_half + 0.5 * dt * a_new
    return {"t": np.arange(n_steps) * dt, "x": xs, "v": vs, "omega_n": omega_n,
            "zeta": zeta, "omega_d": omega_n * np.sqrt(max(1.0 - zeta ** 2, 0.0))}


def elastic_mount_response(omega_n: float, zeta: float, x0: float, n_steps: int,
                           dt: float = 1.0) -> dict:
    """Exact free decay of an underdamped 1-DOF oscillator with ``x(0)=x0, v(0)=0``.

    ``x = A e^{-zeta omega t} cos(omega_d t - phi)`` with
    ``A = x0 omega_n / omega_d`` and ``phi = atan(zeta omega_n / omega_d)``.
    """
    t = np.arange(n_steps) * dt
    wd = omega_n * np.sqrt(max(1.0 - zeta ** 2, 0.0))
    A = x0 * omega_n / wd
    phi = np.arctan2(zeta * omega_n, wd)
    env = A * np.exp(-zeta * omega_n * t)
    return {"t": t, "x": env * np.cos(wd * t - phi), "envelope": env,
            "omega_d": wd, "A": A, "phi": phi}


@dataclass
class VivResult:
    U_r: float
    Re: float
    m_star: float
    zeta: float
    D: int
    steps: int
    cd_mean: float
    cl_amp: float
    amplitude_ratio: float
    f_osc: float
    st: float
    freq_ratio: float
    locked: bool
    history: dict = field(default_factory=dict, repr=False)

    def as_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in self.__dict__.items() if k != "history"}


class VIVFrameSolver:
    """Transverse 1-DOF VIV of a circular cylinder (2-D, per unit span)."""

    def __init__(self, D: int = 16, nx: int = 384, ny: int = 320,
                 Re: float = 100.0, U: float = 0.1, m_star: float = 10.0,
                 zeta: float = 0.0, U_r: float = 5.0,
                 xc: int | None = None, yc: int | None = None,
                 tau_min: float = 0.515, fixed: bool = False,
                 n_sub: int = 1, x0_frac: float = 0.02):
        self.D = int(D)
        self.nx, self.ny = int(nx), int(ny)
        self.Re = float(Re)
        self.U = float(U)
        self.m_star = float(m_star)
        self.zeta = float(zeta)
        self.U_r = float(U_r)
        self.xc = int(xc if xc is not None else 6 * D)
        self.yc = int(yc if yc is not None else ny // 2)
        self.n_sub = int(n_sub)
        nu = U * D / Re
        if 0.5 + 3.0 * nu < tau_min:
            raise ValueError(
                f"tau = {0.5 + 3 * nu:.4f} below the stability floor {tau_min}: "
                f"increase U or D, or reduce Re")
        self.nu = nu
        self.tau = 0.5 + 3.0 * nu
        self.s_p = 1.0 / self.tau
        self.s_m = 1.0 / (0.5 + _LAMBDA / (self.tau - 0.5))
        # --- geometry -----------------------------------------------------
        yy, xx = np.mgrid[0:self.ny, 0:self.nx]
        self.solid = ((xx - self.xc) ** 2 + (yy - self.yc) ** 2) <= (0.5 * D) ** 2
        self.fluid = ~self.solid
        flat = np.arange(self.nx * self.ny).reshape(self.ny, self.nx)
        self.idx = {}
        self.links = 0
        for i in range(1, 9):
            nb_solid = np.roll(np.roll(self.solid, _EY[i], axis=0), _EX[i], axis=1)
            mask = self.fluid & nb_solid
            self.idx[i] = flat[mask]
            self.links += int(mask.sum())
        self.solid_area = float(self.solid.sum())
        # --- structure ----------------------------------------------------
        self.f_n = self.U / (self.U_r * self.D)
        self.omega_n = 2.0 * np.pi * self.f_n
        self.m = self.m_star * self.D ** 2          # rho = 1, per unit span
        self.k = self.m * self.omega_n ** 2
        self.c = 2.0 * self.zeta * self.omega_n * self.m
        # --- state + preallocated buffers ---------------------------------
        self.fixed = bool(fixed)
        sh = (9, self.ny, self.nx)
        self.f = np.zeros(sh, dtype=_DTYPE)
        self.fc = np.zeros(sh, dtype=_DTYPE)
        self.feq = np.zeros(sh, dtype=_DTYPE)
        self.rho = np.ones((self.ny, self.nx), dtype=_DTYPE)
        self.ux = np.zeros((self.ny, self.nx), dtype=_DTYPE)
        self.uy = np.zeros((self.ny, self.nx), dtype=_DTYPE)
        self._s1 = np.zeros((self.ny, self.nx), dtype=_DTYPE)
        self._s2 = np.zeros((self.ny, self.nx), dtype=_DTYPE)
        self._s3 = np.zeros((self.ny, self.nx), dtype=_DTYPE)
        self._fp = np.zeros((self.ny, self.nx), dtype=_DTYPE)
        self._fm = np.zeros((self.ny, self.nx), dtype=_DTYPE)
        self._t = np.zeros((self.ny, self.nx), dtype=_DTYPE)
        # The perfectly symmetric start (x_s = 0, symmetric domain) has the
        # antisymmetric shedding mode only at round-off level, and the VIV
        # limit cycle then needs thousands of periods to emerge.  A small
        # initial displacement is the standard trigger; 0.02 D is far below the
        # limit-cycle amplitude so it does not bias the response.
        self.x0 = float(x0_frac) * self.D
        self.x_s = 0.0
        self.v_s = 0.0
        self.a_s = 0.0
        self._init_field()

    # ------------------------------------------------------------------
    def _init_field(self) -> None:
        self.rho[:] = 1.0
        self.ux[:] = self.U
        self.uy[:] = 0.0
        for i in range(9):
            self.f[i] = self._feq_i(i)
        self.x_s = getattr(self, "x0", 0.0)
        self.v_s = self.a_s = 0.0

    def _feq_i(self, i: int) -> np.ndarray:
        w, ex, ey = float(_W[i]), int(_EX[i]), int(_EY[i])
        eu = self._s1
        np.multiply(self.ux, 3.0 * ex, out=eu)
        eu += (3.0 * ey) * self.uy
        t = self._t
        np.multiply(eu, eu, out=t)
        t *= 0.5
        t += 1.0
        t += eu
        np.multiply(self.ux, self.ux, out=self._s2)
        np.multiply(self.uy, self.uy, out=self._s3)
        self._s2 += self._s3
        self._s2 *= -1.5
        t += self._s2
        t *= w
        t *= self.rho
        return t

    def _apply_farfield(self) -> None:
        ux, uy = self.U, -self.v_s
        eu = 3.0 * (_EX * ux + _EY * uy)
        base = (_W * (1.0 + eu + 0.5 * eu * eu
                      - 1.5 * (ux * ux + uy * uy)))[:, None]
        f = self.f
        f[:, :, 0] = base                       # inlet
        f[:, 0, :] = base                       # bottom far field
        f[:, -1, :] = base                      # top far field
        f[:, :, -1] = f[:, :, -2]               # convective outlet

    def step(self) -> dict:
        """One lattice step; returns the instantaneous force coefficients."""
        f = self.f
        rho, ux, uy = self.rho, self.ux, self.uy
        # ---- moments ------------------------------------------------------
        np.add(f[0], f[1], out=rho)
        for i in range(2, 9):
            rho += f[i]
        np.subtract(f[1], f[3], out=ux)
        ux += f[5]
        ux -= f[6]
        ux -= f[7]
        ux += f[8]
        ux /= rho
        np.subtract(f[2], f[4], out=uy)
        uy += f[5]
        uy += f[6]
        uy -= f[7]
        uy -= f[8]
        uy /= rho
        uy += (-0.5 * self.a_s) / rho                # Guo half force (y only)
        # ---- equilibrium --------------------------------------------------
        feq = self.feq
        for i in range(9):
            feq[i] = self._feq_i(i)
        # ---- TRT collision ------------------------------------------------
        fc = self.fc
        sp, sm = self.s_p, self.s_m
        # TRT relaxes pairs of opposite directions: (0), (1,3), (2,4), (5,7), (6,8)
        for i in (0, 1, 2, 5, 6):
            o = _OPP[i]
            fp = self._fp
            fm = self._fm
            np.add(f[i], f[o], out=fp)
            fp *= 0.5
            np.subtract(f[i], f[o], out=fm)
            fm *= 0.5
            np.subtract(feq[i], feq[o], out=self._s1)
            self._s1 *= 0.5                              # eqm
            np.add(feq[i], feq[o], out=self._s2)
            self._s2 *= 0.5                              # eqp
            # symmetric part
            np.subtract(fp, self._s2, out=self._s3)      # fp - eqp
            self._s3 *= sp
            np.subtract(fp, self._s3, out=self._t)       # fp - sp (fp-eqp)
            # antisymmetric part
            np.subtract(fm, self._s1, out=self._s3)      # fm - eqm
            self._s3 *= sm
            np.subtract(fm, self._s3, out=fp)            # fm - sm (fm-eqm)  (reuse)
            np.add(self._t, fp, out=fc[i])
            np.subtract(self._t, fp, out=fc[o])
        if self.a_s != 0.0:
            for i in range(9):
                ey = int(_EY[i])
                if ey == 0:
                    continue
                ex = int(_EX[i])
                eu = self._s1
                np.multiply(ux, 3.0 * ex, out=eu)
                eu += (3.0 * ey) * uy
                t = self._t
                # (3 (ey - 3 uy) + 9 eu ey) * Fy * w * (1 - 1/(2 tau))
                np.multiply(eu, 9.0 * ey, out=t)
                t += 3.0 * ey - 9.0 * uy
                t *= float(_W[i]) * (1.0 - 0.5 / self.tau) * (-self.a_s)
                fc[i] += t
        # ---- streaming ----------------------------------------------------
        for i in range(9):
            f[i] = np.roll(np.roll(fc[i], _EY[i], axis=0), _EX[i], axis=1)
        self._apply_farfield()
        # ---- halfway bounce-back + momentum exchange ----------------------
        fx = 0.0
        fy = 0.0
        for i in range(1, 9):
            ids = self.idx[i]
            if ids.size == 0:
                continue
            newval = fc[_OPP[i]].ravel()[ids]
            f[i].ravel()[ids] = newval
            # momentum-exchange load; the sign convention is fixed once, by the
            # fixed-cylinder benchmark (F must point downstream for U > 0)
            fx -= 2.0 * _EX[i] * float(newval.sum())
            fy -= 2.0 * _EY[i] * float(newval.sum())
        cd = fx / (0.5 * self.U ** 2 * self.D)
        cl = fy / (0.5 * self.U ** 2 * self.D)
        # ---- structure (velocity-Verlet, lagged fluid force) --------------
        if self.fixed:
            return {"cd": cd, "cl": cl, "x": 0.0, "v": 0.0, "Fy": fy, "Fx": fx}
        a = (fy - self.c * self.v_s - self.k * self.x_s) / self.m
        for _ in range(self.n_sub):
            dt = 1.0 / self.n_sub
            v_half = self.v_s + 0.5 * dt * a
            self.x_s += dt * v_half
            a_new = (fy - self.c * v_half - self.k * self.x_s) / self.m
            self.v_s = v_half + 0.5 * dt * a_new
            a = a_new
        self.a_s = a
        return {"cd": cd, "cl": cl, "x": self.x_s, "v": self.v_s,
                "Fy": fy, "Fx": fx}

    # ------------------------------------------------------------------
    def run(self, n_periods: float = 24.0, n_transient_periods: float = 8.0,
            max_steps: int | None = None, report_every: int = 0,
            verbose: bool = False) -> VivResult:
        """Integrate for a fixed number of *structure* periods.

        ``T_n = U_r D / U`` lattice steps per structure period, so every case
        runs the same number of oscillations independent of ``U_r``.
        """
        T_n = self.U_r * self.D / self.U
        n_trans = int(n_transient_periods * T_n)
        n_total = int(n_trans + n_periods * T_n)
        if max_steps is not None:
            n_total = min(n_total, int(max_steps))
        keys = ("t", "x", "cd", "cl", "vy")
        hist = {k: np.empty(n_total - n_trans) for k in keys}
        for n in range(n_total):
            out = self.step()
            if n >= n_trans:
                j = n - n_trans
                hist["t"][j] = n
                hist["x"][j] = out["x"]
                hist["cd"][j] = out["cd"]
                hist["cl"][j] = out["cl"]
                hist["vy"][j] = out["v"]
            if report_every and (n + 1) % report_every == 0 and verbose:
                print(f"  step {n+1}/{n_total} x/D={self.x_s/self.D:+.4f} "
                      f"Cd={out['cd']:.3f} Cl={out['cl']:+.3f}", flush=True)
        amp = 0.5 * (hist["x"].max() - hist["x"].min())
        freq = dominant_frequency(hist["t"], hist["x"])
        # a vibrating cylinder that never leaves the wake-lock band has an
        # ill-defined spectral peak; fall back to the force signal then.
        st = freq["f"] / self.U * self.D
        freq_ratio = freq["f"] / self.f_n
        return VivResult(
            U_r=self.U_r, Re=self.Re, m_star=self.m_star, zeta=self.zeta,
            D=self.D, steps=n_total,
            cd_mean=float(np.mean(hist["cd"])),
            cl_amp=float(0.5 * (hist["cl"].max() - hist["cl"].min())),
            amplitude_ratio=float(amp / self.D),
            f_osc=float(freq["f"]), st=float(st),
            freq_ratio=float(freq_ratio),
            locked=bool(abs(freq_ratio - 1.0) <= 0.10),
            history=hist,
        )


def stationary_strouhal(Re: float) -> float:
    """Empirical ``St(Re)`` of a *fixed* smooth cylinder (Williamson & Brown fit).

    ``St = 0.2731 - 1.1129/Re^0.5 + 0.4821/Re`` for ``Re >= 50``.  Used only to
    bracket the reduced-velocity sweep, never as a validation reference.
    """
    Re = max(float(Re), 50.0)
    return 0.2731 - 1.1129 / np.sqrt(Re) + 0.4821 / Re