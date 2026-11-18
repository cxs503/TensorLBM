import io, math, pathlib, sys

p = pathlib.Path("/srv/tensor/agent/work/m20929/src/tensorlbm/phasefield/laplace_droplet_calibration.py")
src = p.read_text()

def rep(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:80])
    src = src.replace(old, new)


# ---------------------------------------------------------------- __all__
rep(
    '''    "bulk_pressure_jump",
    "run_droplet_case",
    "grade_case",
    "grade_report",
]''',
    '''    "bulk_pressure_jump",
    "capillary_pressure_from_profile",
    "diffuse_interface_correction",
    "run_droplet_case",
    "grade_case",
    "grade_report",
    "default_cases",
    "smoke_cases",
    "run_suite",
]''',
)

# ---------------------------------------------------- analytic DP reference
rep(
    '''# --------------------------------------------------------------------------- #
# configuration / result containers
# --------------------------------------------------------------------------- #''',
    '''def capillary_pressure_from_profile(
    R: float, A: float, kappa: float, *, n: int = 400001, r_span: float = 8.0
) -> float:
    """Curved-interface (cylinder) pressure jump of the *analytic* tanh profile.

    For a radially symmetric field the Korteweg part of the pressure tensor is
    ``P_rr - P_tt = kappa (d phi/dr)^2``, so the exact radial momentum balance
    ``dP_rr/dr + (P_rr - P_tt)/r = 0`` integrated across the interface gives

        ``dp = integral_0^inf kappa (d phi/dr)^2 / r  dr``

    which is **not** exactly ``sigma / R`` for a diffuse interface: it is the
    finite-``xi`` reference used to separate discretisation error from the
    curvature correction.  The integral is evaluated on ``phi = -tanh((r-R)/xi)``
    (the profile that extremises the free energy), never on simulated data.
    """
    _require_positive(R=R, A=A, kappa=kappa)
    xi = xi_analytic(A, kappa)
    if n < 3:
        raise ValueError("n must be >= 3 for the trapezoid rule")
    r = torch.linspace(0.0, r_span * R, n, dtype=torch.float64)
    sech2 = 1.0 / torch.cosh((r - R) / xi) ** 2
    weight = (r[1] - r[0]).item()
    integrand = kappa * (sech2 / xi) ** 2
    return float((integrand / torch.clamp(r, min=1.0e-9)).sum().item() * weight)


def diffuse_interface_correction(
    R: float, A: float, kappa: float, *, n: int = 400001
) -> float:
    """Ratio ``(dp_profile * R) / sigma`` for the analytic diffuse interface.

    Equals 1 in the sharp-interface limit ``R/xi -> inf`` and grows like
    ``1 + c (xi/R)^2`` for finite ``xi`` (``+2.2 %`` at ``R/xi = 4``,
    ``+0.9 %`` at ``R/xi = 6`` for this free-energy functional).  It is the
    reference a finite-``xi`` simulation should approach, tighter than the plain
    sharp-interface law.
    """
    return capillary_pressure_from_profile(R, A, kappa, n=n) * R / sigma_analytic(A, kappa)


# --------------------------------------------------------------------------- #
# configuration / result containers
# --------------------------------------------------------------------------- #''',
)

# --------------------------------------------------------------- DropletCase
rep(
    '''    tau_f: float = 1.0
    tau_phi: float = 1.0
    steps: int = 2000
    sample_every: int = 50
    bulk_offset_xi: float = 3.0
    bin_width: float = 0.25
''',
    '''    tau_f: float = 1.0
    #: Order-parameter relaxation time.  The default is 2.0 (not 1.0): the explicit
    #: Cahn-Hilliard branch of the production kernel is *measured* to blow up at
    #: ``tau_phi = 1.0`` for ``A = 0.1, kappa = 0.2`` (divergence at step ~350 with
    #: ``Gamma = 0.05``) and at any ``tau_phi`` for ``Gamma >= 1.0``.
    tau_phi: float = 2.0
    steps: int = 2000
    sample_every: int = 50
    #: Number of trailing steps over which the steady-state observables are
    #: time-averaged.  A single-snapshot reading of the bulk pressure is noisy
    #: (measured +-10..30 % scatter step-to-step on this model), so the headline
    #: KPI uses the time mean and reports the noise band next to it.
    steady_window: int = 500
    #: Stride (in steps) of the averaging samples inside ``steady_window``.
    averaging_every: int = 10
    bulk_offset_xi: float = 3.0
    bin_width: float = 0.25
''',
)

rep(
    '''        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
''',
    '''        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if not isinstance(self.steady_window, int) or self.steady_window < 1:
            raise ValueError("steady_window must be a positive integer")
        if self.steady_window >= self.steps:
            raise ValueError("steady_window must be smaller than steps (no transient left)")
        if not isinstance(self.averaging_every, int) or self.averaging_every < 1:
            raise ValueError("averaging_every must be a positive integer")
''',
)

rep(
    '''            "steps": self.steps,
            "sample_every": self.sample_every,
            "bulk_offset_xi": self.bulk_offset_xi,''',
    '''            "steps": self.steps,
            "sample_every": self.sample_every,
            "steady_window": self.steady_window,
            "averaging_every": self.averaging_every,
            "bulk_offset_xi": self.bulk_offset_xi,''',
)

# ------------------------------------------------------------ DropletMeasures
rep(
    '''    u_max: float
    u_rms: float
    sigma_analytic: float''',
    '''    u_max: float
    u_rms: float
    #: Time mean of ``dp_bulk`` over the declared steady window (headline input).
    dp_bulk_mean: float | None = None
    #: Standard deviation of the ``dp_bulk`` samples inside the steady window.
    dp_bulk_std: float | None = None
    #: Number of averaging samples that entered ``dp_bulk_mean``.
    n_samples: int = 0
    #: Time mean of the maximum speed inside the steady window.
    u_max_mean: float | None = None
    #: Length of the averaging window in steps (0 -> ``dp_bulk_mean`` is unset).
    steady_window: int = 0
    sigma_analytic: float''',
)

rep(
    '''    @property
    def sigma_capillary(self) -> float:
        """``sigma`` recovered from ``Delta p * R_fit`` (the calibration KPI)."""
        return self.dp_bulk * self.R_fit

    @property
    def sigma_capillary_contour(self) -> float:
        return self.dp_bulk * self.R_contour

    @property
    def sigma_capillary_surface_of_tension(self) -> float:
        return self.dp_bulk * self.R_sigma
''',
    '''    @property
    def dp_kpi(self) -> float:
        """Pressure jump entering the calibration KPI (time mean when available).

        Falls back to the final-snapshot value so hand-built measures (tests,
        external callers) keep working unchanged.
        """
        return self.dp_bulk if self.dp_bulk_mean is None else self.dp_bulk_mean

    @property
    def dp_relative_std(self) -> float:
        """``std / |mean|`` of the steady-window pressure-jump samples."""
        if self.dp_bulk_mean is None or self.dp_bulk_std is None:
            return 0.0
        if self.dp_bulk_mean == 0.0:
            return 0.0
        return self.dp_bulk_std / abs(self.dp_bulk_mean)

    @property
    def dp_relative_standard_error(self) -> float:
        """Standard error of the time mean, ``std / (|mean| sqrt(n))``."""
        if self.n_samples <= 0:
            return 0.0
        return self.dp_relative_std / math.sqrt(self.n_samples)

    @property
    def sigma_capillary(self) -> float:
        """``sigma`` recovered from ``Delta p * R_fit`` (the calibration KPI)."""
        return self.dp_kpi * self.R_fit

    @property
    def sigma_capillary_contour(self) -> float:
        return self.dp_kpi * self.R_contour

    @property
    def sigma_capillary_surface_of_tension(self) -> float:
        return self.dp_kpi * self.R_sigma
''',
)

rep(
    '''            "xi_fit": (self.xi_fit - self.xi_analytic) / self.xi_analytic,
            "R_fit_vs_contour_cells": self.R_fit - self.R_contour,
        }''',
    '''            "xi_fit": (self.xi_fit - self.xi_analytic) / self.xi_analytic,
            "R_fit_vs_contour_cells": self.R_fit - self.R_contour,
            # measurement-noise diagnostics: how far the single final snapshot of
            # the pressure jump is from the steady-window time mean
            "dp_snapshot_vs_mean": (self.dp_bulk - self.dp_kpi) / self.dp_kpi,
            "dp_relative_std": self.dp_relative_std,
            "dp_relative_standard_error": self.dp_relative_standard_error,
        }''',
)

rep(
    '''            "u_max": self.u_max,
            "u_rms": self.u_rms,
            "phi_drift": self.phi_drift,''',
    '''            "u_max": self.u_max,
            "u_rms": self.u_rms,
            "dp_bulk_mean": self.dp_bulk_mean,
            "dp_bulk_std": self.dp_bulk_std,
            "n_samples": self.n_samples,
            "u_max_mean": self.u_max_mean,
            "steady_window": self.steady_window,
            "phi_drift": self.phi_drift,''',
)

# ------------------------------------------------------------------ criteria
rep(
    '''    spurious_velocity_over_cs: float = 0.01
    conservation_drift: float = 1.0e-8
''',
    '''    spurious_velocity_over_cs: float = 0.01
    conservation_drift: float = 1.0e-8
    #: Largest admissible relative standard error of the time-averaged pressure
    #: jump; above this the steady window is too short/noisy to grade a 3 % band.
    noise_band: float = 0.02
''',
)

# -------------------------------------------------------- run_droplet_case
rep(
    '''    history: list[dict] = []
    with torch.no_grad():
        for step in range(1, case.steps + 1):
            f, g = collide_and_stream(f, g, p)
            if not bool(torch.isfinite(f).all().item() and torch.isfinite(g).all().item()):
                raise FloatingPointError(f"free-energy LBM diverged at step {step}")
            if step % case.sample_every == 0 or step == case.steps:
                rho, ux, uy, phi = macroscopic(f, g)
                pr = mean_pressure(f)
                p_in, p_out, dp = bulk_pressure_jump(
                    pr, radius, case.R, case.xi, offset_xi=case.bulk_offset_xi, r_max=r_max
                )
                speed = torch.sqrt(ux * ux + uy * uy)
                history.append(
                    {
                        "step": step,
                        "dp_bulk": dp,
                        "sigma_capillary_at_R_target": dp * case.R,
                        "u_max": float(speed.max().item()),
                        "u_rms": float(torch.sqrt((speed * speed).mean()).item()),
                    }
                )
''',
    '''    history: list[dict] = []
    dp_samples: list[float] = []
    u_samples: list[float] = []
    steady_start = case.steps - case.steady_window
    with torch.no_grad():
        for step in range(1, case.steps + 1):
            f, g = collide_and_stream(f, g, p)
            if not bool(torch.isfinite(f).all().item() and torch.isfinite(g).all().item()):
                raise FloatingPointError(f"free-energy LBM diverged at step {step}")
            steady = step > steady_start
            if steady and step % case.averaging_every == 0:
                rho, ux, uy, phi = macroscopic(f, g)
                pr = mean_pressure(f)
                _, _, dp = bulk_pressure_jump(
                    pr, radius, case.R, case.xi, offset_xi=case.bulk_offset_xi, r_max=r_max
                )
                speed = torch.sqrt(ux * ux + uy * uy)
                dp_samples.append(float(dp))
                u_samples.append(float(speed.max().item()))
            if step % case.sample_every == 0 or step == case.steps:
                rho, ux, uy, phi = macroscopic(f, g)
                pr = mean_pressure(f)
                p_in, p_out, dp = bulk_pressure_jump(
                    pr, radius, case.R, case.xi, offset_xi=case.bulk_offset_xi, r_max=r_max
                )
                speed = torch.sqrt(ux * ux + uy * uy)
                history.append(
                    {
                        "step": step,
                        "dp_bulk": dp,
                        "sigma_capillary_at_R_target": dp * case.R,
                        "u_max": float(speed.max().item()),
                        "u_rms": float(torch.sqrt((speed * speed).mean()).item()),
                        "in_steady_window": bool(steady),
                    }
                )
''',
)

rep(
    '''    speed = torch.sqrt(ux * ux + uy * uy)
    return DropletMeasures(
        R_contour=R_contour,''',
    '''    speed = torch.sqrt(ux * ux + uy * uy)
    if not dp_samples:
        raise RuntimeError("no steady-window samples collected; check steady_window/averaging_every")
    dp_mean = sum(dp_samples) / len(dp_samples)
    dp_var = sum((d - dp_mean) ** 2 for d in dp_samples) / len(dp_samples)
    return DropletMeasures(
        R_contour=R_contour,''',
)

rep(
    '''        u_max=float(speed.max().item()),
        u_rms=float(torch.sqrt((speed * speed).mean()).item()),
        sigma_analytic=case.sigma,''',
    '''        u_max=float(speed.max().item()),
        u_rms=float(torch.sqrt((speed * speed).mean()).item()),
        dp_bulk_mean=dp_mean,
        dp_bulk_std=math.sqrt(dp_var),
        n_samples=len(dp_samples),
        u_max_mean=sum(u_samples) / len(u_samples),
        steady_window=case.steady_window,
        sigma_analytic=case.sigma,''',
)

# --------------------------------------------------------------- new checks
rep(
    '''        "C7_conservation": {
            "value": max(measures.phi_drift, measures.rho_drift),
            "abs_value": max(measures.phi_drift, measures.rho_drift),
            "band": criteria.conservation_drift,
            "passed": max(measures.phi_drift, measures.rho_drift) <= criteria.conservation_drift,
        },
    }''',
    '''        "C7_conservation": {
            "value": max(measures.phi_drift, measures.rho_drift),
            "abs_value": max(measures.phi_drift, measures.rho_drift),
            "band": criteria.conservation_drift,
            "passed": max(measures.phi_drift, measures.rho_drift) <= criteria.conservation_drift,
        },
        "C8_steady_window_noise": {
            "value": measures.dp_relative_standard_error,
            "abs_value": measures.dp_relative_standard_error,
            "band": criteria.noise_band,
            "passed": measures.dp_relative_standard_error <= criteria.noise_band,
        },
    }''',
)

# ------------------------------------------------------- suite convenience
rep(
    '''def grade_report(
    rows: Sequence[dict], criteria: LaplaceCriteria | None = None
) -> dict:''',
    '''def default_cases() -> list[DropletCase]:
    """The declared calibration case set (three radii, two resolutions).

    ``N64_R9``/``N64_R12`` give the finite-``xi`` curvature trend at one
    resolution, ``N96_R18`` adds a grid-refinement point at larger ``R/xi`` so a
    resolution artefact cannot masquerade as a curvature correction.  Each case is
    run past a transient of ``steps - steady_window`` steps and then time-averaged.
    """
    return [
        DropletCase(N=64, R=9.0, steps=5000, sample_every=250, steady_window=3000),
        DropletCase(N=64, R=12.0, steps=6000, sample_every=250, steady_window=3000),
        DropletCase(N=96, R=18.0, steps=9000, sample_every=250, steady_window=4000),
    ]


def smoke_cases() -> list[DropletCase]:
    """Reduced case set for CI/smoke runs (same protocol, shorter windows)."""
    return [
        DropletCase(
            N=48, R=8.0, steps=900, sample_every=150, steady_window=500, averaging_every=10
        ),
    ]


def run_suite(
    cases: Sequence[DropletCase] | None = None,
    criteria: LaplaceCriteria | None = None,
    *,
    case_ids: Sequence[str] | None = None,
) -> list[dict]:
    """Run every case and grade it; returns the per-case rows.

    Thin driver over :func:`run_droplet_case` + :func:`grade_case` so the XVal
    entry and the benchmark-pool entry execute *the same* protocol.
    """
    criteria = criteria or LaplaceCriteria()
    cases = list(cases) if cases is not None else default_cases()
    if case_ids is not None and len(case_ids) != len(cases):
        raise ValueError("case_ids must be parallel to cases")
    rows = []
    for k, case in enumerate(cases):
        measures = run_droplet_case(case)
        graded = grade_case(case, measures, criteria)
        graded["case"]["case_id"] = (
            case_ids[k] if case_ids is not None else f"N{case.N}_R{case.R:g}"
        )
        graded["measures_history"] = measures.history
        graded["measures_profile"] = measures.profile
        rows.append(graded)
    return rows


def grade_report(
    rows: Sequence[dict], criteria: LaplaceCriteria | None = None
) -> dict:''',
)

# report: add diffuse-interface reference per row
rep(
    '''    return {
        "verdict": "PASS" if all(r["passed"] for r in rows) else "FAIL",''',
    '''    diffuse = {}
    for r in rows:
        R_fit = r["measures"]["R_fit"]
        A = r["case"]["A"]
        kappa = r["case"]["kappa"]
        ratio = diffuse_interface_correction(R_fit, A, kappa)
        sigma = sigma_analytic(A, kappa)
        dp_ref = ratio * sigma / R_fit
        mean = r["measures"]["dp_bulk_mean"]
        diffuse[r["case"].get("case_id", "?")] = {
            "dp_diffuse_exact": dp_ref,
            "dp_times_R_over_sigma_profile": ratio,
            "rel_error_vs_diffuse_exact": None if mean is None else (mean - dp_ref) / dp_ref,
        }
    diff_errors = [
        abs(v["rel_error_vs_diffuse_exact"])
        for v in diffuse.values()
        if v["rel_error_vs_diffuse_exact"] is not None
    ]
    return {
        "verdict": "PASS" if all(r["passed"] for r in rows) else "FAIL",''',
)

rep(
    '''        "radius_trend": trend,''',
    '''        "diffuse_interface_reference": diffuse,
        "worst_abs_rel_error_vs_diffuse_exact": max(diff_errors) if diff_errors else None,
        "radius_trend": trend,''',
)

p.write_text(src)
print("patched", len(src.splitlines()), "lines")