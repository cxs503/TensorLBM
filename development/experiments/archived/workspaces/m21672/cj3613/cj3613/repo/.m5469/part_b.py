class HybridWindEngine:
    """Hybrid empirical/LBM wind-coefficient engine for one ship type.

    Parameters
    ----------
    ship
        Catalogued superstructure (supplies the complexity metric).
    database
        Per-angle LBM records with keys ``angle_deg``, ``C_X``, ``C_Y``,
        ``C_K``, ``C_N`` and optionally ``C_X_std``/``C_Y_std``.  May be empty,
        in which case the engine is empirical-only.
    correlation
        Output of :func:`fit_blendermann_style` (the empirical branch).  When
        omitted it is fitted on ``database`` itself.
    route
        ``"auto"`` (default), ``"empirical"`` or ``"lbm"``.
    complexity_threshold
        ``stow_ratio`` above which the complex (LBM) branch is used.
    blend_width
        Width of the linear blending band in ``stow_ratio`` units, centred on
        ``complexity_threshold``.
    """

    def __init__(self, ship: ShipSuperstructure, database: Sequence[Mapping] = (),
                 correlation: Mapping | None = None, *, route: str = "auto",
                 complexity_threshold: float = 0.25, blend_width: float = 0.10):
        if route not in ("auto", "empirical", "lbm"):
            raise ValueError("route must be one of auto/empirical/lbm")
        self.ship = ship
        self.route = route
        self.threshold = float(complexity_threshold)
        self.blend_width = float(blend_width)
        self.reference = dict(ship.reference) if ship.reference else dict(
            projected_areas(ship))
        self.complexity = complexity_index(ship, self.reference)

        self.database = [dict(r) for r in database]
        self.table = _mirror_table(self.database) if self.database else None
        self.correlation = (dict(correlation) if correlation is not None
                            else (fit_blendermann_style(self.database)
                                  if self.database else None))
        if self.route == "lbm" and self.table is None:
            raise ValueError("route='lbm' needs a non-empty LBM database")

    # -- routing ---------------------------------------------------------
    def lbm_weight(self) -> float:
        """Weight (0..1) on the LBM branch implied by the complexity metric."""

        if self.route == "lbm":
            return 1.0
        if self.route == "empirical":
            return 0.0
        if self.table is None:
            return 0.0
        r = self.complexity["stow_ratio"]
        lo = self.threshold - 0.5 * self.blend_width
        hi = self.threshold + 0.5 * self.blend_width
        if r <= lo:
            return 0.0
        if r >= hi:
            return 1.0
        return float((r - lo) / (hi - lo))

    def routing_report(self) -> dict:
        w = self.lbm_weight()
        branch = ("lbm" if w >= 0.999 else
                  "empirical" if w <= 0.001 else "blend")
        return {"branch": branch, "lbm_weight": w,
                "threshold": self.threshold, "blend_width": self.blend_width,
                "complexity": dict(self.complexity),
                "has_lbm_table": self.table is not None,
                "n_lbm_angles": 0 if self.table is None else
                len(self.table["angles_deg"])}

    # -- branches --------------------------------------------------------
    def _lbm_eval(self, theta_deg: float) -> dict:
        t = self.table
        out = {}
        for key in ("C_X", "C_Y", "C_K", "C_N"):
            vals = [t["rows"][a][key] for a in t["angles_deg"]]
            out[key] = _catmull_rom_periodic(theta_deg, vals, t["angles_deg"])
        for key in ("C_X_std", "C_Y_std"):
            vals = [float(t["rows"][a].get(key, 0.0)) for a in t["angles_deg"]]
            out[key] = _catmull_rom_periodic(theta_deg, vals, t["angles_deg"])
        out["method"] = "lbm_table"
        return out

    def _emp_eval(self, theta_deg: float) -> dict:
        if self.correlation is None:
            raise RuntimeError("no empirical correlation available")
        return blendermann_coefficients(self.correlation, theta_deg)

    # -- public ----------------------------------------------------------
    def coefficients(self, theta_deg: float) -> dict:
        """Four wind coefficients at ``theta_deg`` with provenance."""

        w = self.lbm_weight()
        emp = self._emp_eval(theta_deg) if w < 1.0 else None
        lbm = self._lbm_eval(theta_deg) if w > 0.0 else None
        out = {"angle_deg": float(theta_deg) % 360.0, "lbm_weight": w}
        if w <= 0.0:
            src = emp
            out["method"] = "empirical"
        elif w >= 1.0:
            src = lbm
            out["method"] = "lbm_table"
        else:
            src = {k: w * lbm[k] + (1.0 - w) * emp[k]
                   for k in ("C_X", "C_Y", "C_K", "C_N")}
            src["C_X_std"] = w * lbm.get("C_X_std", 0.0)
            src["C_Y_std"] = w * lbm.get("C_Y_std", 0.0)
            out["method"] = "blend"
        for k in ("C_X", "C_Y", "C_K", "C_N", "C_X_std", "C_Y_std"):
            if k in src:
                out[k] = float(src[k])
        if "C_X_std" not in out:
            out["C_X_std"] = 0.0
            out["C_Y_std"] = 0.0
        return out

    def polar(self, angles: Iterable[float] | None = None) -> list[dict]:
        ang = list(angles) if angles is not None else list(range(0, 361, 10))
        return [self.coefficients(a) for a in ang]

    def force(self, theta_deg: float, wind_speed_ms: float, *,
              rho_air: float = RHO_AIR) -> dict:
        """Dimensional forces/moments from the coefficients (ship frame)."""

        c = self.coefficients(theta_deg)
        q = 0.5 * rho_air * wind_speed_ms ** 2
        A_F = self.reference["area_frontal_m2"]
        A_L = self.reference["area_lateral_m2"]
        B = self.reference["breadth_m"]
        H = self.reference["height_m"]
        return {"q_Pa": q, "X_N": c["C_X"] * q * A_F,
                "Y_N": c["C_Y"] * q * A_L,
                "K_Nm": c["C_K"] * q * A_L * H,
                "N_Nm": c["C_N"] * q * A_L * B,
                "coefficients": c}

    def to_dict(self) -> dict:
        return {"ship": self.ship.key, "routing": self.routing_report(),
                "correlation": self.correlation,
                "reference": self.reference}

    @classmethod
    def from_records(cls, ship: ShipSuperstructure,
                     records: Sequence[Mapping], **kw) -> "HybridWindEngine":
        """Build the engine from LBM run records (``coefficients`` sub-dict)."""

        samples = []
        for r in records:
            coeff = r.get("coefficients", r)
            samples.append({"angle_deg": float(r["angle_deg"]),
                            "C_X": float(coeff["C_X"]),
                            "C_Y": float(coeff["C_Y"]),
                            "C_K": float(coeff.get("C_K", 0.0)),
                            "C_N": float(coeff.get("C_N", 0.0)),
                            "C_X_std": float(coeff.get("C_X_std", 0.0)),
                            "C_Y_std": float(coeff.get("C_Y_std", 0.0))})
        return cls(ship, samples, **kw)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def sign_conventions(samples: Sequence[Mapping] | None = None) -> dict:
    """Document and (when data is supplied) check the coefficient conventions.

    Definitions (ship frame, wind blowing *from* ``theta`` measured from the
    bow): ``C_X > 0`` is a retarding longitudinal force, ``C_Y`` follows the
    transverse force on the ship (sign taken from the LBM data, see the
    ``C_Y_at_90`` entry when samples are given), ``C_K`` is the heel moment
    about ``+x`` and ``C_N`` the yaw moment about ``+z``.

    When ``samples`` are given, the port/starboard mirror symmetry of the
    catalogue geometry is used as a numerical self-check: for a symmetric
    assembly ``C_X(180 - a) = C_X(a)`` and ``C_Y(180 - a) = -C_Y(a)``.  The
    returned ``mirror_pairs`` report the measured deviation, which is therefore
    an *a posteriori* measure of the numerical asymmetry (grid, sampling) of
    the LBM campaign.
    """

    out: dict = {
        "frame": {"x": "+forward (bow)", "y": "+port", "z": "+up (deck at z=0)"},
        "theta_deg": "wind direction measured from the bow (0 = head wind, "
                     "90 = beam wind from starboard)",
        "C_X": "X_wind / (q A_F), positive = retarding",
        "C_Y": "Y_wind / (q A_L), sign follows the transverse force in +y",
        "C_K": "heel moment about +x / (q A_L H)",
        "C_N": "yaw moment about +z / (q A_L B)",
        "mirror_pairs": [],
    }
    if not samples:
        return out
    by = {round(float(s["angle_deg"]) % 360.0, 6): s for s in samples}
    pairs = []
    for a in sorted(by):
        if not (0.0 < a < 90.0 + 1e-9):
            continue
        b = round(180.0 - a, 6)
        if b not in by:
            continue
        sa, sb = by[a], by[b]
        dev_cx = abs(sb["C_X"] - sa["C_X"]) / max(abs(sa["C_X"]), 1e-9)
        dev_cy = abs(-sb["C_Y"] - sa["C_Y"]) / max(abs(sa["C_Y"]), 1e-9)
        pairs.append({"angles_deg": [a, b], "C_X": [sa["C_X"], sb["C_X"]],
                      "C_Y": [sa["C_Y"], sb["C_Y"]],
                      "rel_dev_C_X": dev_cx, "rel_dev_C_Y": dev_cy})
    out["mirror_pairs"] = pairs
    for a in (0.0, 90.0):
        if a in by:
            out[f"C_X_at_{int(a)}"] = by[a]["C_X"]
            out[f"C_Y_at_{int(a)}"] = by[a]["C_Y"]
    return out


def _weibull_pdf(v: float, k: float, c: float) -> float:
    if v <= 0.0:
        return 0.0
    return (k / c) * (v / c) ** (k - 1.0) * math.exp(-(v / c) ** k)


def synthetic_wind_rose(name: str = "north_atlantic") -> dict:
    """A representative open-ocean wind rose (documented, *not* measured data).

    Eight 45 deg sectors, occurrence probability, and a Weibull (k, c) pair per
    sector.  The numbers are representative of North Atlantic route statistics
    used for annual wind-resistance studies: prevailing westerlies (sectors
    around 270 deg) with ``c ~ 10 m/s``, the eastern quadrant calmer, mean wind
    speed over the rose ~7.5 m/s.  They are **documented representative values**
    and must be replaced by measured route statistics for a real ship study.
    """

    if name != "north_atlantic":
        raise ValueError("only the 'north_atlantic' representative rose is built in")
    sectors = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    prob = [0.085, 0.075, 0.080, 0.100, 0.130, 0.150, 0.210, 0.170]
    c = [9.0, 8.0, 7.0, 7.5, 9.0, 9.5, 10.5, 10.0]
    k = [2.0, 2.0, 1.9, 1.9, 2.0, 2.1, 2.2, 2.1]
    s = sum(prob)
    prob = [p / s for p in prob]
    mean = 0.0
    for p, kk, cc in zip(prob, k, c):
        mean += p * cc * math.gamma(1.0 + 1.0 / kk)
    return {"name": name, "sectors_deg_from": sectors, "probability": prob,
            "weibull_shape_k": k, "weibull_scale_c": c,
            "mean_wind_speed_ms": mean,
            "provenance": "representative North Atlantic route statistics "
                          "(documented values, not measured data)"}


def calm_water_resistance_estimate(hull: HullReference, speed_ms: float,
                                   *, rho_water: float = 1025.0,
                                   form_factor: float = 1.15,
                                   wetted_area_factor: float = 0.90) -> dict:
    """Order-of-magnitude calm-water resistance (ITTC-57 friction dominated).

    ``S = wetted_area_factor * Lpp (2T + B)`` (a coarse wetted-area
    approximation, ~10% for conventional hulls, stated as such) and
    ``R = 0.5 rho S V^2 C_F (1+k)`` with the ITTC-57 friction line.  Used only
    to quote the wind resistance as a percentage; it is *not* a resistance
    prediction tool.
    """

    S = wetted_area_factor * hull.length_pp_m * (2.0 * hull.draught_m + hull.breadth_m)
    Re = speed_ms * hull.length_pp_m / 1.19e-6
    cf = 0.075 / (math.log10(Re) - 2.0) ** 2
    r = 0.5 * rho_water * S * speed_ms ** 2 * cf * form_factor
    return {"wetted_area_m2": S, "Re": Re, "C_F": cf,
            "form_factor": form_factor, "R_calm_water_N": r,
            "provenance": "ITTC-57 friction + coarse wetted-area estimate"}


def wind_rose_annual_resistance(engine: HybridWindEngine, rose: Mapping,
                                *, ship_speed_ms: float, heading_deg: float = 0.0,
                                rho_air: float = RHO_AIR,
                                n_speed: int = 48, v_max_factor: float = 3.0,
                                baseline_resistance_N: float | None = None
                                ) -> dict:
    """Annual mean wind added resistance from a wind rose.

    For every rose sector the *apparent* wind is obtained from the true wind
    vector minus the ship velocity vector (no current, no wave-orbit
    correction), the apparent wind angle feeds :meth:`HybridWindEngine.
    coefficients`, and the longitudinal wind force is

        ``dR = 0.5 rho_air A_F C_X(theta_a) V_a^2``

    (positive = retarding, negative = thrust).  The sector Weibull speed
    distribution is integrated with a midpoint rule on ``n_speed`` bins out to
    ``v_max_factor * c``.  Reported: the probability-weighted means of ``dR``
    (net, positive part only), the probability of a propulsive wind, and -- if
    a baseline resistance is supplied -- the ratio.
    """

    A_F = engine.reference["area_frontal_m2"]
    psi = math.radians(heading_deg)
    hx, hy = math.cos(psi), math.sin(psi)
    prob = list(rose["probability"])
    sectors = list(rose["sectors_deg_from"])
    ks = list(rose["weibull_shape_k"])
    cs = list(rose["weibull_scale_c"])
    total = sum(prob)
    prob = [p / total for p in prob]

    mean_net = 0.0
    mean_pos = 0.0
    p_thrust = 0.0
    per_sector = []
    weights = []
    for ang_from, p_s, k, c in zip(sectors, prob, ks, cs):
        # unit vector of the true wind *velocity* (blowing towards ang_from+180)
        wdir = math.radians(ang_from + 180.0)
        wx, wy = math.cos(wdir), math.sin(wdir)
        vmax = v_max_factor * c
        dv = vmax / n_speed
        s_net = s_pos = s_w = s_thrust = 0.0
        for i in range(n_speed):
            v = (i + 0.5) * dv
            w = _weibull_pdf(v, k, c) * dv
            if w <= 0.0:
                continue
            ax = v * wx - ship_speed_ms * hx
            ay = v * wy - ship_speed_ms * hy
            v_a = math.hypot(ax, ay)
            if v_a < 1e-9:
                continue
            # wind blows along (ax, ay); the coefficient angle is measured from
            # the bow for a wind blowing *from* the opposite direction
            theta = math.degrees(math.atan2(-ay, -ax)) % 360.0
            cx = engine.coefficients(theta)["C_X"]
            dR = 0.5 * rho_air * A_F * cx * v_a ** 2
            s_net += w * dR
            s_pos += w * max(dR, 0.0)
            s_thrust += w * (1.0 if dR < 0.0 else 0.0)
            s_w += w
        if s_w > 0:
            s_net /= s_w
            s_pos /= s_w
            s_thrust /= s_w
        mean_net += p_s * s_net
        mean_pos += p_s * s_pos
        p_thrust += p_s * s_thrust
        per_sector.append({"sector_from_deg": ang_from, "probability": p_s,
                           "mean_dR_net_N": s_net, "mean_dR_positive_N": s_pos,
                           "thrust_fraction": s_thrust})
        weights.append({"sector_from_deg": ang_from, "p": p_s})

    out = {
        "ship": engine.ship.key,
        "heading_deg": heading_deg,
        "ship_speed_ms": ship_speed_ms,
        "mean_wind_added_resistance_N": mean_net,
        "mean_wind_added_resistance_positive_N": mean_pos,
        "thrust_probability": p_thrust,
        "equivalent_added_power_kW": mean_net * ship_speed_ms / 1e3,
        "rose": dict(rose),
        "per_sector": per_sector,
        "engine_routing": engine.routing_report(),
    }
    if baseline_resistance_N:
        out["wind_over_calm_water_pct"] = 100.0 * mean_net / baseline_resistance_N
        out["baseline_resistance_N"] = baseline_resistance_N
    return out


# ---------------------------------------------------------------------------
# Acceptance evaluation
# ---------------------------------------------------------------------------

def evaluate_acceptance(criteria: Sequence[Mapping]) -> dict:
    """Assemble the machine-checkable acceptance table.

    ``criteria`` entries are ``{"id", "description", "value", "threshold",
    "operator" ("<=", ">=", "<", ">"), "unit",
    "hard" (bool: gates the node)}``.  Returns the table plus ``all_hard_passed``
    so the report can state pass/FAIL without rhetoric.
    """

    rows = []
    for c in criteria:
        v = c.get("value")
        th = c.get("threshold")
        op = c.get("operator", "<=")
        if v is None or th is None:
            passed = None
            verdict = "NOT_EVALUATED"
        else:
            passed = {"<=": v <= th, "<": v < th, ">=": v >= th, ">": v > th}[op]
            verdict = "PASS" if passed else "FAIL"
        rows.append({"id": c["id"], "description": c["description"],
                     "value": v, "threshold": th, "operator": op,
                     "unit": c.get("unit", ""), "hard": bool(c.get("hard", True)),
                     "verdict": verdict, "passed": passed})
    hard = [r for r in rows if r["hard"]]
    return {"criteria": rows,
            "all_hard_passed": all(r["passed"] for r in hard) if hard else False,
            "n_hard": len(hard),
            "n_hard_passed": sum(1 for r in hard if r["passed"]),
            "n_soft": len(rows) - len(hard),
            "n_soft_passed": sum(1 for r in rows if not r["hard"] and r["passed"])}