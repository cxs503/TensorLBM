"""Patch run_case.py: single source of truth for the reference constant."""
from pathlib import Path
import re

p = Path("benchmarks/case_pool/rayleigh_taylor/run_case.py")
t = p.read_text()
lines = t.split("\n")

# ---- 1. replace the round-2 comment block + `C_2D = 1.0` with an import-backed
#         single source of truth
idx = lines.index("C_2D = 1.0")
start = idx
while start > 0 and lines[start - 1].startswith("#:"):
    start -= 1
new_block = '''#: Reference constant of the saturated-velocity gate.  It is **not** defined
#: here: the single source of truth is ``ref/ref_lock.py``, which also records
#: why the two literature conventions disagree (``CONVENTION_CONFLICT``).  The
#: previous revision re-defined it locally as ``C_2D = 1.0`` after back-solving
#: ``C = 0.90`` (A = 1/2) and ``C = 1.01`` (A = 1/3) *from the solver's own
#: terminal velocities* -- a circular reference that manufactured a PASS
#: (-0.6 % at A = 1/3) out of a 1.7x velocity overshoot.  Importing the locked
#: value makes that failure mode impossible, and the alternate convention is
#: still evaluated (``C_2D_ALT``) so its influence on the verdict is visible.
from ref_lock import C_2D, C_2D_ALT, CONVENTION_CONFLICT  # noqa: E402
from ref_lock import boussinesq_velocity as _locked_boussinesq_velocity  # noqa: E402
'''
lines[start:idx + 1] = new_block.split("\n")
t = "\n".join(lines)

# ---- 2. the local wrappers delegate to the lock (kept: other scripts import them)
old = '''def goncharov_velocity(atwood: float, g: float, lam: float, branch: str, c: float = C_2D) -> float:
    """One branch of the two-fluid potential-flow saturated velocity."""
    k = 2.0 * math.pi / float(lam)
    if branch == "b":
        return math.sqrt(2.0 * atwood * g / ((1.0 + atwood) * c * k))
    if branch == "s":
        return math.sqrt(2.0 * atwood * g / ((1.0 - atwood) * c * k))
    raise ValueError(branch)
'''
assert old in t
new = '''def goncharov_velocity(atwood: float, g: float, lam: float, branch: str,
                       c: float = C_2D) -> float:
    """One branch of the two-fluid potential-flow saturated velocity."""
    k = 2.0 * math.pi / float(lam)
    if branch == "b":
        return math.sqrt(2.0 * atwood * g / ((1.0 + atwood) * c * k))
    if branch == "s":
        return math.sqrt(2.0 * atwood * g / ((1.0 - atwood) * c * k))
    raise ValueError(branch)
'''
t = t.replace(old, new)

# ---- 3. boussinesq_velocity: keep the (correct) symmetry argument, delegate
start = t.index("def boussinesq_velocity(atwood: float, g: float, lam: float, c: float = C_2D) -> float:")
end = t.index("def froude_normalisation(")
new_bv = '''def boussinesq_velocity(atwood: float, g: float, lam: float, c: float = C_2D) -> float:
    """Reference velocity for a *Boussinesq* single-mode RT bubble/spike.

    ``tensorlbm.rayleigh_taylor`` solves the Boussinesq equations: the density is
    ``rho = 1`` everywhere and the Atwood number enters only through the buoyancy
    force.  Those equations are invariant under ``(phi, y) -> (-phi, -y)``, which
    exchanges bubble and spike, so the solver *must* return one common velocity

        u_b = u_s = sqrt(2 A g / (C k)).

    That symmetry is proved (and tested to machine precision) in
    ``tests/test_case_pool_rayleigh_taylor_reference_audit.py``: it is a property
    of the Boussinesq model class, **not** a measurement, so comparing a
    Boussinesq run against the unequally split two-fluid branches would build a
    spurious +/- (1 +/- A)^{1/2} (+/-22 % at A = 1/2) into the gate.  Both
    branches are still reported (``u_*_goncharov_twofluid``) for provenance.

    The *value of C* comes from ``ref/ref_lock.py`` (primary ``C = 3``, the
    2-D-specific literature statement) -- see ``CONVENTION_CONFLICT`` there.
    """
    return _locked_boussinesq_velocity(atwood, g, lam, c)


def boussinesq_velocity_alt(atwood: float, g: float, lam: float) -> float:
    """Same reference under the competing convention ``C = 1`` (arXiv:2009.14655)."""
    return _locked_boussinesq_velocity(atwood, g, lam, C_2D_ALT)


'''
t = t[:start] + new_bv + t[end:]

# ---- 4. report the alternate-convention error alongside the primary one
old_ref = '''    ub_ref = boussinesq_velocity(spec.atwood, spec.g, spec.lam)
    us_ref = ub_ref
'''
assert old_ref in t
new_ref = '''    ub_ref = boussinesq_velocity(spec.atwood, spec.g, spec.lam)
    us_ref = ub_ref
    ub_ref_alt = boussinesq_velocity_alt(spec.atwood, spec.g, spec.lam)
'''
t = t.replace(old_ref, new_ref)

old_model = '''        "reference_model": "boussinesq-symmetric sqrt(2Ag/(Ck)), C_2D=1", '''
assert old_model in t, "reference_model line changed"
new_model = '''        "reference_model": f"boussinesq-symmetric sqrt(2Ag/(Ck)), C_2D={C_2D} (locked in ref/ref_lock.py)",
        "reference_C_2D": C_2D,
        "reference_C_2D_alt": C_2D_ALT,
        "u_ref_alt": ub_ref_alt,
        "u_bubble_rel_error_pct_alt_C1": 100.0 * (ub / ub_ref_alt - 1.0) if math.isfinite(ub) else float("nan"),
        "convention_conflict_status": CONVENTION_CONFLICT["status"],'''
t = t.replace(old_model, new_model)

# ---- 5. gates block records both conventions
old_gate = '''            "goncharov_C_2D": C_2D,'''
assert old_gate in t
new_gate = '''            "goncharov_C_2D": C_2D,
            "goncharov_C_2D_alt": C_2D_ALT,
            "convention_conflict": CONVENTION_CONFLICT["status"],'''
t = t.replace(old_gate, new_gate)

# ---- 6. sys.path so `ref_lock` is importable in every invocation mode
old_sys = '''for _p in (str(SRC_ROOT), str(REPO_ROOT)):'''
assert old_sys in t
t = t.replace(old_sys, '''for _p in (str(SRC_ROOT), str(REPO_ROOT), str(BENCH_DIR / "ref")):''', 1)

p.write_text(t)
print("patched", p)