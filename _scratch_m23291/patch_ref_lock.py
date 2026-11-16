"""Patch ref_lock.py: record the C-convention conflict explicitly, expose both."""
from pathlib import Path

p = Path("benchmarks/case_pool/rayleigh_taylor/ref/ref_lock.py")
t = p.read_text()

old_c = 'C_2D = 3.0          # Goncharov geometric constant, 2-D (locked below)\n'
assert old_c in t
new_c = '''#: Goncharov geometric constant used by the gates.  Two of the repo's own locked
#: sources disagree on its 2-D value, so both conventions are carried explicitly
#: instead of one being quietly overwritten (see CONVENTION_CONFLICT below):
#:
#: * ``C = 3`` (primary, ``C_2D``): arXiv:1910.03663 Fig. 3 caption (p. 6) is a
#:   *two-dimensional* RTI figure (A = 0.04) whose dashed potential-model
#:   prediction is ``Fr_B = sqrt(1/(3 pi))``.  With the Froude normalisation
#:   ``Fr_B == U_B/sqrt(A g lam/(1+A))`` this reproduces ``C = 3`` exactly
#:   (checked to 1e-15 in :func:`main`).  The classic 2-D Layzer value
#:   ``u_b = sqrt(2 A g/(3 k))`` is the same statement.
#: * ``C = 1`` (alternate, ``C_2D_ALT``): arXiv:2009.14655 Eq. (1), p. 3 states
#:   verbatim "C is a constant given by 1 or 3 for the two-dimensional or
#:   three-dimensional geometry", i.e. the opposite assignment.  Under
#:   ``Fr_B == U_B/sqrt(A g lam)`` that convention predicts ``Fr_B = sqrt(1/pi)``
#:   for 2-D, which no 2-D figure in either excerpt supports.
#:
#: The conflict cannot be settled from this sandbox (publisher copies of
#: Goncharov 2002 PRL 88:134502 are behind a 403).  Policy: (a) the gates use
#: the convention backed by the *2-D-specific* primary source (C = 3);
#: (b) every report also quotes the C = 1 errors, so the reader sees how much of
#: the verdict is convention-dependent; (c) no constant is ever fitted to the
#: solver's own output -- that is what produced the false PASS this file now
#: guards against (round 2, m2026-11-13, back-solved C = 0.90/1.01 from its own
#: runs).
C_2D = 3.0          # Goncharov geometric constant, 2-D (primary, locked below)
C_2D_ALT = 1.0      # the competing assignment of arXiv:2009.14655 Eq. (1)


def boussinesq_velocity(atwood: float, g: float, lam: float, c: float = C_2D) -> float:
    """Symmetric (``u_b = u_s``) reference of the single-density Boussinesq model.

    ``tensorlbm.rayleigh_taylor`` keeps ``rho = 1`` everywhere and lets the
    Atwood number act only through the buoyancy force; that model is invariant
    under ``(phi, y) -> (-phi, -y)``, which exchanges bubble and spike, so its
    terminal velocities must coincide and the A -> 0 limit of the two-fluid
    formula, ``sqrt(2 A g/(C k))``, is the reference (proof + numerical test:
    ``tests/test_case_pool_rayleigh_taylor_reference_audit.py``).
    """
    return math.sqrt(2.0 * atwood * g / (c * 2.0 * math.pi / float(lam)))


#: Machine-readable record of the disagreement, written into ref_lock.json.
CONVENTION_CONFLICT = {
    "quantity": "Goncharov geometric constant C in u_b = sqrt(2Ag/((1+A)Ck))",
    "primary": {"c": 3.0, "source": "arXiv:1910.03663 Fig. 3 caption p.6 (2-D RTI, A = 0.04), "
                                   "Fr_B = sqrt(1/(3*pi)); equivalent to the classic 2-D Layzer "
                                   "value u_b = sqrt(2Ag/(3k))"},
    "alternate": {"c": 1.0, "source": "arXiv:2009.14655 Eq. (1) p.3, verbatim 'C is a constant "
                                      "given by 1 or 3 for the two-dimensional or "
                                      "three-dimensional geometry'"},
    "fr_b_primary": math.sqrt(1.0 / (3.0 * math.pi)),
    "fr_b_alternate": math.sqrt(2.0 / (1.0 * 2.0 * math.pi)),
    "ratio_primary_over_alternate": math.sqrt(3.0),
    "status": "UNRESOLVED (publisher copy of Goncharov 2002 PRL 88:134502 not reachable: HTTP 403)",
    "policy": "gates use C = 3 (2-D-specific primary source); both errors are reported; "
              "no constant is fitted to solver output",
}
'''
t = t.replace(old_c, new_c)

# expose a branch helper that takes an explicit constant
old_g = '''def goncharov(atwood: float, g: float, lam: float, branch: str) -> float:
    k = 2.0 * math.pi / lam
    denom = (1.0 + atwood) if branch == "b" else (1.0 - atwood)
    return math.sqrt(2.0 * atwood * g / (denom * C_2D * k))
'''
assert old_g in t
new_g = '''def goncharov(atwood: float, g: float, lam: float, branch: str,
             c: float = C_2D) -> float:
    k = 2.0 * math.pi / lam
    denom = (1.0 + atwood) if branch == "b" else (1.0 - atwood)
    return math.sqrt(2.0 * atwood * g / (denom * c * k))
'''
t = t.replace(old_g, new_g)

# record both Fr_B values and the conflict in the JSON lock
old_lock = '''            "R2_nonlinear": {
                "formula": "u_s = sqrt(2Ag/((1-A)Ck)); u_b = sqrt(2Ag/((1+A)Ck)); C = 3 (2-D)",'''
assert old_lock in t
new_lock = '''            "R2_nonlinear": {
                "convention_conflict": CONVENTION_CONFLICT,
                "formula": "u_s = sqrt(2Ag/((1-A)Ck)); u_b = sqrt(2Ag/((1+A)Ck)); "
                           "C = 1 (2-D, alternate) or 3 (2-D, primary) -- see convention_conflict",'''
t = t.replace(old_lock, new_lock)

# also lock the alternate branch numerically so neither side is a bare number
old_assert = '''    assert abs(fr_c3 - fr_quoted) < 1e-15, (fr_c3, fr_quoted)
'''
assert old_assert in t
new_assert = '''    assert abs(fr_c3 - fr_quoted) < 1e-15, (fr_c3, fr_quoted)
    # the competing convention is locked the same way, so the disagreement is a
    # checked property of the two source excerpts rather than a memory claim
    fr_c1 = froude_bubble_theory(C_2D_ALT)
    assert abs(fr_c1 - CONVENTION_CONFLICT["fr_b_alternate"]) < 1e-15, fr_c1
    assert abs(fr_c3 / fr_c1 - math.sqrt(3.0)) < 1e-15
'''
t = t.replace(old_assert, new_assert)

old_print = '''    print(f"[ref_lock] C={C_2D} check {fr_c3:.15f} vs quoted {fr_quoted:.15f} -> OK")'''
assert old_print in t
new_print = '''    print(f"[ref_lock] C={C_2D} check {fr_c3:.15f} vs quoted {fr_quoted:.15f} -> OK")
    print(f"[ref_lock] competing convention C={C_2D_ALT} -> Fr_B={fr_c1:.6f} "
          f"(ratio sqrt(3)={math.sqrt(3.0):.6f}); conflict recorded as UNRESOLVED")'''
t = t.replace(old_print, new_print)

p.write_text(t)
print("patched", p)