P = "src/tensorlbm/boundary/boundary_capability_contract.py"
src = open(P, encoding="utf-8").read()

OLD = '''    ("wall_partial_slip", "D3Q27"): (
        "NO_IMPLEMENTATION",
        None,
        "No mixed bounce-back/specular wall for D3Q27",
    ),
'''

NEW = '''    ("wall_partial_slip", "D3Q27"): (
        "PHYSICS_VALIDATED",
        "tensorlbm.boundary.partial_slip_wall3d.mixed_wall_row_lattice / "
        "channel_poiseuille_lattice (LATTICE_27)",
        "test_partial_slip_wall3dq27.py (14 cases) + artifacts/P106W1-N04/"
        "d3q27_slip_wall.json: the D3Q19 mixed wall lifted to the 27-velocity "
        "set by making the tables (weights/opposites/specular mirror) an "
        "argument instead of a module constant. sigma=0 reproduces the "
        "analytic no-slip parabola to 0.2311% (criterion 0.5%), |b|/H = "
        "1.156e-3; the full D3Q27 field matches the 1D-column reduction to "
        "5.8e-14 and the D3Q19 field to 4.1e-14; plane periodicity is "
        "*reached* from a 5% x/z wave start (profile dev 3.0e-14, plane "
        "spread 3.7e-14); b(sigma) = a sigma/(1-sigma) + c with a = "
        "0.29999999 and c*H = -0.0650003, identical to D3Q19 to 1.6e-13 / "
        "9.6e-13 (the law is an operator property, not a lattice one); "
        "curvature-free estimators agree with the quadratic fit to 4.5e-7 lu "
        "on the same fields. BGK/single-phase/float64, stationary tangential "
        "wall only, D3Q27 corners streamed link-by-link. Docs: "
        "docs/P106W1_N04_SLIP_WALL_3D_D3Q27.md",
    ),
'''

assert src.count(OLD) == 1
src = src.replace(OLD, NEW)
open(P, "w", encoding="utf-8").write(src)
print("contract updated")

P2 = "tests/test_boundary_capability_contract.py"
s2 = open(P2, encoding="utf-8").read()
OLD2 = '''def test_wall_partial_slip_statuses() -> None:
    """The mixed slip wall: D2Q9 reduction tested, D3Q19 validated, D3Q27 absent."""

    assert matrix["wall_partial_slip"]["D2Q9"].implementation_status == "MECHANICS_TESTED"
    assert matrix["wall_partial_slip"]["D3Q19"].implementation_status == "PHYSICS_VALIDATED"
    assert matrix["wall_partial_slip"]["D3Q27"].implementation_status == "NO_IMPLEMENTATION"
'''
NEW2 = '''def test_wall_partial_slip_statuses() -> None:
    """The mixed slip wall: D2Q9 reduction tested, D3Q19 and D3Q27 validated."""

    assert matrix["wall_partial_slip"]["D2Q9"].implementation_status == "MECHANICS_TESTED"
    assert matrix["wall_partial_slip"]["D3Q19"].implementation_status == "PHYSICS_VALIDATED"
    assert matrix["wall_partial_slip"]["D3Q27"].implementation_status == "PHYSICS_VALIDATED"
    cap27 = matrix["wall_partial_slip"]["D3Q27"]
    assert "partial_slip_wall3d" in cap27.entrypoint
    assert "test_partial_slip_wall3dq27.py" in cap27.verification_evidence
    assert cap27.available is False            # no complete composition contract
'''
assert s2.count(OLD2) == 1
s2 = s2.replace(OLD2, NEW2)
open(P2, "w", encoding="utf-8").write(s2)
print("contract test updated")