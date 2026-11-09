P2 = "tests/test_boundary_capability_contract.py"
s2 = open(P2, encoding="utf-8").read()
OLD2 = '''def test_wall_partial_slip_statuses() -> None:
    """The mixed slip wall: D2Q9 reduction tested, D3Q19 validated, D3Q27 absent."""
    matrix = boundary_capability_matrix()
    assert matrix["wall_partial_slip"]["D2Q9"].implementation_status == "MECHANICS_TESTED"
    assert matrix["wall_partial_slip"]["D3Q19"].implementation_status == "PHYSICS_VALIDATED"
    assert matrix["wall_partial_slip"]["D3Q27"].implementation_status == "NO_IMPLEMENTATION"
    ev = matrix["wall_partial_slip"]["D3Q19"].verification_evidence
    assert "partial_slip_wall3d" in matrix["wall_partial_slip"]["D3Q19"].entrypoint
    assert "P106W1_N04" in ev
    # still fail-closed: no composition contract, so not claimable as available
    assert matrix["wall_partial_slip"]["D3Q19"].available is False
'''
NEW2 = '''def test_wall_partial_slip_statuses() -> None:
    """The mixed slip wall: D2Q9 reduction tested, D3Q19 and D3Q27 validated."""
    matrix = boundary_capability_matrix()
    assert matrix["wall_partial_slip"]["D2Q9"].implementation_status == "MECHANICS_TESTED"
    assert matrix["wall_partial_slip"]["D3Q19"].implementation_status == "PHYSICS_VALIDATED"
    assert matrix["wall_partial_slip"]["D3Q27"].implementation_status == "PHYSICS_VALIDATED"
    ev = matrix["wall_partial_slip"]["D3Q19"].verification_evidence
    assert "partial_slip_wall3d" in matrix["wall_partial_slip"]["D3Q19"].entrypoint
    assert "P106W1_N04" in ev
    cap27 = matrix["wall_partial_slip"]["D3Q27"]
    assert "partial_slip_wall3d" in cap27.entrypoint
    assert "mixed_wall_row_lattice" in cap27.entrypoint
    assert "test_partial_slip_wall3dq27.py" in cap27.verification_evidence
    # still fail-closed: no composition contract, so not claimable as available
    assert matrix["wall_partial_slip"]["D3Q19"].available is False
    assert cap27.available is False
'''
assert s2.count(OLD2) == 1
s2 = s2.replace(OLD2, NEW2)
open(P2, "w", encoding="utf-8").write(s2)
print("ok")