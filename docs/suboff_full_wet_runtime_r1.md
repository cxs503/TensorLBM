# SUBOFF full-wet runtime R1

`tensorlbm.suboff_full_wet_runtime` is a minimal **software/artifact runtime**
chain.  It is not a SUBOFF physical simulation, CFD validation, or a validated
resistance result.

## Scope

1. Accept an immutable `GeometryAsset` and compile its D3Q19 wall links with
   `compile_d3q19_wall_links`.
2. Construct deterministic, uniform synthetic D3Q19 equilibrium populations.
3. For every owned solid-to-fluid link, read the opposing incident population
   at the compiled fluid neighbour and explicitly calculate stationary
   bounce-back momentum exchange: `F_body = -2 f_incident c_q`.
4. Sum the per-link values into a force sample, construct a measured
   `ForceObservation`, and pass it to `build_resistance_force_contract`.

The returned serialisable artifact records the case and runtime hashes,
geometry hash, link count/ownership, method, sample phase, per-link exchanges,
force series, force observation, and `Ct` classification.

## Deliberate limits

* No collision, streaming update, boundary update, obstacle update, or
  population/cell reset occurs in this runtime.
* The source `SuboffCaseDefinition` reference defaults to `withheld`; output
  sets `physical_validation: false` and the resistance contract remains
  `measured_candidate`, never validated.
* Uniform synthetic populations are deterministic test evidence only.  They
  do not establish SUBOFF hydrodynamics or a physical resistance coefficient.

## Minimal use

```python
result = run_suboff_full_wet_runtime(asset)
assert result["contract"]["status"] == "measured_candidate"
assert result["physical_validation"] is False
```
