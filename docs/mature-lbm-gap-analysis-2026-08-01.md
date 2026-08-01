# Mature LBM gap analysis for external resistance prediction

This note records publicly documented design choices used to guide TensorLBM.
It does not claim access to, or reproduce, proprietary PowerFLOW/XFlow code.

## Public reference points

- Dassault Systèmes describes PowerFLOW as a transient LBM solver with
  automated domain discretization, turbulence modelling and wall treatment,
  without manual volume or boundary-layer meshing:
  <https://www.3ds.com/products/simulia/powerflow>
- Dassault Systèmes describes XFlow as using automatic lattice generation,
  GPU computing, WALE-based WMLES and a unified non-equilibrium wall
  function:
  <https://www.3ds.com/products/simulia/xflow>
- OpenLB 1.8 user guide, sections 6.4 and 9.12, publicly documents a curved
  Bouzidi + exchange-location Spalding wall model and an
  equilibrium-difference damping layer:
  <https://www.openlb.net/wp-content/uploads/2025/08/olb_ug-1.8r0.pdf>

Marketing statements are architecture hints, not validation evidence.  The
OpenLB equations and benchmark hierarchy are the reproducible technical
reference.

## Gap found in the former SUBOFF path

| concern | former path | mature direction | TensorLBM common module |
|---|---|---|---|
| curved wall | BFL, but startup blended with streamed solid data | full BFL at all times | `bfl_d3q19.py`, corrected ramp in `wall_model.py` |
| wall model | first-cell log/Reichardt stress imposed as Guo force | velocity sampled at exchange location; unified Spalding law; non-equilibrium assimilation | `spalding_wall_model.py` |
| LES | constant Smagorinsky default | near-wall WALE WMLES candidate | existing `collide_wale_mrt3d`, exposed by runners |
| open boundary | direct distribution-to-equilibrium blend | equilibrium-difference absorbing source with smooth strength | `sponge_layer.py` |
| force | link force alone | independent momentum balance | `control_volume_force.py` |
| resolution | uniform voxel grid | automatic nested lattice around surface/wake | `static_block_amr.py`, `suboff_static_amr.py` |
| AMR reflux | uniform absolute population correction | population-proportional, positivity-limited correction; later face-local flux register | `static_block_amr.py` |

## Mandatory validation ladder

1. Manufactured Spalding states and equilibrium fixed points.
2. Laminar Couette/Poiseuille wall stress.
3. Turbulent periodic channel mean profile and friction Reynolds number.
4. Cylinder and sphere drag with BFL link force agreeing with an interior
   kinetic control volume.
5. SUBOFF AFF-1, followed by AFF-8/AFF-1 force ratio.
6. At least three effective hull resolutions and three settled time windows.

No SUBOFF force is admitted merely because one parameter set is close to a
tow-tank value.  Force-observer agreement, conservation, positivity, domain
independence and grid/time convergence are independent gates.
