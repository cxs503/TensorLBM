# OSS (joconnor22/LIFE) vs TensorLBM — difference list / transplant trade-offs
Source read: src/Grid.cpp:386-466 (regularisedBC), inc/Grid.h:124-125, call sites src/Grid.cpp:321/333/351/368.
1 closure | node closure: Zou/He density (velocity) or normal velocity (pressure) + anti-symmetric completion + 2nd-order Hermite rebuild | identical (scalar transcription agrees to max|df| = 1.39e-17) | the point of the anchor
2 where | post-propagation applyBCs on every non-fluid site | post-streaming on the boundary plane (BB control stays pre-streaming) | each closure where it is defined
3 lattice | compile-time D2Q9/D3Q19 | runtime LatticeSpec, both lattices, tables pinned by tests | one implementation
4 equilibrium | 2nd-order Hermite with 4.5(u_x^2(cx^2-1/3)+...); optional CENTRAL_MOMENTS | same 2nd-order form; CENTRAL_MOMENTS branch NOT transplanted | changes the collision operator, not the closure (WITHHELD)
5 pressure BC | rho_in fixed, tangential from Utils::zeroGradient(order=2), u_n from the closure | implemented; zero_gradient_tangential(order=2) exposed, caller supplies it | closure stays stencil-free
6 free slip | u_n=0, tangential extrapolated, then regularised | implemented | same
7 corners | two non-zero normal components -> density extrapolated, buried link = feq | NOT transplanted (single-axis planes only) | scope limit, not approximated
8 wall location | the boundary node IS the wall (0, ny-1) | same for A1; BB keeps the repo convention (0.5, ny-1.5) | convention, not physics; fitted wall origin reported
9 wall velocity | u_in*rampCoefficient (compile-time ramp) | arbitrary u_wall(t) | needed by the oscillating-wall arm
10 forcing | latticeForce with (1-omega/2) split | Guo split post-collision, u = sum(fc)/rho (channel_kernel_numpy convention) | control arm comparability
11 precision | double | float64 | same
Take-aways: local and stencil-free (drops into any loop); it IS the wall so the declared channel height changes by one lattice unit;
corners and CENTRAL_MOMENTS are the two omitted pieces and are named, not approximated.
