# PREREG — XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC
Predeclared before the runs (2026-11-13, m18423); no threshold changed after seeing data.
Decision: does transplanting joconnor22/LIFE `GridClass::regularisedBC` (src/Grid.cpp:386-466) into TensorLBM
reproduce the OSS closure exactly, and does it beat the in-repo half-way bounce back on analytic channel flow?
Arms: A1 regularised (Zou/He density closure -> anti-symmetric completion -> S=sum cc(f-feq) -> f=feq+w/(2cs^4)Q:S);
A2 bounce_back (repo-validated pre-streaming half-way BB + Ladd moving-wall term, benchmarks/verified/couette_2d/run.py);
A3 closed form (Couette, Poiseuille, Stokes layer) evaluated in-process.
Hypotheses: H1 production vectorised closure == literal scalar transcription of the OSS routine (max|df|<1e-12);
H2 the closure preserves S and removes the ghost third moment of f-feq (rel<1e-13, all four wall normals);
H3 steady channel: both closures <3% and the fitted wall origin within 0.05 lu of the declared 0 (regularised) / 0.5 (BB);
H4 unsteady Stokes layer: l2(A1) < l2(A2) at equal grid; H5 non-finite/|rho-1|>0.05 or period-to-period change >1e-3 = declared failure.
Metrics: identity {stress_rel, ghost_after, oss_transcription_abs}; steady {l2_rel, fitted_wall_low/high, mass_drift};
oscillating {l2_rel_complex, amp_max_rel, phase_max_abs (|A_ref|>0.1U0), period_to_period_change}.
Thresholds: owner 15%, stretch 3%; identity 1e-13; OSS transcription 1e-12.
Stopping rule: steady arms stop on max|u(n)-u(n-200)|/max|u| < 1e-10 (never extended after looking at KPIs);
oscillating arms run a fixed 6T and are analysed over the final period only. CPU float64, single process.
Declared limit: LIFE itself is not executed (C++/Boost/LAPACK FSI solver, no off-the-shelf case at these parameters);
OSS semantics are cross-checked by the line-cited scalar transcription, recorded as a comparison limit.
