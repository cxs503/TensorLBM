-- oss_solutions deposit for XVAL-LBM-TWO-CONSERVATION-ZOUHE-WITH-CHEMICAL-POT (m18753)
-- DB role has no write permission to oss_solutions (same friction as m17112/mcmp_laplace);
-- this file is the machine-readable deposit and is applied when the role is granted write.
INSERT INTO oss_solutions (repo, feature_slug, tag, method, files, conventions, transfer, evidence)
VALUES (
 'ebinan92/Fingering_dynamics',
 'XVAL-LBM-TWO-CONSERVATION-ZOUHE-WITH-CHEMICAL-POT',
 'two-conservation-zouhe-with-chemical-potential-flux',
 'Free-energy two-distribution LBM: force-consistent Zou-He inlet/outlet. rho=(Kf+f3+f6+f7-Fx/2)/(1-ux) with Kf=sum known f; f1=f3+(2/3)rho ux; f5,f8 from the two momentum balances with the Guo half-force rho u = sum c f + F/2. Order parameter uses the same algebra on g: psi=((g0+g2+g4)+2(g3+g6+g7))/(1-ux) so that sum g = psi AND sum c_x g = psi u_x. Outlet psi closure must be pressure/back-pressure type; keep corners and avoid the (1+ux) division near interface (see failure record).',
 'src/tensorlbm/phasefield/zouhe_two_conservation.py, src/tensorlbm/phasefield/free_energy_channel.py, tests/test_zouhe_two_conservation_chemical_potential.py, xval/XVAL-LBM-TWO-CONSERVATION-ZOUHE-WITH-CHEMICAL-POT/run_xval.py',
 'mu = a psi (1-psi^2) - kappa lap psi, a=-2kappa/xi^2, kappa=0.75 sigma xi; p = rho/3 + psi mu/3; g^eq carries 3 gamma mu; F = mu grad psi; float64; half-way bounce-back walls.',
 'adapt xval:XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION',
 '{"b2_max_rel_mismatch": 9.711053671427633e-15, "b4_all_stable": false, "classic_ux_rel_max": 3.248791434318108, "corrected_mass_flux_rel_max": 5.094356239863502e-15, "corrected_psi_flux_rel_max": 1.2766658370446992e-13, "impose_psi_flux_rel_max": 2120.835130156092}'
);
