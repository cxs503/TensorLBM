INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
VALUES (
 'XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S',
 'nebb_force_corrected_poiseuille_2d',
 'closed-form steady plane Poiseuille ux(y)=Gx/(2 nu) y (H-y), H=ny-1 (no-slip planes on the wall rows)',
 'max|got-ref|/max|ref| of the steady ux(y) profile at ny=40, nu=0.2, Gx=1e-3',
 '0.03',
 'TensorLBM',
 'active'
)
ON CONFLICT DO NOTHING;
