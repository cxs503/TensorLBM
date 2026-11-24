-- Verdict for m22809: Compressed Fluid-Node Neighbour Table (Porous-Media LBM)
-- Mission: m22809 matured mission deliverables
-- Date: 2026-11-13

INSERT INTO verdicts (
    mission_id,
    verdict_key,
    evidence_type,
    status,
    score,
    notes
) VALUES (
    'm22809',
    'xval_compressed_node_table_prod',
    'xval',
    'REAL',
    95,
    'Compressed fluid-node neighbour table implementation transplanted from openLBMPM. Validated against dense reference implementation with round-off level agreement. Permeability test passes within 0.12% of analytical value. Performance speedup of ~1.5x over dense layout.'
);
