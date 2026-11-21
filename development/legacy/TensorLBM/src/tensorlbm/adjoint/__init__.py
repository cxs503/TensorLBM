"""Deprecated location -- the adjoint kernels now live in the installable tree.

The D2Q9 forward model, the BPTT reference gradient and the out-of-core
checkpointed gradient were moved out of this nested ``TensorLBM/src``
copy into the package that actually ships::

    src/tensorlbm/adjoint_ootc/{lbm2d,bptt,ootc,inversion}.py

so that ``from tensorlbm.adjoint_ootc import ootc_gradient`` works from
the installed library instead of only via a ``sys.path`` insert into this
directory (the importability defect recorded in
``docs/ADJOINT_OOTC_INVERSION_CONVERGENCE.md``).

Nothing is executed here for that reason; import from
``tensorlbm.adjoint_ootc`` instead.  ``README.md`` in this directory is
kept as the method description.
"""

__all__: list[str] = []