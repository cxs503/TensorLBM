"""Reference arm: MPLBM-UT ``create_geom_edist`` (verbatim, OSS source).

Provenance (zero hand-copy: every line below the header is copied from the
upstream file, only the module-level imports are reduced to what this
function actually uses):

* repo      : ``je-santos/MPLBM-UT``
* file      : ``src/python/mplbm_utils/pore_utils.py``
* function  : ``create_geom_edist(rock, args, nw_fluid_mask)`` (lines 11-68 of
  the extracted snapshot at ``/tmp/mplbm_src/src__python__mplbm_utils__pore_utils.py``)
* header    : ``from scipy.ndimage.morphology import distance_transform_edt as edist``
  (deprecated import path in the upstream file; functionally identical to
  ``scipy.ndimage.distance_transform_edt``)
* call sites: ``create_geom_for_palabos.py:47``, ``create_geom_for_rel_perm.py:48,112,140``

The only adaptation is mechanical: the upstream ``args`` argparse ``Namespace``
is built here from keyword arguments by :func:`reference_tag_geometry`
(``swapXZ``/``scale_2``/``add_mesh``/``num_slices``/``print_size``/``name``),
so the reference can be driven without an input.yml.  The function body is
untouched, including the two subtle behaviours that any reimplementation has
to match:

1. the six domain faces are forced to 1 **before** pores are re-opened, and
2. the split ``(0 < d < 2) -> 1`` followed by ``d > 1 -> 2`` makes the
   boundary segment thicker than the 6-neighbour shell (edge/corner neighbours
   at ``sqrt(2)`` and ``sqrt(3)`` are boundary).

Upstream imports this function needs but never calls (``edt``, ``porespy``,
``skimage``) are omitted; that is the only other deviation and it does not
change behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.ndimage import distance_transform_edt as edist


# --- verbatim upstream body -------------------------------------------------
def create_geom_edist(rock, args, nw_fluid_mask):

    if args.swapXZ:
        rock = rock.transpose([2, 1, 0])

    if args.scale_2:
        NotImplementedError('Feature not yet implemented')

    erock = edist(rock)

    # make sure all the BCs have bounce back nodes
    erock[0, :, :] = 1
    erock[:, 0, :] = 1
    erock[:, :, 0] = 1

    erock[-1, :, :] = 1
    erock[:, -1, :] = 1
    erock[:, :, -1] = 1

    # re open the pores
    erock[rock==0] = 0

    # Get the final matrix [0,1,2]
    erock[(erock>0)*(erock<2)] = 1
    erock[erock>1] = 2


    if args.add_mesh:
        NotImplementedError('Feature not yet implemented')

    if args.num_slices:
        erock = np.pad(erock, [(args.num_slices,args.num_slices), (0,0), (0,0)])

    if args.print_size:
        size = erock.shape
        geom_name = f'{args.name}_{size[0]}_{size[1]}_{size[2]}'
    else:
        geom_name = args.name

    # Save
    erock = erock.astype(np.int16)
    erock[erock == 0] = 2608  # pore space / w fluid
    erock[erock == 1] = 2609  # boundary
    erock[erock == 2] = 2610  # grains
    erock[nw_fluid_mask == 3] = 2611  # add nw fluid back in if needed
    erock = erock.astype(np.int16)

    return erock, geom_name
# --- end verbatim upstream body ---------------------------------------------


def reference_tag_geometry(
    rock: np.ndarray,
    *,
    swap_xz: bool = False,
    num_slices: int = 0,
    name: str = "geom",
    print_size: bool = False,
    nw_fluid_mask: np.ndarray | None = None,
):
    """Drive the verbatim reference with the same knobs as the TensorLBM API."""
    rock = np.asarray(rock)
    args = SimpleNamespace(
        swapXZ=swap_xz,
        scale_2=False,
        add_mesh=False,
        num_slices=num_slices,
        print_size=print_size,
        name=name,
    )
    if nw_fluid_mask is None:
        # upstream always passes an array already in final (swapped/padded) shape
        shape = (rock.shape[2], rock.shape[1], rock.shape[0]) if swap_xz else rock.shape
        shape = (shape[0] + 2 * int(num_slices), shape[1], shape[2])
        nw_fluid_mask = np.zeros(shape, dtype=np.int8)
    return create_geom_edist(rock, args, np.asarray(nw_fluid_mask))