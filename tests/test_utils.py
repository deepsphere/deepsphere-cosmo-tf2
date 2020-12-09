import numpy as np
import healpy as hp

from deepsphere import utils


def test_extend_indices():
    # defs
    nside_in = 4
    nside_out = 2

    # create a set of indices
    indices = np.arange(hp.nside2npix(nside_in))[::4]

    # get the expanded set
    new_indices = utils.extend_indices(indices, nside_in=nside_in, nside_out=nside_out)

    assert len(new_indices) == hp.nside2npix(nside_in)

    # this should also work reorderd
    m_nest = np.zeros(hp.nside2npix(nside_in))
    m_nest[::4] = 1.0
    m_ring = hp.reorder(map_in=m_nest, n2r=True)

    # get the indices
    indices = np.arange(hp.nside2npix(nside_in))[m_ring > 0.0]

    # get the expanded set
    new_indices = utils.extend_indices(indices, nside_in=nside_in, nside_out=nside_out, nest=False)

    assert len(new_indices) == hp.nside2npix(nside_in)
